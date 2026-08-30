"""End-to-end behaviour: trust boundary, failure handling, overload bounds."""

import time

import pytest

from inference.buffer import BoundedFifo, LatestFrameBuffer, UnboundedFifo
from inference.clock import monotonic
from inference.engine import EngineDied, EngineTimeout, InferenceEngine, MockEngine
from inference.pipeline import Health, Pipeline
from inference.record import Frame, RawModelOutput
from inference.schema import Failure
from telemetry.metrics import Metrics


class ScriptedEngine(InferenceEngine):
    """Returns a fixed sequence of responses (or raises). No timing noise."""

    name = "scripted"

    def __init__(self, responses, latency=0.01):
        self.responses = list(responses)
        self.latency = latency
        self.i = 0

    def describe(self):
        return {"engine": "scripted"}

    def infer(self, image, deadline=None):
        start = monotonic()
        time.sleep(self.latency)
        item = self.responses[min(self.i, len(self.responses) - 1)]
        self.i += 1
        if isinstance(item, Exception):
            raise item
        return RawModelOutput(text=item, inference_start_ts=start, inference_end_ts=monotonic())


def build(engine, policy=None, deadline_s=5.0):
    published = []
    metrics = Metrics("test", sample_resources=False)
    pipe = Pipeline(
        engine=engine,
        policy=policy or LatestFrameBuffer(),
        metrics=metrics,
        consumer=published.append,
        deadline_s=deadline_s,
        watchdog_s=0.2,
    )
    return pipe, metrics, published


def drive(pipe, n, fps=200.0, settle=1.0):
    """Feed n frames at `fps` and wait for the worker to drain."""
    period = 1.0 / fps
    for i in range(n):
        pipe.submit(Frame(frame_id=i, capture_ts=monotonic(), payload=None))
        time.sleep(period)
    time.sleep(settle)


def test_metadata_is_attached_by_the_wrapper_not_the_model():
    """The model emits semantics only; identifiers and clocks are ours."""
    engine = ScriptedEngine(
        # A model trying to assert its own frame_id and timestamp.
        ['{"path_status": "clear", "obstacle_location": "none", '
         '"frame_id": 999999, "capture_ts": 1.0}']
    )
    pipe, _, published = build(engine)
    pipe.start(warmup=False)
    pipe.submit(Frame(frame_id=7, capture_ts=monotonic(), payload=None))
    time.sleep(0.5)
    pipe.stop()

    assert len(published) == 1
    result = published[0]
    assert result.frame_id == 7                     # ours, not the model's
    assert "frame_id" not in result.semantic        # the model's copy never survives
    assert "capture_ts" not in result.semantic
    assert result.validation["extra_keys_stripped"] == ["capture_ts", "frame_id"]


def test_timestamps_are_ordered_and_durations_are_non_negative():
    pipe, _, published = build(ScriptedEngine(['{"path_status": "clear", "obstacle_location": "none"}']))
    pipe.start(warmup=False)
    drive(pipe, 5, fps=20, settle=0.5)
    pipe.stop()

    assert published
    for r in published:
        assert r.capture_ts <= r.inference_start_ts <= r.inference_end_ts <= r.publish_ts
        assert r.queue_age >= 0
        assert r.inference_latency >= 0
        assert r.post_processing >= 0
        # The decomposition has to add up, or the three metrics are not comparable.
        assert r.result_age == pytest.approx(
            r.queue_age + r.inference_latency + r.post_processing, abs=1e-9
        )


@pytest.mark.parametrize(
    "response, expected",
    [
        ("not json", Failure.MALFORMED_JSON),
        ('{"path_status": "sideways", "obstacle_location": "left"}', Failure.SCHEMA_VIOLATION),
        ('{"path_status": "clear", "obstacle_location": "left"}', Failure.UNUSABLE_SEMANTICS),
        (EngineTimeout("deadline"), Failure.INFERENCE_TIMEOUT),
        (EngineDied("cuda oom"), Failure.ENGINE_ERROR),
        (ValueError("engine bug"), Failure.ENGINE_ERROR),
    ],
)
def test_every_failure_still_publishes_an_explicit_unknown(response, expected):
    """A failure is a published outcome, never a silent drop and never a retry."""
    engine = ScriptedEngine([response])
    pipe, metrics, published = build(engine)
    pipe.start(warmup=False)
    pipe.submit(Frame(frame_id=1, capture_ts=monotonic(), payload=None))
    time.sleep(0.5)
    pipe.stop()

    assert len(published) == 1
    result = published[0]
    assert result.validation["failure"] == expected.value
    assert result.validation["ok"] is False
    assert result.semantic == {"path_status": "unknown", "obstacle_location": "unknown"}
    assert result.frame_id == 1  # accounting survives the failure
    assert engine.i <= 1         # exactly one invocation: no retry
    assert metrics.summarize().invalid_output_rate == 1.0


def test_engine_death_flips_health_and_is_not_restarted():
    pipe, _, published = build(ScriptedEngine([EngineDied("gone")]))
    pipe.start(warmup=False)
    pipe.submit(Frame(frame_id=1, capture_ts=monotonic(), payload=None))
    time.sleep(0.4)
    assert pipe.health is Health.ENGINE_DEAD
    pipe.stop()
    assert len(published) == 1


def test_watchdog_reports_a_stall_without_acting_on_it():
    pipe, _, _ = build(ScriptedEngine(['{"path_status": "clear", "obstacle_location": "none"}']))
    pipe.start(warmup=False)
    time.sleep(0.6)  # nothing submitted, watchdog_s is 0.2
    assert pipe.health is Health.STALLED
    pipe.stop()


def test_capture_is_not_blocked_by_a_slow_engine():
    """submit() does bounded work; a 200 ms engine must not stall the producer."""
    pipe, metrics, _ = build(ScriptedEngine(['{"path_status": "clear", "obstacle_location": "none"}'],
                                            latency=0.2))
    pipe.start(warmup=False)
    start = monotonic()
    for i in range(30):
        pipe.submit(Frame(frame_id=i, capture_ts=monotonic(), payload=None))
    elapsed = monotonic() - start
    pipe.stop()

    assert elapsed < 0.05, f"submit() blocked for {elapsed:.3f}s"
    assert metrics.summarize().frames_captured == 30


def test_latest_frame_bounds_result_age_where_fifo_does_not():
    """The headline claim, asserted rather than asserted-about."""
    service = 0.08
    ages = {}
    for policy in (LatestFrameBuffer(), UnboundedFifo()):
        engine = MockEngine(mean_latency=service, sigma=0.01, seed=3)
        pipe, metrics, _ = build(engine, policy=policy)
        pipe.start(warmup=False)
        drive(pipe, 60, fps=100, settle=0.3)
        pipe.stop()
        ages[policy.name] = metrics.summarize().result_age["max"]

    # latest-frame: bounded by ~one service time plus one inter-frame interval
    assert ages["latest"] < 3 * service
    # unbounded FIFO: the backlog is the age
    assert ages["fifo_unbounded"] > 5 * service


def test_bounded_fifo_result_age_is_capped_by_capacity_not_unbounded():
    """The precise version of the claim: a bounded queue does not grow forever.

    The bound to expect is (capacity + 1) x service, not capacity x service: a
    frame admitted to a full queue waits behind `capacity` frames and then pays
    for its own inference. Two extra service times of slack absorb scheduler
    jitter without letting an unbounded-style regression through -- the
    unbounded policy reaches several times this under the same conditions.
    """
    service, capacity = 0.05, 4
    engine = MockEngine(mean_latency=service, sigma=0.01, seed=5)
    pipe, metrics, _ = build(engine, policy=BoundedFifo(capacity))
    pipe.start(warmup=False)
    drive(pipe, 120, fps=100, settle=0.4)
    pipe.stop()

    max_age = metrics.summarize().result_age["max"]
    assert max_age < (capacity + 3) * service  # capped, as the capacity predicts


def test_drop_rate_is_computed_against_frames_actually_captured():
    engine = MockEngine(mean_latency=0.05, sigma=0.01, seed=1)
    pipe, metrics, published = build(engine, policy=LatestFrameBuffer())
    pipe.start(warmup=False)
    drive(pipe, 40, fps=100, settle=0.3)
    pipe.stop()

    s = metrics.summarize()
    assert s.frames_captured == 40
    assert 0.0 < s.drop_rate < 1.0
    assert s.frames_dropped + s.results_published == pytest.approx(s.frames_captured, abs=1)
