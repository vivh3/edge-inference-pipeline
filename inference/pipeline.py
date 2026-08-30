"""The asynchronous inference stage.

One worker thread, one admission policy, one engine.  Capture runs on its own
thread and never blocks: `submit` does bounded work and returns.  That
separation is the whole reason result age stays bounded -- if capture waited
on inference, the camera's own buffering would become the queue and the
admission policy would have nothing left to decide.

Failure handling is not an error path bolted on the side.  Every one of the
five failure modes in `inference.schema.Failure` produces a published record
with explicit unknown semantics and a `validation` block naming the cause.
The alternatives -- dropping the frame silently, or retrying until the model
says something parseable -- both lie: the first makes a broken model look
like a slow one, and the second hides invalid outputs inside a latency
number that now covers several model invocations.
"""

from __future__ import annotations

import enum
import threading
from typing import Callable, Optional

from .buffer import Admission, AdmissionPolicy
from .clock import monotonic
from .config import DEFAULT_POLICY, GenerationPolicy
from .engine import EngineDied, EngineTimeout, InferenceEngine
from .preprocess import Preprocessor
from .record import Frame, PublishedResult, RawModelOutput, attach_metadata
from .schema import Failure, failure_report, validate

__all__ = ["Health", "Pipeline"]


class Health(str, enum.Enum):
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"    # publishing, but recent outputs are failing validation
    STALLED = "stalled"      # nothing published within the watchdog interval
    ENGINE_DEAD = "engine_dead"


class Pipeline:
    """Admission -> preprocess -> inference -> validation -> publish.

    Parameters
    ----------
    deadline_s
        Per-frame inference budget.  Exceeding it is `INFERENCE_TIMEOUT`, a
        reported outcome, not a retry.  Set it from measured baseline latency
        (Gate 1), not from a wished-for number.
    watchdog_s
        If nothing is published for this long the health state goes STALLED.
        The watchdog observes and reports; it does not restart anything.
        Automatic restart is out of scope and would obscure exactly the
        failures this project exists to make visible.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        policy: AdmissionPolicy,
        metrics,
        consumer: Callable[[PublishedResult], None],
        preprocessor: Optional[Preprocessor] = None,
        generation: GenerationPolicy = DEFAULT_POLICY,
        deadline_s: float = 5.0,
        watchdog_s: float = 10.0,
        degraded_window: int = 20,
        degraded_threshold: float = 0.25,
    ) -> None:
        self.engine = engine
        self.policy = policy
        self.metrics = metrics
        self.consumer = consumer
        self.preprocessor = preprocessor or Preprocessor(generation)
        self.generation = generation
        self.deadline_s = deadline_s
        self.watchdog_s = watchdog_s
        self.degraded_window = degraded_window
        self.degraded_threshold = degraded_threshold

        self.health = Health.STARTING
        self.last_publish_ts = monotonic()
        self.preprocess_total = 0.0
        self._recent_ok: list[bool] = []
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._watchdog: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # -- producer entry point ---------------------------------------------

    def submit(self, frame: Frame) -> None:
        """Called from the capture thread. Bounded work only; never blocks."""
        self.metrics.on_capture(frame.capture_ts)
        outcome = self.policy.offer(frame)
        self.metrics.on_admission(
            accepted=outcome.admission is not Admission.DROPPED_INCOMING,
            dropped=outcome.dropped,
        )

    # -- lifecycle ---------------------------------------------------------

    def start(self, warmup: bool = True) -> None:
        if warmup:
            # Discarded before any measurement: the first invocations pay for
            # CUDA context creation and allocator growth, which describe
            # startup rather than steady-state service time.
            self.engine.warmup(self.generation.warmup_runs)
        self.health = Health.HEALTHY
        self.last_publish_ts = monotonic()
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        self._watchdog = threading.Thread(target=self._watch, daemon=True)
        self._watchdog.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self.policy.close()
        for t in (self._worker, self._watchdog):
            if t is not None:
                t.join(timeout)

    # -- worker ------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            frame = self.policy.take(timeout=0.1)
            if frame is None:
                continue
            self._process(frame)

    def _process(self, frame: Frame) -> None:
        start = monotonic()
        deadline = start + self.deadline_s

        pre = self.preprocessor.run(frame.payload)
        self.preprocess_total += pre.duration

        try:
            raw = self.engine.infer(pre.image, deadline=deadline)
        except EngineTimeout as exc:
            raw = RawModelOutput("", start, monotonic(), engine_error=str(exc))
            semantic, report = failure_report(Failure.INFERENCE_TIMEOUT, str(exc))
        except EngineDied as exc:
            raw = RawModelOutput("", start, monotonic(), engine_error=str(exc))
            semantic, report = failure_report(Failure.ENGINE_ERROR, str(exc))
            with self._lock:
                self.health = Health.ENGINE_DEAD
        except Exception as exc:  # an engine bug is still an engine failure
            raw = RawModelOutput("", start, monotonic(), engine_error=repr(exc))
            semantic, report = failure_report(Failure.ENGINE_ERROR, repr(exc))
            with self._lock:
                self.health = Health.ENGINE_DEAD
        else:
            # `raw.text` is untrusted until this call returns.
            semantic, report = validate(raw.text)

        # Trusted metadata is attached here and only here.
        result = attach_metadata(frame, raw, semantic, report)
        self.metrics.on_publish(result)
        self._note_outcome(report.failure is Failure.NONE, result.publish_ts)
        self.consumer(result)

    def _note_outcome(self, ok: bool, publish_ts: float) -> None:
        with self._lock:
            self.last_publish_ts = publish_ts
            self._recent_ok.append(ok)
            if len(self._recent_ok) > self.degraded_window:
                self._recent_ok.pop(0)
            if self.health is Health.ENGINE_DEAD:
                return
            if len(self._recent_ok) == self.degraded_window:
                bad = self._recent_ok.count(False) / self.degraded_window
                self.health = Health.DEGRADED if bad >= self.degraded_threshold else Health.HEALTHY
            elif self.health is Health.STALLED:
                self.health = Health.HEALTHY

    # -- watchdog ----------------------------------------------------------

    def _watch(self) -> None:
        while not self._stop.wait(min(1.0, self.watchdog_s / 2)):
            with self._lock:
                if self.health is Health.ENGINE_DEAD:
                    continue
                if monotonic() - self.last_publish_ts > self.watchdog_s:
                    self.health = Health.STALLED

    # -- telemetry ---------------------------------------------------------

    def health_snapshot(self) -> dict:
        with self._lock:
            return {
                "health": self.health.value,
                "seconds_since_publish": monotonic() - self.last_publish_ts,
                "admission_policy": self.policy.name,
                "deadline_s": self.deadline_s,
                "preprocess_total_s": self.preprocess_total,
            }
