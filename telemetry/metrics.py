"""Telemetry: the latency metrics, the rates, and resource sampling.

Durations are reported separately because an average of the wrong one hides
the behaviour that matters:

  queue age         capture -> inference start. Waiting, plus this frame's
                    preprocessing. Where the admission policy shows up.
  preprocess        broken out of queue age so it stays attributable.
  inference latency inference start -> result complete. Model execution.
                    Should be roughly invariant to the admission policy.
  post-processing   result complete -> publish. Validation and serialisation,
                    kept separate so "the bottleneck was JSON parsing, not the
                    model" is a conclusion the data can support.
  result age        capture -> publish. Primary: the only one a consumer
                    actually experiences.

Rates: drop rate is dropped / captured, against frames that actually arrived
rather than a nominal 30 fps. Invalid-output rate is broken out by failure
kind, because "8% invalid" and "8% timeouts" need different fixes.

Frames admitted is deliberately not reported. Under latest-frame a frame can
be admitted and then evicted before it runs, so the count means something
different per policy and is not comparable across the policies being
compared. Captured, dropped and published are unambiguous, and
`dropped + published` accounts for every captured frame bar the one in
flight.

Queue depth is not reported either: in a correct one-slot buffer it is 0 or 1
and carries no information.
"""

from __future__ import annotations

import csv
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from inference.clock import CLOCK_NAME, monotonic, wall_clock_iso
from inference.schema import Failure

__all__ = [
    "ResultRow", "Summary", "Metrics", "percentile",
    "read_rss_bytes", "read_jetson_power_w", "jetson_power_rail_names",
]


def percentile(values: List[float], q: float) -> float:
    """Nearest-rank percentile. No numpy: the runtime core stays stdlib-only."""
    if not values:
        return float("nan")
    import math

    ordered = sorted(values)
    rank = math.ceil(q / 100.0 * len(ordered))
    rank = max(1, min(len(ordered), rank))
    return ordered[rank - 1]


@dataclass
class ResultRow:
    """One published result, flattened for CSV. Durations in seconds."""

    frame_id: int
    capture_ts: float
    inference_start_ts: float
    inference_end_ts: float
    publish_ts: float
    queue_age: float
    preprocess_s: float
    inference_latency: float
    post_processing: float
    result_age: float
    ok: bool
    failure: str
    extracted: bool
    path_status: str
    obstacle_location: str
    rss_mb: float = 0.0
    power_w: float = 0.0


@dataclass
class Summary:
    policy: str
    duration_s: float
    frames_captured: int
    frames_dropped: int
    results_published: int
    drop_rate: float
    invalid_output_rate: float
    extraction_rate: float
    failures: Dict[str, int]
    capture_fps_measured: float
    publish_fps_measured: float
    queue_age: Dict[str, Optional[float]]
    preprocess: Dict[str, Optional[float]]
    inference_latency: Dict[str, Optional[float]]
    post_processing: Dict[str, Optional[float]]
    result_age: Dict[str, Optional[float]]
    peak_rss_mb: float
    mean_power_w: float
    clock: str = CLOCK_NAME
    wall_clock_start: str = ""
    notes: Dict[str, object] = field(default_factory=dict)


def _stats(values: List[float]) -> Dict[str, Optional[float]]:
    # None, not NaN: bare NaN is not valid JSON and json.dump emits it anyway.
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "p99": None, "max": None}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "max": max(values),
    }


# --------------------------------------------------------------------------
# Resource sampling
# --------------------------------------------------------------------------


def read_rss_bytes() -> int:
    """Resident set size of this process, from /proc. 0 if unavailable.

    Deliberately process RSS and not "GPU memory used": on a Jetson the GPU
    and CPU share one physical pool, so a separate device-memory figure would
    invite double counting.  The headroom number that matters is total system
    memory, sampled alongside this (see docs/SETUP-jetson.md).
    """
    try:
        with open(f"/proc/{os.getpid()}/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


# INA3221 exposes per-rail channels alongside an aggregate. Summing all of
# them double-counts the total against its own constituents, so aggregates are
# skipped by label.
_AGGREGATE_RAIL = ("total", "sum", "all")
_POWER_GLOB = "/sys/bus/i2c/drivers/ina3221*/*/hwmon/hwmon*/power*_input"


def _power_rails() -> List[Tuple[str, float]]:
    """(label, watts) per Jetson power rail, aggregates excluded. Empty off-target."""
    import glob
    import os

    rails = []
    for path in sorted(glob.glob(_POWER_GLOB)):
        label_path = path.replace("_input", "_label")
        try:
            with open(path) as fh:
                watts = int(fh.read().strip()) / 1000.0
        except (OSError, ValueError):
            continue
        label = ""
        if os.path.exists(label_path):
            try:
                with open(label_path) as fh:
                    label = fh.read().strip()
            except OSError:
                pass
        if any(word in label.lower() for word in _AGGREGATE_RAIL):
            continue
        rails.append((label or os.path.basename(path), watts))
    return rails


def read_jetson_power_w() -> float:
    """Module power in watts, summed over the non-aggregate rails. 0.0 off-target."""
    return sum(watts for _, watts in _power_rails())


def jetson_power_rail_names() -> List[str]:
    """Labels of the rails being summed. Recorded once in the run notes."""
    return [label for label, _ in _power_rails()]


# --------------------------------------------------------------------------
# Collector
# --------------------------------------------------------------------------


class Metrics:
    """Thread-safe collector. Capture and inference report from different threads."""

    def __init__(self, policy_name: str, sample_resources: bool = True) -> None:
        self.policy_name = policy_name
        self.sample_resources = sample_resources
        self._lock = threading.Lock()
        self.rows: List[ResultRow] = []
        self.frames_captured = 0
        self.frames_dropped = 0
        self.failures: Dict[str, int] = {f.value: 0 for f in Failure}
        self.extracted_count = 0
        self.peak_rss = 0
        self.power_samples: List[float] = []
        # Resource reads touch /proc and /sys. Doing that per result puts two
        # syscall-heavy reads on the worker thread between frames, which shows
        # up as queue age on the next one. Sample on an interval instead and
        # reuse the last value in between.
        self.sample_interval = 1.0
        self._last_sample_ts = -1e9
        self._last_rss = 0
        self._last_power = 0.0
        self.start_ts = monotonic()
        self.start_wall = wall_clock_iso()
        self.first_capture_ts: Optional[float] = None
        self.last_capture_ts: Optional[float] = None
        self.capture_intervals: List[float] = []

    # -- capture side ------------------------------------------------------

    def on_capture(self, capture_ts: float) -> None:
        with self._lock:
            self.frames_captured += 1
            if self.first_capture_ts is None:
                self.first_capture_ts = capture_ts
            else:
                self.capture_intervals.append(capture_ts - self.last_capture_ts)
            self.last_capture_ts = capture_ts

    def on_drop(self) -> None:
        """One captured frame will never reach inference."""
        with self._lock:
            self.frames_dropped += 1

    # -- publish side ------------------------------------------------------

    def _sample_resources(self) -> Tuple[int, float]:
        if not self.sample_resources:
            return 0, 0.0
        now = monotonic()
        if now - self._last_sample_ts >= self.sample_interval:
            self._last_sample_ts = now
            self._last_rss = read_rss_bytes()
            self._last_power = read_jetson_power_w()
        return self._last_rss, self._last_power

    def on_publish(self, result) -> ResultRow:
        rss, power = self._sample_resources()
        v = result.validation
        row = ResultRow(
            frame_id=result.frame_id,
            capture_ts=result.capture_ts,
            inference_start_ts=result.inference_start_ts,
            inference_end_ts=result.inference_end_ts,
            publish_ts=result.publish_ts,
            queue_age=result.queue_age,
            preprocess_s=result.preprocess_s,
            inference_latency=result.inference_latency,
            post_processing=result.post_processing,
            result_age=result.result_age,
            ok=bool(v.get("ok", False)),
            failure=str(v.get("failure", Failure.NONE.value)),
            extracted=bool(v.get("extracted", False)),
            path_status=result.semantic.get("path_status", "unknown"),
            obstacle_location=result.semantic.get("obstacle_location", "unknown"),
            rss_mb=rss / 1e6,
            power_w=power,
        )
        with self._lock:
            self.rows.append(row)
            self.failures[row.failure] = self.failures.get(row.failure, 0) + 1
            if row.extracted:
                self.extracted_count += 1
            self.peak_rss = max(self.peak_rss, rss)
            if power:
                self.power_samples.append(power)
        return row

    # -- reporting ---------------------------------------------------------

    def summarize(self, notes: Optional[dict] = None) -> Summary:
        with self._lock:
            rows = list(self.rows)
            captured = self.frames_captured
            dropped = self.frames_dropped
            failures = dict(self.failures)
            extracted = self.extracted_count
            intervals = list(self.capture_intervals)
            peak_rss = self.peak_rss
            power = list(self.power_samples)
            first_cap = self.first_capture_ts
            last_cap = self.last_capture_ts

        duration = monotonic() - self.start_ts
        published = len(rows)
        invalid = sum(1 for r in rows if not r.ok)
        capture_span = (last_cap - first_cap) if (first_cap is not None and last_cap and captured > 1) else 0.0

        return Summary(
            policy=self.policy_name,
            duration_s=duration,
            frames_captured=captured,
            frames_dropped=dropped,
            results_published=published,
            # Against frames actually captured, never against a nominal rate.
            drop_rate=(dropped / captured) if captured else 0.0,
            invalid_output_rate=(invalid / published) if published else 0.0,
            extraction_rate=(extracted / published) if published else 0.0,
            failures={k: v for k, v in failures.items() if v},
            capture_fps_measured=((captured - 1) / capture_span) if capture_span > 0 else 0.0,
            publish_fps_measured=(published / duration) if duration > 0 else 0.0,
            queue_age=_stats([r.queue_age for r in rows]),
            preprocess=_stats([r.preprocess_s for r in rows]),
            inference_latency=_stats([r.inference_latency for r in rows]),
            post_processing=_stats([r.post_processing for r in rows]),
            result_age=_stats([r.result_age for r in rows]),
            peak_rss_mb=peak_rss / 1e6,
            mean_power_w=(sum(power) / len(power)) if power else 0.0,
            wall_clock_start=self.start_wall,
            notes=dict(notes or {}),
        )

    def inter_frame_intervals(self) -> List[float]:
        """Measured camera arrival intervals. Nominal 30 fps is a claim, not data."""
        with self._lock:
            return list(self.capture_intervals)

    # -- persistence -------------------------------------------------------

    def write_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with self._lock:
            rows = list(self.rows)
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(ResultRow.__annotations__))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

    def write_summary(self, path: str, notes: Optional[dict] = None) -> Summary:
        summary = self.summarize(notes)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(summary), fh, indent=2, sort_keys=False)
            fh.write("\n")
        return summary
