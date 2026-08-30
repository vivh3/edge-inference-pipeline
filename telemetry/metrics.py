"""Telemetry: the three latency metrics, the rates, and resource sampling.

Three durations are reported separately because they answer different
questions and an average of the wrong one hides the interesting behaviour:

  queue age        capture -> inference start.  How long a frame waited to be
                   admitted.  This is where an admission policy shows up.
  inference latency inference start -> result complete.  Model execution.
                   Roughly invariant to the admission policy; if it moves
                   when the policy changes, something else is wrong.
  result age       capture -> publish.  The primary metric, because it is the
                   only one a downstream consumer actually experiences.

Post-processing overhead (result complete -> publish) is derived too, so that
"the bottleneck was JSON parsing, not the model" is a conclusion the data can
support rather than one that has to be assumed away.

Two rates:

  drop rate            dropped / captured, computed against frames actually
                       captured.  A cheap USB camera does not deliver a
                       steady 30 fps, and computing against a nominal rate
                       would fabricate or hide drops.
  invalid-output rate  results whose validation failed / results published.
                       Broken out by failure kind, because "8% invalid" and
                       "8% timeouts" call for different fixes.

Queue depth is deliberately not reported: in a correct one-slot latest-value
buffer it is 0 or 1 and carries no information.
"""

from __future__ import annotations

import csv
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from inference.clock import CLOCK_NAME, monotonic, wall_clock_iso
from inference.schema import Failure

__all__ = ["ResultRow", "Summary", "Metrics", "percentile", "read_rss_bytes", "read_jetson_power_w"]


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
    frames_admitted: int
    frames_dropped: int
    results_published: int
    drop_rate: float
    invalid_output_rate: float
    extraction_rate: float
    failures: Dict[str, int]
    capture_fps_measured: float
    publish_fps_measured: float
    queue_age: Dict[str, float]
    inference_latency: Dict[str, float]
    post_processing: Dict[str, float]
    result_age: Dict[str, float]
    peak_rss_mb: float
    mean_power_w: float
    clock: str = CLOCK_NAME
    wall_clock_start: str = ""
    notes: Dict[str, object] = field(default_factory=dict)


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"),
                "p90": float("nan"), "p99": float("nan"), "max": float("nan")}
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


def read_jetson_power_w() -> float:
    """Instantaneous module power in watts from the Jetson INA3221 rails.

    Returns 0.0 off-target.  Path layout differs between JetPack releases, so
    this scans rather than hard-coding an index; the rail actually read is
    recorded in the run notes.
    """
    import glob

    total_mw = 0
    found = False
    for path in glob.glob("/sys/bus/i2c/drivers/ina3221*/*/hwmon/hwmon*/power*_input"):
        try:
            with open(path) as fh:
                total_mw += int(fh.read().strip())
                found = True
        except (OSError, ValueError):
            continue
    return (total_mw / 1000.0) if found else 0.0


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
        self.frames_admitted = 0
        self.frames_dropped = 0
        self.failures: Dict[str, int] = {f.value: 0 for f in Failure}
        self.extracted_count = 0
        self.peak_rss = 0
        self.power_samples: List[float] = []
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

    def on_admission(self, accepted: bool, dropped: bool) -> None:
        with self._lock:
            if accepted:
                self.frames_admitted += 1
            if dropped:
                self.frames_dropped += 1

    # -- publish side ------------------------------------------------------

    def on_publish(self, result) -> ResultRow:
        rss = read_rss_bytes() if self.sample_resources else 0
        power = read_jetson_power_w() if self.sample_resources else 0.0
        v = result.validation
        row = ResultRow(
            frame_id=result.frame_id,
            capture_ts=result.capture_ts,
            inference_start_ts=result.inference_start_ts,
            inference_end_ts=result.inference_end_ts,
            publish_ts=result.publish_ts,
            queue_age=result.queue_age,
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
            admitted = self.frames_admitted
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
            frames_admitted=admitted,
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
