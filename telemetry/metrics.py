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

Frames admitted is not reported. Under latest-frame a frame can be admitted
and then evicted before it runs, so the count means something different per
policy and is not comparable across them. Captured, dropped and published are
unambiguous, and `dropped + published` accounts for every captured frame bar
the one in flight.

Queue depth is not reported either: in a one-slot buffer it is 0 or 1 and
carries no information.
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
    "read_rss_bytes", "read_system_available_bytes",
    "read_jetson_power_w", "jetson_power_rail_names",
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
    # On a Jetson this is the number that matters, not rss_mb. See
    # read_system_available_bytes.
    system_available_mb: float = 0.0
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
    min_system_available_mb: Optional[float]
    mean_power_w: float
    clock: str = CLOCK_NAME
    wall_clock_start: str = ""
    notes: Dict[str, object] = field(default_factory=dict)


# Below this many samples, nearest-rank p99 picks rank ceil(0.99n) == n, which
# is the maximum. Reporting it as "p99" names the single slowest result a tail
# statistic, and one scheduler hiccup becomes the headline. Under heavy
# overload a run publishes few results by design, so this is the normal case
# here rather than an edge one.
_P99_MIN_SAMPLES = 100


def _stats(values: List[float]) -> Dict[str, Optional[float]]:
    # None, not NaN: bare NaN is not valid JSON and json.dump emits it anyway.
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "max": None}
    out = {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "max": max(values),
    }
    if len(values) >= _P99_MIN_SAMPLES:
        out["p99"] = percentile(values, 99)
    return out


# --------------------------------------------------------------------------
# Resource sampling
# --------------------------------------------------------------------------


def read_rss_bytes() -> int:
    """Resident set size of this process, from /proc. 0 if unavailable.

    RSS badly understates footprint on a Jetson. CUDA allocations live in the
    pool the CPU shares but are not mapped into the process, so they never
    appear here. Loading a 2.2B model at float16 moved RSS by 1.3 GB while
    system available memory fell by 6.4 GB. Read this next to
    read_system_available_bytes, and quote that one for headroom.
    """
    try:
        with open(f"/proc/{os.getpid()}/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def read_system_available_bytes() -> int:
    """MemAvailable from /proc/meminfo. 0 if unavailable.

    MemAvailable, not MemFree: reading several GB of weights off disk fills
    the page cache, which is not free but is reclaimable, and MemFree would
    report a shortage that does not exist.

    This is the headroom figure on a board whose GPU and CPU share one pool.
    It counts the model's device memory, which process RSS does not.
    """
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


# The Jetson's INA3221 monitor is exposed through hwmon, which reports bus
# voltage (mV) and current (mA) on separate channels and leaves the
# multiplication to the caller. There is no power channel to read.
_HWMON_ROOT = "/sys/class/hwmon"
_INA3221_NAME = "ina3221"

# The rails overlap. On an Orin Nano, VDD_CPU_GPU_CV and VDD_SOC sit
# downstream of VDD_IN. Summing all three counts the same current twice;
# summing the children alone read 2.5 W against an actual 4.4 W at idle. So
# the input rail is reported by itself, and the sum is a fallback for boards
# that expose no input rail.
_INPUT_RAIL_LABELS = ("vdd_in", "vdd_sys", "pom_5v_in")


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _read_int(path: str) -> Optional[int]:
    try:
        return int(_read_text(path))
    except (TypeError, ValueError):
        return None


def _power_rails(root: str = _HWMON_ROOT) -> List[Tuple[str, float]]:
    """(label, watts) per INA3221 channel. Empty off-target."""
    import glob

    rails = []
    for hwmon in sorted(glob.glob(os.path.join(root, "hwmon*"))):
        if _read_text(os.path.join(hwmon, "name")) != _INA3221_NAME:
            continue
        for label_path in sorted(glob.glob(os.path.join(hwmon, "in*_label"))):
            channel = os.path.basename(label_path)[len("in") : -len("_label")]
            millivolts = _read_int(os.path.join(hwmon, "in%s_input" % channel))
            milliamps = _read_int(os.path.join(hwmon, "curr%s_input" % channel))
            if millivolts is None or milliamps is None:
                # e.g. "sum of shunt voltages", a voltage channel with no
                # current channel. Not a rail, so not a power reading.
                continue
            label = _read_text(label_path) or os.path.basename(label_path)
            rails.append((label, millivolts * milliamps / 1_000_000.0))
    return rails


def _reported_rails(root: str = _HWMON_ROOT) -> List[Tuple[str, float]]:
    """The rails that make up the reported figure: the input rail, or all."""
    rails = _power_rails(root)
    for rail in rails:
        if rail[0].lower() in _INPUT_RAIL_LABELS:
            return [rail]
    return rails


def read_jetson_power_w() -> float:
    """Total module power in watts. 0.0 off-target."""
    return sum(watts for _, watts in _reported_rails())


def jetson_power_rail_names() -> List[str]:
    """Labels of the rails behind the figure. Recorded once in the run notes."""
    return [label for label, _ in _reported_rails()]


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
        # syscall-heavy reads on the worker thread between frames, which show
        # up as queue age on the next one. Sample on an interval instead.
        self.sample_interval = 1.0
        self._last_sample_ts = -1e9
        self._last_rss = 0
        self._last_available = 0
        self._last_power = 0.0
        self.min_available = 0
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

    def _sample_resources(self) -> Tuple[int, int, float]:
        if not self.sample_resources:
            return 0, 0, 0.0
        now = monotonic()
        if now - self._last_sample_ts >= self.sample_interval:
            self._last_sample_ts = now
            self._last_rss = read_rss_bytes()
            self._last_available = read_system_available_bytes()
            self._last_power = read_jetson_power_w()
        return self._last_rss, self._last_available, self._last_power

    def on_publish(self, result) -> ResultRow:
        rss, available, power = self._sample_resources()
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
            system_available_mb=available / 1e6,
            power_w=power,
        )
        with self._lock:
            self.rows.append(row)
            self.failures[row.failure] = self.failures.get(row.failure, 0) + 1
            if row.extracted:
                self.extracted_count += 1
            self.peak_rss = max(self.peak_rss, rss)
            if available:
                self.min_available = (
                    available if not self.min_available else min(self.min_available, available)
                )
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
            min_available = self.min_available
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
            # The low-water mark, because headroom is about the worst moment.
            min_system_available_mb=(min_available / 1e6) if min_available else None,
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
