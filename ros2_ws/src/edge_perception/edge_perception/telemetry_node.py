"""telemetry_node: persist the published contract to disk as it arrives.

A consumer that records rather than acts. It subscribes to the results topic,
writes one CSV row per record and a summary JSON on an interval, and counts
what it could not parse.

**Reliability is the opposite of the frames topic, deliberately.** `frames`
is BEST_EFFORT because a stale frame has negative value: losing one is the
design. Here a lost record would silently remove a sample from the very
distribution being measured, biasing it toward whatever the middleware happens
to keep. So this subscriber is RELIABLE with a deep queue. Results arrive
every six seconds against a queue of 100, so reliability costs nothing.

**Rows are written and flushed as they arrive.** The measurement this node
exists for is a ten-minute sustained run, and a run that dies at minute nine
should still have nine minutes of data. Buffering in memory until shutdown
would trade the whole measurement against a tidier write path.

**Resource figures are the system-wide ones only.** `system_available_mb` and
power come from `/proc` and `/sys` and are the same from any process on the
board. This node's own RSS is not the inference process's and is not recorded,
because a column named `rss_mb` filled from the wrong process is worse than no
column.

**`semantic` is untrusted.** It is whatever the model produced, and the
validator may have replaced it with the unknown-state fallback. Fields are
read with `.get` and coerced to `str`; nothing here assumes a shape.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from inference.clock import monotonic
from telemetry.metrics import (
    percentile,
    read_jetson_power_w,
    read_system_available_bytes,
)

PARAMETERS = {
    "results_topic": "perception",
    # Relative to the working directory. The absolute path is logged at
    # startup so there is no guessing which one got written.
    "out_dir": "results/sustained",
    "csv_name": "results.csv",
    "summary_name": "summary.json",
    "report_every_s": 30.0,
    # The summary is rewritten on this interval as well as at shutdown, so a
    # kill that skips the shutdown path still leaves a readable summary.
    "summary_every_s": 60.0,
}

# A record every six seconds against a queue of 100. RELIABLE here for the
# reason in the module docstring: a dropped sample biases the distribution.
RESULTS_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=100,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

# `Failure.NONE.value`, spelled out rather than imported so the node keeps no
# dependency on the enum. Counting any non-empty failure string as a failure
# reported {"none": 96} for a run where nothing failed.
NO_FAILURE = "none"

FIELDS = [
    "frame_id",
    "wall_clock",
    "capture_ts",
    "inference_start_ts",
    "inference_end_ts",
    "publish_ts",
    "received_ts",
    "queue_age",
    "preprocess_s",
    "inference_latency",
    # Both inside inference_latency, either side of the CPU/GPU boundary.
    "processor_s",
    "generate_s",
    "post_processing",
    "result_age",
    # capture -> this subscriber. result_age ends at publish inside the
    # inference process; a consumer waits for the hop back as well, and the
    # project claims result_age is "what a consumer experiences". This is the
    # column that checks the claim.
    "consumer_age",
    "ok",
    "failure",
    # Why validation rejected it. Without this the CSV says 96 records failed
    # and gives no way to tell which of the six ways, or on what.
    "detail",
    "extracted",
    "path_status",
    "obstacle_location",
    "system_available_mb",
    "power_w",
]


def _optional(value) -> Optional[float]:
    """A missing split stays missing. A zero would read as "the CPU phase is
    free", which is the opposite of what an absent measurement means."""
    return None if value is None else round(float(value), 6)


def row_from_record(record: dict, received_ts: float) -> dict:
    """Flatten one published record. Trusted timings, untrusted semantics."""
    capture = float(record["capture_ts"])
    start = float(record["inference_start_ts"])
    end = float(record["inference_end_ts"])
    publish = float(record["publish_ts"])
    validation = record.get("validation") or {}
    semantic = record.get("semantic") or {}
    return {
        "frame_id": int(record["frame_id"]),
        "wall_clock": record.get("wall_clock", ""),
        "capture_ts": round(capture, 6),
        "inference_start_ts": round(start, 6),
        "inference_end_ts": round(end, 6),
        "publish_ts": round(publish, 6),
        "received_ts": round(received_ts, 6),
        "queue_age": round(start - capture, 6),
        "preprocess_s": round(float(record.get("preprocess_s", 0.0)), 6),
        "inference_latency": round(end - start, 6),
        "processor_s": _optional(record.get("processor_s")),
        "generate_s": _optional(record.get("generate_s")),
        "post_processing": round(publish - end, 6),
        "result_age": round(publish - capture, 6),
        "consumer_age": round(received_ts - capture, 6),
        "ok": bool(validation.get("ok", False)),
        # The contract's own value, not a re-encoding: a passing record says
        # "none", which is a failure *name*, not an absence of one.
        "failure": str(validation.get("failure") or NO_FAILURE),
        "detail": str(validation.get("detail") or ""),
        "extracted": bool(validation.get("extracted", False)),
        "path_status": str(semantic.get("path_status", "")),
        "obstacle_location": str(semantic.get("obstacle_location", "")),
    }


def window_p50(rows: list, seconds: float, last: bool) -> Optional[float]:
    """result age p50 over the first or last `seconds` of received records.

    The budget's sustained-load row is "p50 within 10% first minute vs last",
    which needs the two ends compared rather than the whole run averaged.
    """
    if not rows:
        return None
    edge = (
        rows[-1]["received_ts"] - seconds if last else rows[0]["received_ts"] + seconds
    )
    ages = [
        r["result_age"]
        for r in rows
        if (r["received_ts"] >= edge if last else r["received_ts"] <= edge)
    ]
    return percentile(ages, 50) if ages else None


DISTRIBUTIONS = (
    "queue_age",
    "preprocess_s",
    "inference_latency",
    "processor_s",
    "generate_s",
    # Both inside inference_latency, either side of the CPU/GPU boundary.
    "processor_s",
    "generate_s",
    "post_processing",
    "result_age",
    "consumer_age",
    "system_available_mb",
    "power_w",
)

# Below this many samples nearest-rank p99 is the maximum, and naming the
# slowest record a tail statistic is how one hiccup becomes the headline.
P99_MIN_SAMPLES = 100


def summarize(rows: list, duration_s: float, unparseable: int = 0) -> dict:
    failures: dict = {}
    for row in rows:
        failure = row["failure"]
        if failure and failure != NO_FAILURE:
            failures[failure] = failures.get(failure, 0) + 1
    summary = {
        "duration_s": round(duration_s, 3),
        "records": len(rows),
        "unparseable": unparseable,
        "failures": failures,
        "invalid_output_rate": (
            round(sum(1 for r in rows if not r["ok"]) / len(rows), 4) if rows else None
        ),
        "first_minute_result_age_p50": window_p50(rows, 60.0, last=False),
        "last_minute_result_age_p50": window_p50(rows, 60.0, last=True),
    }
    for field in DISTRIBUTIONS:
        values = [r[field] for r in rows if r.get(field) is not None]
        if not values:
            continue
        stats = {
            "n": len(values),
            "min": min(values),
            "p50": percentile(values, 50),
            "p90": percentile(values, 90),
            "max": max(values),
        }
        if len(values) >= P99_MIN_SAMPLES:
            stats["p99"] = percentile(values, 99)
        summary[field] = stats
    summary["note"] = (
        "consumer_age is capture -> this subscriber. result_age ends at "
        "publish inside the inference process."
    )
    return summary


class TelemetryNode(Node):
    def __init__(self) -> None:
        super().__init__("telemetry_node")
        for name, default in PARAMETERS.items():
            self.declare_parameter(name, default)
        value = lambda name: self.get_parameter(name).value  # noqa: E731

        self.out_dir = os.path.abspath(value("out_dir"))
        os.makedirs(self.out_dir, exist_ok=True)
        self.csv_path = os.path.join(self.out_dir, value("csv_name"))
        self.summary_path = os.path.join(self.out_dir, value("summary_name"))

        self.records = 0
        self.unparseable = 0
        self.rows: list = []
        self.start_ts = monotonic()

        self._fh = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=FIELDS)
        self._writer.writeheader()
        self._fh.flush()

        self.create_subscription(
            String, value("results_topic"), self.on_result, RESULTS_QOS
        )
        self.create_timer(value("report_every_s"), self.report)
        self.create_timer(value("summary_every_s"), self.write_summary)
        self.get_logger().info(f"writing {self.csv_path} and {self.summary_path}")

    def on_result(self, message: String) -> None:
        received_ts = monotonic()
        try:
            record = json.loads(message.data)
            row = row_from_record(record, received_ts)
        except (ValueError, KeyError, TypeError) as exc:
            # A consumer that crashes on a malformed record is a consumer that
            # loses the rest of the run. Count it and carry on.
            self.unparseable += 1
            self.get_logger().warn(f"unparseable record ({exc}); counted, not fatal")
            return

        # Sampled here rather than on the inference worker, where two
        # syscall-heavy reads between frames land on the next frame's queue age.
        row["system_available_mb"] = round(read_system_available_bytes() / 1e6, 1)
        row["power_w"] = round(read_jetson_power_w(), 2)

        self.records += 1
        self.rows.append(row)
        self._writer.writerow(row)
        self._fh.flush()

    def summarize(self) -> dict:
        return summarize(self.rows, monotonic() - self.start_ts, self.unparseable)

    def write_summary(self) -> None:
        with open(self.summary_path, "w") as fh:
            json.dump(self.summarize(), fh, indent=2, sort_keys=False)
            fh.write("\n")

    def report(self) -> None:
        if not self.records:
            self.get_logger().warn("no records yet -- is inference_node publishing?")
            return
        summary = self.summarize()
        age = summary["result_age"]
        self.get_logger().info(
            f"records={self.records} "
            f"unparseable={self.unparseable} "
            f"result_age p50={age['p50']:.3f}s p90={age['p90']:.3f}s "
            f"invalid={100.0 * summary['invalid_output_rate']:.1f}% "
            f"avail={summary['system_available_mb']['min']:.0f}MB min"
        )

    def destroy_node(self) -> bool:
        self.write_summary()
        self._fh.close()
        self.get_logger().info(
            f"{self.records} records to {self.csv_path} "
            f"({self.unparseable} unparseable)"
        )
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TelemetryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
