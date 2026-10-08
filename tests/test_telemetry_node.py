"""The record-flattening and summary logic behind telemetry_node.

Imported by path rather than as a package: the module lives in the ROS
workspace, and `rclpy` is not importable in CI. Only the parts that do not
touch ROS are exercised here, which is the point of keeping the node thin.
"""

import importlib.util
import os
import sys
import types

import pytest

PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ros2_ws", "src", "edge_perception", "edge_perception", "telemetry_node.py",
)


class _AnyMeta(type):
    """Attribute access on the class yields another stub, so enum members like
    `HistoryPolicy.KEEP_LAST` resolve without enumerating them here."""

    def __getattr__(cls, name):
        return cls()


class _Any(metaclass=_AnyMeta):
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return _Any()


def load():
    """Import the node module with the ROS imports stubbed out."""
    stubs = {
        "rclpy": ["init", "spin", "shutdown", "ok"],
        "rclpy.node": ["Node"],
        "rclpy.qos": [
            "DurabilityPolicy", "HistoryPolicy", "QoSProfile", "ReliabilityPolicy"
        ],
        "std_msgs": [],
        "std_msgs.msg": ["String"],
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        for attr in attrs:
            setattr(module, attr, _Any)
        sys.modules[name] = module
    try:
        spec = importlib.util.spec_from_file_location("telemetry_node", PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


node = load()


RECORD = {
    "frame_id": 41,
    "capture_ts": 100.0,
    "inference_start_ts": 100.02,
    "inference_end_ts": 106.156,
    "publish_ts": 106.157,
    "wall_clock": "2026-10-08T04:35:02.000Z",
    "preprocess_s": 0.004,
    "semantic": {"path_status": "blocked", "obstacle_location": "front_left"},
    "validation": {"ok": True, "failure": None, "extracted": False},
}


def test_durations_are_differences_of_the_published_stamps():
    row = node.row_from_record(RECORD, received_ts=106.2)
    assert row["queue_age"] == pytest.approx(0.02)
    assert row["inference_latency"] == pytest.approx(6.136)
    assert row["post_processing"] == pytest.approx(0.001)
    assert row["result_age"] == pytest.approx(6.157)


def test_the_validator_detail_is_recorded():
    # A run of 96 rejected records with no detail column says only that
    # something failed, which is where the first sustained run landed.
    record = dict(
        RECORD,
        validation={
            "ok": False,
            "failure": "unusable_semantics",
            "detail": "path_status=blocked with obstacle_location=none",
            "extracted": False,
        },
    )
    row = node.row_from_record(record, received_ts=106.2)
    assert row["detail"] == "path_status=blocked with obstacle_location=none"


def test_consumer_age_extends_past_publish_to_this_subscriber():
    # result_age ends at publish inside the inference process. A consumer waits
    # for the hop back too, and the README claims result_age is what a consumer
    # experiences, so the gap is worth a column rather than an assumption.
    row = node.row_from_record(RECORD, received_ts=106.2)
    assert row["consumer_age"] == pytest.approx(6.2)
    assert row["consumer_age"] > row["result_age"]


def test_missing_semantics_do_not_raise():
    # The validator substitutes an unknown-state fallback, and the model can
    # return anything. Neither case may take the node down.
    record = dict(RECORD, semantic={}, validation={})
    row = node.row_from_record(record, received_ts=106.2)
    assert row["path_status"] == ""
    assert row["ok"] is False
    assert row["failure"] == node.NO_FAILURE


def test_a_passing_record_is_not_counted_as_a_failure_kind():
    # ValidationReport spells a pass as failure="none", which is a failure
    # *name*, not an absence of one. Counting non-empty strings reported
    # {"none": 96} for a run in which nothing failed.
    data = rows([6.1] * 3)
    for row in data:
        row["failure"] = node.NO_FAILURE
    summary = node.summarize(data, duration_s=18.0)
    assert summary["failures"] == {}


def test_real_failures_are_still_counted_beside_passing_records():
    data = rows([6.1] * 4)
    data[0]["failure"] = node.NO_FAILURE
    data[1]["failure"] = "unusable_semantics"
    data[1]["ok"] = False
    data[2]["failure"] = node.NO_FAILURE
    data[3]["failure"] = "malformed_json"
    data[3]["ok"] = False
    summary = node.summarize(data, duration_s=24.0)
    assert summary["failures"] == {"unusable_semantics": 1, "malformed_json": 1}
    assert summary["invalid_output_rate"] == pytest.approx(0.5)


def test_semantics_are_coerced_not_trusted():
    record = dict(RECORD, semantic={"path_status": 7, "obstacle_location": None})
    row = node.row_from_record(record, received_ts=106.2)
    assert row["path_status"] == "7"
    assert isinstance(row["obstacle_location"], str)


def test_a_missing_timestamp_is_an_error_not_a_zero():
    # Silently writing 0.0 would put a 100-second result age in the CSV and
    # move the p50 of the run.
    with pytest.raises(KeyError):
        node.row_from_record({"frame_id": 1, "capture_ts": 1.0}, received_ts=2.0)


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def rows(ages, start=0.0, step=6.1):
    out = []
    for i, age in enumerate(ages):
        out.append(
            {
                "received_ts": start + i * step,
                "result_age": age,
                "failure": "",
                "ok": True,
                "queue_age": 0.02,
                "preprocess_s": 0.004,
                "inference_latency": age - 0.021,
                "post_processing": 0.001,
                "consumer_age": age + 0.04,
                "system_available_mb": 450.0,
                "power_w": 15.6,
            }
        )
    return out


def test_first_and_last_minute_are_compared_not_the_whole_run():
    # Ten records at 6.1 s: the first minute is the first ten, the last minute
    # the last ten, and they overlap only if the run is shorter than two.
    early = [6.1] * 10
    late = [7.0] * 10
    summary = node.summarize(rows(early + late), duration_s=122.0)
    assert summary["first_minute_result_age_p50"] == pytest.approx(6.1)
    assert summary["last_minute_result_age_p50"] == pytest.approx(7.0)


def test_p99_is_omitted_below_a_hundred_samples():
    summary = node.summarize(rows([6.1] * 40), duration_s=244.0)
    assert "p99" not in summary["result_age"]
    summary = node.summarize(rows([6.1] * 120), duration_s=732.0)
    assert "p99" in summary["result_age"]


def test_failures_are_counted_by_kind():
    data = rows([6.1] * 4)
    data[0]["failure"] = "unusable_semantics"
    data[0]["ok"] = False
    data[2]["failure"] = "unusable_semantics"
    data[2]["ok"] = False
    summary = node.summarize(data, duration_s=24.0)
    assert summary["failures"] == {"unusable_semantics": 2}
    assert summary["invalid_output_rate"] == pytest.approx(0.5)


def test_an_empty_run_summarises_without_dividing_by_zero():
    summary = node.summarize([], duration_s=5.0)
    assert summary["records"] == 0
    assert summary["invalid_output_rate"] is None
    assert summary["first_minute_result_age_p50"] is None
