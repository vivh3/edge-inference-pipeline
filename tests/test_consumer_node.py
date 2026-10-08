"""The decision table behind consumer_node.

The node is a thin wrapper; `decide` is the whole of it, and it takes a record
and a clock reading rather than any ROS object, so every case is a direct call.
"""

import importlib.util
import os
import sys
import types

import pytest

PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ros2_ws", "src", "edge_perception", "edge_perception", "consumer_node.py",
)


class _AnyMeta(type):
    def __getattr__(cls, name):
        return cls()


class _Any(metaclass=_AnyMeta):
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return _Any()


def load():
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
        spec = importlib.util.spec_from_file_location("consumer_node", PATH)
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

STALE, NO_DATA_AFTER = 13.0, 30.0


def record(capture_ts=100.0, ok=True, failure=None, status="clear", where="none"):
    return {
        "frame_id": 7,
        "capture_ts": capture_ts,
        "validation": {"ok": ok, "failure": failure, "extracted": False},
        "semantic": {"path_status": status, "obstacle_location": where},
    }


def call(rec, now):
    return node.decide(rec, now, STALE, NO_DATA_AFTER)


def test_a_fresh_clear_record_proceeds():
    assert call(record(), 106.5)[0] == node.PROCEED


def test_an_obstacle_holds():
    decision, reason = call(record(status="blocked", where="front_left"), 106.5)
    assert decision == node.HOLD
    assert "front_left" in reason


def test_unusable_semantics_hold():
    decision, reason = call(
        record(ok=False, failure="unusable_semantics", status="unknown", where="unknown"),
        106.5,
    )
    assert decision == node.HOLD
    assert "unusable_semantics" in reason


def test_nothing_received_is_no_data_not_proceed():
    assert call(None, 106.5)[0] == node.NO_DATA


# --------------------------------------------------------------------------
# Ageing. The consumer decides at 10 Hz against a result every 6.4 s, so most
# ticks have no new information and the only question is how old the newest
# record is.
# --------------------------------------------------------------------------


def test_the_same_record_goes_stale_with_no_new_input():
    rec = record()
    assert call(rec, 106.5)[0] == node.PROCEED        # 6.5 s old
    assert call(rec, 120.0)[0] == node.HOLD           # 20 s old
    assert call(rec, 140.0)[0] == node.NO_DATA        # 40 s old


def test_stale_beats_valid_semantics():
    # A clear path reported twenty seconds ago is not a clear path now.
    decision, reason = call(record(), 125.0)
    assert decision == node.HOLD
    assert "not a current view" in reason


def test_staleness_is_checked_before_the_model_output():
    # Age comes from capture_ts, which no model output ever touched, so the
    # check still works when everything downstream of the camera is suspect.
    decision, reason = call(record(ok=False, failure="malformed_json"), 125.0)
    assert decision == node.HOLD
    assert "not a current view" in reason
    assert "malformed_json" not in reason


def test_no_data_and_hold_are_distinguished():
    # Different causes: hold means the system told us something, no_data means
    # it has stopped telling us anything.
    assert call(record(status="blocked", where="left"), 106.5)[0] == node.HOLD
    assert call(record(), 200.0)[0] == node.NO_DATA


# --------------------------------------------------------------------------
# Not trusting upstream
# --------------------------------------------------------------------------


def test_a_contradiction_the_validator_missed_still_holds():
    # `clear` with an obstacle localised is the exact shape the validator
    # rejects. A consumer that assumes its upstream is correct fails when it
    # is not, so this is handled here too.
    decision, reason = call(record(status="clear", where="front_left"), 106.5)
    assert decision == node.HOLD
    assert "contradictory" in reason


def test_missing_semantics_hold_rather_than_proceed():
    rec = record()
    rec["semantic"] = {}
    assert call(rec, 106.5)[0] == node.HOLD


def test_missing_validation_holds_rather_than_proceeding():
    # Absent means unproven, not fine.
    rec = record()
    rec["validation"] = {}
    assert call(rec, 106.5)[0] == node.HOLD


def test_every_decision_carries_a_reason():
    for rec, now in (
        (None, 100.0),
        (record(), 106.5),
        (record(), 125.0),
        (record(), 200.0),
        (record(status="blocked", where="left"), 106.5),
        (record(ok=False, failure="engine_error"), 106.5),
    ):
        decision, reason = call(rec, now)
        assert decision in node.DECISIONS
        assert reason and isinstance(reason, str)
