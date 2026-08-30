"""Admission policies. The behavioural difference between them is the project."""

from inference.buffer import (
    Admission,
    BoundedFifo,
    LatestFrameBuffer,
    UnboundedFifo,
    make_policy,
)
from inference.record import Frame


def frame(i, ts=None):
    return Frame(frame_id=i, capture_ts=float(i) / 30.0 if ts is None else ts, payload=None)


def test_latest_frame_keeps_the_newest_and_reports_what_it_dropped():
    buf = LatestFrameBuffer()
    assert buf.offer(frame(0)).admission is Admission.ACCEPTED

    outcome = buf.offer(frame(1))
    assert outcome.admission is Admission.DROPPED_STALE
    assert outcome.dropped_frame_id == 0  # the older frame, not the new one

    outcome = buf.offer(frame(2))
    assert outcome.dropped_frame_id == 1
    assert buf.depth() == 1
    assert buf.take(timeout=0).frame_id == 2


def test_latest_frame_never_grows():
    buf = LatestFrameBuffer()
    for i in range(1000):
        buf.offer(frame(i))
    assert buf.depth() == 1


def test_bounded_fifo_tail_drops_and_serves_the_oldest():
    buf = BoundedFifo(capacity=3)
    for i in range(3):
        assert buf.offer(frame(i)).admission is Admission.ACCEPTED

    outcome = buf.offer(frame(3))
    assert outcome.admission is Admission.DROPPED_INCOMING
    assert outcome.dropped_frame_id == 3  # the new frame is refused, not an old one

    assert buf.depth() == 3
    assert [buf.take(timeout=0).frame_id for _ in range(3)] == [0, 1, 2]


def test_unbounded_fifo_never_drops():
    buf = UnboundedFifo()
    for i in range(500):
        assert buf.offer(frame(i)).admission is Admission.ACCEPTED
    assert buf.depth() == 500


def test_take_returns_none_when_empty():
    assert LatestFrameBuffer().take(timeout=0.01) is None


def test_make_policy_round_trip():
    for name in ("latest", "fifo_bounded", "fifo_unbounded"):
        assert make_policy(name).name == name
