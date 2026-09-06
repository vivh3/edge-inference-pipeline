"""Admission policies: what happens to a frame between capture and inference.

Deliberately *admission and drop policy*, not backpressure. Backpressure
slows the producer; a camera cannot be slowed, so the only lever is deciding
which frames to admit and which to discard. Saying "backpressure" would name a
mechanism this system does not have.

Three policies, so the design choice can be defended with measurements rather
than assertion:

  LatestFrameBuffer   capacity 1, newest wins. The design choice.
  BoundedFifo(n)      classic tail-drop queue. The realistic alternative.
  UnboundedFifo       never drops. A deliberately pathological baseline.

One shared interface, so an experiment swaps between them with nothing else
changing.
"""

from __future__ import annotations

import enum
import threading
from collections import deque
from typing import Deque, List, Optional

from .record import Frame

__all__ = [
    "Admission",
    "OfferOutcome",
    "AdmissionPolicy",
    "LatestFrameBuffer",
    "BoundedFifo",
    "UnboundedFifo",
    "make_policy",
    "POLICY_NAMES",
]


class Admission(str, enum.Enum):
    ACCEPTED = "accepted"
    DROPPED_INCOMING = "dropped_incoming"  # queue full, the new frame is discarded
    DROPPED_STALE = "dropped_stale"        # a pending older frame was evicted


class OfferOutcome:
    """Result of offering one frame, including which frame (if any) was lost."""

    __slots__ = ("admission", "dropped_frame_id", "dropped_age")

    def __init__(
        self,
        admission: Admission,
        dropped_frame_id: Optional[int] = None,
        dropped_age: float = 0.0,
    ):
        self.admission = admission
        self.dropped_frame_id = dropped_frame_id
        self.dropped_age = dropped_age

    @property
    def dropped(self) -> bool:
        return self.admission is not Admission.ACCEPTED or self.dropped_frame_id is not None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"OfferOutcome({self.admission.value}, dropped={self.dropped_frame_id})"


class AdmissionPolicy:
    """Base class. Thread-safe: capture and inference run on different threads."""

    name = "base"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._q: Deque[Frame] = deque()
        self._closed = False

    # -- producer side ----------------------------------------------------

    def offer(self, frame: Frame) -> OfferOutcome:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- consumer side ----------------------------------------------------

    def take(self, timeout: float = 0.5) -> Optional[Frame]:
        """Block until a frame is available. Returns None on timeout or close."""
        with self._not_empty:
            if not self._q and not self._closed:
                self._not_empty.wait(timeout)
            if not self._q:
                return None
            return self._q.popleft()

    def close(self) -> None:
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()

    def depth(self) -> int:
        """Present for the FIFO baselines only.

        Deliberately not a reported metric: in a correct one-slot
        latest-value buffer it is 0 or 1 and says nothing.  Queue *age* is
        the informative quantity, and it is measured on the record itself.
        """
        with self._lock:
            return len(self._q)

    def drain(self) -> List[Frame]:
        with self._lock:
            items = list(self._q)
            self._q.clear()
            return items


class LatestFrameBuffer(AdmissionPolicy):
    """Capacity-1 overwrite buffer. A newer frame always displaces an older one.

    A stale frame has negative value here: the consumer cannot tell a
    3-second-old view of the world from a current one by reading the
    semantics, so a confidently wrong old answer is worse than none. When a
    new frame arrives and inference is busy, the pending one is discarded.

    The cost, stated plainly: result age stays near one inference service time
    plus one inter-frame interval, and drop rate rises with the overload
    ratio. Drops are the design working, which is why drop rate is a headline
    number rather than a buried one.
    """

    name = "latest"

    def offer(self, frame: Frame) -> OfferOutcome:
        with self._not_empty:
            evicted = self._q.popleft() if self._q else None
            self._q.append(frame)
            self._not_empty.notify()
        if evicted is not None:
            return OfferOutcome(
                Admission.DROPPED_STALE,
                evicted.frame_id,
                frame.capture_ts - evicted.capture_ts,
            )
        return OfferOutcome(Admission.ACCEPTED)


class BoundedFifo(AdmissionPolicy):
    """Tail-drop FIFO: when full, the *incoming* frame is discarded.

    The honest comparison for latest-frame. It drops too, so result age does
    not grow without bound -- it saturates near (capacity + 1) x service time,
    since a frame admitted to a full queue waits behind `capacity` others and
    then pays for its own inference. That ceiling is the point: it makes the
    difference between the policies a matter of *which* frames survive, not
    whether anything is dropped.

    Dropping the oldest instead converges on latest-frame as capacity falls to
    1; not implemented, because it adds a variant without adding an argument.
    """

    name = "fifo_bounded"

    def __init__(self, capacity: int = 8) -> None:
        super().__init__()
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity

    def offer(self, frame: Frame) -> OfferOutcome:
        with self._not_empty:
            if len(self._q) >= self.capacity:
                return OfferOutcome(Admission.DROPPED_INCOMING, frame.frame_id, 0.0)
            self._q.append(frame)
            self._not_empty.notify()
        return OfferOutcome(Admission.ACCEPTED)


class UnboundedFifo(AdmissionPolicy):
    """Never drops. A deliberately pathological baseline, labelled as one.

    The only configuration where "result age grows without bound" is literally
    true, included so the claim attaches to where it holds rather than to
    queueing in general. Memory grows without bound too; do not run it long.
    """

    name = "fifo_unbounded"

    def offer(self, frame: Frame) -> OfferOutcome:
        with self._not_empty:
            self._q.append(frame)
            self._not_empty.notify()
        return OfferOutcome(Admission.ACCEPTED)


POLICY_NAMES = ("latest", "fifo_bounded", "fifo_unbounded")


def make_policy(name: str, capacity: int = 8) -> AdmissionPolicy:
    if name == "latest":
        return LatestFrameBuffer()
    if name == "fifo_bounded":
        return BoundedFifo(capacity)
    if name == "fifo_unbounded":
        return UnboundedFifo()
    raise ValueError(f"unknown policy {name!r}; expected one of {POLICY_NAMES}")
