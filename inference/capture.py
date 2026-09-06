"""Capture sources.

The producer's contract is narrow on purpose: stamp a trusted `frame_id` and
a monotonic `capture_ts` at the moment the frame is obtained, hand it to the
admission policy, and never block on inference.  Everything that makes the
system interesting happens downstream of this file.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from .clock import monotonic
from .record import Frame

__all__ = ["SyntheticCamera", "WebcamSource"]


class _BaseSource:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._next_id = 0

    def _emit(self, sink: Callable[[Frame], Any], payload: Any, w: int, h: int) -> None:
        frame = Frame(
            frame_id=self._next_id,
            capture_ts=monotonic(),  # stamped at acquisition, not at admission
            payload=payload,
            width=w,
            height=h,
        )
        self._next_id += 1
        sink(frame)

    def start(self, sink: Callable[[Frame], Any]) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(sink,), daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self, sink) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class SyntheticCamera(_BaseSource):
    """Paced frame source with no camera attached.

    Used for the overload experiments so the arrival process is a controlled
    input rather than whatever a particular USB webcam felt like doing.  Jitter
    is included because a real UVC camera is not periodic, and a policy that
    only works against a perfectly periodic producer has not been tested.
    """

    def __init__(self, fps: float = 30.0, jitter: float = 0.002, seed: int = 0) -> None:
        super().__init__()
        self.fps = fps
        self.jitter = jitter
        import random

        self._rng = random.Random(seed)

    def _run(self, sink) -> None:
        import time

        period = 1.0 / self.fps
        next_at = monotonic()
        while not self._stop.is_set():
            next_at += period + self._rng.uniform(-self.jitter, self.jitter)
            sleep_for = next_at - monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_at = monotonic()  # fell behind; do not spiral
            self._emit(sink, {"synthetic": self._next_id}, 0, 0)


class WebcamSource(_BaseSource):
    """UVC webcam via OpenCV.

    No fps is requested from the driver beyond a hint: the actual inter-frame
    arrival times are measured (telemetry.Metrics.inter_frame_intervals) and
    reported, because a cheap USB camera will not hold a steady 30 fps and
    drop rate must be computed against frames that actually arrived.
    """

    def __init__(self, device: int = 0, width: int = 640, height: int = 480, fps_hint: float = 30.0) -> None:
        super().__init__()
        self.device = device
        self.width = width
        self.height = height
        self.fps_hint = fps_hint
        self._cap = None

    def open(self) -> None:
        import cv2  # type: ignore

        self._cap = cv2.VideoCapture(self.device)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps_hint)
        # Ask the driver for the shallowest capture queue available. Frames
        # buffered inside the driver are already stale by the time we read
        # them, and no application-level policy can undo that staleness.
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video device {self.device}")

    def _run(self, sink) -> None:
        if self._cap is None:
            self.open()
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                continue
            self._emit(sink, frame, self.width, self.height)

    def stop(self, timeout: float = 2.0) -> None:
        super().stop(timeout)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
