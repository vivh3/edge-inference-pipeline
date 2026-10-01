"""Capture sources.

The producer's contract is narrow on purpose: stamp a trusted `frame_id` and
a monotonic `capture_ts` at the moment the frame is obtained, hand it to the
admission policy, and never block on inference.  Everything that makes the
system interesting happens downstream of this file.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional, Tuple

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


def _frame_size(frame: Any, fallback: Tuple[int, int]) -> Tuple[int, int]:
    """(width, height) from the frame itself, falling back to what was asked.

    A driver is free to hand back a resolution other than the one requested.
    `Frame.width`/`height` are trusted metadata, so they have to describe the
    array that arrived rather than repeat the request that was made.
    """
    shape = getattr(frame, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1]), int(shape[0])
    return fallback


class WebcamSource(_BaseSource):
    """UVC webcam via OpenCV.

    No fps is requested from the driver beyond a hint: the actual inter-frame
    arrival times are measured (telemetry.Metrics.inter_frame_intervals) and
    reported, because a cheap USB camera will not hold a steady 30 fps and
    drop rate must be computed against frames that actually arrived.

    The pixel format is requested explicitly. OpenCV defaults to uncompressed
    YUYV, and USB 2.0 bandwidth limits most webcams to a few frames per second
    that way -- some offer no YUYV mode at the requested size at all. MJPG is
    what a UVC camera will actually sustain at 30 fps. The cost is a JPEG
    decode per frame, which lands in preprocessing and is measured there
    rather than being hidden.

    The capture backend is named explicitly too. JetPack ships an OpenCV built
    with GStreamer and prefers it, and GStreamer does not honour the pixel
    format request -- it reports the property as unhandled and, on the camera
    used here, fails to start a pipeline at all. V4L2 is the backend that
    talks to a UVC webcam directly and accepts the format.
    """

    def __init__(
        self,
        device: int = 0,
        width: int = 640,
        height: int = 480,
        fps_hint: float = 30.0,
        fourcc: str = "MJPG",
        backend: str = "v4l2",
    ) -> None:
        super().__init__()
        self.device = device
        self.width = width
        self.height = height
        self.fps_hint = fps_hint
        self.fourcc = fourcc
        self.backend = backend
        # What the driver settled on, filled in by open(). Recorded in run
        # notes, because the requested format is a claim and this is the fact.
        self.negotiated: Optional[dict] = None
        self._cap = None

    def open(self) -> None:
        import cv2  # type: ignore

        api = {"v4l2": "CAP_V4L2", "gstreamer": "CAP_GSTREAMER", "any": "CAP_ANY"}
        preference = getattr(cv2, api.get(self.backend, "CAP_ANY"), 0)
        cap = cv2.VideoCapture(self.device, preference)
        # Pixel format first. The backend resolves resolution and frame rate
        # within the chosen format, so setting it afterwards can renegotiate
        # the size that was just requested.
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps_hint)
        # Ask the driver for the shallowest capture queue available. Frames
        # buffered inside the driver are already stale by the time we read
        # them, and no application-level policy can undo that staleness.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            raise RuntimeError(
                f"could not open video device {self.device} via the "
                f"{self.backend} backend. Check `v4l2-ctl --list-devices`, and "
                f"that nothing else holds the device."
            )
        self._cap = cap
        self.negotiated = self._read_back(cv2, cap)

    @staticmethod
    def _read_back(cv2: Any, cap: Any) -> dict:
        """What the driver agreed to, which need not be what was asked for."""
        code = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        fourcc = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "fourcc": fourcc.strip("\x00 ") or None,
            "backend": getattr(cap, "getBackendName", lambda: None)(),
        }

    def _run(self, sink) -> None:
        if self._cap is None:
            self.open()
        fallback = (self.width, self.height)
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                continue
            width, height = _frame_size(frame, fallback)
            self._emit(sink, frame, width, height)

    def stop(self, timeout: float = 2.0) -> None:
        super().stop(timeout)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
