"""Capture sources.

The producer's contract is narrow: stamp a trusted `frame_id` and a monotonic
`capture_ts` when the frame is obtained, hand it to the admission policy, and
never block on inference.
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

    Used for the overload experiments, so the arrival process is a controlled
    input. Jitter is included because a real UVC camera is not periodic, and a
    policy tested only against a periodic producer is not tested.
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

    A driver may return a different resolution than requested. `Frame.width`
    and `height` are trusted metadata, so they describe the array that
    arrived, not the request.
    """
    shape = getattr(frame, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1]), int(shape[0])
    return fallback


class WebcamSource(_BaseSource):
    """UVC webcam via OpenCV. Three defaults are wrong here, so all are set.

    MJPG, because OpenCV defaults to uncompressed YUYV, which USB bandwidth
    caps well under 30 fps; some cameras offer no YUYV mode at 640x480 at all.
    It costs a JPEG decode per frame, charged to preprocessing.

    V4L2, because JetPack's OpenCV prefers GStreamer, which ignores the format
    request and on the camera used here would not start a pipeline at all.

    Two capture buffers, because with one the driver has nowhere to write
    while we hold the only buffer, and drops every other frame. Those frames
    are lost before `frame_id` is stamped, so nothing downstream can count
    them. Staleness belongs to the admission policy, where it is recorded.

    fps is a hint. Arrival times are measured, because drop rate is computed
    against frames that arrived, not a nominal rate.
    """

    def __init__(
        self,
        device: int = 0,
        width: int = 640,
        height: int = 480,
        fps_hint: float = 30.0,
        fourcc: str = "MJPG",
        backend: str = "v4l2",
        buffer_frames: int = 2,
    ) -> None:
        super().__init__()
        self.device = device
        self.width = width
        self.height = height
        self.fps_hint = fps_hint
        self.fourcc = fourcc
        self.backend = backend
        self.buffer_frames = buffer_frames
        # What the driver settled on, filled in by open(). Recorded in run
        # notes, because the requested format is a claim and this is the fact.
        self.negotiated: Optional[dict] = None
        self._cap = None

    def open(self) -> None:
        import cv2  # type: ignore

        api = {"v4l2": "CAP_V4L2", "gstreamer": "CAP_GSTREAMER", "any": "CAP_ANY"}
        preference = getattr(cv2, api.get(self.backend, "CAP_ANY"), 0)
        cap = cv2.VideoCapture(self.device, preference)
        # Format first: the backend resolves size and rate within the chosen
        # format, so setting it later can renegotiate the size just asked for.
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps_hint)
        # Two is the shallowest depth that does not starve the driver: one to
        # hold, one for the sensor to fill. 1 stays reachable so the
        # measurement behind that can be reproduced.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_frames)
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
            "buffer_frames": int(cap.get(cv2.CAP_PROP_BUFFERSIZE) or 0),
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
