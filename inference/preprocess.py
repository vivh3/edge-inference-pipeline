"""Preprocessing: bring every frame to the one fixed input form the model sees.

Kept as its own timed stage because it is a plausible bottleneck.  On a
Jetson, a naive resize plus colour conversion plus a host-to-device copy on
the CPU can cost tens of milliseconds per frame, and if that turns out to
dominate, the fix is nothing to do with the model.  Gate 3 needs to be able
to attribute time here.

The image backend is resolved lazily so the pipeline core, the tests, and the
overload simulation all run on a machine with no OpenCV, no PIL, and no
camera -- which is what makes the architecture portable ahead of hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .clock import monotonic
from .config import DEFAULT_POLICY, GenerationPolicy

__all__ = ["PreprocessResult", "Preprocessor"]


@dataclass
class PreprocessResult:
    image: Any
    duration: float
    backend: str


class Preprocessor:
    """Resize to the frozen resolution and hand the model a consistent input.

    `backend="auto"` picks OpenCV, then PIL, then a passthrough for the
    synthetic payloads used by the simulation.  Whichever is used is recorded
    on every result, because a measurement taken with one backend is not
    comparable to a measurement taken with another.
    """

    def __init__(
        self,
        policy: GenerationPolicy = DEFAULT_POLICY,
        backend: str = "auto",
    ) -> None:
        self.policy = policy
        self._requested = backend
        self._backend: Optional[str] = None if backend == "auto" else backend
        self._cv2 = None
        self._pil = None

    # -- backend resolution ------------------------------------------------

    def _resolve(self) -> str:
        if self._backend is not None:
            return self._backend
        try:
            import cv2  # type: ignore

            self._cv2 = cv2
            self._backend = "opencv"
        except ImportError:
            try:
                from PIL import Image  # type: ignore

                self._pil = Image
                self._backend = "pillow"
            except ImportError:
                self._backend = "passthrough"
        return self._backend

    @property
    def backend(self) -> str:
        return self._resolve()

    # -- the stage ---------------------------------------------------------

    def run(self, payload: Any) -> PreprocessResult:
        start = monotonic()
        backend = self._resolve()
        size = (self.policy.image_width, self.policy.image_height)

        if backend == "opencv" and _looks_like_array(payload):
            # Camera frames arrive BGR from OpenCV; the model expects RGB.
            image = self._cv2.resize(payload, size, interpolation=self._cv2.INTER_AREA)
            image = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
        elif backend == "pillow" and hasattr(payload, "resize"):
            image = payload.convert("RGB").resize(size)
        else:
            # Synthetic payload (simulation) or an already-prepared image.
            image = payload

        return PreprocessResult(image=image, duration=monotonic() - start, backend=backend)


def _looks_like_array(payload: Any) -> bool:
    return hasattr(payload, "shape") and hasattr(payload, "dtype")
