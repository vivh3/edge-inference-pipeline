"""Preprocessing: bring every frame to the one fixed input form the model sees.

Its own timed stage because it is a plausible bottleneck: on a Jetson a naive
resize plus colour conversion plus a host-to-device copy can cost tens of
milliseconds per frame, and if that dominates, the fix has nothing to do with
the model. Gate 3 needs to be able to attribute time here.

The backend is resolved lazily so the core, the tests, and the overload
simulation all run with no OpenCV, no Pillow, and no camera -- which is what
makes the architecture portable ahead of hardware.
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
        """Resize to the frozen resolution. Reports the backend actually used.

        The reported backend is per call, not the one resolved at startup: a
        payload the resolved backend cannot handle falls through unresized,
        and labelling that "opencv" would break the frozen-resolution
        guarantee silently. `backend == "passthrough"` on a real frame means
        the image did not get resized -- treat it as a bug, not a fallback.
        """
        start = monotonic()
        size = (self.policy.image_width, self.policy.image_height)
        resolved = self._resolve()
        used = "passthrough"
        image = payload

        if resolved == "opencv" and _looks_like_array(payload):
            # OpenCV hands back BGR; the model expects RGB. Getting this wrong
            # still runs and still answers, just worse -- a silent failure that
            # looks like a bad model rather than a bug.
            image = self._cv2.resize(payload, size, interpolation=self._cv2.INTER_AREA)
            image = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
            used = "opencv"
        elif resolved == "pillow" and _is_pil_image(payload):
            image = payload.convert("RGB").resize(size)
            used = "pillow"

        return PreprocessResult(image=image, duration=monotonic() - start, backend=used)


def _looks_like_array(payload: Any) -> bool:
    return hasattr(payload, "shape") and hasattr(payload, "dtype")


def _is_pil_image(payload: Any) -> bool:
    # numpy arrays also have .resize, so duck-typing on that alone sends an
    # array down the Pillow path and raises on .convert().
    return hasattr(payload, "convert") and hasattr(payload, "resize")
