"""Preprocessing: bring every frame to the one fixed input form the model sees.

Timed as its own stage because it is a plausible bottleneck. On a Jetson,
resize plus colour conversion plus a host-to-device copy can cost tens of
milliseconds per frame, and if that dominates, the fix has nothing to do with
the model.

The backend resolves lazily, so the core, the tests and the overload
simulation all run with no OpenCV, no Pillow and no camera.
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
    synthetic payloads the simulation uses. Whichever is used is recorded on
    every result: a measurement taken with one backend is not comparable to
    one taken with another.
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

    def warmup(self) -> str:
        """Resolve the backend before any frame is timed.

        `_resolve` imports OpenCV, which takes about 2.7 s on a Jetson reading
        from an SD card. Left to the first `run`, that one-time library load
        lands inside a per-frame duration: the first published record reported
        2.709 s of preprocessing against a steady-state 0.004 s, and the
        outlier was mistaken for a real cost for most of an evening.

        The engine's warmup runs are discarded for the same reason. This is
        the same cost, in a stage that was not getting the same treatment.
        """
        return self._resolve()

    # -- the stage ---------------------------------------------------------

    def run(self, payload: Any) -> PreprocessResult:
        """Resize to the frozen resolution. Reports the backend actually used.

        Per call, not the backend resolved at startup. A payload the resolved
        backend cannot handle falls through unresized, and labelling that
        "opencv" would silently break the frozen-resolution guarantee.
        "passthrough" on a real frame means the image was never resized: a
        bug, not a fallback.
        """
        start = monotonic()
        size = (self.policy.image_width, self.policy.image_height)
        resolved = self._resolve()
        used = "passthrough"
        image = payload

        if resolved == "opencv" and _looks_like_array(payload):
            # OpenCV hands back BGR; the model expects RGB. Getting this
            # wrong still runs and still answers, just worse: a silent failure
            # that reads as a bad model.
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
