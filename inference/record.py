"""Record types, organised around the trust boundary.

The model emits semantic content and nothing else.  Identifiers and
timestamps are attached here, after validation, by trusted code.

If the model emitted `frame_id`, a hallucinated value would match a result to
the wrong capture and corrupt latency accounting -- and the error would look
like jitter, not a bug.  So `frame_id` is assigned at capture and carried
alongside the model, never through it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .clock import monotonic, wall_clock_iso

# --------------------------------------------------------------------------
# Trusted: produced by the capture stage.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """One captured image plus the trusted metadata that travels with it."""

    frame_id: int
    capture_ts: float  # monotonic
    payload: Any  # raw image; opaque to the pipeline until preprocessing
    width: int = 0
    height: int = 0

    def age(self, now: Optional[float] = None) -> float:
        return (monotonic() if now is None else now) - self.capture_ts


# --------------------------------------------------------------------------
# Untrusted: produced by the model, before validation.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RawModelOutput:
    """Exactly what the engine produced, plus timing the wrapper observed.

    `text` is untrusted.  Nothing downstream may read it without going
    through `inference.schema.validate`.
    """

    text: str
    inference_start_ts: float
    inference_end_ts: float
    first_token_ts: Optional[float] = None  # None if the runtime does not expose it
    output_tokens: Optional[int] = None
    engine_error: Optional[str] = None  # set when the engine raised or timed out


# --------------------------------------------------------------------------
# Trusted: what the pipeline publishes.
# --------------------------------------------------------------------------


@dataclass
class PublishedResult:
    """The wire record. Trusted metadata + validated (or unknown) semantics."""

    frame_id: int
    capture_ts: float
    inference_start_ts: float
    inference_end_ts: float
    publish_ts: float
    wall_clock: str
    semantic: dict
    # Trusted validation report, on every record, so a consumer never has to
    # guess whether `semantic` came from the model or from the unknown-state
    # fallback.
    validation: dict = field(default_factory=dict)
    # Preprocessing cost for this frame. It falls inside `queue_age` (the
    # worker preprocesses before the engine stamps its start), so it is
    # reported separately rather than being lost inside the wait.
    preprocess_s: float = 0.0

    # --- derived durations (all from the monotonic clock) ------------------

    @property
    def queue_age(self) -> float:
        """capture -> inference start: waiting, plus this frame's preprocessing.

        Subtract `preprocess_s` for the pure queueing component.
        """
        return self.inference_start_ts - self.capture_ts

    @property
    def inference_latency(self) -> float:
        """inference start -> model result complete. Model execution only."""
        return self.inference_end_ts - self.inference_start_ts

    @property
    def post_processing(self) -> float:
        """model result complete -> publish. Validation, serialisation, publish."""
        return self.publish_ts - self.inference_end_ts

    @property
    def result_age(self) -> float:
        """capture -> publish. The primary metric: what a consumer experiences."""
        return self.publish_ts - self.capture_ts

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "capture_ts": round(self.capture_ts, 6),
            "inference_start_ts": round(self.inference_start_ts, 6),
            "inference_end_ts": round(self.inference_end_ts, 6),
            "publish_ts": round(self.publish_ts, 6),
            "wall_clock": self.wall_clock,
            "semantic": self.semantic,
            "validation": self.validation,
            "preprocess_s": round(self.preprocess_s, 6),
        }


def attach_metadata(
    frame: Frame,
    raw: RawModelOutput,
    semantic: dict,
    validation: dict,
    preprocess_s: float = 0.0,
) -> PublishedResult:
    """Wrap validated semantics in trusted metadata and stamp the publish time."""
    return PublishedResult(
        frame_id=frame.frame_id,
        capture_ts=frame.capture_ts,
        inference_start_ts=raw.inference_start_ts,
        inference_end_ts=raw.inference_end_ts,
        publish_ts=monotonic(),
        wall_clock=wall_clock_iso(),
        semantic=semantic,
        validation=validation,
        preprocess_s=preprocess_s,
    )
