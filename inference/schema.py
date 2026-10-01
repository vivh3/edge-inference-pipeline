"""Output contract, validation, and the failure taxonomy.

The model emits only untrusted semantic fields:

    {"path_status": "blocked", "obstacle_location": "front_left"}

and must be able to express ignorance:

    {"path_status": "unknown", "obstacle_location": "unknown"}

`Failure` enumerates every way a result can be unusable. Each is a published
outcome with its own counter, and inference is never retried to get parseable
output: a retry would hide two or three model invocations inside one published
latency.

On failure the pipeline still publishes a record, with semantics set to the
explicit unknown state and `validation` naming the cause, so a consumer can
tell "the model says it does not know" from "the model produced garbage".
"""

from __future__ import annotations

import enum
import json
from typing import Optional, Tuple

__all__ = [
    "Failure",
    "PATH_STATUS_VALUES",
    "OBSTACLE_LOCATION_VALUES",
    "UNKNOWN_SEMANTIC",
    "ValidationReport",
    "validate",
    "extract_json_object",
]


class Failure(str, enum.Enum):
    """Every way a published result can fail to be usable.

    The first three are the model's fault, the next two the engine's, and
    `PIPELINE_ERROR` is ours. They stay separate because "8% invalid" and "8%
    timeouts" need different fixes, and folding our own bugs into
    `ENGINE_ERROR` would blame the model for them.
    """

    NONE = "none"
    MALFORMED_JSON = "malformed_json"          # no parseable JSON object
    SCHEMA_VIOLATION = "schema_violation"      # parses, wrong shape/keys/values
    UNUSABLE_SEMANTICS = "unusable_semantics"  # valid values, self-contradictory
    INFERENCE_TIMEOUT = "inference_timeout"    # exceeded the per-frame deadline
    ENGINE_ERROR = "engine_error"              # engine raised / process died
    PIPELINE_ERROR = "pipeline_error"          # a stage around the engine raised


# A closed vocabulary is what makes "schema violation" detectable at all.
# Free-text semantics would make every output trivially valid and the
# invalid-output rate meaningless.

PATH_STATUS_VALUES = ("clear", "blocked", "unknown")
OBSTACLE_LOCATION_VALUES = (
    "front_left",
    "front_center",
    "front_right",
    "left",
    "right",
    "none",
    "unknown",
)

REQUIRED_KEYS = ("path_status", "obstacle_location")

UNKNOWN_SEMANTIC = {"path_status": "unknown", "obstacle_location": "unknown"}

# No confidence field, by design. A VLM emitting "confidence": "high" is
# producing a token, not a calibrated probability, and publishing it would
# invite a consumer to threshold on it. Such keys are stripped and counted.


class ValidationReport(dict):
    """Trusted description of what validation did. Serialised with the record."""

    def __init__(
        self,
        failure: Failure = Failure.NONE,
        detail: str = "",
        extracted: bool = False,
        extra_keys_stripped: Tuple[str, ...] = (),
    ):
        super().__init__(
            ok=failure is Failure.NONE,
            failure=failure.value,
            detail=detail,
            extracted=extracted,
            extra_keys_stripped=list(extra_keys_stripped),
        )
        self.failure = failure

    @property
    def ok(self) -> bool:
        return self["ok"]


def extract_json_object(text: str) -> Tuple[Optional[str], bool]:
    """Find the JSON object in a model response.

    Returns ``(json_text, needed_extraction)``.

    Instruction-tuned models wrap JSON in prose or a ```json fence even when
    told not to, so this scans for the first balanced top-level ``{...}``. It
    is one deterministic parsing step, not a retry: the model is invoked once
    per admitted frame either way. ``needed_extraction`` is reported
    separately, so a prompt that keeps losing the output format stays visible
    instead of averaging into "valid".
    """
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped, False

    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue  # stray closer, e.g. a truncated object before this one
            depth -= 1
            if depth == 0:
                return text[start : i + 1], True
    return None, False


def _semantically_usable(path_status: str, location: str) -> Optional[str]:
    """Cross-field consistency. Returns a reason string if unusable, else None.

    Individually legal values can still combine into an answer a consumer
    cannot act on. These three rules are the whole check, and they are stated
    in the README so a reviewer can disagree with them explicitly.
    """
    if path_status == "blocked" and location in ("none", "unknown"):
        return "path_status=blocked with no obstacle location"
    if path_status == "clear" and location != "none":
        return f"path_status=clear with obstacle_location={location!r}"
    if path_status == "unknown" and location != "unknown":
        return f"path_status=unknown with obstacle_location={location!r}"
    return None


def validate(text: str) -> Tuple[dict, ValidationReport]:
    """Validate one untrusted model response.

    Always returns a publishable ``semantic`` dict: the model's values when
    they are usable, the explicit unknown state otherwise.
    """
    candidate, extracted = extract_json_object(text or "")
    if candidate is None:
        return dict(UNKNOWN_SEMANTIC), ValidationReport(
            Failure.MALFORMED_JSON, "no JSON object found in response"
        )

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return dict(UNKNOWN_SEMANTIC), ValidationReport(
            Failure.MALFORMED_JSON, f"json decode error: {exc.msg}", extracted
        )

    if not isinstance(parsed, dict):
        return dict(UNKNOWN_SEMANTIC), ValidationReport(
            Failure.SCHEMA_VIOLATION,
            f"top level is {type(parsed).__name__}, expected object",
            extracted,
        )

    missing = [k for k in REQUIRED_KEYS if k not in parsed]
    if missing:
        return dict(UNKNOWN_SEMANTIC), ValidationReport(
            Failure.SCHEMA_VIOLATION, f"missing keys: {', '.join(missing)}", extracted
        )

    extra = tuple(sorted(k for k in parsed if k not in REQUIRED_KEYS))

    for key, allowed in (
        ("path_status", PATH_STATUS_VALUES),
        ("obstacle_location", OBSTACLE_LOCATION_VALUES),
    ):
        value = parsed[key]
        if not isinstance(value, str):
            return dict(UNKNOWN_SEMANTIC), ValidationReport(
                Failure.SCHEMA_VIOLATION,
                f"{key} is {type(value).__name__}, expected string",
                extracted,
                extra,
            )
        if value not in allowed:
            return dict(UNKNOWN_SEMANTIC), ValidationReport(
                Failure.SCHEMA_VIOLATION,
                f"{key}={value!r} not in {list(allowed)}",
                extracted,
                extra,
            )

    reason = _semantically_usable(parsed["path_status"], parsed["obstacle_location"])
    if reason is not None:
        return dict(UNKNOWN_SEMANTIC), ValidationReport(
            Failure.UNUSABLE_SEMANTICS, reason, extracted, extra
        )

    semantic = {k: parsed[k] for k in REQUIRED_KEYS}
    return semantic, ValidationReport(Failure.NONE, "", extracted, extra)


def failure_report(failure: Failure, detail: str) -> Tuple[dict, ValidationReport]:
    """Report for a failure detected before any text existed (timeout, crash)."""
    return dict(UNKNOWN_SEMANTIC), ValidationReport(failure, detail)
