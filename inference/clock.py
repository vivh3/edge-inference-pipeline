"""Single clock source for the whole pipeline.

Every timestamp that is ever subtracted from another timestamp comes from
`monotonic()`.  Wall-clock time is recorded once per published record, for
human readability only, and is never used in a duration.

Why: the system clock is subject to NTP slew and step corrections.  A step
correction in the middle of an inference silently corrupts a latency
measurement and can produce a negative duration.  CLOCK_MONOTONIC never
jumps, so a duration computed from it is always meaningful.

`time.monotonic()` in CPython on Linux is CLOCK_MONOTONIC.  If a stage of
this pipeline is ever reimplemented in C++ or as a ROS 2 node, use
CLOCK_MONOTONIC / the ROS 2 steady clock respectively so the same epoch is
shared end to end.
"""

from __future__ import annotations

import datetime as _dt
import time

__all__ = ["monotonic", "wall_clock_iso", "CLOCK_NAME"]

CLOCK_NAME = "time.monotonic (CLOCK_MONOTONIC)"


def monotonic() -> float:
    """Seconds from an arbitrary fixed epoch. The only source for durations."""
    return time.monotonic()


def wall_clock_iso() -> str:
    """UTC wall time, ISO 8601 with milliseconds. Display only, never a duration."""
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
