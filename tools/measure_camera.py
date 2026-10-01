#!/usr/bin/env python3
"""Measure what the camera actually delivers, not what it claims.

Gate 1, step 6. A cheap USB webcam does not hold 30 fps -- exposure lengthens
in low light, the driver drops frames, USB bandwidth caps the format. Drop rate
in this project is computed against frames that actually arrived, so the real
arrival rate has to be measured before any of the overload numbers mean
anything. Computing against a nominal 30 would fabricate drops that never
happened.

    python3 tools/measure_camera.py --seconds 30
    python3 tools/measure_camera.py --seconds 30 --device 0 --width 640 --height 480
    python3 tools/measure_camera.py --seconds 5 --synthetic   # no camera needed

Writes results/baseline/camera.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.capture import SyntheticCamera, WebcamSource
from inference.clock import CLOCK_NAME, monotonic, wall_clock_iso
from telemetry.metrics import percentile

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "baseline"
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=30.0, help="how long to sample")
    p.add_argument("--device", type=int, default=0, help="/dev/videoN index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps-hint", type=float, default=30.0, help="requested, not guaranteed")
    p.add_argument("--backend", default="v4l2", choices=["v4l2", "gstreamer", "any"],
                   help="JetPack's OpenCV prefers GStreamer, which ignores the "
                        "pixel format request. v4l2 talks to a UVC camera directly.")
    p.add_argument("--fourcc", default="MJPG",
                   help="pixel format to request; MJPG is what most UVC cameras "
                        "sustain at 30 fps. Empty string leaves it to the driver.")
    p.add_argument("--synthetic", action="store_true", help="fake source, to test this script")
    p.add_argument("--out", default=os.path.join(OUT, "camera.json"))
    args = p.parse_args()

    if args.synthetic:
        source = SyntheticCamera(fps=args.fps_hint)
        label = f"synthetic @ {args.fps_hint:g} fps"
    else:
        source = WebcamSource(args.device, args.width, args.height, args.fps_hint,
                              args.fourcc, args.backend)
        label = f"/dev/video{args.device} {args.width}x{args.height} {args.fourcc or 'driver default'}"

    stamps = []
    lock = threading.Lock()

    def record(frame):
        with lock:
            stamps.append(frame.capture_ts)

    print(f"sampling {label} for {args.seconds:g}s ...", flush=True)
    started = wall_clock_iso()
    source.start(record)
    deadline = monotonic() + args.seconds
    while monotonic() < deadline:
        time.sleep(0.1)
    source.stop()

    with lock:
        stamps = sorted(stamps)
    if len(stamps) < 3:
        print(f"only {len(stamps)} frames captured -- is the camera connected?", file=sys.stderr)
        return 1

    intervals = [b - a for a, b in zip(stamps, stamps[1:])]
    span = stamps[-1] - stamps[0]
    # Frames arriving more than 1.5 inter-frame periods apart: the driver
    # skipped one. Worth knowing, because those gaps are camera-side staleness
    # no admission policy can undo.
    nominal = 1.0 / args.fps_hint
    gaps = [i for i in intervals if i > 1.5 * nominal]

    report = {
        "source": label,
        "clock": CLOCK_NAME,
        "wall_clock_start": started,
        "requested_fps": args.fps_hint,
        # What the driver agreed to, read back after opening. The request
        # above is a claim; this is what the camera is actually doing.
        "negotiated": getattr(source, "negotiated", None),
        "sample_seconds": round(span, 3),
        "frames": len(stamps),
        "effective_fps": round(len(intervals) / span, 3) if span > 0 else 0.0,
        "interval_s": {
            "mean": round(sum(intervals) / len(intervals), 5),
            "min": round(min(intervals), 5),
            "p50": round(percentile(intervals, 50), 5),
            "p90": round(percentile(intervals, 90), 5),
            "p99": round(percentile(intervals, 99), 5),
            "max": round(max(intervals), 5),
        },
        "long_gaps": {
            "threshold_s": round(1.5 * nominal, 5),
            "count": len(gaps),
            "fraction": round(len(gaps) / len(intervals), 4),
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    iv = report["interval_s"]
    print()
    print(f"  frames captured     {report['frames']} over {report['sample_seconds']}s")
    print(f"  effective fps       {report['effective_fps']}  (requested {args.fps_hint:g})")
    if report["negotiated"]:
        n = report["negotiated"]
        print(f"  negotiated format   {n['fourcc']} {n['width']}x{n['height']} "
              f"@ {n['fps']:g} fps via {n['backend']}")
    print(f"  interval p50 / p99  {iv['p50'] * 1000:.1f} ms / {iv['p99'] * 1000:.1f} ms")
    print(f"  interval min / max  {iv['min'] * 1000:.1f} ms / {iv['max'] * 1000:.1f} ms")
    print(f"  long gaps           {report['long_gaps']['count']} "
          f"({report['long_gaps']['fraction'] * 100:.1f}% of intervals)")
    print(f"\nwrote {args.out}")
    print("Put effective_fps in the README next to \"nominal 30 fps\". That is the")
    print("denominator every drop rate in this project is computed against.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
