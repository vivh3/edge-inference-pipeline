#!/usr/bin/env python3
"""Capture the probe set used to compare candidate models.

Frames come through `WebcamSource`, the same path live frames take, at the
same negotiated format. A set shot on a phone would be a comparison against
pictures the pipeline never sees -- different sensor, different resolution,
different JPEG encoder, different colour handling. The point of the set is to
predict how a model behaves on *this* camera.

Mean brightness is recorded per frame because an unusable set is cheap to
catch here and expensive to discover after a model comparison. A blown-out or
near-black frame tells you about the lighting, not about the model.

    python3 tools/capture_probe_set.py --count 20 --interval 4

Move the camera, or the scene, between shots. Mix the obvious against the
genuinely ambiguous: a clear hallway, a doorway, a bag on the floor, a dark
room, a blank wall. A set of twenty easy frames distinguishes nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.buffer import LatestFrameBuffer
from inference.capture import WebcamSource
from inference.clock import wall_clock_iso

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "probe"
)

# Outside this range the frame says more about the lighting than the scene.
USABLE_MEAN = (40.0, 200.0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--interval", type=float, default=4.0, help="seconds between shots")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    import cv2  # type: ignore

    os.makedirs(args.out, exist_ok=True)
    buffer = LatestFrameBuffer()
    source = WebcamSource(args.device, args.width, args.height)
    source.start(buffer.offer)

    # Auto white balance and any remaining auto controls need a moment, and
    # the first frames off a UVC camera are routinely junk.
    print("warming up ...", flush=True)
    time.sleep(2.0)

    shots, unusable = [], 0
    try:
        for i in range(args.count):
            for remaining in range(int(args.interval), 0, -1):
                print(f"\r  {i + 1:2d}/{args.count}  in {remaining}s ", end="", flush=True)
                time.sleep(1.0)

            frame = buffer.take(timeout=2.0)
            if frame is None:
                print("\n  no frame available -- is the camera still attached?")
                return 1

            name = f"probe_{i:02d}.jpg"
            cv2.imwrite(os.path.join(args.out, name), frame.payload)
            mean = round(float(frame.payload.mean()), 1)
            usable = USABLE_MEAN[0] <= mean <= USABLE_MEAN[1]
            unusable += not usable
            shots.append({"file": name, "mean_pixel": mean, "usable": usable})
            flag = "" if usable else "  <- too dark" if mean < USABLE_MEAN[0] else "  <- blown out"
            print(f"\r  {i + 1:2d}/{args.count}  {name}  mean {mean}{flag}")
    finally:
        source.stop()

    manifest = {
        "captured_at": wall_clock_iso(),
        "negotiated": source.negotiated,
        "usable_mean_range": list(USABLE_MEAN),
        "shots": shots,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    print(f"\nwrote {len(shots)} frames and manifest.json to {args.out}")
    if unusable:
        print(f"{unusable} frame(s) outside the usable brightness range.")
        print("Fix the lighting or exposure and retake those, rather than asking a")
        print("model to describe a picture of nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
