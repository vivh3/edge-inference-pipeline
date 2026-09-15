#!/usr/bin/env python3
"""Gate 1 baseline: one image, one model, no pipeline.

Answers the three questions Gate 1 exists to answer, and nothing else:

  1. Does the model clear the output contract? Not "did it produce plausible
     text" -- does inference.schema.validate accept it.
  2. How much memory is left with the model loaded? Weights on disk are not
     runtime footprint, and the Jetson shares one 8 GB pool with the OS.
  3. How long does one inference take, warmup discarded?

Answer 3 is what the performance budget gets set from in Gate 2. It is
deliberately measured here, with no camera and no pipeline attached, so it is
the model's cost and nothing else.

    python3 tools/gate1_baseline.py --model <org>/<model> --image hallway.jpg
    python3 tools/gate1_baseline.py --mock --image anything.jpg   # smoke test

Writes results/baseline/model.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.clock import CLOCK_NAME, wall_clock_iso
from inference.config import DEFAULT_POLICY
from inference.engine import MockEngine, VlmEngine
from inference.schema import validate
from telemetry.metrics import jetson_power_rail_names, percentile, read_jetson_power_w, read_rss_bytes

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "baseline"
)


def system_memory_mb() -> dict:
    """Total and available system memory. On a Jetson this pool is shared with the GPU."""
    fields = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    fields[key] = int(rest.split()[0]) / 1024.0
    except OSError:
        pass
    return {
        "total_mb": round(fields.get("MemTotal", 0.0), 1),
        "available_mb": round(fields.get("MemAvailable", 0.0), 1),
    }


def snapshot(label: str) -> dict:
    mem = system_memory_mb()
    snap = {
        "stage": label,
        "process_rss_mb": round(read_rss_bytes() / 1e6, 1),
        "system_available_mb": mem["available_mb"],
        "system_total_mb": mem["total_mb"],
        "power_w": round(read_jetson_power_w(), 2),
    }
    print(f"  {label:26} rss {snap['process_rss_mb']:8.1f} MB   "
          f"system free {snap['system_available_mb']:8.1f} MB", flush=True)
    return snap


def load_image(path: str):
    from PIL import Image

    return Image.open(path).convert("RGB")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="Hugging Face model id, e.g. org/model")
    p.add_argument("--image", required=True, help="one representative photo")
    p.add_argument("--runs", type=int, default=20, help="measured runs after warmup")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--mock", action="store_true", help="synthetic engine, to test this script")
    p.add_argument("--out", default=os.path.join(OUT, "model.json"))
    args = p.parse_args()

    if not args.mock and not args.model:
        p.error("--model is required unless --mock is given")

    print("memory, stage by stage:")
    before = snapshot("before model load")

    if args.mock:
        engine = MockEngine(mean_latency=0.45, sigma=0.2, seed=0)
        image = None
    else:
        engine = VlmEngine(args.model, device=args.device, dtype=args.dtype)
        image = load_image(args.image)

    # Warmup is discarded: the first invocations pay for CUDA context creation,
    # kernel autotuning and allocator growth, which describe startup rather than
    # steady-state service time.
    print(f"\nwarmup: {DEFAULT_POLICY.warmup_runs} runs, discarded", flush=True)
    engine.warmup(DEFAULT_POLICY.warmup_runs)
    loaded = snapshot("after load + warmup")

    print(f"\nmeasuring {args.runs} runs ...", flush=True)
    latencies, ttfts, tokens = [], [], []
    outcomes = Counter()
    extractions = 0
    samples = []

    for i in range(args.runs):
        raw = engine.infer(image)
        latency = raw.inference_end_ts - raw.inference_start_ts
        latencies.append(latency)
        if raw.first_token_ts is not None:
            ttfts.append(raw.first_token_ts - raw.inference_start_ts)
        if raw.output_tokens is not None:
            tokens.append(raw.output_tokens)

        _, report = validate(raw.text)
        outcomes[report.failure.value] += 1
        extractions += bool(report["extracted"])
        if len(samples) < 3:
            samples.append({"text": raw.text[:300], "failure": report.failure.value})
        print(f"  run {i + 1:3d}/{args.runs}  {latency * 1000:7.1f} ms  {report.failure.value}", flush=True)

    steady = snapshot("steady state")

    def stats(values):
        if not values:
            return None
        return {
            "n": len(values),
            "mean": round(sum(values) / len(values), 4),
            "p50": round(percentile(values, 50), 4),
            "p90": round(percentile(values, 90), 4),
            "p99": round(percentile(values, 99), 4),
            "max": round(max(values), 4),
        }

    valid = outcomes.get("none", 0)
    report = {
        "wall_clock": wall_clock_iso(),
        "clock": CLOCK_NAME,
        "model": "MOCK -- not a real model" if args.mock else args.model,
        "device": args.device,
        "dtype": args.dtype,
        "image": os.path.basename(args.image),
        "generation_policy": DEFAULT_POLICY.describe(),
        "runs": args.runs,
        "inference_latency_s": stats(latencies),
        "time_to_first_token_s": stats(ttfts),
        "output_tokens": stats([float(t) for t in tokens]),
        "contract": {
            "valid": valid,
            "valid_rate": round(valid / args.runs, 4) if args.runs else 0.0,
            "outcomes": dict(outcomes),
            "extraction_rate": round(extractions / args.runs, 4) if args.runs else 0.0,
        },
        "memory": {"before_load": before, "after_load": loaded, "steady": steady},
        "power_rails": jetson_power_rail_names(),
        "sample_outputs": samples,
    }
    if args.mock:
        report["WARNING"] = "synthetic engine; these numbers measure nothing real"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    lat = report["inference_latency_s"]
    headroom = steady["system_available_mb"]
    print("\n" + "=" * 62)
    print(f"  inference latency p50   {lat['p50'] * 1000:.0f} ms")
    print(f"  inference latency p99   {lat['p99'] * 1000:.0f} ms")
    if report["output_tokens"]:
        print(f"  output tokens p50       {report['output_tokens']['p50']:.0f}")
    print(f"  clears the contract     {valid}/{args.runs}  ({report['contract']['valid_rate'] * 100:.0f}%)")
    if outcomes and set(outcomes) != {"none"}:
        print(f"  failures                {dict((k, v) for k, v in outcomes.items() if k != 'none')}")
    print(f"  memory headroom         {headroom:.0f} MB free of {steady['system_total_mb']:.0f} MB")
    print("=" * 62)
    print(f"\nwrote {args.out}")
    print("\nGate 1 passes when the contract rate is high and headroom is comfortable.")
    print("The p50 above is what the Gate 2 performance budget gets set from --")
    print("call it a budget, never an SLO: it derives from what this board can do,")
    print("not from a requirement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
