#!/usr/bin/env python3
"""Overload experiment: latest-frame admission vs FIFO, under a slow engine.

This is the headline comparison, run against a *synthetic* engine so it is
reproducible on any machine with a stdlib Python and no accelerator.  Read the
caveat and then read it again:

    The service-time distribution here is an input, not a measurement.  These
    runs demonstrate the admission policy's behaviour; they say nothing about
    how fast any real model is.  Results land in results/simulated/ and are
    never reported as baseline or optimised performance.

What it does show, and what the same code will show on hardware once the real
engine is substituted, is the shape of the argument:

  latest          result age stays near one inference service time; drop rate
                  rises with the overload ratio.
  fifo_bounded    result age saturates near capacity x service time -- bounded,
                  but bounded at a much worse value, and the surviving frames
                  are the oldest ones.
  fifo_unbounded  result age grows without bound.  Labelled pathological; it
                  is here so the "grows without bound" claim is attached to
                  the one configuration where it is literally true.

Usage:
    python3 tools/run_overload_sim.py --fps 30 --latency 0.6 --duration 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.buffer import POLICY_NAMES, make_policy
from inference.capture import SyntheticCamera
from inference.clock import monotonic
from inference.engine import MockEngine
from inference.pipeline import Pipeline
from telemetry.metrics import Metrics

RESULTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "simulated"
)


def run_one(policy_name: str, args) -> dict:
    engine = MockEngine(
        mean_latency=args.latency,
        sigma=args.sigma,
        p_malformed=args.p_malformed,
        p_schema_violation=args.p_schema_violation,
        p_unusable=args.p_unusable,
        p_timeout=args.p_timeout,
        seed=args.seed,
    )
    policy = make_policy(policy_name, capacity=args.fifo_capacity)
    metrics = Metrics(policy_name, sample_resources=True)
    published: list = []

    pipeline = Pipeline(
        engine=engine,
        policy=policy,
        metrics=metrics,
        consumer=published.append,  # the mock consumer: a logger, nothing more
        deadline_s=args.deadline,
        watchdog_s=args.deadline * 4,
    )
    # No warmup for the synthetic engine: there is no CUDA context to build,
    # and skipping it keeps the run length honest.
    pipeline.start(warmup=False)

    camera = SyntheticCamera(fps=args.fps, seed=args.seed)
    camera.start(pipeline.submit)

    deadline = monotonic() + args.duration
    while monotonic() < deadline:
        time.sleep(0.05)

    camera.stop()
    backlog = policy.depth()
    pipeline.stop()

    notes = {
        "SIMULATED": "synthetic engine; service time is an input, not a measurement",
        "engine": engine.describe(),
        "preprocess_backend": pipeline.preprocessor.backend,
        "offered_fps": args.fps,
        "fifo_capacity": args.fifo_capacity if policy_name == "fifo_bounded" else None,
        "frames_left_in_queue_at_stop": backlog,
        "overload_ratio": round(args.fps * args.latency, 2),
        "health": pipeline.health_snapshot(),
    }

    outdir = os.path.join(RESULTS_ROOT, policy_name)
    metrics.write_csv(os.path.join(outdir, "results.csv"))
    summary = metrics.write_summary(os.path.join(outdir, "summary.json"), notes)
    return asdict(summary)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fps", type=float, default=30.0, help="synthetic capture rate")
    p.add_argument("--latency", type=float, default=0.6, help="mean synthetic service time (s)")
    p.add_argument("--sigma", type=float, default=0.25, help="lognormal sigma of service time")
    p.add_argument("--duration", type=float, default=30.0, help="seconds per policy")
    p.add_argument("--deadline", type=float, default=5.0, help="per-frame inference budget (s)")
    p.add_argument("--fifo-capacity", type=int, default=8)
    p.add_argument("--policies", nargs="*", default=list(POLICY_NAMES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p-malformed", type=float, default=0.0)
    p.add_argument("--p-schema-violation", type=float, default=0.0)
    p.add_argument("--p-unusable", type=float, default=0.0)
    p.add_argument("--p-timeout", type=float, default=0.0)
    args = p.parse_args()

    print(f"offered {args.fps:g} fps against a {args.latency:g}s mean service time "
          f"-> overload ratio {args.fps * args.latency:.1f}x\n")

    summaries = {}
    for name in args.policies:
        print(f"running {name} for {args.duration:g}s ...", flush=True)
        summaries[name] = run_one(name, args)

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    with open(os.path.join(RESULTS_ROOT, "comparison.json"), "w") as fh:
        json.dump({"args": vars(args), "summaries": summaries}, fh, indent=2)
        fh.write("\n")

    print()
    hdr = f"{'policy':16} {'captured':>9} {'published':>10} {'drop rate':>10} " \
          f"{'result age p50':>15} {'p99':>9} {'max':>9} {'infer p50':>10}"
    print(hdr)
    print("-" * len(hdr))
    for name, s in summaries.items():
        print(f"{name:16} {s['frames_captured']:9d} {s['results_published']:10d} "
              f"{s['drop_rate']*100:9.1f}% {s['result_age']['p50']:14.3f}s "
              f"{s['result_age']['p99']:8.3f}s {s['result_age']['max']:8.3f}s "
              f"{s['inference_latency']['p50']:9.3f}s")
    print(f"\nwrote {RESULTS_ROOT}/<policy>/{{results.csv,summary.json}} and comparison.json")
    print("reminder: synthetic engine. these are admission-policy results, not model performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
