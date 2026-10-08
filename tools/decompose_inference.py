#!/usr/bin/env python3
"""Separate prefill from decode using baselines at different output lengths.

Generative latency is prefill (vision encoding and prompt processing, once)
plus decode (autoregressive, per token). Telling them apart normally needs
time to first token, which `transformers.generate` does not expose without
instrumenting the generation loop.

Baselines do it instead, as long as everything except output length is held
constant. GPU time is then linear in output length:

    generate = prefill + per_token x tokens

so a least-squares line through the runs gives the per-token cost as its slope
and prefill as its intercept -- the GPU time at zero tokens.

    python3 tools/decompose_inference.py run_a.json run_b.json [run_c.json ...]

**Give it more than two runs where you can.** Two points always fit a line
exactly, so a two-point estimate cannot be wrong and cannot be checked. The
first three runs taken here had pairwise slopes of 73, 89 and 97 ms per token
while the three-point fit was 90 ms with residuals under 10 ms on a 6 s
quantity. Either pair alone would have looked authoritative and two of the
three would have been off by 15%.

`processor_s` is the CPU phase and is subtracted first. Runs taken before that
field existed are handled by passing `--processor-s`, since the phase depends
on resolution and prompt -- both frozen -- and not on output length.

Assumes prefill is identical across runs, which the frozen 448x448 makes
defensible because SmolVLM tiles by resolution, and that per-token cost is
flat over the range compared -- which the residuals now test rather than
assert. A profiler would measure the split outright; this says where to point
one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Everything that must match for the comparison to mean anything. A baseline
# measured on a 640x480 image was once compared against a 448x448 pipeline and
# the 25% difference was chased through three wrong hypotheses, so the policy
# is checked rather than assumed.
MUST_MATCH = (
    ("generation_policy", "prompt_sha256_prefix"),
    ("generation_policy", "image_resolution"),
    ("generation_policy", "max_new_tokens"),
    ("generation_policy", "decoding"),
    ("generation_policy", "seed"),
    ("model",),
)


def dig(report: dict, path):
    for key in path:
        if not isinstance(report, dict) or key not in report:
            return None
        report = report[key]
    return report


def check_comparable(a: dict, b: dict, names) -> None:
    differences = []
    for path in MUST_MATCH:
        left, right = dig(a, path), dig(b, path)
        if left != right:
            differences.append(f"  {'.'.join(path)}: {names[0]} {left!r}, {names[1]} {right!r}")
    if differences:
        raise SystemExit(
            "these runs are not comparable; the generation policy differs:\n"
            + "\n".join(differences)
            + "\n\nOnly output length may vary between the two."
        )


def generate_s(report: dict, fallback_processor: float, name: str) -> float:
    """GPU time: inference minus the CPU processor phase."""
    measured = dig(report, ("generate_s", "p50"))
    if measured is not None:
        return measured
    inference = dig(report, ("inference_latency_s", "p50"))
    if inference is None:
        raise SystemExit(f"{name} has no inference_latency_s")
    if fallback_processor is None:
        raise SystemExit(
            f"{name} predates the CPU/GPU split. Pass --processor-s with the "
            "value measured by a run that has it; the phase depends on "
            "resolution and prompt, both frozen, not on output length."
        )
    return inference - fallback_processor


def fit(points):
    """Least squares through (tokens, generate_s). Returns slope, intercept."""
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    sxx = sum((x - mean_x) ** 2 for x, _ in points)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in points)
    slope = sxy / sxx
    return slope, mean_y - slope * mean_x


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baselines", nargs="+", help="two or more baseline JSON files")
    p.add_argument(
        "--processor-s",
        type=float,
        help="CPU processor phase in seconds, for runs taken before the split "
        "was reported",
    )
    p.add_argument("--out", help="write the decomposition as JSON")
    args = p.parse_args()

    if len(args.baselines) < 2:
        raise SystemExit("at least two baselines are needed to separate the two costs")

    names = [os.path.basename(path) for path in args.baselines]
    reports = [json.load(open(path)) for path in args.baselines]
    # Every run is checked against the first, so one mismatched file cannot
    # slip through by matching its neighbour.
    for report, name in zip(reports[1:], names[1:]):
        check_comparable(reports[0], report, (names[0], name))

    processor = args.processor_s
    for report in reports:
        measured = dig(report, ("processor_s", "p50"))
        if measured is not None:
            processor = measured
            break

    points = []
    for report, name in zip(reports, names):
        tokens = dig(report, ("output_tokens", "p50"))
        if tokens is None:
            raise SystemExit(f"{name} has no output_tokens")
        points.append((tokens, generate_s(report, processor, name)))

    if len({x for x, _ in points}) < 2:
        raise SystemExit(
            f"every run produced {points[0][0]:.0f} tokens, so they cannot "
            "separate a per-token cost from a fixed one. Vary the scene."
        )

    per_token, prefill = fit(points)
    if per_token <= 0:
        raise SystemExit(
            "GPU time falls as output length rises across these runs, so "
            "something other than output length differs between them."
        )

    residuals = [(x, y, y - (prefill + per_token * x)) for x, y in points]
    worst = max(abs(r) for _, _, r in residuals)

    # The decomposition is reported for the last run named, which is normally
    # the newest.
    tokens = points[-1][0]
    inference = dig(reports[-1], ("inference_latency_s", "p50"))
    decode = per_token * tokens
    measured_prefill = points[-1][1] - decode

    for (tokens_i, gen), name in zip(points, names):
        print(f"{name}: {gen:.3f} s of GPU for {tokens_i:.0f} tokens")
    print(f"\nfit over {len(points)} run(s): {per_token * 1000:.1f} ms per output token")
    print(f"prefill (the intercept, GPU time at zero tokens): {prefill:.3f} s")
    if len(points) < 3:
        print(
            "\nTwo points fit a line exactly, so this cannot be checked. A third\n"
            "run at a different output length would test it."
        )
    else:
        print(f"\nresiduals, worst {worst * 1000:.0f} ms:")
        for tokens_i, gen, residual in residuals:
            print(
                f"  {tokens_i:3.0f} tokens  predicted {gen - residual:.3f}  "
                f"actual {gen:.3f}  {residual * 1000:+.0f} ms"
            )

    print(f"\ndecomposition of {names[-1]}'s {inference:.3f} s:")
    for label, value in (
        ("processor (CPU)", processor),
        ("prefill (GPU)", measured_prefill),
        ("decode (GPU)", decode),
    ):
        print(f"  {label:<18} {value:6.3f} s  {100 * value / inference:5.1f}%")

    result = {
        "runs": names,
        "points": [[x, round(y, 6)] for x, y in points],
        "per_token_s": round(per_token, 6),
        "prefill_s_fitted": round(prefill, 6),
        "worst_residual_s": round(worst, 6),
        "reported_for": names[-1],
        "inference_s": inference,
        "processor_s": round(processor, 6) if processor is not None else None,
        "prefill_s": round(measured_prefill, 6),
        "decode_s": round(decode, 6),
        "method": "least squares over (tokens, GPU time); see module docstring",
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
