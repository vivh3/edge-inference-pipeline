#!/usr/bin/env python3
"""Separate prefill from decode using two baselines at different output lengths.

Generative latency is prefill (vision encoding and prompt processing, once)
plus decode (autoregressive, per token). Telling them apart normally needs
time to first token, which `transformers.generate` does not expose without
instrumenting the generation loop.

Two runs do it instead, as long as everything except output length is held
constant. Decode cost is per token and prefill is not, so:

    per_token = (generate_b - generate_a) / (tokens_b - tokens_a)
    decode    = per_token * tokens
    prefill   = generate - decode

    python3 tools/decompose_inference.py run_a.json run_b.json

`processor_s` is the CPU phase and is subtracted first. Runs taken before that
field existed are handled by passing `--processor-s`, since the phase depends
on resolution and prompt -- both frozen -- and not on output length.

This is a two-point estimate. It assumes prefill is identical across the two
runs, which the frozen 448x448 resolution makes defensible because SmolVLM
tiles by resolution, and that per-token decode cost does not change over the
token range compared. A profiler would measure the split outright; this says
where to point one.
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline_a")
    p.add_argument("baseline_b")
    p.add_argument(
        "--processor-s",
        type=float,
        help="CPU processor phase in seconds, for runs taken before the split "
        "was reported",
    )
    p.add_argument("--out", help="write the decomposition as JSON")
    args = p.parse_args()

    names = (os.path.basename(args.baseline_a), os.path.basename(args.baseline_b))
    a = json.load(open(args.baseline_a))
    b = json.load(open(args.baseline_b))
    check_comparable(a, b, names)

    # Either run may carry the measured phase; a run predating the split uses
    # the other run's value, which is sound because the phase depends on
    # resolution and prompt, both frozen, and not on output length.
    processor = (
        dig(b, ("processor_s", "p50"))
        or dig(a, ("processor_s", "p50"))
        or args.processor_s
    )

    tokens_a = dig(a, ("output_tokens", "p50"))
    tokens_b = dig(b, ("output_tokens", "p50"))
    if tokens_a is None or tokens_b is None:
        raise SystemExit("both runs need output_tokens")
    if tokens_a == tokens_b:
        raise SystemExit(
            f"both runs produced {tokens_a:.0f} tokens, so they cannot separate "
            "a per-token cost from a fixed one. Vary the scene."
        )

    gen_a = generate_s(a, processor, names[0])
    gen_b = generate_s(b, processor, names[1])
    per_token = (gen_b - gen_a) / (tokens_b - tokens_a)
    if per_token <= 0:
        raise SystemExit(
            f"the run with more tokens was faster ({gen_b:.3f} s for "
            f"{tokens_b:.0f} vs {gen_a:.3f} s for {tokens_a:.0f}), so something "
            "other than output length differs between them."
        )

    inference = dig(b, ("inference_latency_s", "p50"))
    decode = per_token * tokens_b
    prefill = gen_b - decode
    parts = [
        ("processor (CPU)", processor),
        ("prefill (GPU)", prefill),
        ("decode (GPU)", decode),
    ]

    print(f"{names[0]}: {gen_a:.3f} s of GPU for {tokens_a:.0f} tokens")
    print(f"{names[1]}: {gen_b:.3f} s of GPU for {tokens_b:.0f} tokens")
    print(f"\nmarginal cost per output token  {per_token * 1000:.1f} ms\n")
    print(f"decomposition of {names[1]}'s {inference:.3f} s:")
    for label, value in parts:
        print(f"  {label:<18} {value:6.3f} s  {100 * value / inference:5.1f}%")
    print(
        "\nTwo-point estimate. It assumes prefill is the same in both runs and\n"
        "that per-token decode cost is flat across the range compared."
    )

    result = {
        "runs": list(names),
        "tokens": [tokens_a, tokens_b],
        "generate_s": [round(gen_a, 6), round(gen_b, 6)],
        "per_token_s": round(per_token, 6),
        "inference_s": inference,
        "processor_s": round(processor, 6) if processor is not None else None,
        "prefill_s": round(prefill, 6),
        "decode_s": round(decode, 6),
        "method": "two baselines at different output lengths; see module docstring",
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
