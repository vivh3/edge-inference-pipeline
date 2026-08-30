"""Frozen generation policy.

Everything in this file is held constant across every configuration that is
ever compared -- baseline, optimised, FIFO, latest-frame.  It lives in one
module so "was anything else different?" has a one-file answer.

The reason this matters: generative inference latency is dominated by how
many tokens are produced.  A configuration that happens to emit a chattier
answer will look slower, and the difference will be read as a system effect
when it is a generation-policy artifact.  Fixing the prompt, the image
resolution, the maximum output length, and the decoding strategy removes
that confound.

Note the limit of the guarantee: GPU execution is not necessarily
bit-deterministic even under greedy decoding with a fixed seed, because
reduction orders in fused kernels can vary.  Minor output variation is
expected and does not invalidate the timing work.  What is being controlled
here is output *length* and generation policy, not bit-exact output.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["GenerationPolicy", "DEFAULT_POLICY", "PROMPT"]

# Kept short on purpose: a long instruction inflates prefill cost, and prefill
# is one of the stages the profiling in Gate 3 needs to be able to see.
PROMPT = (
    "You are a robot's forward-facing camera analyser. "
    "Answer only with a JSON object and nothing else, in this exact form:\n"
    '{"path_status": "<clear|blocked|unknown>", '
    '"obstacle_location": "<front_left|front_center|front_right|left|right|none|unknown>"}\n'
    "Use \"unknown\" for both fields if the image is too unclear to judge. "
    "Use \"none\" for obstacle_location only when path_status is \"clear\"."
)


@dataclass(frozen=True)
class GenerationPolicy:
    prompt: str = PROMPT
    image_width: int = 448
    image_height: int = 448
    max_new_tokens: int = 48  # the schema needs ~25; the margin catches run-on output
    greedy: bool = True       # temperature 0 / do_sample=False where supported
    seed: int = 0
    warmup_runs: int = 5      # discarded before any measurement (see docs/PERFORMANCE.md)

    def describe(self) -> dict:
        return {
            "prompt_sha256_prefix": _sha_prefix(self.prompt),
            "prompt_chars": len(self.prompt),
            "image_resolution": f"{self.image_width}x{self.image_height}",
            "max_new_tokens": self.max_new_tokens,
            "decoding": "greedy" if self.greedy else "sampled",
            "seed": self.seed,
            "warmup_runs": self.warmup_runs,
        }


def _sha_prefix(text: str, n: int = 12) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


DEFAULT_POLICY = GenerationPolicy()
