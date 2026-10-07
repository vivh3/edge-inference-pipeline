"""Frozen generation policy.

Held constant across every configuration that is ever compared: baseline,
optimised, FIFO, latest-frame. One module, so "was anything else different?"
has a one-file answer.

Generative latency is dominated by how many tokens come out. A configuration
that emits a chattier answer looks slower, and that reads as a system effect
when it is a generation artifact. Fixing the prompt, resolution, output length
and decoding strategy removes the confound.

The guarantee has a limit. GPU execution is not bit-deterministic even under
greedy decoding with a fixed seed, because reduction orders in fused kernels
vary. What is controlled here is output length and generation policy, not
bit-exact output.
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
