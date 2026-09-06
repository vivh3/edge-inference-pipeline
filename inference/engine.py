"""Inference engines.

The pipeline talks to one interface, so admission, validation, telemetry and
failure handling can be built and tested before any model exists -- and the
model can be swapped in Gate 1 without touching anything else.

  MockEngine  synthetic slow component: controllable service-time
              distribution, injectable faults. Exercises overload and failure
              paths deterministically on any machine. It measures nothing
              about a real model and its numbers are never reported as
              performance results.

  VlmEngine   the Hugging Face adapter. Gate 1 confirms the concrete model and
              processor classes on hardware; the seam is fixed here so nothing
              downstream depends on that choice.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from .clock import monotonic
from .config import DEFAULT_POLICY, GenerationPolicy
from .record import RawModelOutput

__all__ = ["InferenceEngine", "EngineTimeout", "EngineDied", "MockEngine", "VlmEngine"]


class EngineTimeout(RuntimeError):
    """Generation exceeded the per-frame deadline."""


class EngineDied(RuntimeError):
    """The engine is no longer usable (process died, CUDA error, OOM)."""


class InferenceEngine:
    """Interface. `infer` returns untrusted text; it never parses or judges it."""

    name = "base"

    def describe(self) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def warmup(self, runs: int) -> None:
        """Discarded runs before measurement.

        The first few invocations pay for lazy CUDA context creation, kernel
        autotuning, and allocator growth.  Including them in a latency
        distribution produces a long tail that describes startup, not
        steady-state service time.
        """

    def infer(self, image: Any, deadline: Optional[float] = None) -> RawModelOutput:  # pragma: no cover - abstract
        raise NotImplementedError

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------
# Synthetic engine
# --------------------------------------------------------------------------


class MockEngine(InferenceEngine):
    """A deliberately slow component with controllable pathologies.

    Service time is drawn from a lognormal distribution, which is a
    reasonable shape for generative inference: a floor set by prefill and a
    right tail set by how many tokens the model decides to emit.  The
    parameters are inputs to an experiment, not measurements of anything.

    Fault probabilities let every branch of the failure taxonomy be reached
    without waiting for a real model to misbehave.  Defaults are zero: faults
    are opt-in so an overload experiment is not silently also a failure
    experiment.
    """

    name = "mock"

    def __init__(
        self,
        mean_latency: float = 0.60,
        sigma: float = 0.25,
        policy: GenerationPolicy = DEFAULT_POLICY,
        p_malformed: float = 0.0,
        p_schema_violation: float = 0.0,
        p_unusable: float = 0.0,
        p_timeout: float = 0.0,
        p_die: float = 0.0,
        seed: int = 0,
        sleep: bool = True,
    ) -> None:
        self.mean_latency = mean_latency
        self.sigma = sigma
        self.policy = policy
        self.p_malformed = p_malformed
        self.p_schema_violation = p_schema_violation
        self.p_unusable = p_unusable
        self.p_timeout = p_timeout
        self.p_die = p_die
        self.sleep = sleep
        self._rng = random.Random(seed)
        self._dead = False
        self.calls = 0

    def describe(self) -> dict:
        return {
            "engine": "mock",
            "mean_latency_s": self.mean_latency,
            "sigma": self.sigma,
            "faults": {
                "malformed": self.p_malformed,
                "schema_violation": self.p_schema_violation,
                "unusable": self.p_unusable,
                "timeout": self.p_timeout,
                "die": self.p_die,
            },
            "note": "synthetic; not a measurement of any real model",
        }

    def _service_time(self) -> float:
        # lognormal with the requested arithmetic mean
        import math

        mu = math.log(self.mean_latency) - 0.5 * self.sigma**2
        return self._rng.lognormvariate(mu, self.sigma)

    def _spend(self, seconds: float) -> None:
        if not self.sleep:
            return
        # Busy-free wait; the pipeline is I/O-shaped here, not CPU-bound.
        import time

        time.sleep(max(0.0, seconds))

    def infer(self, image: Any, deadline: Optional[float] = None) -> RawModelOutput:
        if self._dead:
            raise EngineDied("engine previously died and was not restarted")

        self.calls += 1
        start = monotonic()
        service = self._service_time()
        roll = self._rng.random()

        if roll < self.p_die:
            self._spend(service * 0.3)
            self._dead = True
            raise EngineDied("simulated engine process death")

        if roll < self.p_die + self.p_timeout:
            # Overrun the deadline rather than finishing late by a hair.
            budget = (deadline - start) if deadline is not None else service
            self._spend(min(budget * 1.5, budget + 0.5))
            raise EngineTimeout(f"exceeded deadline after {monotonic() - start:.3f}s")

        self._spend(service)
        end = monotonic()

        if deadline is not None and end > deadline:
            raise EngineTimeout(f"exceeded deadline after {end - start:.3f}s")

        roll2 = self._rng.random()
        if roll2 < self.p_malformed:
            text = "The path ahead appears blocked on the left side."
        elif roll2 < self.p_malformed + self.p_schema_violation:
            text = '{"path_status": "obstructed", "obstacle_location": "left"}'
        elif roll2 < self.p_malformed + self.p_schema_violation + self.p_unusable:
            text = '{"path_status": "clear", "obstacle_location": "front_left"}'
        else:
            status, location = self._rng.choice(
                [
                    ("clear", "none"),
                    ("blocked", "front_left"),
                    ("blocked", "front_center"),
                    ("blocked", "front_right"),
                    ("unknown", "unknown"),
                ]
            )
            text = f'{{"path_status": "{status}", "obstacle_location": "{location}"}}'

        return RawModelOutput(
            text=text,
            inference_start_ts=start,
            inference_end_ts=end,
            output_tokens=len(text) // 4,
        )


# --------------------------------------------------------------------------
# Hugging Face VLM adapter (Gate 1)
# --------------------------------------------------------------------------


class VlmEngine(InferenceEngine):
    """Adapter for an open-weights vision-language model via `transformers`.

    Deliberately thin.  Everything interesting -- admission, validation,
    telemetry, failure handling -- lives outside it, so replacing this class
    with a different runtime is a contained change.

    Gate 1's job on hardware is to confirm three things and then stop:
    the concrete model id, that `AutoModelForImageTextToText` is the right
    class for it, and the real memory headroom.  Do not tune here.

    The deadline is enforced inside generation via a stopping criterion that
    checks the monotonic clock between tokens.  That is the honest place for
    it: a wall-clock alarm outside the call cannot interrupt a kernel that is
    already running, so a deadline enforced from outside would only ever be
    detected after the fact.
    """

    name = "vlm"

    def __init__(
        self,
        model_id: str,
        policy: GenerationPolicy = DEFAULT_POLICY,
        device: str = "cuda",
        dtype: str = "float16",
    ) -> None:
        self.model_id = model_id
        self.policy = policy
        self.device = device
        self.dtype = dtype
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # type: ignore
        from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore

        torch_dtype = getattr(torch, self.dtype)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, torch_dtype=torch_dtype
        ).to(self.device)
        self._model.eval()

    def describe(self) -> dict:
        return {
            "engine": "vlm",
            "model_id": self.model_id,
            "device": self.device,
            "dtype": self.dtype,
            "generation": self.policy.describe(),
        }

    def warmup(self, runs: int) -> None:
        self._load()
        blank = _blank_image(self.policy.image_width, self.policy.image_height)
        for _ in range(runs):
            self.infer(blank)

    def infer(self, image: Any, deadline: Optional[float] = None) -> RawModelOutput:
        self._load()
        import torch  # type: ignore

        start = monotonic()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.policy.prompt},
                ],
            }
        ]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        stopping = _deadline_stopping_criteria(deadline)
        try:
            with torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=self.policy.max_new_tokens,
                    do_sample=not self.policy.greedy,
                    stopping_criteria=stopping,
                )
        except RuntimeError as exc:  # CUDA OOM and friends are not recoverable here
            raise EngineDied(str(exc)) from exc

        end = monotonic()
        if deadline is not None and end > deadline:
            raise EngineTimeout(f"exceeded deadline after {end - start:.3f}s")

        new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
        text = self._processor.decode(new_tokens, skip_special_tokens=True)
        return RawModelOutput(
            text=text,
            inference_start_ts=start,
            inference_end_ts=end,
            output_tokens=int(new_tokens.shape[-1]),
        )

    def close(self) -> None:
        self._model = None
        self._processor = None


def _deadline_stopping_criteria(deadline: Optional[float]):
    if deadline is None:
        return None
    from transformers import StoppingCriteria, StoppingCriteriaList  # type: ignore

    class _Deadline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs) -> bool:
            return monotonic() >= deadline

    return StoppingCriteriaList([_Deadline()])


def _blank_image(width: int, height: int):
    from PIL import Image  # type: ignore

    return Image.new("RGB", (width, height), color=(0, 0, 0))
