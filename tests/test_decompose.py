"""Separating prefill from decode with two baselines.

The arithmetic is three lines; the guards are the point. A decomposition over
two runs that differ in anything but output length is not a decomposition, and
this project has already published a 25% discrepancy that turned out to be a
baseline measured on 640x480 compared against a 448x448 pipeline.
"""

import importlib.util
import json
import os
import sys

import pytest

PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "decompose_inference.py",
)
spec = importlib.util.spec_from_file_location("decompose_inference", PATH)
decompose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decompose)


def report(tokens, inference, processor=0.2641, generate=None, **overrides):
    out = {
        "model": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "generation_policy": {
            "prompt_sha256_prefix": "3b0badbc7204",
            "image_resolution": "448x448",
            "max_new_tokens": 48,
            "decoding": "greedy",
            "seed": 0,
        },
        "inference_latency_s": {"p50": inference},
        "output_tokens": {"p50": float(tokens)},
    }
    if processor is not None:
        out["processor_s"] = {"p50": processor}
    if generate is not None:
        out["generate_s"] = {"p50": generate}
    out.update(overrides)
    return out


def run(tmp_path, a, b, argv_extra=()):
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    out = tmp_path / "decomp.json"
    saved = sys.argv
    sys.argv = ["decompose_inference", str(pa), str(pb), "--out", str(out), *argv_extra]
    try:
        decompose.main()
    finally:
        sys.argv = saved
    return json.loads(out.read_text())


def test_the_real_numbers_reproduce():
    # The committed runs: 5.824 s of GPU for 19 tokens, 6.091 for 22.
    per_token = (6.0911 - 5.8237) / (22 - 19)
    assert per_token == pytest.approx(0.0891, abs=0.0002)
    assert 6.0911 - per_token * 22 == pytest.approx(4.130, abs=0.002)


def test_a_run_predating_the_split_uses_the_other_run_s_processor(tmp_path):
    old = report(19, 6.0878, processor=None)
    new = report(22, 6.3551, generate=6.0911)
    result = run(tmp_path, old, new)
    assert result["generate_s"][0] == pytest.approx(6.0878 - 0.2641)
    assert result["per_token_s"] == pytest.approx(0.0891, abs=0.0002)
    assert result["prefill_s"] == pytest.approx(4.130, abs=0.002)
    assert result["decode_s"] == pytest.approx(1.961, abs=0.002)


def test_prefill_and_decode_and_processor_account_for_inference(tmp_path):
    result = run(
        tmp_path, report(19, 6.0878, processor=None), report(22, 6.3551, generate=6.0911)
    )
    total = result["processor_s"] + result["prefill_s"] + result["decode_s"]
    assert total == pytest.approx(result["inference_s"], abs=0.001)


def test_a_different_resolution_is_refused(tmp_path):
    # The exact mistake that produced the 25% gap.
    a = report(19, 6.0878, generate=5.8237)
    b = report(22, 6.3551, generate=6.0911)
    b["generation_policy"]["image_resolution"] = "640x480"
    with pytest.raises(SystemExit, match="not comparable"):
        run(tmp_path, a, b)


def test_a_different_prompt_is_refused(tmp_path):
    a = report(19, 6.0878, generate=5.8237)
    b = report(22, 6.3551, generate=6.0911)
    b["generation_policy"]["prompt_sha256_prefix"] = "deadbeef0000"
    with pytest.raises(SystemExit, match="not comparable"):
        run(tmp_path, a, b)


def test_a_different_model_is_refused(tmp_path):
    a = report(19, 6.0878, generate=5.8237)
    b = report(22, 6.3551, generate=6.0911)
    b["model"] = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    with pytest.raises(SystemExit, match="not comparable"):
        run(tmp_path, a, b)


def test_equal_token_counts_cannot_separate_anything(tmp_path):
    a = report(22, 6.3551, generate=6.0911)
    b = report(22, 6.3600, generate=6.0960)
    with pytest.raises(SystemExit, match="cannot separate"):
        run(tmp_path, a, b)


def test_more_tokens_but_faster_means_something_else_differs(tmp_path):
    a = report(19, 6.5000, generate=6.2359)
    b = report(22, 6.3551, generate=6.0911)
    with pytest.raises(SystemExit, match="other than output length"):
        run(tmp_path, a, b)


def test_a_run_with_no_split_and_no_fallback_says_so(tmp_path):
    a = report(19, 6.0878, processor=None)
    b = report(22, 6.3551, processor=None)
    with pytest.raises(SystemExit, match="predates the CPU/GPU split"):
        run(tmp_path, a, b)
