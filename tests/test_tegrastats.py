"""Parsing a tegrastats log.

The sample lines are copied from a real log on a Jetson Orin Nano Super,
JetPack 6.2 / L4T 36.4.3, because the field layout varies between releases and
a hand-written approximation would test the approximation.
"""

import importlib.util
import os

import pytest

PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools",
    "summarize_tegrastats.py",
)
spec = importlib.util.spec_from_file_location("summarize_tegrastats", PATH)
tegrastats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tegrastats)

IDLE = (
    "10-08-2026 04:38:21 RAM 1343/7620MB (lfb 157x4MB) SWAP 165/3810MB (cached 3MB) "
    "CPU [0%@1497,0%@729,0%@729,0%@729,0%@729,0%@729] EMC_FREQ 15%@204 "
    "GR3D_FREQ 0%@[305] NVDEC off NVJPG off VIC off OFA off APE 200 "
    "cpu@48.562C soc2@46.593C soc0@48.625C gpu@49.625C tj@49.312C soc1@47.937C "
    "VDD_IN 5251mW/8535mW VDD_CPU_GPU_CV 512mW/1944mW VDD_SOC 2720mW/3659mW"
)
BUSY = (
    "10-08-2026 04:35:02 RAM 6712/7620MB (lfb 12x4MB) SWAP 369/3810MB (cached 8MB) "
    "CPU [38%@1497,2%@729,1%@729,0%@729,1%@729,0%@729] EMC_FREQ 61%@2133 "
    "GR3D_FREQ 99%@[612] NVDEC off NVJPG off VIC off OFA off APE 200 "
    "cpu@57.125C soc2@55.093C soc0@56.625C gpu@58.625C tj@58.312C soc1@55.937C "
    "VDD_IN 16321mW/11204mW VDD_CPU_GPU_CV 7104mW/4944mW VDD_SOC 3120mW/3659mW"
)


def test_parses_every_field():
    s = tegrastats.parse_line(BUSY)
    assert s["gpu_pct"] == 99.0
    assert s["gpu_mhz"] == 612.0
    assert s["emc_pct"] == 61.0
    assert s["emc_mhz"] == 2133.0
    assert s["ram_used_mb"] == 6712.0
    assert s["swap_used_mb"] == 369.0
    assert s["tj_c"] == 58.312
    assert s["cpu_busiest_pct"] == 38.0
    assert s["cpu_total_pct"] == 42.0


def test_vdd_in_takes_the_instantaneous_value_not_the_average():
    # tegrastats prints inst/avg and the average is cumulative over the whole
    # log, so using it would fold the idle brackets back into a busy figure.
    assert tegrastats.parse_line(BUSY)["vdd_in_w"] == pytest.approx(16.321)


def test_clock_without_brackets_still_parses():
    # Older L4T releases print GR3D_FREQ without the bracketed clock.
    assert tegrastats.parse_line(BUSY.replace("99%@[612]", "99%@612"))["gpu_mhz"] == 612.0


def test_a_line_without_a_gpu_field_is_not_a_sample():
    assert tegrastats.parse_line("some other log line") is None


def summarize(tmp_path, lines, argv_extra=()):
    log = tmp_path / "tegrastats.log"
    log.write_text("\n".join(lines) + "\n")
    out = tmp_path / "summary.json"
    import sys

    argv = sys.argv
    sys.argv = ["summarize_tegrastats", str(log), "--out", str(out), *argv_extra]
    try:
        tegrastats.main()
    finally:
        sys.argv = argv
    import json

    return json.loads(out.read_text())


def test_splits_busy_from_idle(tmp_path):
    summary = summarize(tmp_path, [IDLE] * 7 + [BUSY] * 3)
    assert summary["samples"] == 10
    assert summary["busy_samples"] == 3
    assert summary["idle_samples"] == 7
    # The whole point of the split: a mean over all ten samples would report a
    # GPU neither busy nor idle ever was.
    assert summary["busy"]["gpu_pct"]["p50"] == 99.0
    assert summary["idle"]["gpu_pct"]["p50"] == 0.0


def test_a_sagging_clock_is_visible(tmp_path):
    sagged = BUSY.replace("99%@[612]", "99%@[510]")
    summary = summarize(tmp_path, [IDLE] + [BUSY] * 2 + [sagged])
    gpu = summary["busy"]["gpu_mhz"]
    assert gpu["min"] == 510.0 and gpu["max"] == 612.0


def test_a_log_with_no_busy_window_is_an_error_not_a_summary(tmp_path):
    # Reporting idle percentiles as if they described the run is the failure
    # mode worth being loud about.
    with pytest.raises(SystemExit):
        summarize(tmp_path, [IDLE] * 5)
