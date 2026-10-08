"""Parsing a tegrastats log.

The sample lines are copied from a real log on a Jetson Orin Nano Super,
JetPack 6.2 / L4T 36.4.3, because the field layout varies between releases and
a hand-written approximation would test the approximation.
"""

import importlib.util
import os
from datetime import datetime, timedelta

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


def stamped(line, when):
    """Replace a sample's timestamp. The fixtures carry different ones."""
    return when + line[len("10-08-2026 04:38:21"):]


def summarize(tmp_path, lines, argv_extra=(), restamp=True):
    if restamp:
        # One second apart, so the log reads as a single contiguous run. The
        # IDLE and BUSY fixtures were copied from different moments of a real
        # log and their stamps are minutes apart, which the segmenter would
        # read as several runs.
        base = datetime(2026, 10, 8, 4, 30)
        lines = [
            stamped(line, (base + timedelta(seconds=i)).strftime("%m-%d-%Y %H:%M:%S"))
            for i, line in enumerate(lines)
        ]
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
    summary = summarize(tmp_path, [IDLE] * 7 + [BUSY] * 5)
    assert summary["samples"] == 12
    assert summary["busy_samples_before_trim"] == 5
    assert summary["busy_samples"] == 3  # one trimmed from each end
    assert summary["idle_samples"] == 7
    # The whole point of the split: a mean over all twelve samples would report
    # a GPU neither busy nor idle ever was.
    assert summary["busy"]["gpu_pct"]["p50"] == 99.0
    assert summary["idle"]["gpu_pct"]["p50"] == 0.0


def test_a_sagging_clock_is_visible(tmp_path):
    # The sagged sample sits mid-stretch so trimming cannot remove it.
    sagged = BUSY.replace("99%@[612]", "99%@[510]")
    summary = summarize(tmp_path, [IDLE, BUSY, BUSY, sagged, BUSY, IDLE])
    gpu = summary["busy"]["gpu_mhz"]
    assert gpu["min"] == 510.0 and gpu["max"] == 612.0


def test_a_log_with_no_busy_window_is_an_error_not_a_summary(tmp_path):
    # Reporting idle percentiles as if they described the run is the failure
    # mode worth being loud about.
    with pytest.raises(SystemExit):
        summarize(tmp_path, [IDLE] * 5)


# --------------------------------------------------------------------------
# Edge trimming
#
# tegrastats reports the mean over its interval, so a sample straddling the
# start or end of the busy window mixes a working GPU with an idle one. The
# first real log this tool saw reported "CPU busiest core: 0%" and "VDD_IN:
# 11.5 W" as busy minima, neither of which can happen during an inference.
# --------------------------------------------------------------------------

# 99% GPU but idle-looking everything else: what a straddling sample looks like.
EDGE = BUSY.replace("38%@1497", "0%@1497").replace("16321mW", "11500mW")


def test_trimming_drops_the_straddling_samples(tmp_path):
    summary = summarize(tmp_path, [IDLE] + [EDGE] + [BUSY] * 4 + [EDGE] + [IDLE])
    assert summary["busy_samples_before_trim"] == 6
    assert summary["busy_samples"] == 4
    assert summary["busy"]["cpu_busiest_pct"]["min"] == 38.0
    assert summary["busy"]["vdd_in_w"]["min"] == pytest.approx(16.321)


def test_trimming_can_be_turned_off(tmp_path):
    summary = summarize(
        tmp_path, [IDLE] + [EDGE] + [BUSY] * 4 + [EDGE] + [IDLE], ("--trim-edges", "0")
    )
    assert summary["busy_samples"] == 6
    assert summary["busy"]["vdd_in_w"]["min"] == pytest.approx(11.5)


def test_each_stretch_is_trimmed_separately(tmp_path):
    # Warmup and the measured runs are separated by a gap, so a log has more
    # than one busy stretch and each has its own two edges.
    summary = summarize(
        tmp_path, [BUSY] * 5 + [IDLE] * 3 + [BUSY] * 5 + [IDLE]
    )
    assert summary["busy_stretches"] == 2
    assert summary["busy_samples"] == 6


def test_a_stretch_too_short_to_trim_is_dropped_whole(tmp_path):
    summary = summarize(tmp_path, [BUSY] * 5 + [IDLE] * 3 + [BUSY] + [IDLE])
    assert summary["busy_stretches"] == 2
    assert summary["dropped_short_runs"] == 1
    assert summary["busy_samples"] == 3


def test_all_stretches_too_short_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        summarize(tmp_path, [IDLE, BUSY, IDLE, BUSY, IDLE])


# --------------------------------------------------------------------------
# Drift
#
# A distribution says what a run cost; it cannot say whether the run settled.
# The real log shows no clock sag and tj climbing 6.5 C from the first ten
# seconds to the last, which scopes "not thermally limited" to a run of that
# length rather than to the board.
# --------------------------------------------------------------------------

WARM = BUSY.replace("tj@58.312C", "tj@61.400C")


def test_drift_compares_the_ends_of_one_stretch():
    run = [tegrastats.parse_line(BUSY)] * 2 + [tegrastats.parse_line(WARM)] * 2
    moved = tegrastats.drift(run, window=2)
    assert moved["tj_c"]["first"] == pytest.approx(58.312)
    assert moved["tj_c"]["last"] == pytest.approx(61.4)
    assert moved["tj_c"]["delta"] == pytest.approx(3.088)


def test_a_stretch_too_short_for_two_windows_gets_no_drift():
    run = [tegrastats.parse_line(BUSY)] * 5
    assert tegrastats.drift(run, window=2) is not None
    assert tegrastats.drift(run, window=3) is None


def test_drift_uses_the_longest_stretch_not_the_whole_busy_set(tmp_path):
    # Weight loading and warmup are their own short stretches. Comparing
    # first-to-last across all of them would call the difference between
    # loading and inference "drift".
    loading = BUSY.replace("tj@58.312C", "tj@40.000C")
    summary = summarize(
        tmp_path,
        [loading] * 3 + [IDLE] + [BUSY] * 4 + [WARM] * 4 + [IDLE],
        ("--drift-window", "2"),
    )
    assert summary["busy_stretches"] == 2
    assert summary["drift"]["stretch_samples"] == 6
    # 40 C never appears: the loading stretch is not the longest one.
    assert summary["drift"]["fields"]["tj_c"]["first"] == pytest.approx(58.312)
    assert summary["drift"]["fields"]["tj_c"]["last"] == pytest.approx(61.4)


def test_drift_is_absent_rather_than_guessed_when_the_run_is_short(tmp_path):
    summary = summarize(tmp_path, [IDLE] + [BUSY] * 5 + [IDLE])
    assert summary["drift"]["fields"] is None


# --------------------------------------------------------------------------
# Several runs in one logfile
#
# `tegrastats --logfile` appends. Reusing a filename across runs left one file
# holding four, and summarising all of them gave 65 busy stretches instead of
# 33 and an "idle" CPU figure above the busy one -- the idle samples were two
# runs' weight-loading phases, which are CPU-bound with the GPU parked.
# --------------------------------------------------------------------------


def test_a_contiguous_log_is_one_run():
    samples = [
        tegrastats.parse_line(stamped(BUSY, f"10-08-2026 04:38:{s:02d}"))
        for s in range(20, 25)
    ]
    assert tegrastats.segments(samples, max_gap_s=5.0) == [(0, 5)]


def test_a_time_gap_starts_a_new_run():
    times = ["04:38:20", "04:38:21", "04:45:00", "04:45:01"]
    samples = [
        tegrastats.parse_line(stamped(BUSY, f"10-08-2026 {t}")) for t in times
    ]
    assert tegrastats.segments(samples, max_gap_s=5.0) == [(0, 2), (2, 4)]


def test_a_log_without_timestamps_is_still_one_run():
    # Not every release prints a stamp; losing the segmentation is acceptable,
    # crashing is not.
    samples = [{"gpu_pct": 99.0, "gpu_mhz": 612.0}] * 3
    assert tegrastats.segments(samples, max_gap_s=5.0) == [(0, 3)]


def test_the_last_run_is_summarised_by_default(tmp_path, capsys):
    early = [stamped(IDLE, f"10-08-2026 04:30:{s:02d}") for s in range(0, 6)]
    late = [stamped(BUSY, f"10-08-2026 05:00:{s:02d}") for s in range(0, 6)]
    summary = summarize(tmp_path, early + late, restamp=False)
    assert summary["runs_in_logfile"] == 2
    assert summary["run_summarised"] == 2
    assert summary["samples"] == 6
    assert summary["busy"]["gpu_pct"]["p50"] == 99.0


def test_an_earlier_run_can_be_chosen(tmp_path):
    early = [stamped(BUSY, f"10-08-2026 04:30:{s:02d}") for s in range(0, 6)]
    late = [stamped(IDLE, f"10-08-2026 05:00:{s:02d}") for s in range(0, 6)]
    summary = summarize(tmp_path, early + late, ("--segment", "1"), restamp=False)
    assert summary["run_summarised"] == 1
    assert summary["busy"]["gpu_pct"]["p50"] == 99.0


# --------------------------------------------------------------------------
# The verdict reads a percentile, not an extremum
# --------------------------------------------------------------------------


def test_one_dip_in_five_hundred_is_not_throttling():
    # The real run: 533 samples at 611 MHz and one at 509. A min-versus-max
    # verdict called it throttling.
    message = tegrastats.clock_verdict([611.0] * 533 + [509.0])
    assert "held the ceiling" in message
    assert "not clock-limited" in message
    assert "1 of 534 samples dipped" in message


def test_a_sustained_sag_is_still_throttling():
    message = tegrastats.clock_verdict([420.0] * 300 + [612.0] * 100)
    assert "sagged" in message
    assert "Check tj and VDD_IN" in message


def test_a_clock_pinned_throughout_says_so():
    message = tegrastats.clock_verdict([612.0] * 200)
    assert "held 612 MHz across all 200" in message
