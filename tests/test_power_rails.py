"""Reading the INA3221 rails out of sysfs.

Written against the layout on a Jetson Orin Nano Super, JetPack 6.2 / L4T
36.4.3: hwmon reports bus voltage and current on separate channels, and the
rails are nested rather than disjoint.
"""

from telemetry import metrics


def write(path, text):
    path.write_text(str(text))


def ina3221(tmp_path, channels, name="ina3221", index=1):
    """Build a fake hwmon tree. `channels` is {n: (label, mV, mA)}; mA None
    means a voltage-only channel, as the shunt-voltage sum is."""
    root = tmp_path / "hwmon"
    root.mkdir(exist_ok=True)
    hwmon = root / ("hwmon%d" % index)
    hwmon.mkdir()
    write(hwmon / "name", name)
    for n, (label, millivolts, milliamps) in channels.items():
        write(hwmon / ("in%d_label" % n), label)
        write(hwmon / ("in%d_input" % n), millivolts)
        if milliamps is not None:
            write(hwmon / ("curr%d_input" % n), milliamps)
    return str(root)


ORIN_NANO = {
    1: ("VDD_IN", 5000, 880),            # 4.4 W
    2: ("VDD_CPU_GPU_CV", 5000, 240),    # 1.2 W
    3: ("VDD_SOC", 5000, 260),           # 1.3 W
    7: ("sum of shunt voltages", 120, None),
}


def test_watts_come_from_volts_times_amps(tmp_path):
    rails = dict(metrics._power_rails(ina3221(tmp_path, ORIN_NANO)))
    assert rails["VDD_IN"] == 4.4
    assert rails["VDD_CPU_GPU_CV"] == 1.2
    assert rails["VDD_SOC"] == 1.3


def test_a_voltage_channel_with_no_current_is_not_a_rail(tmp_path):
    # "sum of shunt voltages" has in7_input but no curr7_input. Treating it
    # as a rail would multiply a voltage by a missing current.
    rails = dict(metrics._power_rails(ina3221(tmp_path, ORIN_NANO)))
    assert "sum of shunt voltages" not in rails


def test_the_input_rail_is_reported_alone_not_summed_with_its_children(tmp_path):
    # VDD_CPU_GPU_CV and VDD_SOC sit downstream of VDD_IN. Summing all three
    # double-counts (6.9 W); summing only the children undercounts (2.5 W).
    # The board draws 4.4 W.
    root = ina3221(tmp_path, ORIN_NANO)
    assert metrics._reported_rails(root) == [("VDD_IN", 4.4)]


def test_rail_names_say_which_rail_the_figure_came_from(tmp_path):
    root = ina3221(tmp_path, ORIN_NANO)
    assert [label for label, _ in metrics._reported_rails(root)] == ["VDD_IN"]


def test_boards_without_an_input_rail_fall_back_to_summing(tmp_path):
    root = ina3221(tmp_path, {1: ("VDD_GPU", 5000, 200), 2: ("VDD_CPU", 5000, 300)})
    assert sorted(metrics._reported_rails(root)) == [("VDD_CPU", 1.5), ("VDD_GPU", 1.0)]


def test_non_ina3221_hwmon_devices_are_ignored(tmp_path):
    # hwmon0 on this board is the fan controller, hwmon2 a tachometer.
    root = ina3221(tmp_path, {1: ("fan", 1, 1)}, name="pwmfan", index=0)
    ina3221(tmp_path, ORIN_NANO, index=1)
    assert metrics._reported_rails(root) == [("VDD_IN", 4.4)]


def test_absent_sysfs_reads_as_empty_rather_than_zero_watts(tmp_path):
    # Off-target (laptop, CI) there are no rails. An empty list is honest;
    # a 0.0 W reading would look like a measurement.
    assert metrics._power_rails(str(tmp_path / "nothing-here")) == []


def test_available_memory_is_read_and_is_not_memfree(tmp_path, monkeypatch):
    # MemAvailable, not MemFree: reading several GB of weights fills the page
    # cache, which is not free but is reclaimable, so MemFree would report a
    # shortage that is not real.
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:        7802112 kB\nMemFree:          460800 kB\n"
        "MemAvailable:    6900000 kB\nCached:          5000000 kB\n"
    )
    real_open = open

    def fake_open(path, *a, **kw):
        return real_open(meminfo if path == "/proc/meminfo" else path, *a, **kw)

    monkeypatch.setattr("builtins.open", fake_open)
    assert metrics.read_system_available_bytes() == 6900000 * 1024


def test_missing_meminfo_reads_as_zero_rather_than_raising(tmp_path, monkeypatch):
    real_open = open

    def fake_open(path, *a, **kw):
        if path == "/proc/meminfo":
            raise OSError("no /proc here")
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", fake_open)
    assert metrics.read_system_available_bytes() == 0
