# Jetson setup and Gate 1 runbook

Target: **NVIDIA Jetson Orin Nano Super Developer Kit**, 8 GB, JetPack 6.x.

Gate 1 is a feasibility gate, not a tuning gate. It answers three questions and
then stops:

1. Does the model clear the output contract?
2. How much memory is left with it loaded?
3. How long does one inference take?

Work top to bottom. Steps 1–5 are setup, 6–10 are the milestone. Record every
`TBD` as you go; the tables at the end are the Gate 1 deliverable.

> **Verify version-specific details against NVIDIA's current documentation.**
> JetPack releases move, and the PyTorch install path in particular changes
> between them. Where this runbook says "check the current docs", it means the
> exact command is not stable enough to hard-code here.

---

## 0. What you need on the desk

| item | note |
|---|---|
| Orin Nano Super Dev Kit + its power supply | Use the supplied adapter. An underpowered USB-C brick causes brownouts that look like random kernel panics. |
| microSD, 64 GB, U3/A2 | Slower cards make every step painful. |
| **DisplayPort** cable and a monitor | The dev kit has DisplayPort, **not HDMI**. A passive HDMI adapter usually will not work. This catches almost everyone. |
| USB keyboard and mouse | For first boot only. |
| Ethernet | Simpler than wifi for the first hour. |
| USB webcam (UVC) | Any boring one. Avoid CSI cameras: they add driver and GStreamer work that contributes nothing here. |

Headless is possible over the USB-C serial console, but for a first bring-up a
monitor removes a whole class of confusion.

---

## 1. Flash JetPack

Download the **Jetson Orin Nano Developer Kit SD card image** for JetPack 6.x
from NVIDIA, then write it with Balena Etcher or `dd`. Insert, connect
DisplayPort, keyboard, ethernet, then power.

First boot walks through the usual Ubuntu setup: language, user, timezone. It
reboots once or twice on its own.

```bash
cat /etc/nv_tegra_release        # L4T version
lsb_release -a                   # Ubuntu 22.04 on JetPack 6
python3 --version                # 3.10, which matches this repo's floor
free -h                          # confirm ~8 GB total
```

- JetPack / L4T version: `TBD`

**Storage note.** The NVMe SSD is optional and for model weights only. Do not
move the root filesystem onto it unless it is trivial — boot configuration is
not part of this project.

**If you have a Super dev kit and the Super power modes are missing**, the
firmware may need updating before `nvpmodel` offers them. Check NVIDIA's release
notes for your JetPack version.

---

## 2. Fix one power profile and never change it

```bash
sudo nvpmodel -q                 # list modes and show the current one
sudo nvpmodel -m <N>             # select one
sudo nvpmodel -q                 # confirm it took
sudo jetson_clocks --show        # report clock state
```

- Mode selected: `TBD`
- Rationale: `TBD`

**Do not reflexively pick the highest mode.** If part of the story is
constrained compute, locking to the maximum and then hitting thermal throttling
manufactures a share of your own problem — and the throttling shows up as
unexplained variance in the middle of the sustained load test, where it is
hardest to diagnose. A mid mode is usually the better story and the steadier
measurement.

**Methodology consistency matters more than which mode.** Record it once and
hold it across every measurement in this repository: baseline, overload
comparison, post-optimisation. A comparison across two power modes is not a
comparison.

---

## 3. Instrumentation

```bash
sudo pip3 install jetson-stats
sudo systemctl restart jtop.service
# log out and back in, then:
jtop
```

Confirm before continuing:

- [ ] power rails readable
- [ ] GPU clock readable
- [ ] memory total and used readable

This repo reads the INA3221 rails directly from `/sys` so power lands in the
same CSV row as the latency measurement, rather than in a separate tool's
timeline you would then have to align by hand. Check what it finds:

```bash
python3 -c "from telemetry.metrics import jetson_power_rail_names, read_jetson_power_w; \
print(jetson_power_rail_names(), read_jetson_power_w(), 'W')"
```

- Rails discovered: `TBD`

If the list is empty, the rail paths differ on your JetPack version — note it
and fall back to `jtop` for power, rather than reporting zeros as measurements.

---

## 4. Get the repo and prove the core runs

Before any model, before any camera. This takes two minutes and rules out a
whole class of "is it the board or is it my code" confusion later.

```bash
git clone https://github.com/vivh3/edge-inference-pipeline.git
cd edge-inference-pipeline

python3 -m pytest tests/ -q                                  # 40 tests, no GPU needed
python3 tools/run_overload_sim.py --duration 12 --latency 0.4
```

Both should pass on a bare JetPack image with no extra packages — the core is
stdlib-only. If they do not, stop and fix that first.

---

## 5. Find the webcam

```bash
sudo apt install -y v4l-utils
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

Look at the formats. **MJPG usually reaches higher frame rates than YUYV** at
the same resolution, because YUYV is uncompressed and saturates USB bandwidth.
If 640x480 YUYV tops out at 10 fps, that is your camera, not your code.

- Device and format chosen: `TBD`

---

## 6. Measure what the camera actually delivers

```bash
python3 tools/measure_camera.py --seconds 30 --device 0 --width 640 --height 480
```

This is the milestone's first real measurement. A cheap USB webcam will not hold
30 fps, especially in low light where exposure lengthens. **Drop rate in this
project is computed against frames that actually arrived**, so the real rate is
the denominator for every overload number you will report. Computing against a
nominal 30 would fabricate drops that never happened.

| quantity | value |
|---|---|
| requested fps | 30 |
| **effective fps** | `TBD` |
| interval p50 | `TBD` |
| interval p99 | `TBD` |
| long gaps (driver skips) | `TBD` |

Put the effective figure in the README next to "nominal 30 fps". Written to
`results/baseline/camera.json`.

---

## 7. Install PyTorch — the step that actually bites

**`pip install torch` from PyPI will not give you a working GPU build on
Jetson.** You will get a CPU-only ARM wheel at best, and silently run everything
on the CPU at a fraction of the speed. This is the single most common way to
lose a day here.

Two paths:

- **NVIDIA's Jetson wheels.** NVIDIA publishes PyTorch wheels built against the
  CUDA in your JetPack. Get the current index URL from NVIDIA's Jetson PyTorch
  install page for your exact JetPack version — it changes between releases, so
  a URL written down here would rot.
- **A prebuilt container.** `jetson-containers` (dusty-nv) ships images with
  torch, torchvision and transformers already matched to your JetPack. More
  disk, far less version archaeology. If the wheel path fights you for more
  than an hour, switch to this.

Then install the rest:

```bash
pip3 install "transformers>=4.45" accelerate pillow
```

**Verify before going further**, because everything downstream depends on it:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If that prints `False`, stop. Fix it or switch to the container. Do not proceed
and quietly measure CPU inference.

- torch version: `TBD`
- transformers version: `TBD`

---

## 8. Choose a model

Filters, then a measurement. See the project explainer for the full reasoning;
the short version:

- Open weights, **permissive licence** (Apache-2.0 or MIT are unambiguous)
- Loadable through `AutoProcessor` / `AutoModelForImageTextToText`
- Instruction-tuned (`-Instruct` / `-it` in the name)
- Fits 8 GB **with the OS, CUDA context and KV cache** — start around 2B in
  float16 and only move up if headroom is real

Take 20 photos with your webcam first: hallways, doorways, a bag on the floor, a
dark room, a blank wall. Mix easy and genuinely ambiguous. You will use them to
compare candidates, and later as the quality probe if you need one.

- Model id: `TBD`
- Revision / commit SHA: `TBD` — pin it, "latest" is not reproducible
- Licence, and where you read it: `TBD`

---

## 9. Run the baseline

```bash
python3 tools/gate1_baseline.py --model <org>/<model> --image photos/hallway.jpg --runs 20
```

No camera, no pipeline, no ROS. One image, repeated, warmup discarded. This
isolates the model's cost from everything else.

The script reports memory at three points — before load, after load and warmup,
and steady state — because **quantised weight size is not runtime footprint**. A
4B model at 4-bit is 3–4 GB on disk, but the vision encoder activations, KV
cache, CUDA context, image tensors and the OS all share the same 8 GB, and
physical memory on a Jetson is unified so there is no separate device pool to
fall back on.

| quantity | value |
|---|---|
| inference latency p50 / p90 / p99 | `TBD` |
| output tokens per result | `TBD` |
| **clears `validate()`** | `TBD` / 20 |
| failure breakdown | `TBD` |
| extraction rate | `TBD` |
| total system memory | `TBD` |
| free before load | `TBD` |
| free at steady state | `TBD` |
| mean power | `TBD` |

Written to `results/baseline/model.json`.

**Time to first token** is reported separately if the runtime exposes it
cleanly. It separates prefill (vision encoding, one pass) from decode
(autoregressive, per token) — different costs with different fixes. If getting
it requires invasive instrumentation, skip it. It is not worth fighting the
runtime for.

---

## 10. Gate 1 exit criteria

You are done when all five hold:

- [ ] `jtop` reads power, clocks and memory
- [ ] One power profile selected and written down
- [ ] Effective camera fps measured, not assumed
- [ ] One image produces JSON that **clears `inference.schema.validate`** — not
      "plausible text", the actual validator
- [ ] Baseline latency and real memory headroom recorded

Then fill in the tables above, commit `results/baseline/`, and stop. Gate 1 does
not include tuning, ROS, or the pipeline.

---

## The rules that keep this to two weeks

**Day 2 escape hatch.** If the model is not *boringly* working through
`transformers` by the end of day 2, switch model or runtime immediately. Do not
debug an exotic conversion path. This project studies one slow component; it
does not search for the best one.

**Day 3 freeze.** No model changes after day 3 for performance reasons. After
this point the model is a fixed input, and swapping it invalidates every
measurement you have taken.

**Headroom is uncomfortable?** Move down a size. Nobody cares whether it was 2B
or 4B. The architecture is the deliverable; the parameter count is not.

---

## Common failures, and what they actually are

| symptom | cause |
|---|---|
| No display at boot | HDMI adapter. The board needs DisplayPort. |
| Random freezes or reboots under load | Underpowered supply. Use the one in the box. |
| `torch.cuda.is_available()` is `False` | PyPI wheel instead of NVIDIA's Jetson build. See step 7. |
| Inference 10–50x slower than expected | Same cause. You are running on the CPU. |
| Killed mid-inference, no traceback | Out of memory. The OOM killer is silent. Watch `jtop` during a run. |
| Camera stuck around 10 fps | YUYV instead of MJPG, or low light lengthening exposure. |
| `jtop` says "service not running" | Needs `systemctl restart jtop.service` and a re-login. |
| Power rails list is empty | Path layout differs on your JetPack. Note it, use `jtop`, do not report zeros. |
