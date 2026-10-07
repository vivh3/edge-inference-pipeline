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
move the root filesystem onto it unless it is trivial. Boot configuration is
not part of this project.

**If you have a Super dev kit and the Super power modes are missing**, the
firmware may need updating before `nvpmodel` offers them. Check NVIDIA's release
notes for your JetPack version.

---

## 2. Fix one power profile and never change it

```bash
grep -E '^< POWER_MODEL' /etc/nvpmodel.conf   # the modes this board offers
sudo nvpmodel -q                              # the mode it is in now
sudo nvpmodel -m <N>                          # select one
sudo nvpmodel -q                              # confirm it took
sudo jetson_clocks --show                     # report clock state
```

`nvpmodel -q` reports only the current mode, so the `grep` is what lists them.

**Mode selected: `0` (`15W`).** Measured on an Orin Nano 8GB Super, JetPack 6.2
/ L4T R36.4.3, which offers `0: 15W`, `1: 25W`, `2: MAXN_SUPER`.

**Rationale: 15W is the only one of the three that also exists on the non-Super
Orin Nano.** Numbers published at 25W cannot be reproduced on that board at all,
and reproducibility from a fresh clone is a constraint this repository holds
itself to. MAXN_SUPER is uncapped, so it throttles under sustained load and the
throttling arrives as unexplained variance in the latency tail, the exact place
the overload claim is made.

What the choice costs, from `jetson_clocks --show` in each mode:

| cap | mode 0 (15W) | mode 1 (25W) |
| --- | --- | --- |
| CPU max | 1497.6 MHz | 1344.0 MHz |
| GPU max | 612 MHz | 918 MHz |
| EMC max | 2133 MHz | 3199 MHz |

Six A78 cores online and 4 GPU TPCs active in both; governor `schedutil`.

The CPU cap being *higher* in the lower-power mode is not a typo. A power mode
is a budget allocation, not a single dial: 15W spends more of a smaller budget
on CPU and less on GPU and memory. The real cost is the other two rows. Token
decode re-reads the model weights once per token, so it is bound by memory
bandwidth rather than arithmetic, and the 33% lower EMC ceiling is the figure
that will show up in decode latency. Expect inference meaningfully slower than
this board can go. That is acceptable: a wider gap between the 30 fps sensor
and the model sharpens the comparison the project exists to make.

Revisit only if Gate 1 is too slow to iterate against, and then retake every
measurement at the new mode.

**Methodology consistency matters more than which mode.** Record it once and
hold it across every measurement in this repository: baseline, overload
comparison, post-optimisation. A comparison across two power modes is not a
comparison.

---

## 3. Instrumentation

`pip3` is not in the JetPack image, and the `jtop` service is created by the
install rather than existing beforehand:

```bash
sudo apt install -y python3-pip
sudo pip3 install -U jetson-stats
sudo jtop --install-service
# log out and back in, then:
jtop
```

Confirm before continuing:

- [ ] power rails readable
- [ ] GPU clock readable
- [ ] memory total and used readable

Note the idle memory figure. GPU allocations come out of the same pool as the
OS; there is no separate VRAM. Measured on this board, of 7619 MB total:

| state | used | available |
| --- | --- | --- |
| desktop running | ~2300 MB | ~5100 MB |
| `multi-user.target` | 380 MB | 7010 MB |

Run headless before loading a model. `sudo systemctl set-default
multi-user.target`, and `graphical.target` to put it back. Confirm SSH works
first: a headless board you cannot reach needs a monitor to recover.

This repo reads the INA3221 rails directly from `/sys` so power lands in the
same CSV row as the latency measurement, rather than in a separate tool's
timeline you would then have to align by hand. Check what it finds:

```bash
python3 -c "from telemetry.metrics import jetson_power_rail_names, read_jetson_power_w; \
print(jetson_power_rail_names(), read_jetson_power_w(), 'W')"
```

- Rails discovered: `['VDD_IN']`, on JetPack 6.2 / L4T 36.4.3.

The board exposes three channels, `VDD_IN`, `VDD_CPU_GPU_CV` and `VDD_SOC`, but
they overlap: the latter two are measured downstream of the first. Only the input rail is reported, because summing all three counts the
same current twice and summing the two children misses everything on the board
that is neither. Cross-check against `jtop`: the figure should match its
`VDD_IN` row, not the sum of the rows above it.

If the list is empty, the paths differ on your JetPack version. Compare against
`ls /sys/class/hwmon/hwmon*/` and `grep -H . /sys/class/hwmon/hwmon*/in*_label`,
and until it is fixed use `jtop` for power rather than reporting zeros as
measurements.

---

## 4. Get the repo and prove the core runs

Before any model, before any camera. This takes two minutes and rules out a
whole class of "is it the board or is it my code" confusion later.

While the repository is private, clone over SSH. GitHub does not accept
password authentication for git, so the HTTPS URL only prompts and fails:

```bash
ssh-keygen -t ed25519            # then add ~/.ssh/id_ed25519.pub at github.com/settings/keys
ssh -T git@github.com            # expect "Hi <user>! You've successfully authenticated"

git clone git@github.com:vivh3/edge-inference-pipeline.git
cd edge-inference-pipeline

sudo apt install -y python3-pytest     # the only dependency, and only to run the tests
python3 -m pytest tests/ -q            # no GPU needed
python3 tools/run_overload_sim.py --duration 12 --latency 0.4   # quick shape check
```

Both should pass on a bare JetPack image: the core imports nothing outside the
standard library, and a test runner is the only thing added to exercise it. If
they do not pass, stop and fix that first.

The simulation writes into `results/simulated/`, overwriting the committed
summaries that the README table is checked against. Running it here is a
reproduction check, not new data, so discard the result afterwards:

```bash
git checkout -- results/
```

---

## 5. Find the webcam

```bash
sudo apt install -y v4l-utils
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

Plug the camera into a port on the Jetson itself, not a hub. A hub shares
bandwidth across everything on it, and the result arrives as dropped and late
frames in section 6, where the point is to find out what the camera delivers,
not what the hub allows.

Read `--list-formats-ext` for the format that sustains 30 fps at a size at or
above the model's input resolution. On the camera used here (j5 JVCU100) that
is **MJPG at 640x480**: YUYV is offered only at 1024x576 and above, and only at
20 fps or less, so uncompressed capture cannot reach 30 fps at any useful size.

- Camera: `j5create JVCU100`, UVC, on `/dev/video0`.
- Format selected: `MJPG 640x480 @ 30 fps`, V4L2 backend, 2 capture buffers.
- Measured delivery: `30.027 fps`, interval p50 32.1 ms / p99 36.6 ms, no long gaps.

640x480 is the smallest MJPG mode at or above the model's 448x448 input, so it
reaches the model without upscaling and keeps USB bandwidth and JPEG decode to
a minimum. It is 4:3 against a square input, so preprocessing stretches rather
than crops. Uniform across every frame, so it biases no comparison, but it is
a choice rather than an accident.

### One capture buffer halves the frame rate

Found on this hardware, and worth repeating before trusting any camera number.

`CAP_PROP_BUFFERSIZE` sets how many buffers the V4L2 driver gets. With exactly
one, the application holds the only buffer while it works on a frame, the
sensor's next frame arrives with nowhere to go, and the driver drops it. The
result is exactly half the frame rate, at every resolution:

```
buffersize 1: 12.5 fps        buffersize 3: 25.0 fps
buffersize 2: 25.0 fps        buffersize 4: 25.0 fps
```

Resolution independence is what identifies it. Bandwidth, sensor readout and
JPEG decode all scale with frame size; this does not. Streaming the same format
straight from the kernel (`v4l2-ctl --stream-mmap`, which defaults to four
buffers) reached ~28 fps on the same camera, which is what ruled out the camera
itself.

The reason it matters beyond the number: those frames are discarded inside the
driver, before capture stamps a `frame_id`. Nothing downstream can see or count
them, so a reported drop rate would silently describe half the input. A single
buffer looks like it minimises staleness and actually destroys frames with no
accounting, the opposite of what this project argues for. Keep driver-side
queueing at the shallow minimum (two: one held, one filling) and let the
admission policy handle staleness where the decision is explicit and counted.

`--buffer-frames 1` reproduces it.


JetPack ships an OpenCV built with GStreamer, and OpenCV prefers it. GStreamer
does not honour the pixel format request: it logs `unhandled property` and may
fail to start a pipeline at all. The capture code names the V4L2 backend
explicitly for that reason. If `measure_camera.py` prints GStreamer warnings,
that is what is happening; `--backend v4l2` is already the default.

**Exposure is left on auto.** `v4l2-ctl --set-ctrl` changes are runtime state
and reset on reboot, so pinning exposure would mean a setup step to forget or
a service to maintain. It buys nothing here: the frame rate was limited by
buffer starvation, not exposure, so exposure is not a timing variable and no
latency measurement depends on it. Auto also adapts as the camera moves, which
is what a varied probe set wants. Set it manually only if a measurement turns
out to depend on it, and write down that it did.

MJPG costs a JPEG decode per frame. That decode lands in preprocessing, where
it is measured, rather than disappearing into the queue wait. The Orin has
hardware JPEG decoders that OpenCV's `VideoCapture` does not use, which makes
this a candidate for the bottleneck investigation. Note it and move on.

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

## 7. Install PyTorch: the step that actually bites

**`pip install torch` from PyPI will not give you a working GPU build on
Jetson.** You will get a CPU-only ARM wheel at best, and silently run everything
on the CPU at a fraction of the speed. This is the single most common way to
lose a day here.

Two paths:

- **NVIDIA's Jetson wheels.** NVIDIA publishes PyTorch wheels built against the
  CUDA in your JetPack. Get the current index URL from NVIDIA's Jetson PyTorch
  install page for your exact JetPack version. It changes between releases, so
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

- torch version: `2.8.0` (torchvision `0.23.0`), from
  `https://pypi.jetson-ai-lab.io/jp6/cu126`
- transformers version: `5.18.0`, accelerate `1.15.0`
- numpy held at `1.21.5` (`<2`), cv2 `4.5.4` from apt, pillow `12.3.0` in the venv
- Verified: `cuda.is_available()` true, device `Orin`, a real GPU matmul ran

Exact pins and the two-step install are in
[`requirements-jetson.txt`](../requirements-jetson.txt). `pip install -r` on
its own does not reproduce it, and the file says why.

---

## 8. Choose a model

Filters, then a measurement. See the project explainer for the full reasoning;
the short version:

- Open weights, **permissive licence** (Apache-2.0 or MIT are unambiguous)
- Loadable through `AutoProcessor` / `AutoModelForImageTextToText`
- Instruction-tuned (`-Instruct` / `-it` in the name)
- Fits 8 GB **with the OS, CUDA context and KV cache**. Start around 2B in
  float16 and only move up if headroom is real

Capture the probe set first, through the camera the pipeline uses:

```bash
python3 tools/capture_probe_set.py --count 20 --interval 4
```

It counts down between shots so you can move the camera or the scene, and
records each frame's mean brightness so an unusable set is caught at capture
time rather than after a model comparison.

Composition matters more than count; ten varied frames beat twenty easy ones.
Cover the clear case, each obstacle location in the vocabulary, two scenes
where `unknown` is the right answer (a blank wall at close range, a dark
corner), and a couple that are genuinely arguable. Without the `unknown`
frames there is no way to tell a calibrated model from a confident one, which
is most of what this set is for.

At ten frames, one frame of disagreement between two candidates is noise. That
is enough to answer the Gate 1 question, which is whether a model clears the
output contract at all. Capture more only if two candidates come out close
enough that the set has to choose between them.

The frames are photographs of a real room, so `results/probe/*.jpg` is
ignored and only `manifest.json` is committed. It records the negotiated
capture format and each frame's mean brightness, which is what a reader needs
to judge whether the set was usable. Anyone reproducing this shoots their own
set against the table above; the frames were never the reusable part.

Shooting these on a phone would compare candidates on pictures the pipeline
never sees: different sensor, resolution, JPEG encoder and colour handling.
The set exists to predict behaviour on *this* camera.

- Model id: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Revision / commit SHA: `482adb537c021c86670beed01cd58990d01e72e4`
- Licence: Apache 2.0, read on the model card at
  <https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct> on 2026-10-07
- Loads as: `AutoModelForImageTextToText` resolves to
  `SmolVLMForConditionalGeneration` (`model_type: smolvlm`), confirmed from
  `config.json` before downloading any weights

Chosen over Qwen2.5-VL for the fallback rather than for quality. SmolVLM2 ships
at 2.2B, 500M and 256M with one architecture and one loading path, so if 2.2B
will not fit or is too slow to iterate against, the retreat is a model id
string. Nothing in `inference/engine.py` changes, the generation policy does
not change, and the measurements stay comparable in kind.

The card quotes 5.2 GB of GPU RAM, which is for video inference holding many
frames. One image needs considerably less.

Loading it prints a transformers warning that `pad_token_id` 128002 falls
outside a sub-config's vocabulary. Nothing is padded at batch size 1, so it is
inert here. Recorded because it is the first thing to suspect if generation
starts producing garbage.

---

## 9. Run the baseline

```bash
python3 tools/gate1_baseline.py --model <org>/<model> --revision <sha> \
    --image results/probe/probe_00.jpg --runs 20 \
    --probe-dir results/probe
```

No camera, no pipeline, no ROS, so the measurement is the model's cost and
nothing else.

The two flags answer different questions and must not be confused. `--image`
with `--runs` repeats one scene, which keeps the input constant so the latency
distribution is clean; under greedy decoding its validation outcomes are the
same answer twenty times, so they measure determinism. `--probe-dir` runs each
probe frame once, and that is the contract rate. Reporting "20/20 valid" from
the repeated image would be reporting n=1 as n=20.

The script reports memory at three points, before load, after load and warmup,
and at steady state, because **weight size on disk is not runtime footprint**. A
4B model at 4-bit is 3–4 GB on disk, but the vision encoder activations, KV
cache, CUDA context, image tensors and the OS all share the same 8 GB, and
physical memory on a Jetson is unified so there is no separate device pool to
fall back on.

Measured at 15W on an Orin Nano Super, one probe frame repeated 20 times:

| quantity | SmolVLM2-2.2B | SmolVLM2-500M |
| --- | --- | --- |
| inference latency p50 | 4930 ms | 3094 ms |
| inference latency max | 6134 ms | 3112 ms |
| output tokens per result | 19 | 19 |
| **contract rate over 10 probe frames** | **7/10** | **0/10** |
| failure kinds | 3 `unusable_semantics` | 4 `unusable_semantics`, 6 `schema_violation` |
| valid on one frame repeated 20x | 20/20 | 0/20 |
| system memory available, steady | 519 MB | 3819 MB |
| process memory (jtop) | 1.2 GB CPU + 5.1 GB GPU | 2.2 GB CPU + 1.7 GB GPU |
| swap in use during inference | 369 MB | 63 MB |
| mean VDD_IN | 15.1 - 16.4 W | 12.1 W |

**Selected: SmolVLM2-2.2B-Instruct.** It is the only one that perceives. Over
the probe set it returned `clear`/`none`, `blocked`/`left`,
`blocked`/`front_center` and `blocked`/`front_left` on different frames, so
the answers track the scene.

The 500M returned `obstacle_location: "left"` on all ten frames regardless of
content, and six of those put `"none"` in `path_status`, which is not in that
field's vocabulary at all but is in the other field's. It confuses the two
fields and emits a near-constant answer. No prompt change fixes a model that
gives the same reply to an empty floor and a blocked doorway.

The cost of that choice is 519 MB of headroom against 3819 MB, and swapping
during inference. That is a Gate 2 constraint to manage, not a reason to ship
a model that cannot see.

Measured since: four empty `rclpy` nodes cost 108 MB, so the planned node
layout leaves roughly 350 MB with this model loaded. See
[`ros2_ws/README.md`](../ros2_ws/README.md).

4.4x fewer parameters bought 1.6x less time. If decode dominated, latency would
scale roughly with parameter count. It does not, so a large fixed cost sits in
front of decode, and vision encoding in prefill is the candidate. That is
independent support for the section 11 hypothesis, from an experiment run for
another reason.

**The invalid-output rate is 30%**, and it is the first real number for a line
the README has carried as `TBD`. All three failures are the same shape:
`blocked` with `obstacle_location: "none"`, meaning the model saw an
obstruction and would not localise it. The prompt says `Use "none" for
obstacle_location only when path_status is "clear"`, so the rule enforces a
stated instruction that the model broke on three frames out of ten.

Do not relax a rule to make a model pass. That fits the contract to the model
and destroys what the contract is for. If a rule is wrong it is wrong on its
own merits, and the model's output on the other seven frames is the evidence
either way.

A 30% invalid rate is not a disappointing result. It is the result: a real
model on real frames produces unusable output roughly a third of the time,
and the system classifies each one with a reason rather than publishing it as
perception. A pipeline that could not tell the difference would have reported
ten confident answers.

Written to `results/baseline/model-*.json`.

**Time to first token** is reported separately if the runtime exposes it
cleanly. It separates prefill (vision encoding, one pass) from decode
(autoregressive, per token), which are different costs with different fixes. If getting
it requires invasive instrumentation, skip it. It is not worth fighting the
runtime for.

---

## 10. Gate 1 exit criteria

You are done when all five hold:

- [ ] `jtop` reads power, clocks and memory
- [ ] One power profile selected and written down
- [ ] Effective camera fps measured, not assumed
- [ ] The **probe set** clears `inference.schema.validate` at a rate you can
      state, not "plausible text" but the actual validator, and measured across
      varied frames rather than one image repeated. One image is not a gate:
      the 500M candidate here would have passed it on a lucky frame while
      returning the same answer to every scene.
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
| `jtop` says "service not running" | Needs `sudo jtop --install-service` and a re-login. |
| Power rails list is empty | Path layout differs on your JetPack. Check `/sys/class/hwmon/hwmon*/`. Use `jtop` meanwhile, do not report zeros. |
