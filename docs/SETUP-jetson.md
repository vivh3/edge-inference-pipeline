# Jetson setup (Gate 1)

Target: **NVIDIA Jetson Orin Nano Super Developer Kit**, 8 GB, JetPack 6.x.

Gate 1 is a feasibility gate, not a tuning gate. It has one job: prove the
model runs and measure how much room is left. Record the numbers, then stop.

> **Status: not yet executed.** Every `TBD` below is a measurement to be filled
> in on hardware. Nothing in this file is an estimate, and no number appears
> here until it has been measured on the device.

## 1. Flash and boot

- JetPack version: `TBD` (record exact release, e.g. `6.2 / L4T 36.4.x`)
- `cat /etc/nv_tegra_release`
- Storage: microSD 64 GB. An NVMe SSD is optional and used for model weights
  only -- do not move the root filesystem onto it unless it is trivial. Boot
  configuration changes are not part of this project.

## 2. Fix one power profile and never change it

```bash
sudo nvpmodel -q            # list and query current mode
sudo nvpmodel -m <N>        # select
sudo jetson_clocks --show   # report clock state
```

- Selected mode: `TBD`
- Rationale: `TBD`

Do not reflexively select MAXN. If part of the story is *constrained compute*,
locking to the highest power mode and then discovering thermal throttling
manufactures a share of your own problem, and the throttling will show up in
the middle of the sustained load test as unexplained variance.

**Methodology consistency matters more than which mode is chosen.** The mode
is recorded once and held fixed across every measurement in this repository:
baseline, overload comparison, and post-optimisation. A comparison across two
power modes is not a comparison.

## 3. Instrumentation

```bash
sudo pip3 install jetson-stats
sudo systemctl restart jtop.service
jtop                      # confirm it reads power, clocks, memory
```

Confirm before continuing:

- [ ] power rails readable
- [ ] GPU clock readable
- [ ] memory total/used readable

`telemetry/metrics.py:read_jetson_power_w` reads the INA3221 hwmon rails
directly so power lands in the same CSV row as the latency measurement,
rather than in a separate tool's timeline that then has to be aligned.
Record which rails it found:

- Rails discovered: `TBD`

## 4. Memory headroom -- measure this on day 1

**Quantised weight size is not runtime footprint.** A 4B model at 4-bit is
roughly 3-4 GB on disk, but the vision encoder activations, the KV cache, the
CUDA context, runtime allocations, image tensors, the ROS 2 processes, and the
operating system all share the same 8 GB. Physical memory on a Jetson is
unified, so there is no separate device pool to fall back on.

Measure, with the model loaded and after several inferences:

```bash
free -m
tegrastats --interval 1000     # RAM x/y, and the GPU rail
```

| quantity | value |
|---|---|
| total system memory | `TBD` |
| baseline before model load | `TBD` |
| after model load | `TBD` |
| steady state after 20 inferences | `TBD` |
| **headroom remaining** | `TBD` |

If headroom is uncomfortable, move down a size. **Nobody cares whether the
model was 2B or 4B.** The architecture is the deliverable; the parameter count
is not.

## 5. Model, running end to end on one image

- Model id: `TBD` (public, permissively licensed, cited in the README)
- Licence: `TBD`
- Weights SHA / revision: `TBD`
- Runtime and version: `TBD`

Success condition: one image in, **valid JSON out**, parsed by
`inference/schema.py:validate` with `failure == none`. Not "the model produced
plausible text" -- it must clear the actual validator.

```bash
python3 -c "
from inference.engine import VlmEngine
from inference.schema import validate
from PIL import Image
e = VlmEngine('<model-id>')
e.warmup(5)
raw = e.infer(Image.open('docs/test-image.jpg'))
print(raw.text)
print(validate(raw.text))
"
```

## 6. Webcam

Any UVC USB webcam. CSI cameras are deliberately avoided: they drag in driver
and GStreamer work that contributes nothing to this project's argument.

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

**Measure the actual inter-frame arrival times.** A cheap USB camera will not
hold a steady 30 fps, especially in low light where exposure lengthens. Drop
rate is computed against frames that actually arrived, never against a nominal
figure -- computing against nominal would fabricate drops that never happened.

| quantity | value |
|---|---|
| device / format | `TBD` |
| requested fps | 30 |
| measured mean interval | `TBD` |
| measured p99 interval | `TBD` |
| measured effective fps | `TBD` |

The README says "nominal 30 fps" and reports the measured figure next to it.

## 7. Single-image baseline

No pipeline, no ROS, no camera. One image, repeated, warmup discarded
(`GenerationPolicy.warmup_runs = 5`). This is the number the performance
budget is set from in Gate 2.

| quantity | value |
|---|---|
| inference latency p50 / p90 / p99 | `TBD` |
| time to first token (if the runtime exposes it cleanly) | `TBD` |
| output tokens per result | `TBD` |
| peak RSS | `TBD` |
| mean power | `TBD` |

## Day 2 escape hatch

**If the model is not boringly working through the intended runtime by the end
of day 2, switch model or runtime immediately.** Do not debug an exotic
conversion path. The project studies one slow component; it does not search
for the best one.

## Day 3 freeze

No model changes after day 3 for performance reasons. After this point the
model is a fixed input to the experiment.
