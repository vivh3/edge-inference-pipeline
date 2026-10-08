# Edge Multimodal Perception on Constrained Robotics Compute

An asynchronous perception pipeline running a multimodal model **slower than its
own sensor**, engineered to behave correctly anyway.

A camera offers ~30 frames per second. A vision-language model on an 8 GB Jetson
takes hundreds of milliseconds per frame. The system is permanently overloaded by
an order of magnitude, and that is its normal operating condition. This project is
about what it does in that condition: which frames it serves, how old the answer is
when it arrives, and what it tells the consumer when something goes wrong.

Deliverables are defensible overload semantics, explicit failure handling, and one
root-caused bottleneck investigation. It is not a benchmarking project.

---

## Why a VLM at all

A small purpose-built detector would be better at obstacle detection than a
generative VLM. That is true, and not what this tests.

> **This project deliberately uses a computationally expensive multimodal model as a
> representative slow semantic perception workload. It is not proposing a VLM as a
> replacement for real-time obstacle detection or safety-critical perception.**

Robotics systems often run cheap perception continuously and expensive semantic
reasoning rarely. This studies the second kind. The model is slow on purpose: a
component that keeps up with its sensor produces no overload behaviour to engineer.

---

## Status

| gate | scope | state |
|---|---|---|
| 0 | Core: admission policies, output contract, failure taxonomy, telemetry, overload experiment | **done** |
| 1 | Jetson feasibility: model running, memory headroom, measured camera rate, baseline latency | **done** |
| 2 | ROS 2 integration, performance budget, end-to-end on device | **done** |
| 3 | Profiling, bottleneck root cause, one justified fix | sustained load **done**, profiling open |
| 4 | Diagram, demo video, results, v0.1 | not started |

Numbers not yet measured are marked `TBD` in `docs/`. No estimate is recorded as a
measurement. The overload comparison below comes from a synthetic engine and is
labelled `SIMULATED`; the camera and power-profile figures are measured.

The core was built ahead of hardware deliberately. `inference/` and `telemetry/` are
stdlib-only and import nothing from ROS, CUDA, or a camera driver at module scope,
so they port to the Jetson unchanged.

---

## Architecture

```
  USB webcam  (measured 30.027 fps, not the 30 the descriptor claims)
      |
      |  capture thread: stamps trusted frame_id + monotonic capture_ts
      v
  admission policy         <-- LatestFrameBuffer: capacity 1, newest wins
      |                        (baselines: BoundedFifo, UnboundedFifo)
      |  ~~~ thread boundary: capture never blocks on inference ~~~
      v
  preprocess               <-- fixed resolution, fixed colour order, timed stage
      |
      v
  model inference          <-- UNTRUSTED; runs at whatever rate is sustainable
      |                        per-frame deadline enforced inside generation
      v
  schema validation        <-- 6-way failure taxonomy; no retries
      |
      v
  metadata attachment      <-- TRUSTED frame_id + 4 monotonic timestamps
      |
      v
  structured JSON output   <-- explicit unknown states, always well-formed
      |
      v
  mock consumer            <-- logger / dashboard. Nothing safety-flavoured.

  telemetry: queue age, inference latency, post-processing, result age,
             drop rate, invalid-output rate, extraction rate, RSS, power, health
```

Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Admission policy, not backpressure

The term matters. Backpressure slows the producer. A camera cannot be slowed; it
delivers frames whether or not anything is ready. The only lever is deciding which
frames to admit and which to discard: an **admission and drop policy**.

If inference takes 600 ms, the system must not work through eighteen stale frames in
sequence. A stale frame has negative value here, because the consumer cannot tell a
three-second-old view of the world from a current one by reading the semantics. So
the pending frame is discarded and the newest one is served.

The resulting drop rate is the design working, so it is a headline number.

### Headline result

30 fps offered against a 6.09 s mean service time, 240 s per policy. **That service
time is the one measured on hardware in Gate 1**, so the simulation runs at this
project's real overload ratio of 183x rather than a guess:

| policy | captured | published | drop rate | result age p50 | p90 | max |
|---|---|---|---|---|---|---|
| `latest` | 7155 | 39 | 99.4% | **6.123 s** | 6.361 s | 7.919 s |
| `fifo_bounded(8)` | 7198 | 40 | 99.3% | 54.798 s | 55.185 s | 55.354 s |
| `fifo_unbounded` | 7198 | 40 | 0.0% | 121.312 s | 218.572 s | 243.046 s |

> **`SIMULATED`.** The admission policies are real; the engine is synthetic. Its
> service time is calibrated to the measured baseline but it is still an input, so
> these runs show admission behaviour and not model performance. They live in
> `results/simulated/` and are never reported as baseline or optimised performance.
> The same code produces the hardware version in Gate 2 by swapping the engine.

Reproduce:
`python3 tools/run_overload_sim.py --duration 240 --latency 6.09 --sigma 0.02 --deadline 15`

No p99: 39 published results make nearest-rank p99 the maximum, which would name the
single slowest result a tail statistic. Under this much overload a run publishes few
results by design, so p90 and max are what the data supports.

The simulation paces itself with real sleeps, so it is not bit-reproducible. Rerunning
it moves these figures by a millisecond or so, which is why the table and the committed
summaries are regenerated together and checked against each other in CI.

Every captured frame is accounted for: `dropped + published + still queued at stop`
equals `captured` exactly, for all three policies. The summaries carry the queue depth
at stop so you can check it.

**The bound was derived before it was measured.** `fifo_bounded(8)` saturates near
`(capacity + 1) x service time` — a frame admitted to a full queue waits behind eight
others, then pays for its own inference. That predicts 9 x 6.084 = 54.76 s against a
measured 54.798 s p50. Under `latest`, result age sits at one service time plus one
inter-frame interval: 6.117 s predicted, 6.123 s measured. Under `fifo_unbounded` it
grows for as long as the run continues: 40 results served back to back at 6.084 s each
is 243 s of service, and the 243.046 s maximum is the oldest surviving frame carrying
the whole life of the run.

**Precision about the claim.** "Result age grows without bound" holds only for the
*unbounded* FIFO. A bounded FIFO fills and starts dropping, so its age is capped by
capacity. The unbounded case is a deliberately pathological baseline, labelled as
one, so the claim attaches to the configuration where it is literally true.

The real comparison is the bounded one, and at this overload ratio it is stark: both
policies drop ~99% of frames and publish about the same number of results, yet one
answers in 6.1 s and the other in 54.8 s. Dropping is not what separates them.
**Which** frames survive is.

---

## The trust boundary

The model emits semantic content and nothing else:

```json
{ "path_status": "blocked", "obstacle_location": "front_left" }
```

It must be able to express ignorance:

```json
{ "path_status": "unknown", "obstacle_location": "unknown" }
```

The wrapper validates that, then attaches trusted metadata:

```json
{
  "frame_id": 1842,
  "capture_ts": 8134.221,
  "inference_start_ts": 8134.243,
  "inference_end_ts": 8134.698,
  "publish_ts": 8134.710,
  "wall_clock": "2026-08-29T14:20:00.123Z",
  "semantic": { "path_status": "blocked", "obstacle_location": "front_left" },
  "validation": { "ok": true, "failure": "none", "extracted": false,
                  "extra_keys_stripped": [] },
  "preprocess_s": 0.004
}
```

Three decisions:

- **`frame_id` is trusted metadata, never model output.** It is assigned at capture
  and carried alongside the model, not through it. If the model produced the
  identifier used for latency accounting, a mangled value would attribute a result to
  the wrong capture. That would not look like a bug. It would look like jitter, and
  it would be measured, plotted and believed.
- **Every duration comes from one monotonic clock.** Wall-clock time is recorded once
  per record for humans and never subtracted. An NTP step correction mid-inference
  would otherwise corrupt a latency or make it negative.
- **Confidence is not in the schema.** A VLM emitting `"confidence": "high"` produced
  a token, not a calibrated probability. Publishing it invites thresholding on a
  meaningless number. Such keys are stripped and counted, not forwarded.

---

## Failure handling

A deliverable, not an afterthought. Six failure modes, each a published outcome:

| failure | trigger |
|---|---|
| `malformed_json` | no parseable JSON object in the response |
| `schema_violation` | missing keys, wrong types, or values outside the closed vocabulary |
| `unusable_semantics` | legal values that contradict each other (`blocked` with no location, `clear` with a location, `unknown` with a location) |
| `inference_timeout` | exceeded the per-frame deadline |
| `engine_error` | the engine raised, OOMed, or died |
| `pipeline_error` | a stage around the engine raised; our bug, not the model's |

On any failure the pipeline still publishes a record, with semantics set to the
explicit unknown state and a `validation` block naming the cause. The consumer always
gets a well-formed record and can tell "the model says it does not know" from "the
model produced garbage". That distinction disappears if both collapse to the same
unknown.

**Inference is never retried to get parseable output.** Retrying would put several
invocations inside one published latency and hide a real deployment problem inside an
average that looks fine. Invalid-output rate is tracked separately, broken down by
failure kind, because "8% invalid" and "8% timeouts" need different fixes.

One documented exception that is not a retry: JSON wrapped in prose or a
```` ```json ```` fence has its first balanced top-level object extracted. The model
is still invoked once, and the rate at which extraction was *needed* is reported, so
a prompt that fails to hold format stays visible.

The worker survives anything that happens to one frame. A consumer that raises is
isolated and counted, and a crash in our own preprocessing or validation publishes a
`pipeline_error` record rather than killing the thread. A pipeline claiming "every
admitted frame produces a published record" cannot die silently on the first
unexpected exception.

A watchdog reports a `stalled` health state when nothing publishes within its interval.
It reports; it does not restart. Restart is out of scope and would hide the failures this
project exists to expose.

Measured in Gate 1: SmolVLM2-2.2B-Instruct at float16, 15W, over 10 probe
frames from the project's own camera.

| measured on hardware | value |
|---|---|
| invalid-output rate | **30%** (3 of 10 probe frames) |
| breakdown by failure kind | `unusable_semantics` 3; nothing else fired |
| extraction rate | 0% — the model never wrapped its JSON in prose |

All three failures are the same shape: `blocked` with `obstacle_location` set
to `none`. The model saw an obstruction and would not localise it, against a
prompt that says to use `none` only when the path is clear. A pipeline that
could not tell that from a usable answer would have published ten confident
results.

---

## The three metrics

All from one monotonic clock, reported separately:

| metric | definition | what it tells you |
|---|---|---|
| queue age | `inference_start_ts - capture_ts` | waiting to be admitted, plus this frame's preprocessing: where the admission policy shows up |
| preprocess | `preprocess_s` | broken out of queue age so it stays attributable |
| inference latency | `inference_end_ts - inference_start_ts` | model execution alone |
| post-processing | `publish_ts - inference_end_ts` | validation, serialisation, publish |
| **result age** | `publish_ts - capture_ts` | **primary**: what a consumer actually experiences |

The decomposition is exact by construction and asserted in the tests. Preprocessing runs
before the engine stamps its start, so it falls inside queue age, and it is reported
separately rather than lost there because Gate 3 has to be able to blame it.
Post-processing is separate for the same reason, so that "the bottleneck was JSON
parsing, not the model" is a conclusion the data can support.

Two counts are absent on purpose. **Frames admitted**: under latest-frame a frame can be
admitted and then evicted before it runs, so the count means different things per policy
and is not comparable across them. Captured, dropped and published are unambiguous.
**Queue depth**: in a one-slot buffer it is 0 or 1 and carries no information.

---

## Performance budget

Called a *performance budget*, never an SLO. An SLO derives from system or user
requirements. This derives from what the hardware turned out to do, which is a
different thing. Measuring first and then setting a target is normal when there is no
external requirement; the honesty is in the label.

Set from the Gate 1 baseline of 6088 ms, then met end to end over a 12.5-minute run
of 94 published records. Derivation and caveats in
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

| budget | target | measured |
|---|---|---|
| result age p50 | <= 6.5 s | **6.464 s** |
| result age p90 | <= 6.6 s | 6.485 s |
| invalid-output rate | <= 35% | 0% on a representative scene |
| sustained 10 min | p50 within 10% first minute vs last, no `stalled` or `engine_dead` | **0.34% drift**, `healthy` throughout |

The pipeline costs 12 ms per cycle: successive results are 6.447 s apart against
6.438 s of inference, a 99.8% duty cycle. Everything that is not the model —
queueing, preprocessing, validation, publishing — rounds to nothing.

An earlier attempt at the sustained run returned 94 records of which every one failed
validation, because the camera was face down. Re-pointed, the invalid rate is 0% and
latency rises 4%: scene content is a latency input, since output length drives
generation time and the frozen generation policy cannot fix what the camera sees.

---

## The bottleneck investigation

> Gate 3, in progress. Method, hypothesis written before profiling, and the trace
> signature of each candidate are in [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

Four of six candidates are eliminated from a 1 Hz resource log over the sustained run
(`results/sustained/`): the GPU reads 90–100% in 550 of 752 samples with its clock's
10th percentile on the 15W ceiling, tj plateaus at 62.8 °C, and memory does not move.
That rules out host-to-device transfer, synchronisation stalls, thermal throttling and
memory pressure.

What is left is on-GPU work plus one thing the instrumentation cannot currently
separate: `inference_latency` spans both the engine's CPU-side `processor` call and
`generate` on the GPU. A sub-90% GPU window recurs once per 6.4 s inference cycle,
and its depth implies roughly 7% of inference runs off the GPU — inferred from 1 Hz
aliasing, so a direction rather than a measurement. Splitting that stamp is the next
step.

The hypothesis is written down before profiling. Being wrong is a good README
section. If the evidence supports none of the obvious tools, the result is
"profiling showed X dominated, so optimising Y would not have addressed the system
bottleneck". That is a stronger result than forcing a tool in. TensorRT, quantisation and Nsight are
instruments, not success criteria.

---

## Quickstart

No accelerator, camera, or ROS needed for the core and the overload experiment.
Python 3.10+, standard library only.

```bash
git clone https://github.com/vivh3/edge-inference-pipeline.git
cd edge-inference-pipeline

# headline experiment: 30 fps against the 6.09 s service time measured on hardware
python3 tools/run_overload_sim.py --duration 240 --latency 6.09 --sigma 0.02 --deadline 15

# a faster sweep, if you only want to see the shape
python3 tools/run_overload_sim.py --duration 12 --latency 0.4

# faults injected, to exercise the failure taxonomy
python3 tools/run_overload_sim.py --duration 12 --latency 0.4 \
    --p-malformed 0.05 --p-schema-violation 0.05 --p-timeout 0.02

# tests
pip install -e '.[dev]' && python3 -m pytest tests/ -q

# optional plot
pip install -e '.[plot]' && python3 tools/plot_results.py results/simulated
```

Output: `results/simulated/<policy>/{results.csv,summary.json}` and
`comparison.json`.

Jetson setup: [`docs/SETUP-jetson.md`](docs/SETUP-jetson.md).

---

## Repository layout

```
inference/     stdlib-only, hardware-independent core; the argument lives here
  clock.py       one monotonic source for every duration
  record.py      Frame / RawModelOutput / PublishedResult: the trust boundary
  schema.py      output contract, validation, failure taxonomy
  buffer.py      admission policies: latest-frame, bounded FIFO, unbounded FIFO
  capture.py     synthetic and UVC webcam sources
  preprocess.py  fixed-resolution preprocessing as its own timed stage
  engine.py      engine interface, synthetic engine, Hugging Face VLM adapter
  config.py      the frozen generation policy
  pipeline.py    async worker, health state, watchdog
telemetry/     three metrics, rates, resource sampling, CSV/JSON output
tools/         overload experiment, camera and baseline measurement, plotting
tests/         the contract, the policies, the capture path, the failure paths
docs/          architecture, Jetson setup, performance methodology
results/       simulated/ (now), baseline/ optimized/ traces/ (Gate 1-3)
ros2_ws/       Gate 2: integration plumbing, a thin wrapper over the core
```

---

## Versions

| component | version |
|---|---|
| Python (core) | 3.10+, standard library only |
| JetPack / L4T | 6.2 / R36.4.3 |
| model id + revision/SHA | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` @ `482adb5` |
| model licence | Apache 2.0, [model card](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) |
| inference runtime | PyTorch 2.8.0, transformers 5.18.0 |
| baseline inference latency | 6088 ms p50, 6119 ms max (15W, float16) |
| overload ratio, measured | 183x against a 33.3 ms frame period |
| ROS 2 | Humble (Gate 2) |
| power profile | `nvpmodel` mode 0 (15W), held for every measurement |
| camera | j5create JVCU100, MJPG 640x480, measured 30.027 fps |

---

## Known issues and limitations

- **The overload comparison is simulated.** `results/simulated/` comes from a
  synthetic engine whose service time is an experimental input, not a claim about any
  model. The model and camera figures in `results/baseline/` are measured on hardware.
- **The deadline is enforced inside generation**, by a stopping criterion checking the
  clock between tokens. A kernel already executing cannot be interrupted, so an
  externally enforced deadline would only be detected after the fact. A single
  grossly overrunning forward pass is bounded by the watchdog, not the deadline.
- **The watchdog reports; it does not restart.** Deliberate, and out of scope.
- **Single-threaded inference by design.** One model on one GPU is the serial
  resource; overlapping invocations would trade result age for throughput in the
  wrong direction here.
- **`fifo_unbounded` grows memory without bound.** A demonstration baseline. Do not
  run it long.
- **ROS 2 usage is integration plumbing plus one considered QoS decision.** Executors,
  lifecycle nodes, transforms, composition, and DDS internals are where the real depth
  is, and none is exercised here.
- **The `unusable_semantics` rules are a design choice**, stated explicitly so a
  reviewer can disagree explicitly.
- **The bounded FIFO tail-drops.** Dropping the oldest instead converges on
  latest-frame as capacity falls to 1. Not implemented: it adds a variant without
  adding an argument.

---

## Before this could affect actuation

Nothing here is safety architecture, and the mock consumer is a logger on purpose.
Before output of this kind could move an actuator, at minimum:

- **A calibrated, validated confidence signal**, which this deliberately lacks.
  Generated text expressing certainty is not a probability, and the schema refuses to
  publish one rather than offer a number that invites thresholding.
- **A freshness contract enforced at the consumer**, not merely reported. The consumer
  would reject any result older than a bound derived from vehicle dynamics and
  stopping distance, and default to the safe action when none is available. Result age
  is measured here; it is not enforced.
- **Independent cross-checking** against a modality that does not share a failure mode
  with the model. A hallucinated `clear` and a correct `clear` are indistinguishable
  from inside this pipeline.
- **A defined consumer behaviour for every failure and for the degraded and stalled
  health states.** This system reports honestly; it decides nothing.
- **Validation appropriate to the hazard**: coverage of the operational design domain,
  evidence of behaviour at its edges, and a way to detect distribution shift in the
  field.
- **Deterministic worst-case timing.** An autoregressive model whose latency depends on
  the content of its own output cannot be bounded by construction.

---

## Scope

Out of scope and staying there: a quantisation matrix, TensorRT as a required box,
Isaac Sim, training or fine-tuning, custom CUDA kernels, a real robot or planner, a
second model or runtime, ROS 2 lifecycle nodes or custom executors, and process
supervision beyond a basic watchdog.

---

## Licence

MIT, see [`LICENSE`](LICENSE). Model weights carry their own licence, recorded in the
versions table once selected.
