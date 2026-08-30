# Edge Multimodal Perception on Constrained Robotics Compute

An asynchronous perception pipeline that runs a multimodal model **slower than
its own sensor**, and behaves correctly anyway.

A camera offers ~30 frames per second. A vision-language model on an 8 GB
Jetson services frames in hundreds of milliseconds. The system is therefore in
permanent overload by an order of magnitude, and that is its *normal*
operating condition. This project is about what a well-engineered system does
in that condition: which frames it serves, how old the answer is when it
arrives, and what it tells the consumer when something goes wrong.

The thesis: **a well-engineered system stays responsive and honest when an
expensive learned component cannot keep pace with its sensor.**

This is not a benchmarking project. The deliverables are defensible overload
semantics, explicit failure handling, and one root-caused bottleneck
investigation.

---

## Why a VLM at all

A small purpose-built detector would be better at obstacle detection than a
generative vision-language model, faster, and easier to validate. That is
true, and it is not what this project is testing.

> **This project deliberately uses a computationally expensive multimodal
> model as a representative slow semantic perception workload. It is not
> proposing a VLM as a replacement for real-time obstacle detection or
> safety-critical perception.**

One common architecture in robotics runs cheap perception continuously and
invokes expensive semantic reasoning far less often. This project studies the
second kind: a slow semantic component consuming a fast sensor stream. The
model is expensive *on purpose*, because a component that keeps up with its
sensor produces no interesting overload behaviour to engineer.

---

## Status

| gate | scope | state |
|---|---|---|
| 0 | Hardware-independent core: admission policies, output contract, failure taxonomy, telemetry, overload experiment | **done** |
| 1 | Jetson feasibility: model running, memory headroom, measured camera rate, single-image baseline | pending hardware |
| 2 | ROS 2 integration, performance budget, end-to-end run on device | pending hardware |
| 3 | Nsight profiling, bottleneck root cause, one justified fix, sustained load | pending hardware |
| 4 | Diagram, demo video, README results, v0.1 | pending hardware |

Every hardware number in `docs/` is currently marked `TBD`. Nothing in this
repository reports an estimate as a measurement. The only numbers here today
come from the synthetic overload experiment, and they are labelled `SIMULATED`
in their own summary files and below.

The core was built ahead of hardware on purpose: the admission policy, the
trust boundary, the failure taxonomy, and the telemetry are the parts that
carry the argument, and none of them needs an accelerator to be designed,
tested, or defended. `inference/` and `telemetry/` are stdlib-only and import
nothing from ROS, CUDA, or a camera driver at module scope, so they port to
the Jetson unchanged.

---

## Architecture

```
  USB webcam  (nominal 30 fps -- actual arrival rate is measured, not assumed)
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
  schema validation        <-- 5-way failure taxonomy; no retries
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

Full discussion in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Admission policy, not backpressure

The precise term matters. **Backpressure slows the producer.** A camera cannot
be slowed -- it delivers frames at its own rate whether or not anything is
ready to receive them. The only available lever is deciding which frames to
admit and which to discard, which is an **admission and drop policy**. Calling
it backpressure would describe a mechanism this system does not have.

If inference takes 600 ms, the system must not work through eighteen stale
frames in sequence. A stale frame has *negative* value here, because the
consumer cannot distinguish a three-second-old view of the world from a
current one by reading the semantics. So the pending frame is discarded and
the newest one is served.

The resulting drop rate is the design working, not the design failing. That is
why it is reported as a headline number rather than buried.

### Headline result

Offered 30 fps against a synthetic 0.4 s mean service time (a 12x overload),
12 s per policy:

| policy | captured | published | drop rate | result age p50 | p99 | max |
|---|---|---|---|---|---|---|
| `latest` | 361 | 30 | 91.4% | **0.406 s** | 0.852 s | 0.852 s |
| `fifo_bounded(8)` | 360 | 30 | 89.4% | 3.536 s | 3.903 s | 3.903 s |
| `fifo_unbounded` | 361 | 30 | 0.0% | 5.861 s | 11.133 s | 11.133 s |

> **`SIMULATED`.** The service-time distribution is an *input* to this
> experiment, not a measurement of any model. These runs demonstrate the
> admission policy's behaviour; they say nothing about how fast a real model
> is. They live in `results/simulated/` and are never reported as baseline or
> optimised performance. The same code produces the hardware version in Gate 2
> by substituting the engine.

Reproduce: `python3 tools/run_overload_sim.py --duration 12 --latency 0.4`

**Reading it.** Under `latest`, result age sits at roughly one inference
service time and stays there: the consumer's answer is always about the most
recent frame the system could serve. Under `fifo_bounded(8)`, result age
saturates near `capacity x service time` -- and the frames that survive are the
*oldest* ones, which is exactly backwards for this workload. Under
`fifo_unbounded`, result age grows for as long as the run continues.

**Being precise about the claim.** "Result age grows without bound" is true
only for the *unbounded* FIFO. A bounded FIFO fills up and starts dropping, so
its age is capped by queue capacity. The unbounded case is included as a
**deliberately pathological baseline**, labelled as one, so that the claim is
attached to the one configuration where it literally holds. The interesting
comparison is the bounded one: both policies drop at a similar rate, and the
difference is *which* frames survive.

---

## The trust boundary

The single most important line in the system. The model emits semantic content
and nothing else.

The model produces only:

```json
{ "path_status": "blocked", "obstacle_location": "front_left" }
```

and must be able to express ignorance:

```json
{ "path_status": "unknown", "obstacle_location": "unknown" }
```

The wrapper validates that output and then attaches trusted system metadata:

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
                  "extra_keys_stripped": [] }
}
```

Three decisions worth defending:

- **`frame_id` is trusted metadata, never model output.** It is assigned at
  capture and carried *alongside* the model, never *through* it. If the model
  produced the identifier used for latency accounting, a hallucinated or
  mangled value would mis-attribute a result to the wrong capture. That
  corruption would not look like a bug -- it would look like jitter, and it
  would be measured, plotted, and believed.
- **Every duration comes from one monotonic clock.** Wall-clock time is
  recorded once per record for humans and never subtracted from anything. An
  NTP step correction mid-inference would otherwise silently corrupt a latency
  or make it negative.
- **Confidence is not in the schema.** A VLM emitting `"confidence": "high"`
  has produced a token, not a calibrated probability. Publishing it would
  invite a consumer to threshold on a meaningless number. Such keys are
  stripped and counted, not forwarded.

---

## Failure handling

A deliverable, not an afterthought. Five failure modes, each a first-class
published outcome:

| failure | trigger |
|---|---|
| `malformed_json` | no parseable JSON object in the response |
| `schema_violation` | missing keys, wrong types, or values outside the closed vocabulary |
| `unusable_semantics` | legal values that contradict each other (`blocked` with no location, `clear` with a location, `unknown` with a location) |
| `inference_timeout` | exceeded the per-frame deadline |
| `engine_error` | the engine raised, OOMed, or died |

On any failure the pipeline **still publishes a record**, with semantics set to
the explicit unknown state and a `validation` block naming the cause. The
consumer always receives a well-formed record, and can distinguish "the model
says it does not know" from "the model produced garbage" -- a distinction that
disappears if both collapse to the same unknown.

**Inference is never retried to obtain parseable output.** Retrying would
distort every latency measurement (one published result silently covering two
or three invocations) and would hide a real deployment problem inside an
average that looks fine. Invalid-output rate is tracked on its own, broken
down by failure kind, because "8% invalid" and "8% timeouts" call for
different fixes.

One documented exception that is not a retry: a JSON object wrapped in prose
or a ```` ```json ```` fence has its first balanced top-level object
extracted. The model is still invoked exactly once, and the rate at which
extraction was *needed* is reported separately so a prompt that fails to hold
the output format stays visible.

A watchdog reports a `stalled` health state when nothing has published within
its interval. It reports; **it does not restart.** Automatic restart is out of
scope and would obscure exactly the failures this project exists to make
visible.

| measured on hardware | value |
|---|---|
| invalid-output rate | `TBD` (Gate 2) |
| breakdown by failure kind | `TBD` |
| extraction rate | `TBD` |

---

## The three metrics

All from one monotonic clock, all reported separately:

| metric | definition | what it tells you |
|---|---|---|
| queue age | `inference_start_ts - capture_ts` | how long a frame waited to be admitted -- where the admission policy shows up |
| inference latency | `inference_end_ts - inference_start_ts` | model execution alone |
| post-processing | `publish_ts - inference_end_ts` | validation, serialisation, publish |
| **result age** | `publish_ts - capture_ts` | **primary**: what a downstream consumer actually experiences |

The decomposition is exact by construction and asserted in the test suite. It
is kept because "the bottleneck was JSON parsing, not the model" needs to be a
conclusion the data can support.

Queue *depth* is deliberately not reported: in a correct one-slot latest-value
buffer it is 0 or 1 and carries no information.

---

## Performance budget

> To be set in Gate 2 **from the Gate 1 baseline measurement**, not before.
> See [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

Called a *performance budget*, deliberately, and never an SLO. A service-level
objective derives from system or user requirements. This number will derive
from what the hardware turned out to be capable of, which is a different
thing. Measuring first and then setting a defensible target is normal practice
when there is no external requirement; the honesty is in the label.

---

## The bottleneck investigation

> Gate 3, pending hardware. Method, hypothesis template, and the trace
> signature of each candidate are laid out in
> [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

The hypothesis gets written down *before* profiling. Being wrong is a good
README section. And if the evidence supports none of the obvious tools, the
result is "profiling showed X dominated, so optimising Y would not have
addressed the system bottleneck" -- which is a stronger outcome than forcing a
tool in. TensorRT, quantisation, and Nsight are instruments here, not success
criteria.

---

## Quickstart

No accelerator, no camera, no ROS required for the core and the overload
experiment. Python 3.10+, standard library only.

```bash
git clone https://github.com/vivh3/edge-inference-pipeline.git
cd edge-inference-pipeline

# the headline experiment: 30 fps offered against a 0.4 s service time
python3 tools/run_overload_sim.py --duration 12 --latency 0.4

# same thing with faults injected, to exercise the failure taxonomy
python3 tools/run_overload_sim.py --duration 12 --latency 0.4 \
    --p-malformed 0.05 --p-schema-violation 0.05 --p-timeout 0.02

# tests
pip install -e '.[dev]' && python3 -m pytest tests/ -q

# optional plot of the headline figure
pip install -e '.[plot]'
python3 tools/plot_results.py results/simulated
```

Output lands in `results/simulated/<policy>/{results.csv,summary.json}` plus
`comparison.json`.

Jetson setup is a separate document: [`docs/SETUP-jetson.md`](docs/SETUP-jetson.md).

---

## Repository layout

```
inference/     stdlib-only, hardware-independent core -- the argument lives here
  clock.py       one monotonic source for every duration
  record.py      Frame / RawModelOutput / PublishedResult -- the trust boundary
  schema.py      output contract, validation, failure taxonomy
  buffer.py      admission policies: latest-frame, bounded FIFO, unbounded FIFO
  capture.py     synthetic and UVC webcam sources
  preprocess.py  fixed-resolution preprocessing as its own timed stage
  engine.py      engine interface, synthetic engine, Hugging Face VLM adapter
  config.py      the frozen generation policy
  pipeline.py    async worker, health state, watchdog
telemetry/     three metrics, rates, resource sampling, CSV/JSON output
tools/         overload experiment, plotting
tests/         37 tests covering the contract, the policies, and the failure paths
docs/          architecture, Jetson setup, performance methodology
results/       simulated/ (here now), baseline/ optimized/ traces/ (Gate 1-3)
ros2_ws/       Gate 2 -- integration plumbing, a thin wrapper over the core
```

---

## Versions

| component | version |
|---|---|
| Python (core) | 3.10+, standard library only |
| JetPack / L4T | `TBD` (Gate 1) |
| model id + revision/SHA | `TBD` (Gate 1) |
| model licence | `TBD` -- public, permissively licensed, cited |
| inference runtime | `TBD` (Gate 1) |
| ROS 2 | Humble (Gate 2) |
| power profile | `TBD` -- selected in Gate 1, held fixed for every measurement |

---

## Known issues and limitations

- **No hardware measurements yet.** Everything in `results/` today is from a
  synthetic engine. Its service-time distribution is an experimental input,
  not a claim about any model.
- **The deadline is enforced inside generation**, via a stopping criterion that
  checks the clock between tokens. A kernel already executing cannot be
  interrupted, so a deadline enforced from outside the call would only ever be
  detected after the fact. A grossly overrunning single forward pass is
  therefore bounded only by the watchdog and the health state, not by the
  deadline.
- **The watchdog reports; it does not restart.** Deliberate, and out of scope.
- **Single-threaded inference by design.** One model on one GPU is the serial
  resource; overlapping invocations would trade result age for throughput in
  the wrong direction for this workload.
- **`fifo_unbounded` grows memory without bound.** It is a baseline for
  demonstration. Do not run it long.
- **ROS 2 usage is integration plumbing plus one considered QoS decision.**
  That is the accurate description. Executors, lifecycle nodes, transforms,
  composition, and DDS internals are where the real depth is, and none of them
  is exercised here.
- **The consistency rules in `unusable_semantics` are a design choice**, stated
  explicitly so a reviewer can disagree with them explicitly.
- The bounded FIFO tail-drops. Dropping the oldest instead is a third policy
  that converges on latest-frame as capacity falls to 1; it is not implemented
  because it adds a variant without adding an argument.

---

## Before this could affect actuation

Nothing here is safety architecture, and the mock consumer is a logger on
purpose. Before output of this kind could influence a physical actuator, at
minimum:

- **A calibrated, validated confidence signal** -- which this system
  deliberately does not have. Generated text expressing certainty is not a
  probability, and the schema refuses to publish one rather than offering a
  number that invites thresholding.
- **A freshness contract enforced at the consumer**, not merely reported.
  A consumer would have to reject any result older than a bound derived from
  vehicle dynamics and stopping distance, and default to the safe action when
  none is available. Result age is measured here; it is not enforced.
- **Independent cross-checking** against a sensing modality that does not share
  a failure mode with the model -- the whole point being that a hallucinated
  `clear` and a correct `clear` are indistinguishable from inside this
  pipeline.
- **A defined behaviour for every failure and for the degraded and stalled
  health states**, owned by the consumer. This system reports honestly; it does
  not decide anything.
- **Validation appropriate to the hazard**: coverage of the operational design
  domain, evidence of behaviour at the edges of it, and a story for detecting
  distribution shift in the field.
- **Deterministic worst-case timing.** An autoregressive model whose latency
  depends on the content of its own output is not a component whose timing can
  be bounded by construction.

---

## Scope

Deliberately out of scope, and staying there: a quantisation matrix, TensorRT
as a required box, Isaac Sim, training or fine-tuning, custom CUDA kernels, a
real robot or planner, a second model or runtime, ROS 2 lifecycle nodes or
custom executors, and elaborate process supervision beyond a basic watchdog.

---

## Licence

MIT. See [`LICENSE`](LICENSE). Model weights are governed by their own licence,
recorded in the versions table once selected.
