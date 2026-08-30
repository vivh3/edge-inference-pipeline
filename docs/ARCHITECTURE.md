# Architecture

## The problem this system solves

A camera produces frames at a rate the model cannot match. Offered load is
~30 frames/second; a multimodal model on an 8 GB Jetson services frames on
the order of hundreds of milliseconds. The system is therefore in permanent
overload by an order of magnitude, and that is the normal operating
condition, not an exceptional one.

The design question is not "how do we go faster". It is **which frames get
served, how old the answer is when it arrives, and what the consumer is told
when something goes wrong.**

## Dataflow

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
  model inference          <-- untrusted; runs at whatever rate is sustainable
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

  telemetry taps: queue age, inference latency, post-processing, result age,
                  drop rate, invalid-output rate, extraction rate, RSS, power,
                  health state
```

## The trust boundary

This is the single most important line in the system, and it runs between
`RawModelOutput` and `PublishedResult` in `inference/record.py`.

**The model emits semantic content and nothing else.** It does not emit
identifiers, it does not emit timestamps, and it does not emit confidence.

- **`frame_id` is trusted metadata.** It is assigned by the capture stage and
  carried *alongside* the model, never *through* it. If the model were the
  source of the identifier used for latency accounting, a hallucinated or
  mangled value would silently mis-attribute a result to the wrong capture.
  The corruption would not look like a bug; it would look like jitter, and it
  would be measured, plotted, and believed.
- **Every timestamp comes from one monotonic clock** (`inference/clock.py`).
  Wall-clock time is recorded once per record, for humans, and is never
  subtracted from anything. An NTP step correction mid-inference would
  otherwise corrupt a duration or make it negative.
- **Confidence is not in the schema.** A VLM emitting `"confidence": "high"`
  has produced a token, not a calibrated probability. Publishing it would
  invite a consumer to threshold on a number that means nothing. Such keys
  are stripped and counted, not forwarded.

Validation sits exactly on the boundary: nothing downstream of
`inference/schema.py:validate` reads model text.

## Admission policy, not backpressure

The precise term matters. **Backpressure** slows the producer. A camera
cannot be slowed -- it delivers frames at its own rate whether or not anything
is ready to receive them. The only available lever is deciding which frames
to admit and which to discard, which is an **admission and drop policy**.

Three policies are implemented (`inference/buffer.py`), sharing one interface
so an experiment can swap between them with nothing else changing:

| policy | on overload | result age | drop rate |
|---|---|---|---|
| `latest` | newest frame evicts the pending one | ~ one service time | high, and rising with overload ratio |
| `fifo_bounded(n)` | tail-drop: the incoming frame is refused | saturates near `n x service time` | high |
| `fifo_unbounded` | never drops | grows without bound | zero |

`latest` is the design choice. The argument: for this workload a stale frame
has **negative** value, because the consumer cannot tell a 3-second-old view
of the world from a current one by reading the semantics. Discarding stale
work is therefore the correct behaviour, and the resulting drop rate is the
design working rather than the design failing -- which is why drop rate is
reported as a headline number instead of being buried.

`fifo_unbounded` is a **deliberately pathological baseline**, labelled as one
everywhere it appears. It exists so that the claim "result age grows without
bound" is attached to the one configuration where it is literally true. A
bounded FIFO fills and then drops, so its age is capped by capacity; stating
otherwise would be an overclaim, and an overclaim is the worst possible place
to be caught.

Queue *depth* is deliberately not reported. In a correct one-slot
latest-value buffer it is 0 or 1 and carries no information. Queue **age** is
the informative quantity.

## Threading model

Three threads, no shared mutable state outside two locks:

- **capture thread** -- acquires a frame, stamps it, calls `policy.offer()`,
  returns. Bounded work only. `test_capture_is_not_blocked_by_a_slow_engine`
  pins this.
- **inference worker** -- `policy.take()` -> preprocess -> engine -> validate ->
  attach -> publish. One at a time. Concurrency here would not help: a single
  model on a single GPU is the serial resource, and overlapping invocations
  would trade result age for throughput in the wrong direction.
- **watchdog** -- observes time since last publish and sets a health state. It
  reports; it does not restart. Automatic restart is out of scope and would
  obscure exactly the failures this project exists to make visible.

If capture and inference shared a thread, the camera driver's own buffering
would become the queue, and the admission policy would have nothing left to
decide. The thread boundary is what gives the policy something to be a policy
*about*.

## The three metrics

All from one monotonic clock, all reported separately, all derivable from the
four timestamps on every published record:

| metric | definition | what it tells you |
|---|---|---|
| queue age | `inference_start_ts - capture_ts` | how long a frame waited to be admitted -- where the admission policy shows up |
| inference latency | `inference_end_ts - inference_start_ts` | model execution alone; should be roughly invariant to the admission policy |
| post-processing | `publish_ts - inference_end_ts` | validation, serialisation, publish |
| **result age** | `publish_ts - capture_ts` | **primary**: the only one a downstream consumer experiences |

The decomposition is exact by construction:
`result_age == queue_age + inference_latency + post_processing`, asserted in
`tests/test_pipeline.py`. Keeping post-processing separate is what makes
"the bottleneck was JSON parsing, not the model" a conclusion the data can
support rather than one that has to be assumed away.

## Failure handling

Five failure modes, each a first-class published outcome
(`inference/schema.py`):

| failure | trigger |
|---|---|
| `malformed_json` | no parseable JSON object in the response |
| `schema_violation` | parses, but missing keys / wrong types / values outside the closed vocabulary |
| `unusable_semantics` | individually legal values that contradict each other (`blocked` with no location, `clear` with a location, `unknown` with a location) |
| `inference_timeout` | exceeded the per-frame deadline |
| `engine_error` | the engine raised, OOMed, or died |

On any failure the pipeline **still publishes a record**, with semantics set
to the explicit unknown state and a trusted `validation` block naming the
cause. A consumer therefore always receives a well-formed record and can
distinguish "the model says it does not know" from "the model produced
garbage" by reading `validation.failure` -- a distinction that is invisible
if both collapse to the same unknown.

**Inference is never retried to obtain parseable output.** Retrying would
distort every latency measurement (one published result would silently cover
two or three model invocations) and would hide a real deployment problem
inside an average that looks fine. Invalid-output rate is tracked as its own
metric, broken down by failure kind, because "8% invalid" and "8% timeouts"
call for entirely different fixes.

One documented exception that is *not* a retry: if the response contains a
JSON object wrapped in prose or a ```` ```json ```` fence, the first balanced
top-level object is extracted. The model is still invoked exactly once. The
rate at which extraction was *needed* is reported separately, so a prompt that
fails to hold the output format stays visible instead of being averaged into
"valid".

## Why a VLM at all

See the README. Short version: the VLM is a *representative slow semantic
workload*, deliberately chosen because it is expensive. This is not a proposal
to use a generative model for real-time obstacle detection.

## Layering

```
inference/   stdlib-only, hardware-independent core -- the argument lives here
telemetry/   measurement, also stdlib-only
tools/       experiments and plotting
ros2_ws/     integration plumbing (Gate 2), a thin wrapper over the above
```

Nothing in `inference/` or `telemetry/` imports ROS, CUDA, or a camera driver
at module scope. That is deliberate and load-bearing: it is what lets the
admission policy, the output contract, the failure taxonomy, and the telemetry
be built and tested on a laptop with no accelerator, then ported unchanged.
