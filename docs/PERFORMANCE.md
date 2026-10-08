# Performance: methodology and the bottleneck investigation

> **Status: Gates 1-3 measured.** The overload comparison comes from the
> synthetic simulation and is labelled `SIMULATED` in its own summary files.
> No estimate is written here as if it were a measurement, and where one was
> inferred rather than measured the page says which.

## Methodology (fixed before any number is recorded)

Everything below is held constant across every configuration that is
compared: baseline, optimised, and each admission policy.

**Generation policy** (`inference/config.py`, one file so "was anything else
different?" has a one-file answer):

| held constant | value |
|---|---|
| prompt | frozen; SHA-256 prefix recorded in every run summary |
| image resolution | 448 x 448 |
| max new tokens | 48 |
| decoding | greedy (`do_sample=False`), seed 0 |
| warmup runs discarded | 5 |

Generative latency is dominated by how many tokens come out. A configuration
that emits a chattier answer looks slower, and that reads as a system effect
when it is a generation artifact. Fixing prompt, resolution, output length and
decoding removes the confound.

**Limit of the guarantee.** GPU execution is not bit-deterministic even under
greedy decoding with a fixed seed, because reduction orders in fused kernels
vary between runs. Minor output variation is expected and does not invalidate
the timing work. What is controlled here is output *length* and generation
policy, not bit-exact output.

**Warmup.** The first several invocations pay for lazy CUDA context creation,
kernel autotuning and allocator growth. Including them produces a long tail
that describes startup rather than steady-state service time, so five runs are
discarded before measurement.

**Environment.** `nvpmodel` mode 0 (15W) on a Jetson Orin Nano Super, JetPack
6.2 / L4T 36.4.3, held fixed for every measurement (see
`docs/SETUP-jetson.md`). A comparison across two power modes is not a
comparison.

**Clock.** One monotonic source for every duration (`inference/clock.py`).

## Performance budget

> Set here from the Gate 1 baseline measurement, as promised, and not before.

Called a *performance budget* or *project target*, never an SLO. A
service-level objective derives from system or user requirements. This number
will derive from what the hardware turned out to be capable of, which is a
different thing. Measuring first and then setting a defensible target is
normal practice when there is no external requirement; the honesty is in the
label.

Derived from Gate 1: 6088 ms inference p50 on SmolVLM2-2.2B at 15W, and 30%
invalid output over the probe set as it then stood. Every target below is what this board was
measured doing plus room for the pipeline around it, which is exactly why it
is a budget and not an SLO.

| budget | target | derived from | met? |
|---|---|---|---|
| result age p50 | <= 6.5 s | 6.088 s inference + one 33 ms frame interval + preprocessing and publish, with margin | **met**, 6.163 s |
| result age p90 | <= 6.6 s | 6.117 s baseline p90 plus the same pipeline overhead | **met**, 6.32 s worst of 29 |
| invalid-output rate | <= 35% | measured 30%, with room for scenes the probe set does not cover | **met**, 10% over varied frames |
| sustained 10 min | no `stalled` or `engine_dead`, and result age p50 within 10% first minute vs last | thermal and memory behaviour are unmeasured over that span | **met**, 0.34% drift, `healthy` throughout |

**The invalid-output rate is 10%**, from `results/probe-varied`: ten frames
captured with the camera moved between shots, frame means spanning 94.7 to
174.1, answers spanning `clear`, `blocked/left` and `blocked/front_left`. One
frame failed, with `blocked` and `obstacle_location: none`.

Four earlier readings each measured something narrower: 30% (original probe
set, pre-correction tool), 0% (sustained run, one live scene), 100% (camera
face down), 0% (ten frames from a stationary camera, every answer identical --
n~=1 wearing n=10, the error `--probe-dir` exists to prevent). Ten varied
frames is still a small sample, and it is the first one that measures
generalisation rather than repetition.

**The budget is stricter than the method would now produce, and was met
anyway.** It derives from the 6088 ms baseline, taken on a sparse frame. The
representative-scene baseline is 6355 ms, which by the same derivation gives
about 6.8 s. The target stays at 6.5 s: re-deriving it would turn a 36 ms
margin into a comfortable one, which is accurate but indistinguishable from
moving a goalpost after the result. The system met the tighter number.

This table used to read 5.5 s, from a 4.930 s baseline that Gate 2 missed by
12%. The baseline was measuring an input the pipeline never produces (below),
and the budget inherited the error. Restated from the corrected measurement --
not relaxed to fit the result, which is why the old number stays on the page.

No p99 target. Under this overload a ten-minute run publishes roughly 120
results, which is barely above the 100 samples nearest-rank p99 needs to mean
anything, so p90 is the honest tail to commit to.

The invalid-output target deserves a word, because a 35% ceiling looks like
accepting failure. It is not a quality goal. The model emits unusable semantics
on 10% of varied frames and the system's job is to classify every one of them
rather than publish it; the budget exists to catch a *regression* in that rate, which
would mean something changed in preprocessing, the prompt, or the model. Making
the number smaller is a model problem, not a systems one, and out of scope.

## Baseline (Gate 1, single image, no pipeline)

SmolVLM2-2.2B-Instruct at `482adb5`, float16, 15W, one probe frame repeated
20 times with 5 warmup runs discarded.

| metric | p50 | p90 | max |
|---|---|---|---|
| inference latency | 6088 ms | 6117 ms | 6119 ms |
| time to first token | not exposed by this runtime | | |

No p99: nearest-rank p99 of 20 samples is rank 20, which is the maximum, so
quoting one would name the slowest single run as a tail statistic.

**Scene content moves this number.** 6088 ms is a sparse frame; the same
model on a representative scene is 6355 ms with 22 output tokens instead of
19 (`results/baseline/model-2.2b-split.json`). Generation latency follows
output length, and the frozen policy fixes prompt, resolution, token cap,
decoding and seed -- not what the camera sees.

| baseline | p50 | tokens |
| --- | --- | --- |
| sparse frame | 6088 ms | 19 |
| representative scene | 6355 ms | 22 |

**This replaces an earlier 4930 ms.** The first version of
`tools/gate1_baseline.py` opened the probe frame with PIL and handed the model
a 640x480 image, skipping `Preprocessor` and with it the 448x448 resolution
this document holds constant everywhere else. The pipeline feeds 448x448. The
two were never measuring the same input, and the 25% gap between them --
chased through three wrong hypotheses below -- was a bug in the measuring
tool.

The spread is the tell. The old run showed a 6134 ms maximum against a 4930 ms
p50 and that 1.2 s outlier was written off as leftover warmup. But
`VlmEngine.warmup` feeds `_blank_image(448, 448)`: warmup autotuned one tensor
shape and the measured runs arrived with another, so the first of them paid
again. With both at 448x448 the outlier is gone -- 31 ms between p50 and max
across 20 runs. A distribution that tightens when the inputs are made
consistent is independent evidence that the inputs were the problem.

Time to first token is what would separate prefill from decode, and
`transformers.generate` does not expose it without instrumenting the
generation loop. That is the measurement the section below needs, and the
reason NVTX ranges or a profiler are the next step rather than more runs.

Time to first token is reported separately **if the runtime exposes it
cleanly**. It separates prefill (vision encoding and prompt processing, one
pass) from decode (autoregressive, per token), which are different costs with
different fixes. If extracting it needs invasive instrumentation, skip it. It
is not worth fighting the runtime for.

## Overload behaviour

Full comparison in the README. Run on hardware with:

```bash
python3 tools/run_overload_sim.py --policies latest fifo_bounded fifo_unbounded
```

The table lives in the README only, so there is one copy for CI to check
against `results/simulated/` (`tools/check_readme_matches_results.py`).

## Gate 2 end to end

The full graph runs on the Jetson -- camera, ROS 2, admission policy,
SmolVLM2-2.2B, validation, JSON out -- at a 98.7% policy drop rate under a
183x overload ratio.

Twenty-nine consecutive published records, steady state:

| stage | p50 | range |
| --- | --- | --- |
| queue age | 0.020 s | 0.009 - 0.094 s |
| of which preprocess | 0.004 s | 0.003 - 0.005 s |
| inference | 6.136 s | 6.081 - 6.286 s |
| post-processing | 0.001 s | |
| **result age** | **6.163 s** | 6.120 - 6.317 s |

Against the 6.5 s budget, so the budget is met with 0.34 s of margin.

Queue age of 20 ms against a 33 ms frame interval is latest-frame admission
working exactly as designed: the newest frame is taken almost as soon as it
arrives, and preprocessing is 4 ms of it.

Inference in the pipeline is 6.136 s against 6.088 s standalone -- 0.8%, which
is the ROS hop and the per-frame preprocess, and is what "the pipeline costs
almost nothing on top of the model" should look like when it is true.

### The gap that was not there

This section used to claim a 25% unexplained inflation -- 6.136 s in the
pipeline against 4.930 s standalone, on every sample. Three hypotheses, all
wrong:

**GIL contention.** `py-spy` showed the decode loop holding the GIL between
CUDA launches while the executor thread deserialised 30 x 921,600 bytes per
second in the same process. Killed by halving the publish rate to 15 fps:
inference stayed at 6.10 s. (`vmstat` showing five of six cores idle was
briefly read as evidence against contention. It is not -- a GIL-bound process
occupies about one core, so that reading supported the hypothesis.)

**Memory pressure.** 451 MB available with the model resident, and swap in
use. Ruled out by the distribution: 29 consecutive samples inside a 200 ms
band is not paging.

**Thermal throttling.** Ruled out by reading the clock mid-run: 612 MHz,
exactly the 15W ceiling.

The cause was in neither process. The baseline tool was measuring 640x480
while the pipeline measured 448x448, so 4.930 s was never comparable to
anything. Re-measured through the same `Preprocessor`, the baseline is 6.088 s
and the gap is 0.8%.

Three plausible hypotheses, investigated carefully and falsified correctly,
and none of them could have reached the answer -- the discrepancy was
manufactured by the instrument. The tool producing the reference number needed
the same scrutiny as the system, and got none for two days.

### Sustained load: 12.5 minutes, every budget row met

94 published records over 599.5 s of steady state, camera at 30 fps,
`results/sustained/`.

| quantity | p50 | p90 | max | budget |
| --- | --- | --- | --- | --- |
| queue age | 0.019 s | 0.035 s | 0.301 s | |
| of which preprocess | 0.003 s | 0.004 s | 0.114 s | |
| inference | 6.438 s | 6.455 s | 6.649 s | |
| post-processing | 0.0006 s | 0.0007 s | 0.003 s | |
| **result age** | **6.464 s** | 6.485 s | 6.804 s | <= 6.5 / 6.6 s |
| consumer age | 6.474 s | 6.493 s | 6.831 s | |
| invalid output | 0.0% | | | <= 35% |
| available memory | 425.6 MB | | 418.3 MB min | |

Result age p50 moved 6.4438 s to 6.4659 s between the first minute and the
last: **0.34% against a 10% budget**. Health stayed `healthy` for the whole
run, no `degraded`, no `stalled`. Memory held between 418 and 433 MB.

**The pipeline costs 12 ms per cycle.** Successive results are 6.447 s apart
against 6.438 s of inference, so the worker is idle 12 ms between frames -- a
99.8% duty cycle. Everything outside the model is queueing, preprocessing,
validation and publishing, and it rounds to nothing.

**Result age is what a consumer experiences, now measured rather than
asserted.** `consumer_age` -- capture to a separate subscriber process -- is
10 ms above `result_age`, which ends at publish inside the inference node.
0.15%. Worth checking, because the forward hop needed `array.array` to get
from 173 ms to 0.9 ms, so a free hop was not a safe assumption.

### What the scene costs, and what that means for the budget

The first attempt at this run produced 94 records of which **every one failed
validation**, and the cause was that the camera was face down. The model was
being asked whether a path was clear while looking at a desk. Two independent
readings agreed: `telemetry_node` reported 100% invalid, and `inference_node`
went `degraded` at exactly the record where its 20-sample window filled.

Re-pointed at a scene, the invalid rate is 0% -- and latency rose:

| | face down | real scene |
| --- | --- | --- |
| result age p50 | 6.182 s | 6.464 s |
| result age p90 | 6.202 s | 6.485 s |
| invalid output | 100% | 0% |

Generation latency is dominated by output length, and a featureless surface is
cheap to describe. **So scene content is a latency input, which the frozen
generation policy does not control.** The policy fixes the prompt, resolution,
token cap, decoding and seed; it cannot fix what the camera sees.

That leaves 36 ms of margin on the p50 budget, and the budget's own derivation
is now suspect in the same way: the 6.088 s baseline was measured on
`probe_00.jpg`, one still frame. A budget derived from one scene and tested
against another is the same error as a baseline measured on 640x480 and
compared against 448x448, in a less obvious costume. The number stands because
it was met, and the next re-derivation should use a baseline over the probe
set rather than a single frame.

### The startup outlier, explained

An earlier version of this section reported preprocessing at 2.818 s from a
single published record, and three rounds of investigation followed from n=1.
The distribution above shows 4 ms.

The outlier is real and reproducible: **the first published record of every
run** shows about 2.7 s of preprocessing. It is not paging and not
contention. `Preprocessor.run` called `_resolve()` inside its timed region,
and `_resolve()` is what lazily does `import cv2` -- roughly 2.7 s on this
board, reading OpenCV's libraries off an SD card. A one-time library load was
being charged to a per-frame metric, on the one frame where it happened.

That is why `py-spy` never saw it (a one-time cost is rarely sampled) and why
a separate probe process measured 4 ms (it imported cv2 at module scope,
before any timing started).

`Pipeline.start` now warms the preprocessor alongside the engine. The engine's
first invocations were already discarded for exactly this reason; the
preprocessor was paying the same kind of cost without the same treatment.

Two lessons worth keeping. A number describing one sample is not a
measurement, which is the same error the p99 reporting had an hour earlier.
And a stage timer that encloses lazy initialisation will report startup as
throughput, once, convincingly.

## The bottleneck investigation (Gate 3)

### Step 1: hypothesis, written down before profiling

Being wrong here is a good outcome and a good README section. The candidates,
and what each would look like in a trace:

| candidate | signature in the trace |
|---|---|
| CPU preprocessing | large CPU-side NVTX `preprocess` range with the GPU idle |
| host-to-device transfer | memcpy H2D dominating, especially if synchronous |
| vision encoder (prefill) | one large fused block before any decode step |
| autoregressive decode | many small kernels, low occupancy, gaps between them |
| synchronisation stalls | GPU idle gaps aligned with CPU-side waits |
| memory pressure | allocator churn, swap, or throttling under sustained load |

### Evidence already in hand, from the Gate 1 baseline

`tegrastats` at 1 Hz across the 12.5-minute sustained run, 15W:
`results/sustained/tegrastats-sustained.log`, summarised by
`tools/summarize_tegrastats.py`. That file holds four runs, because
`tegrastats --logfile` appends and the filename was reused; the tool splits on
the time gaps and this is run 4. The busy window is the 534 samples above 50%
GPU, one trimmed from each end of each stretch because tegrastats reports the
mean over its interval and a straddling sample averages a working GPU with an
idle one.

| quantity | busy p50 | busy range | idle p50 |
| --- | --- | --- | --- |
| GPU utilisation | **99%** | 56 - 100% | 0% |
| GPU clock | 611 MHz | p10 611, the 15W ceiling | 305 MHz |
| EMC clock | 2133 MHz, its cap | pinned | 204 MHz |
| EMC utilisation | 55% | 29 - 68% | 23% |
| CPU, busiest core | 38% | 15 - 86% | 33% |
| CPU, summed over 6 cores | 77% | 25 - 144% | 93% |
| system RAM used | 7069 MB of 7620 | 6708 - 7076 | 3851 MB |
| swap used | 533 MB | 437 - 536 | 174 MB |
| tj | 62.8 C | 51.2 - 63.8 | 49.9 C |
| VDD_IN | 15.7 W | 11.7 - 17.2 | 6.0 W |

This replaces three jtop screenshots read off at three arbitrary moments. A
screenshot cannot answer the question the table exists to answer -- whether the
clock *held* -- and nobody else can regenerate one.

The idle column describes weight loading, not an idle board: 115 of the run's
152 sub-50% samples are one contiguous stretch at the start, CPU-bound at 94%
summed with the GPU parked at 305 MHz and the module drawing 6 W. That is why
"idle" CPU reads higher than busy CPU.

Within the longest busy stretch, first ten samples against last ten:

| quantity | first | last | delta |
| --- | --- | --- | --- |
| tj | 62.9 C | 63.0 C | +0.1 |
| GPU clock | 611 MHz | 611 MHz | 0 |
| system RAM used | 7069 MB | 7068 MB | -1 |
| swap used | 533 MB | 533 MB | 0 |
| VDD_IN | 15.9 W | 15.8 W | -0.1 |

**Thermal throttling is ruled out, now over twelve minutes.** The clock's 10th
percentile is 611 MHz against a 612 MHz high; one sample in 534 dipped to 509
and the rest held. tj reaches 62.8 C and plateaus -- the drift table moves it
0.1 C -- against a part that throttles far higher. An earlier two-minute run
showed tj still climbing at the end and this document said so, scoping the
claim to that length. Twelve minutes settles it.

Memory is ruled out on the same evidence: RAM and swap do not move across the
measured window. The growth to 533 MB of swap happens during weight loading,
before the first timed result.

Host-to-device transfer and synchronisation stalls are ruled out: both show as
GPU idle gaps aligned with CPU-side waits, and the GPU reads 90-100% in 550 of
752 samples.

**CPU preprocessing is only partly ruled out, and this is the open lead.**
Our `Preprocessor` is measured at 3 ms, so it is not the cost. But
`inference_latency` is one number spanning the engine's `processor` call
(tokenisation and image tensor preparation, on the CPU) and `generate` (on the
GPU), and nothing separates them. The log says there is something there:

- A window below 90% GPU recurs every 6.4 s -- once per inference cycle, mode
  6 s over 87 occurrences.
- The deeper sub-50% dips recur every 19 s, which is the beat between a 6.45 s
  cycle and 1 Hz sampling rather than a three-cycle period.
- For a one-second window to average 51%, the non-GPU phase inside it must be
  at least 0.44 s. Against a 6.44 s cycle that is **roughly 7% of inference
  spent off the GPU**.

7% inferred from 1 Hz aliasing is a direction, not a measurement. Stamping
either side of the processor call would measure it outright, costs two lines,
and splits the headline number into CPU and GPU parts -- which is the same
question time to first token was wanted for, answerable without fighting the
runtime. That is step 2, ahead of NVTX.

What is left is on-GPU work: prefill, decode, or both. The arithmetic says to
check prefill first. 6088 ms for 19 output tokens is 320 ms per token if decode
were the whole cost. Decode re-reads the weights once per token, so 4.4 GB
against this mode's memory bandwidth bounds a token well under 100 ms. Decode
alone does not explain the measurement, and SmolVLM tiles an image into many
vision tokens, all of which are processed in one prefill pass.

**Hypothesis:** prefill (vision encoding plus prompt processing) dominates, not
decode. Time to first token separates them, which is why it is a reported
metric. If TTFT is a large fraction of 6088 ms, the fix space is image
tiling, resolution and the vision tower. If it is small, decode is slower than
bandwidth explains and the fix space is the generation path.

Written before profiling. Being wrong here is a good outcome.

### Step 4: what the evidence showed -- the hypothesis held

No profiler needed. GPU time is linear in output length, so a least-squares
line through several baselines gives the per-token cost as its slope and
prefill as its intercept:

    generate = prefill + per_token x tokens

`tools/decompose_inference.py` does this and refuses any pair whose model or
generation policy differs -- the guard this project learned to need. All three
runs: same model at `482adb5`, 448x448, greedy, seed 0. Only the scene differs.

| run | GPU time | tokens |
| --- | --- | --- |
| `model-2.2b.json` | 5.821 s | 19 |
| `model-2.2b-varied.json` | 5.897 s | 20 |
| `model-2.2b-split.json` | 6.091 s | 22 |

**91.0 ms per output token**, against the "well under 100 ms" the bandwidth
arithmetic above predicted before any of it was measured. Residuals +6, -9 and
+3 ms on a 6 s quantity, so the line holds to 0.15%.

| stage | | share of 6.355 s |
| --- | --- | --- |
| processor, CPU | 0.267 s | 4.2% |
| **prefill, GPU** | **4.090 s** | **64.4%** |
| decode, GPU | 2.002 s | 31.5% |

Prefill dominates, as written down. So the fix space is image tiling,
resolution and the vision tower -- not the generation path, and not the CPU
phase, which is worth 267 ms at most.

**Two points were not enough, and the third proved it.** With only the 19- and
22-token runs this read 89.1 ms and 65.0% prefill; with the 19- and 20-token
runs, 75.3 ms and 71.2%. Pairwise slopes across the three are 73.3, 89.1 and
97.0 ms. Two points fit a line exactly, so a two-point estimate cannot be
wrong and cannot be checked -- the earlier version of this section labelled
that caveat and then relied on the estimate anyway. The fit now uses every run
and prints residuals, which makes "per-token cost is flat over this range" a
claim the output tests rather than one the reader has to take on trust.

A profiler would still measure the split directly. This says where to point
one, for two stamps and two extra baselines.

**The 7% that was wrong.** The utilisation log was read as implying ~7% of
inference off the GPU, since a one-second window averaging 51% needs ~490 ms
of non-GPU time. Measured: 4.2%. The reasoning assumed the GPU was saturated
the rest of the time, and it is not -- decode's many small kernels leave real
gaps, so `GR3D_FREQ` sits below 100% with no CPU phase at all. An unstated
premise is an unchecked one.

### Not the bottleneck, but worth recording

The GPU sits at exactly the 15W mode's 612 MHz ceiling and EMC at its cap, so
this measurement is clock-limited by a deliberate choice. Mode 1 (25W) offers
918 MHz and 3199 MHz. That is a different operating point, not a fix, and
changing it would invalidate every measurement taken so far. Noted so the
figure is read as "6088 ms at 15W" rather than "6088 ms".

### Step 2: NVTX ranges

Optional, and skipped if instrumentation fights back. When cheap, annotate
`capture`, `admission`, `preprocess`, `model_invoke`, `generate`, `parse` and
`publish`. NVTX ranges make an Nsight timeline legible: without them the trace
shows kernels, and the question is which *application stage* they belong to.

### Step 3: profile -- not run, and why

```bash
nsys profile --trace=cuda,nvtx,osrt --output=results/traces/baseline \
    python3 tools/run_pipeline.py --duration 60
```

Not run. The question a profiler was wanted for -- how much of inference is
prefill and how much is decode -- was answered above from three baselines at
different output lengths, to residuals of 9 ms on a 6 s quantity. An Nsight
trace would confirm the split and subdivide prefill further, which would be
worth doing if the next step were optimising prefill. It is not; see step 5.

Recorded as a decision rather than a gap. A profiler is an instrument, and the
measurement it was for came in cheaper by other means.

### Step 5: the one change -- deliberately not made

**Change: none.**

The evidence says prefill is 64% of inference and the fix space is image
tiling, resolution and the vision tower. Each of those changes what the model
sees, so each one invalidates the frozen generation policy that every number
in this document depends on, and the comparison would have to be rebuilt from
the baseline up.

That is affordable. What makes it the wrong call is what it would add: one
more latency number. The project's claim is that a system can stay responsive
and honest when a learned component cannot keep pace with its sensor, and that
claim is already carried by the admission policy, the failure taxonomy, the
loss accounting and a bottleneck investigation whose prediction held. A 20%
faster prefill would not strengthen any of it.

So this is the Day 10 rule applied rather than quoted: the investigation is
compelling and the remaining time is worth more spent on communication.
Written down because "we ran out of time" and "we decided not to" look
identical in a repository, and only one of them is a judgement.

### Step 6: re-measure, identical methodology

Nothing to re-measure, since step 5 made no change. The methodology is on
record for whoever does: same power mode, same generation policy, same warmup,
same duration, same seed, only the one change differing.

### Step 7: sustained load

Twelve and a half minutes continuous at `nvpmodel` mode 0, logging clocks,
memory and power. Artifacts in `results/sustained/`. Full write-up in the
Gate 2 section above.

| quantity | value |
|---|---|
| duration | 599.5 s of steady state, 94 published records |
| result age p50, first minute vs last | 6.4438 s -> 6.4659 s, **0.34%** |
| clock throttling observed | none; p10 611 MHz against a 612 MHz ceiling |
| memory at start vs end | 7069 MB used -> 7068 MB; swap 533 -> 533 MB |
| mean / peak power | 15.51 W / 15.59 W from the rails; 15.7 / 17.2 W by tegrastats |
| health state transitions | none; `healthy` throughout |
| tj | 62.9 C -> 63.0 C across the longest stretch: plateaued |

The two power figures differ because they measure different things:
`telemetry_node` reads the INA3221 `VDD_IN` rail per published record, and
tegrastats averages over its own interval including the idle gaps. Both are
recorded rather than reconciled, because picking one would hide that the
question exists.

## Day 10 rule

If the system runs and the bottleneck investigation is compelling, **stop
engineering.** The remaining days are worth more spent on communication than
on another optimisation.
