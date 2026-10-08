# Performance: methodology and the bottleneck investigation

> **Status: Gate 1 and Gate 2 measured; Gate 3 open.**
> Every `TBD` is a measurement still to be taken. The overload comparison
> comes from the synthetic simulation and is labelled `SIMULATED` in its own
> summary files. No estimate is written here as if it were a measurement.

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
invalid output over the probe set. Every target below is what this board was
measured doing plus room for the pipeline around it, which is exactly why it
is a budget and not an SLO.

| budget | target | derived from | met? |
|---|---|---|---|
| result age p50 | <= 6.5 s | 6.088 s inference + one 33 ms frame interval + preprocessing and publish, with margin | **met**, 6.163 s |
| result age p90 | <= 6.6 s | 6.117 s baseline p90 plus the same pipeline overhead | **met**, 6.32 s worst of 29 |
| invalid-output rate | <= 35% | measured 30%, with room for scenes the probe set does not cover | **met**, 30% |
| sustained 10 min | no `stalled` or `engine_dead`, and result age p50 within 10% first minute vs last | thermal and memory behaviour are unmeasured over that span | `TBD` |

This table used to read 5.5 s, from a 4.930 s baseline that Gate 2 missed by
12%. The baseline was measuring an input the pipeline never produces (below),
and the budget inherited the error. Restated from the corrected measurement --
not relaxed to fit the result, which is why the old number stays on the page.

No p99 target. Under this overload a ten-minute run publishes roughly 120
results, which is barely above the 100 samples nearest-rank p99 needs to mean
anything, so p90 is the honest tail to commit to.

The invalid-output target deserves a word, because a 35% ceiling looks like
accepting failure. It is not a quality goal. The model emits unusable semantics
on 30% of frames and the system's job is to classify every one of them rather
than publish it; the budget exists to catch a *regression* in that rate, which
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

`jtop` during 20 measured inferences of SmolVLM2-2.2B at float16, 15W. Taken
during the pre-correction run, so the memory rows describe its 640x480 input.
Utilisation, clocks, power and temperature are not latency figures, so the
conclusions below stand.

| quantity | during inference | idle |
| --- | --- | --- |
| GPU utilisation | **99.7 - 99.8%** | 0% |
| GPU clock | 612 MHz, the 15W ceiling | 306 MHz |
| EMC | 2.1 GHz, against a 2133 MHz cap | 204 MHz |
| CPU, busiest core | 38% (the python process) | 14% |
| process memory | 1.2 GB CPU **+ 5.1 GB GPU** | - |
| system memory | 6.9 GB of 7.4 GB, 369 MB swapped | 3.5 GB, no swap |
| VDD_IN | 15.9 - 16.9 W | 5.9 W |
| tj | 57 - 59 C, fan 38 - 43% | 50 C |

This rules out three of the six candidates before any profiler runs. A GPU at
99.8% is not idling behind CPU preprocessing, not waiting on a synchronous
host-to-device copy, and not stalling on synchronisation. Those all show as
GPU idle gaps, and there are none. CPU is 38% on one core and near zero on the
other five.

It also rules out thermal throttling. 59 C on a part that throttles far higher,
with the fan at 43%, is not a thermally limited board.

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

### Step 3: profile

```bash
nsys profile --trace=cuda,nvtx,osrt --output=results/traces/baseline \
    python3 tools/run_pipeline.py --duration 60
```

- Trace: `results/traces/TBD`
- Screenshot: `TBD`

### Step 4: what the evidence showed

`TBD`

### Step 5: the one change

One justified change, whichever the evidence supports.

**If the evidence supports none of the obvious tools, say so plainly.**
"Profiling showed X dominated, so optimising Y would not have addressed the
system bottleneck" is a stronger result than forcing a tool in, and it is the
judgement this project exists to demonstrate. TensorRT, quantisation and
Nsight are instruments here, not success criteria.

- Change: `TBD`
- Justification from the trace: `TBD`

### Step 6: re-measure, identical methodology

Same power mode, same generation policy, same warmup, same duration, same
seed. Only the one change differs.

| metric | before | after | delta |
|---|---|---|---|
| inference latency p50 | `TBD` | `TBD` | `TBD` |
| result age p50 | `TBD` | `TBD` | `TBD` |
| result age p99 | `TBD` | `TBD` | `TBD` |
| drop rate | `TBD` | `TBD` | `TBD` |
| peak RSS | `TBD` | `TBD` | `TBD` |

### Step 7: sustained load

Ten minutes continuous in the fixed power profile, logging clocks, memory and
power. What a short run misses: thermal throttling, memory growth, allocator
fragmentation and health-state flapping.

| quantity | value |
|---|---|
| duration | `TBD` |
| result age p50, first minute vs last minute | `TBD` |
| clock throttling observed | `TBD` |
| RSS at start vs end | `TBD` |
| mean / peak power | `TBD` |
| health state transitions | `TBD` |

## Day 10 rule

If the system runs and the bottleneck investigation is compelling, **stop
engineering.** The remaining days are worth more spent on communication than
on another optimisation.
