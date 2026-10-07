# Performance: methodology and the bottleneck investigation

> **Status: methodology fixed, model measurements pending (Gate 1/3).**
> Every `TBD` is a measurement still to be taken. The overload numbers in this
> repository come from the synthetic simulation and are labelled `SIMULATED`
> in their own summary files. No estimate is ever written into this document
> as if it were a measurement.

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

> Set in Gate 2 **from the Gate 1 baseline measurement**. Not before.

Called a *performance budget* or *project target*, never an SLO. A
service-level objective derives from system or user requirements. This number
will derive from what the hardware turned out to be capable of, which is a
different thing. Measuring first and then setting a defensible target is
normal practice when there is no external requirement; the honesty is in the
label.

| budget | target | met? |
|---|---|---|
| result age p50 | `TBD` | `TBD` |
| result age p99 | `TBD` | `TBD` |
| invalid-output rate | `TBD` | `TBD` |
| sustained 10 min without health-state degradation | `TBD` | `TBD` |

## Baseline (Gate 1, single image, no pipeline)

SmolVLM2-2.2B-Instruct at `482adb5`, float16, 15W, one probe frame repeated
20 times with 5 warmup runs discarded.

| metric | p50 | p90 | max |
|---|---|---|---|
| inference latency | 4930 ms | 4944 ms | 6134 ms |
| time to first token | not exposed by this runtime | | |

No p99: nearest-rank p99 of 20 samples is rank 20, which is the maximum, so
quoting one would name the slowest single run as a tail statistic. The 6134 ms
maximum is the first measured run, which still carries warmup the five
discarded runs did not absorb.

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

| policy | drop rate | result age p50 | result age p99 | result age max |
|---|---|---|---|---|
| `latest` | `TBD` | `TBD` | `TBD` | `TBD` |
| `fifo_bounded(8)` | `TBD` | `TBD` | `TBD` | `TBD` |
| `fifo_unbounded` | `TBD` | `TBD` | `TBD` | `TBD` |

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

`jtop` during 20 measured inferences of SmolVLM2-2.2B at float16, 15W:

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
check prefill first. 4907 ms for 19 output tokens is 258 ms per token if decode
were the whole cost. Decode re-reads the weights once per token, so 4.4 GB
against this mode's memory bandwidth bounds a token well under 100 ms. Decode
alone does not explain the measurement, and SmolVLM tiles an image into many
vision tokens, all of which are processed in one prefill pass.

**Hypothesis:** prefill (vision encoding plus prompt processing) dominates, not
decode. Time to first token separates them, which is why it is a reported
metric. If TTFT is a large fraction of 4907 ms, the fix space is image
tiling, resolution and the vision tower. If it is small, decode is slower than
bandwidth explains and the fix space is the generation path.

Written before profiling. Being wrong here is a good outcome.

### Not the bottleneck, but worth recording

The GPU sits at exactly the 15W mode's 612 MHz ceiling and EMC at its cap, so
this measurement is clock-limited by a deliberate choice. Mode 1 (25W) offers
918 MHz and 3199 MHz. That is a different operating point, not a fix, and
changing it would invalidate every measurement taken so far. Noted so the
figure is read as "4907 ms at 15W" rather than "4907 ms".

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
