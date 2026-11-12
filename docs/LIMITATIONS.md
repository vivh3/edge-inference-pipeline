# Limitations

## Known issues

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

## Before this could affect actuation

Nothing here is safety architecture, and `consumer_node` only logs an advisory, on purpose.
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

## Scope

Out of scope and staying there: a quantisation matrix, TensorRT as a required box,
Isaac Sim, training or fine-tuning, custom CUDA kernels, a real robot or planner, a
second model or runtime, ROS 2 lifecycle nodes or custom executors, and process
supervision beyond a basic watchdog.
