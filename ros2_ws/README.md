# ROS 2 workspace (Gate 2)

Not yet implemented. Scoped, so it does not quietly grow.

ROS 2 Humble is used here as **integration plumbing plus one considered QoS
decision**. That is the accurate description of the depth involved, and it is
the one this project claims. Executors, lifecycle nodes, transforms,
composition, and DDS internals are where the real ROS 2 depth lives, and none
of them is exercised.

## Planned nodes

Thin wrappers over `inference/` and `telemetry/`, which already hold all the
behaviour. A node should own message conversion and nothing else.

| node | wraps |
|---|---|
| `capture_node` | `inference.capture.WebcamSource` |
| `inference_node` | `inference.pipeline.Pipeline` (admission policy, engine, validation) |
| `consumer_node` | mock consumer: logs the published JSON record |
| `telemetry_node` | `telemetry.metrics.Metrics` snapshots on a timer |

## The one QoS decision

The question to answer, and to be able to defend in a sentence: does ROS 2's
`KEEP_LAST` with `depth=1` plus an appropriate reliability setting already
implement part of the latest-frame semantics that `inference/buffer.py`
implements at the application level?

The defensible statement is of the form: *"the camera publishes faster than
inference consumes; I chose these history and reliability semantics because
stale sensor data is less useful than dropped sensor data for this
workload."*

If an application-level buffer is still required because of the executor
architecture -- a single-threaded executor will not deliver a newer message
while a callback is still running, so the newest frame can sit in the
middleware queue behind work already in progress -- then **say so and explain
why**. That is a better answer than claiming QoS alone solved it.

Timebox: a few hours. This is one decision, not a DDS study.

## Deliberately excluded

Lifecycle nodes, node composition, custom executors, custom message types
beyond what the pipeline needs, and any second middleware configuration.
