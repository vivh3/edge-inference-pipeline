# ROS 2 workspace (Gate 2)

Not yet implemented. Scoped, so it does not quietly grow.

ROS 2 Humble is used here as **integration plumbing plus one considered QoS
decision**. That is the accurate description of the depth involved, and it is
the one this project claims. Executors, lifecycle nodes, transforms,
composition, and DDS internals are where the real ROS 2 depth lives, and none
of them is exercised.

## Measured before building anything

Two questions had to be answered before the node layout was worth designing,
on the Jetson at 15W, headless, with ROS 2 Humble from apt.

**Can one interpreter hold rclpy, torch and cv2 together?** Yes.
`import rclpy, torch, cv2` succeeds with `/opt/ros/humble/setup.bash` sourced
and the project venv (`--system-site-packages`) active. rclpy is built against
system Python 3.10 and lives outside the venv while torch lives inside it, so
this was not a given. Had it failed, `inference_node` could not both subscribe
to frames and run the model, and the nodes would have had to be separate
processes talking over DDS.

**What do the nodes cost?** 108 MB for four. Four `rclpy` nodes that construct
and spin and do nothing else moved system memory from 403 MB to 511 MB used,
about 27 MB each.

That matters against the Gate 1 figure of 519 MB available with
SmolVLM2-2.2B loaded. Four nodes plus cv2 in the capture node and some frame
buffers leaves roughly 350 MB. Tight, and worth watching during the
end-to-end run, but the 2.2B stays.

## Planned nodes

Thin wrappers over `inference/` and `telemetry/`, which already hold all the
behaviour. A node should own message conversion and nothing else.

| node | wraps | state |
|---|---|---|
| `capture_node` | `inference.capture.WebcamSource` | **written, not yet run** |
| `inference_node` | `inference.pipeline.Pipeline` (admission policy, engine, validation) | planned |
| `consumer_node` | mock consumer: logs the published JSON record | planned |
| `telemetry_node` | `telemetry.metrics.Metrics` snapshots on a timer | planned |

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

## Why there is a custom message

`StampedFrame` exists because two pieces of trusted metadata have nowhere
standard to go.

ROS 2 removed `Header.seq`, so the capture sequence number has no home. And
`header.stamp` carries the ROS clock, which is system time by default, while
every duration in this project is a difference of CLOCK_MONOTONIC readings so
that an NTP step mid-inference cannot corrupt a latency or make it negative.
`StampedFrame` therefore carries `frame_id` and `capture_ts_monotonic`
alongside the image, and fills `header` conventionally for tooling that
expects it.

This is the one custom message. The exclusions above still hold.

## Build and run

The core is a normal Python package, so the nodes import it rather than
vendoring it:

```bash
cd ~/edge-inference-pipeline
source .venv/bin/activate
pip install -e .                      # makes inference/ and telemetry/ importable

source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build
source install/setup.bash

ros2 run edge_perception capture_node
```

In a second shell, with the same two setups sourced:

```bash
ros2 topic hz /frames          # should sit near the measured 30.027 fps
ros2 topic echo /frames --field frame_id --once
ros2 topic info /frames -v     # confirms the QoS actually in force
```

`ros2 topic hz` is the check that matters. The camera measured 30.027 fps in
Gate 1, and anything well below that means the publish path is now the
bottleneck rather than the camera.

## What to watch on the first run

A 640x480 bgr8 frame is 921,600 bytes, so 30 fps is about 27 MB/s crossing the
middleware. Intra-host that should use shared memory rather than the network
stack, but it is unverified. If `ros2 topic hz` comes in low, publishing
compressed frames and decoding in the subscriber is the obvious next
experiment -- and it would move the JPEG decode cost from `capture_node` into
the node that already pays for preprocessing.

Memory: four nodes measured 108 MB empty, leaving roughly 350 MB with the
2.2B model loaded. `capture_node` holds cv2 and frame buffers on top of that,
so watch `jtop` during the first end-to-end run.
