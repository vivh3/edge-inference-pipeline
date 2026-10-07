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

`colcon` is not part of `ros-humble-ros-base`:

```bash
sudo apt install -y python3-colcon-common-extensions
```

**Build with the venv active, and source ROS before it.** `ament_python`
bakes whichever `python3` is on PATH into each node's launcher shebang, and
only the venv's interpreter can see all three of ROS, torch and the core at
once. It was created with `--system-site-packages` precisely so it can; system
Python has no torch, which `inference_node` needs. Building with the venv
deactivated produces launchers that re-exec `/usr/bin/python3` and fail on
`No module named 'inference'` no matter what is activated afterwards.

```bash
cd ~/edge-inference-pipeline
source .venv/bin/activate
pip install -e .                      # makes inference/ and telemetry/ importable

source /opt/ros/humble/setup.bash     # ROS first
source .venv/bin/activate             # then the venv, so python3 resolves to it
cd ros2_ws
colcon build
source install/setup.bash

python3 -m edge_perception.capture_node
```

Run nodes with `python3 -m`, not `ros2 run`. `ament_python` bakes a shebang
into each launcher from whichever interpreter built the package, and colcon
from apt runs under `/usr/bin/python3` regardless of what is active, so
`ros2 run` launches a node under an interpreter that cannot see the venv.
`python3 -m` uses whatever is active, which is the only interpreter that sees
ROS, torch and the core at once.

If a build ever picked up the wrong interpreter, `rm -rf build install log`
before rebuilding; the shebangs are written at install time and are not
regenerated otherwise.

In a second shell, with the same setups sourced:

```bash
ros2 topic hz /frames          # should sit near the measured 30.027 fps
ros2 topic echo /frames --field frame_id --once
ros2 topic info /frames -v     # confirms the QoS actually in force
```

`ros2 topic hz` is the check that matters. The camera measured 30.027 fps in
Gate 1, and anything well below that means the publish path is now the
bottleneck rather than the camera.

## What the first run found

`capture_node` throttled the camera from 30.027 fps to 5.64. `frame_id` is
what made it visible: a probe subscribing to `/frames` can compare how many
frames capture *published* against how many *arrived*, which `ros2 topic hz`
cannot do, because it only ever sees what arrived.

Splitting the sink into stages put the cost somewhere nobody would have
guessed:

| | before | after |
| --- | --- | --- |
| convert to `sensor_msgs/Image` | 173.2 ms | **0.9 ms** |
| publish | 0.3 ms | 1.0 ms |
| inter-frame interval | 176.7 ms | **33.3 ms** |
| effective capture rate | 5.64 fps | **30.03 fps** |

"27 MB/s is too much for the middleware" was a plausible theory and wrong by
two orders of magnitude: publishing cost 0.3 ms. The time was in a single
assignment. rclpy's generated setter for a `uint8[]` field short-circuits when
handed an `array.array` and otherwise validates every element in a Python loop
under `__debug__` -- 921,600 of them per 640x480 bgr8 frame.

The node now costs 1.9 ms of a 33.3 ms frame budget, and 33.3 ms is 30.03 fps,
which is the camera's measured 30.027 fps reproduced inside ROS.

This is the same shape as the capture-buffer bug in Gate 1: work placed where
it silently degrades capture, with nothing reporting the degradation. Both
were found by measuring frames published against frames delivered, and neither
would have appeared in a test.

## The subscriber saturates before inference exists

With `capture_node` publishing at a steady 30 fps and a subscriber that does
nothing but increment a counter:

| elapsed | received | published | lost in the middleware | probe RSS |
| --- | --- | --- | --- | --- |
| 10 s | 273 | 297 | 8.4% | - |
| 30 s | 681 | 897 | 24.2% | 57 MB |
| 60 s | 1276 | 1797 | 29.0% | 57 MB |
| 70 s | 1530 | 2097 | 27.1% | - |

RSS is flat and the loss plateaus near 28%, so this is CPU saturation rather
than a leak: deserialising 30 x 921,600 bytes per second is more than six A78
cores at 15W will do alongside the publisher. Capture itself never wavered,
publishing exactly 300 frames per 10 s throughout.

**Those 28% are lost silently.** Nothing in ROS counts them, and `ros2 topic
hz` would report a healthy-looking 21 Hz with no hint that 500 frames had
vanished. Only `frame_id` exposes it, by letting a subscriber compare what was
published against what arrived.

That has a consequence for the telemetry. This project computes drop rate
against frames captured, and once capture and inference are separate
processes, frames can disappear before the admission policy ever sees them.
`inference_node` therefore has to report two different losses under two
different names: frames the middleware dropped, recoverable from gaps in
`frame_id`, and frames the admission policy dropped on purpose. Reporting one
number would attribute the middleware's losses to a design decision.

Whether to reduce the 28% at all is a separate question. Under overload
`inference_node` will drop ~99% of frames by design, so middleware loss may be
irrelevant to the result. Measure it in place before optimising it.

## What to watch next

A 640x480 bgr8 frame is 921,600 bytes, so 30 fps is about 27 MB/s crossing the
middleware. Intra-host that should use shared memory rather than the network
stack, but it is unverified. If `ros2 topic hz` comes in low, publishing
compressed frames and decoding in the subscriber is the obvious next
experiment -- and it would move the JPEG decode cost from `capture_node` into
the node that already pays for preprocessing.

Memory: four nodes measured 108 MB empty, leaving roughly 350 MB with the
2.2B model loaded. `capture_node` holds cv2 and frame buffers on top of that,
so watch `jtop` during the first end-to-end run.
