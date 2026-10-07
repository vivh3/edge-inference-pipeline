"""capture_node: publish camera frames with the metadata the pipeline trusts.

This node owns message conversion and nothing else. The frame source, the
sequence numbering and the timestamping all live in `inference/capture.py`
unchanged, so what runs on the robot is what the tests cover on a laptop.

Two decisions are made here, and both have to survive being questioned.

**QoS is the sensor-data profile: KEEP_LAST depth 1, BEST_EFFORT, VOLATILE.**
A stale frame has negative value for this workload, so the middleware should
discard rather than retransmit, and a subscriber that joins late should not be
handed a backlog.

That does **not** replace `inference.buffer.LatestFrameBuffer`, for two
reasons. A single-threaded executor will not deliver a newer message while a
callback is still running, so under this project's overload the newest frame
sits in the middleware queue behind work already in progress -- QoS depth 1
bounds the queue but does not make the delivered frame current. More
importantly, a frame the middleware discards is discarded without being
counted. The reported drop rate would then describe only the frames DDS chose
to deliver, which is the same silent loss as the one-buffer bug found in Gate
1. The admission policy exists so the decision is explicit and lands in the
telemetry.

**The monotonic capture time travels in its own field, not in `header.stamp`.**
ROS 2's default clock is system time, and durations in this project are always
differences of CLOCK_MONOTONIC readings, because an NTP step mid-inference
would otherwise corrupt a latency or make it negative. `header.stamp` is
filled with the ROS clock for tooling that expects it, and nothing subtracts
it. ROS 2 also removed `Header.seq`, so `frame_id` has nowhere standard to go
either; both are why `StampedFrame` exists rather than a bare
`sensor_msgs/Image`.
"""

from __future__ import annotations

import array

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from edge_perception_msgs.msg import StampedFrame
from inference.capture import WebcamSource

PARAMETERS = {
    "device": 0,
    "width": 640,
    "height": 480,
    "fps_hint": 30.0,
    "fourcc": "MJPG",
    "backend": "v4l2",
    "buffer_frames": 2,
    "topic": "frames",
    # A TF frame name, which is what header.frame_id means in ROS. It is not
    # the capture sequence number; that is StampedFrame.frame_id.
    "optical_frame": "camera_optical_frame",
}


def to_image_msg(payload) -> Image:
    """BGR array -> sensor_msgs/Image, without cv_bridge.

    cv_bridge is compiled against a particular numpy, and this project holds
    numpy below 2.0 so that apt's OpenCV keeps working. Ten lines here is
    cheaper than a dependency that can break on an unrelated upgrade.

    `data` is assigned an `array.array` rather than `bytes`, and the
    difference is 173 ms per frame. rclpy's generated setter for a `uint8[]`
    field short-circuits on `array.array` and otherwise validates every
    element in a Python loop under `__debug__` -- 921,600 of them for one
    640x480 bgr8 frame. Measured on the Jetson: 173.2 ms with `bytes`,
    against 0.3 ms to publish the result. Serialisation was never the
    problem.
    """
    height, width = payload.shape[:2]
    msg = Image()
    msg.height = height
    msg.width = width
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = width * 3
    msg.data = array.array("B", payload.tobytes())
    return msg


class CaptureNode(Node):
    def __init__(self) -> None:
        super().__init__("capture_node")
        for name, default in PARAMETERS.items():
            self.declare_parameter(name, default)
        value = lambda name: self.get_parameter(name).value  # noqa: E731

        self._optical_frame = value("optical_frame")
        self._published = 0

        self._publisher = self.create_publisher(
            StampedFrame,
            value("topic"),
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )

        self._source = WebcamSource(
            device=value("device"),
            width=value("width"),
            height=value("height"),
            fps_hint=value("fps_hint"),
            fourcc=value("fourcc"),
            backend=value("backend"),
            buffer_frames=value("buffer_frames"),
        )
        self._source.open()
        # What the driver agreed to, not what was asked for. Gate 1 found a
        # camera silently delivering half its advertised rate, so this belongs
        # in the log of every run.
        self.get_logger().info(f"camera negotiated {self._source.negotiated}")
        self._source.start(self.on_frame)

    def on_frame(self, frame) -> None:
        """Called on the capture thread. Bounded work only; never blocks.

        The same contract `Pipeline.submit` has: stamp, hand off, return. A
        capture thread that blocked here would put the camera driver's own
        buffering back in charge of staleness, which is exactly what the
        admission policy exists to take over.
        """
        # rclpy's SIGINT handler invalidates the context before this node's
        # own shutdown runs, so a frame already in flight on the capture
        # thread would publish into a dead context and raise. Ctrl-C is a
        # normal exit, not a stack trace.
        if not rclpy.ok():
            return
        message = StampedFrame()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._optical_frame
        message.image = to_image_msg(frame.payload)
        message.frame_id = frame.frame_id
        message.capture_ts_monotonic = frame.capture_ts
        try:
            self._publisher.publish(message)
        except Exception:
            if rclpy.ok():
                raise
            return  # lost the race with shutdown
        self._published += 1

    def destroy_node(self) -> bool:
        self._source.stop()
        self.get_logger().info(f"published {self._published} frames")
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
