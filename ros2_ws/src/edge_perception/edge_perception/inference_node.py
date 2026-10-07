"""inference_node: run the pipeline on subscribed frames, publish the contract.

A thin wrapper, like `capture_node`. The admission policy, the engine, the
validation and the telemetry all live in `inference/` and `telemetry/`
unchanged; this node converts messages and nothing else.

**Two losses, two names.** Frames can now disappear twice over. The middleware
drops them before this node ever sees them -- measured at about 28% with a
subscriber doing nothing, on a board that cannot deserialise 30 x 921,600
bytes per second -- and the admission policy drops them on purpose once they
arrive. Reporting one number would attribute the middleware's losses to a
design decision. Gaps in `frame_id` recover the first, and
`telemetry.Metrics` already counts the second.

**The subscriber QoS must match the publisher's BEST_EFFORT.** A RELIABLE
subscriber does not merely lose frames against a BEST_EFFORT publisher, it
never connects at all, and the node sits silently receiving nothing while
every process involved looks healthy.

**The callback stays bounded.** `Pipeline.submit` offers to the admission
policy and returns; inference happens on the pipeline's own worker thread.
Running the model in the subscriber callback would block the executor for
4.9 s per frame and put the middleware queue back in charge of staleness,
which is what the admission policy exists to take over.
"""

from __future__ import annotations

import json

import numpy
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from edge_perception_msgs.msg import StampedFrame
from inference.buffer import make_policy
from inference.engine import MockEngine, VlmEngine
from inference.pipeline import Pipeline
from inference.record import Frame
from telemetry.metrics import Metrics

PARAMETERS = {
    "frames_topic": "frames",
    "results_topic": "perception",
    "policy": "latest",
    "fifo_capacity": 8,
    "deadline_s": 15.0,
    # Mock by default so the graph can be exercised without a 4.9 s model in
    # the loop. Gate 2's end-to-end run sets this false.
    "use_mock_engine": True,
    "mock_latency_s": 0.4,
    "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "revision": "482adb537c021c86670beed01cd58990d01e72e4",
    "report_every_s": 10.0,
}

SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


def to_payload(image) -> numpy.ndarray:
    """sensor_msgs/Image -> the BGR array the preprocessor expects."""
    flat = numpy.frombuffer(image.data, dtype=numpy.uint8)
    return flat.reshape(image.height, image.width, 3)


class InferenceNode(Node):
    def __init__(self) -> None:
        super().__init__("inference_node")
        for name, default in PARAMETERS.items():
            self.declare_parameter(name, default)
        value = lambda name: self.get_parameter(name).value  # noqa: E731

        # Middleware loss, which the admission policy cannot see because the
        # frames never reach it. Recovered from gaps in the trusted sequence.
        self.middleware_lost = 0
        self.received = 0
        self._last_frame_id = None

        self._results = self.create_publisher(String, value("results_topic"), 10)
        self.metrics = Metrics(policy_name=value("policy"))

        if value("use_mock_engine"):
            engine = MockEngine(mean_latency=value("mock_latency_s"), seed=0)
            self.get_logger().warn("MOCK engine: these are not model measurements")
        else:
            engine = VlmEngine(value("model_id"), revision=value("revision"))

        self.pipeline = Pipeline(
            engine=engine,
            policy=make_policy(value("policy"), value("fifo_capacity")),
            metrics=self.metrics,
            consumer=self.publish_result,
            deadline_s=value("deadline_s"),
        )
        self.pipeline.start()

        self.create_subscription(
            StampedFrame, value("frames_topic"), self.on_frame, SENSOR_QOS
        )
        self.create_timer(value("report_every_s"), self.report)

    def on_frame(self, message: StampedFrame) -> None:
        """Bounded work only. Inference runs on the pipeline's worker thread."""
        if self._last_frame_id is not None:
            # Every frame_id the publisher assigned but we never received.
            self.middleware_lost += message.frame_id - self._last_frame_id - 1
        self._last_frame_id = message.frame_id
        self.received += 1

        self.pipeline.submit(
            Frame(
                frame_id=message.frame_id,
                # The publisher's monotonic reading, not ours. Both processes
                # share CLOCK_MONOTONIC, so result age spans the whole path
                # from acquisition rather than from arrival here.
                capture_ts=message.capture_ts_monotonic,
                payload=to_payload(message.image),
                width=message.image.width,
                height=message.image.height,
            )
        )

    def publish_result(self, result) -> None:
        """The pipeline's consumer. Publishes the record the contract defines."""
        self._results.publish(String(data=json.dumps(result.to_dict())))

    def report(self) -> None:
        offered = self.received + self.middleware_lost
        if not offered:
            self.get_logger().warn("no frames yet -- is the subscriber QoS BEST_EFFORT?")
            return
        summary = self.metrics.summarize()
        health = self.pipeline.health_snapshot()
        self.get_logger().info(
            f"health={health['health']} "
            f"published={summary.results_published} "
            f"policy_dropped={summary.frames_dropped} "
            f"({100.0 * summary.drop_rate:.1f}% of frames that arrived) "
            f"middleware_lost={self.middleware_lost}/{offered} "
            f"({100.0 * self.middleware_lost / offered:.1f}% before the policy saw them)"
        )

    def destroy_node(self) -> bool:
        self.pipeline.stop()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InferenceNode()
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
