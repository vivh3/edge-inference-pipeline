"""consumer_node: act on the contract, at a rate the producer cannot match.

A demonstration consumer. It turns published records into an advisory state and
publishes that. **Nothing here is a safety function** -- no actuation, no
interlock, no claim that `proceed` means anything is safe. It exists to make
the output contract concrete: a consumer has to be written against it, and
writing one is what shows whether the contract carries enough to decide with.

**The consumer runs at its own rate, and that is the point.** Results arrive
every ~6.4 s; this decides at 10 Hz. So on 98% of ticks there is no new
information and the only honest question is how old the newest record is. That
mirrors the admission policy at the other end of the pipeline: upstream, the
newest frame wins and stale frames are dropped; downstream, the newest result
is reused until it ages out, and then the state falls back rather than
pretending.

A result therefore goes stale without any new input arriving, and the advisory
flips to `hold` on a timer. A consumer that only re-evaluated on arrival would
hold a six-second-old view of the world as current for as long as the engine
stayed quiet.

**Staleness is checked before semantics, deliberately.** Age comes from
`capture_ts`, which the capture node stamped and the pipeline never let the
model touch. Deciding on it requires trusting no model output at all, so it is
the check that still works when everything downstream of the camera is
suspect.

Three states, and the two that are not `proceed` are the useful ones:

    proceed   fresh, valid, path_status clear with no obstacle
    hold      stale, or unusable semantics, or an obstacle reported
    no_data   nothing received yet, or the last record aged past the limit

`no_data` is distinct from `hold` because they have different causes: `hold`
means the system told us something, and `no_data` means it has stopped telling
us anything.
"""

from __future__ import annotations

import json
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from inference.clock import monotonic, wall_clock_iso

PARAMETERS = {
    "results_topic": "perception",
    "advisory_topic": "advisory",
    # Deliberately faster than the producer, which is the situation worth
    # demonstrating. 10 Hz against a result every 6.4 s.
    "decide_hz": 10.0,
    # A result older than this is not usable as a current view. Set from the
    # measured result age (6.46 s p50) plus one service time, so a single
    # slow inference does not flip the state while the pipeline is healthy.
    "stale_after_s": 13.0,
    # Past this, the pipeline is not merely slow, it has stopped.
    "no_data_after_s": 30.0,
    "report_every_s": 30.0,
}

RESULTS_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

PROCEED, HOLD, NO_DATA = "proceed", "hold", "no_data"
DECISIONS = (PROCEED, HOLD, NO_DATA)


def decide(
    record: Optional[dict], now: float, stale_after_s: float, no_data_after_s: float
) -> Tuple[str, str]:
    """The whole decision, as a function of the latest record and the clock.

    Returns `(decision, reason)`. No ROS, no state, so the table of cases is
    testable directly.
    """
    if record is None:
        return NO_DATA, "no result received yet"

    # Trusted metadata first: this holds whatever the model emitted.
    age = now - float(record["capture_ts"])
    if age > no_data_after_s:
        return NO_DATA, f"last result {age:.1f}s old, past the {no_data_after_s:.0f}s limit"
    if age > stale_after_s:
        return HOLD, f"last result {age:.1f}s old, not a current view"

    validation = record.get("validation") or {}
    if not validation.get("ok", False):
        failure = validation.get("failure") or "unknown"
        return HOLD, f"semantics unusable ({failure})"

    semantic = record.get("semantic") or {}
    status = str(semantic.get("path_status", ""))
    where = str(semantic.get("obstacle_location", ""))
    if status == "clear" and where == "none":
        return PROCEED, "path reported clear"
    if status == "clear":
        # The validator should have caught this; a consumer that assumes its
        # upstream is correct is a consumer that fails when it is not.
        return HOLD, f"contradictory: clear with obstacle_location {where}"
    return HOLD, f"path {status or 'unreported'}, obstacle {where or 'unreported'}"


class ConsumerNode(Node):
    def __init__(self) -> None:
        super().__init__("consumer_node")
        for name, default in PARAMETERS.items():
            self.declare_parameter(name, default)
        value = lambda name: self.get_parameter(name).value  # noqa: E731

        self.stale_after_s = value("stale_after_s")
        self.no_data_after_s = value("no_data_after_s")
        self._latest: Optional[dict] = None
        self._decision: Optional[str] = None
        self.unparseable = 0
        # Time spent in each state, which is what the overload costs a
        # consumer. A count of decisions would just measure the tick rate.
        self.time_in: dict = {name: 0.0 for name in DECISIONS}
        self._last_tick = monotonic()

        self._advisory = self.create_publisher(String, value("advisory_topic"), 10)
        self.create_subscription(
            String, value("results_topic"), self.on_result, RESULTS_QOS
        )
        self.create_timer(1.0 / value("decide_hz"), self.tick)
        self.create_timer(value("report_every_s"), self.report)

    def on_result(self, message: String) -> None:
        try:
            self._latest = json.loads(message.data)
        except ValueError:
            # Keep the previous record: it will age out on its own, which is a
            # better failure than discarding what we have for a bad frame.
            self.unparseable += 1
            self.get_logger().warn("unparseable record; keeping the previous one")

    def tick(self) -> None:
        now = monotonic()
        decision, reason = decide(
            self._latest, now, self.stale_after_s, self.no_data_after_s
        )

        self.time_in[decision] += now - self._last_tick
        self._last_tick = now

        self._advisory.publish(
            String(
                data=json.dumps(
                    {
                        "decision": decision,
                        "reason": reason,
                        "wall_clock": wall_clock_iso(),
                        "based_on_frame_id": (
                            None if self._latest is None else self._latest.get("frame_id")
                        ),
                        "record_age_s": (
                            None
                            if self._latest is None
                            else round(now - float(self._latest["capture_ts"]), 3)
                        ),
                    }
                )
            )
        )

        # Logged on change only. At 10 Hz, logging every decision would bury
        # the transitions that matter in 600 lines a minute.
        if decision != self._decision:
            self.get_logger().info(f"{self._decision or 'start'} -> {decision}: {reason}")
            self._decision = decision

    def report(self) -> None:
        total = sum(self.time_in.values()) or 1.0
        shares = "  ".join(
            f"{name} {100.0 * self.time_in[name] / total:.1f}%" for name in DECISIONS
        )
        self.get_logger().info(f"time in state over {total:.0f}s:  {shares}")

    def destroy_node(self) -> bool:
        self.report()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConsumerNode()
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
