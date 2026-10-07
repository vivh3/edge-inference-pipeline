# One captured frame and the metadata the pipeline trusts.

# Wall clock, for ROS tooling that expects it. Nothing in this project
# subtracts it: see capture_ts_monotonic.
std_msgs/Header header

# The frame itself, bgr8.
sensor_msgs/Image image

# Assigned at capture by trusted code, never by the model. Identifies which
# capture a published result belongs to. ROS 2 removed Header.seq, so there is
# no standard field for this.
uint64 frame_id

# CLOCK_MONOTONIC seconds, from inference.clock.monotonic(). Every duration in
# this project is a difference of two of these, because wall clock can step
# backwards mid-inference and make a latency negative.
float64 capture_ts_monotonic
