"""Hardware-independent core of the edge perception pipeline.

Nothing in this package imports ROS, CUDA, or a camera driver at module
scope.  That is deliberate: the admission policy, the output contract, the
failure taxonomy, and the telemetry are the parts of this project that carry
the argument, and they are testable on a laptop with no accelerator.  ROS 2
is integration plumbing wrapped around this core, not a dependency of it.
"""

from .clock import monotonic, wall_clock_iso  # noqa: F401
from .record import Frame, PublishedResult, RawModelOutput  # noqa: F401
from .schema import Failure, validate  # noqa: F401
