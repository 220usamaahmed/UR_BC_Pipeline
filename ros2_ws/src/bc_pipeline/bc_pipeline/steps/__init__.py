"""
Step package — registry, base class, and factory.

Importing the concrete step modules here runs their @register decorators so the
factory (build_step) knows about every type.  Add new step modules to the import
list below.
"""

from .base import STEP_REGISTRY, Step, StepConfigError, build_step, register

# Importing each module populates STEP_REGISTRY via the @register decorator.
from . import checkpoint        # noqa: F401  (Checkpoint)
from . import orientation_lock  # noqa: F401  (OrientationLockCheckpoint)
from . import wait              # noqa: F401  (Wait)
from . import gripper           # noqa: F401  (Gripper)

__all__ = [
    'STEP_REGISTRY',
    'Step',
    'StepConfigError',
    'build_step',
    'register',
]
