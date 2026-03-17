"""control – Motor and steering control for the MSDC platform.

Public API
----------
MotorController
    Controls the drive motors (speed / direction).
SteeringController
    Controls the front-wheel steering angle.
"""

from msdc_core.control.controller import MotorController, SteeringController

__all__ = ["MotorController", "SteeringController"]
