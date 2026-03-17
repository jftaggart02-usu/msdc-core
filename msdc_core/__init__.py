"""msdc_core – Core functionality for the Mini Self Driving Car (MSDC).

Submodules
----------
control
    Motor and steering control primitives.
navigation
    Path planning and waypoint navigation.
perception
    Sensor data abstractions and object detection.
"""

from msdc_core import control, navigation, perception

__all__ = ["control", "navigation", "perception"]
__version__ = "0.1.0"
