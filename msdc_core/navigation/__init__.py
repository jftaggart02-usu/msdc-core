"""navigation – Waypoint path planning for the MSDC platform.

Public API
----------
Waypoint
    A single (x, y, heading) navigation target.
PathPlanner
    Computes and manages an ordered list of :class:`Waypoint` objects.
"""

from msdc_core.navigation.navigator import PathPlanner, Waypoint

__all__ = ["PathPlanner", "Waypoint"]
