"""Waypoint data class and path planner implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Waypoint:
    """A single navigation target in 2-D space.

    Parameters
    ----------
    x:
        X-coordinate in metres (positive = east).
    y:
        Y-coordinate in metres (positive = north).
    heading_rad:
        Desired heading at the waypoint in radians, measured counter-clockwise
        from the positive X-axis.  Defaults to ``0.0``.

    Examples
    --------
    >>> wp = Waypoint(x=1.0, y=2.0, heading_rad=0.0)
    >>> wp.x
    1.0
    """

    x: float
    y: float
    heading_rad: float = 0.0


class PathPlanner:
    """Manages an ordered sequence of :class:`Waypoint` objects.

    The planner tracks a *current* waypoint index.  Callers advance through
    the path by calling :meth:`advance` once the vehicle is sufficiently close
    to the current target.

    Parameters
    ----------
    waypoints:
        Initial list of waypoints.  May be empty; waypoints can be appended
        via :meth:`add_waypoint`.
    arrival_threshold_m:
        Distance in metres below which the vehicle is considered to have
        *arrived* at the current waypoint.

    Examples
    --------
    >>> planner = PathPlanner([Waypoint(0.0, 0.0), Waypoint(1.0, 0.0)])
    >>> planner.current_waypoint
    Waypoint(x=0.0, y=0.0, heading_rad=0.0)
    >>> planner.advance()
    >>> planner.current_waypoint
    Waypoint(x=1.0, y=0.0, heading_rad=0.0)
    """

    def __init__(
        self,
        waypoints: list[Waypoint] | None = None,
        arrival_threshold_m: float = 0.1,
    ) -> None:
        if arrival_threshold_m <= 0:
            raise ValueError("arrival_threshold_m must be positive")
        self._waypoints: list[Waypoint] = list(waypoints) if waypoints else []
        self._arrival_threshold_m: float = arrival_threshold_m
        self._index: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """``True`` when all waypoints have been visited."""
        return self._index >= len(self._waypoints)

    @property
    def current_waypoint(self) -> Waypoint:
        """The waypoint currently being targeted.

        Raises
        ------
        IndexError
            If the path is empty or all waypoints have been visited.
        """
        if self.is_complete:
            raise IndexError("No remaining waypoints in path")
        return self._waypoints[self._index]

    @property
    def remaining(self) -> int:
        """Number of waypoints not yet visited (including the current one)."""
        return max(0, len(self._waypoints) - self._index)

    # ------------------------------------------------------------------
    # Path management
    # ------------------------------------------------------------------

    def add_waypoint(self, waypoint: Waypoint) -> None:
        """Append *waypoint* to the end of the path.

        Parameters
        ----------
        waypoint:
            The :class:`Waypoint` to append.
        """
        self._waypoints.append(waypoint)

    def reset(self) -> None:
        """Reset the planner to the first waypoint without clearing the path."""
        self._index = 0

    def clear(self) -> None:
        """Remove all waypoints and reset the index."""
        self._waypoints.clear()
        self._index = 0

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def distance_to_current(self, x: float, y: float) -> float:
        """Return the Euclidean distance from *(x, y)* to the current waypoint.

        Parameters
        ----------
        x:
            Current vehicle X-position in metres.
        y:
            Current vehicle Y-position in metres.

        Raises
        ------
        IndexError
            If there is no current waypoint.
        """
        wp = self.current_waypoint
        return math.hypot(wp.x - x, wp.y - y)

    def has_arrived(self, x: float, y: float) -> bool:
        """Return ``True`` if the vehicle is within the arrival threshold.

        Parameters
        ----------
        x:
            Current vehicle X-position in metres.
        y:
            Current vehicle Y-position in metres.
        """
        if self.is_complete:
            return False
        return self.distance_to_current(x, y) <= self._arrival_threshold_m

    def advance(self) -> None:
        """Mark the current waypoint as visited and move to the next one.

        Has no effect if the path is already complete.
        """
        if not self.is_complete:
            self._index += 1

    def __iter__(self) -> Iterator[Waypoint]:
        """Iterate over *all* waypoints (visited and unvisited)."""
        return iter(self._waypoints)

    def __len__(self) -> int:
        return len(self._waypoints)
