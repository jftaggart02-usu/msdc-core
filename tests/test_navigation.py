"""Unit tests for the navigation submodule."""

import math

import pytest

from msdc_core.navigation.navigator import PathPlanner, Waypoint


# ---------------------------------------------------------------------------
# Waypoint
# ---------------------------------------------------------------------------


class TestWaypoint:
    """Tests for :class:`Waypoint`."""

    def test_construction(self) -> None:
        wp = Waypoint(x=1.0, y=2.0, heading_rad=math.pi / 4)
        assert wp.x == 1.0
        assert wp.y == 2.0
        assert wp.heading_rad == pytest.approx(math.pi / 4)

    def test_default_heading(self) -> None:
        wp = Waypoint(x=0.0, y=0.0)
        assert wp.heading_rad == 0.0

    def test_equality(self) -> None:
        assert Waypoint(1.0, 2.0) == Waypoint(1.0, 2.0)

    def test_inequality(self) -> None:
        assert Waypoint(1.0, 2.0) != Waypoint(1.0, 3.0)


# ---------------------------------------------------------------------------
# PathPlanner
# ---------------------------------------------------------------------------


class TestPathPlanner:
    """Tests for :class:`PathPlanner`."""

    def _two_wp_planner(self) -> PathPlanner:
        return PathPlanner([Waypoint(0.0, 0.0), Waypoint(1.0, 0.0)])

    def test_current_waypoint_at_start(self) -> None:
        planner = self._two_wp_planner()
        assert planner.current_waypoint == Waypoint(0.0, 0.0)

    def test_advance_moves_to_next(self) -> None:
        planner = self._two_wp_planner()
        planner.advance()
        assert planner.current_waypoint == Waypoint(1.0, 0.0)

    def test_is_complete_after_all_waypoints(self) -> None:
        planner = self._two_wp_planner()
        planner.advance()
        planner.advance()
        assert planner.is_complete

    def test_remaining_count(self) -> None:
        planner = self._two_wp_planner()
        assert planner.remaining == 2
        planner.advance()
        assert planner.remaining == 1

    def test_advance_does_nothing_when_complete(self) -> None:
        planner = PathPlanner([Waypoint(0.0, 0.0)])
        planner.advance()
        assert planner.is_complete
        planner.advance()  # should not raise
        assert planner.is_complete

    def test_empty_path_is_complete(self) -> None:
        planner = PathPlanner()
        assert planner.is_complete

    def test_current_waypoint_raises_when_complete(self) -> None:
        planner = PathPlanner()
        with pytest.raises(IndexError):
            _ = planner.current_waypoint

    def test_add_waypoint(self) -> None:
        planner = PathPlanner()
        assert planner.is_complete
        planner.add_waypoint(Waypoint(5.0, 5.0))
        assert not planner.is_complete

    def test_distance_to_current(self) -> None:
        planner = PathPlanner([Waypoint(3.0, 4.0)])
        # distance from origin to (3, 4) = 5
        assert planner.distance_to_current(0.0, 0.0) == pytest.approx(5.0)

    def test_has_arrived_true(self) -> None:
        planner = PathPlanner([Waypoint(0.0, 0.0)], arrival_threshold_m=0.5)
        assert planner.has_arrived(0.1, 0.1)

    def test_has_arrived_false(self) -> None:
        planner = PathPlanner([Waypoint(0.0, 0.0)], arrival_threshold_m=0.1)
        assert not planner.has_arrived(5.0, 5.0)

    def test_has_arrived_when_complete_returns_false(self) -> None:
        planner = PathPlanner()
        assert not planner.has_arrived(0.0, 0.0)

    def test_reset_restarts_index(self) -> None:
        planner = self._two_wp_planner()
        planner.advance()
        planner.reset()
        assert planner.current_waypoint == Waypoint(0.0, 0.0)

    def test_clear_empties_path(self) -> None:
        planner = self._two_wp_planner()
        planner.clear()
        assert planner.is_complete
        assert len(planner) == 0

    def test_len(self) -> None:
        planner = self._two_wp_planner()
        assert len(planner) == 2

    def test_iter(self) -> None:
        waypoints = [Waypoint(0.0, 0.0), Waypoint(1.0, 1.0)]
        planner = PathPlanner(waypoints)
        assert list(planner) == waypoints

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            PathPlanner(arrival_threshold_m=0.0)
