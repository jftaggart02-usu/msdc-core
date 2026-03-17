"""Unit tests for the control submodule."""

import math

import pytest

from msdc_core.control.controller import MotorController, SteeringController


# ---------------------------------------------------------------------------
# MotorController
# ---------------------------------------------------------------------------


class TestMotorController:
    """Tests for :class:`MotorController`."""

    def test_default_throttle_is_zero(self) -> None:
        mc = MotorController()
        assert mc.throttle == 0.0

    def test_set_throttle_forward(self) -> None:
        mc = MotorController(max_speed_mps=2.0)
        mc.set_throttle(0.5)
        assert mc.throttle == 0.5

    def test_set_throttle_reverse(self) -> None:
        mc = MotorController()
        mc.set_throttle(-0.8)
        assert mc.throttle == -0.8

    def test_throttle_clamped_above_one(self) -> None:
        mc = MotorController()
        mc.set_throttle(1.5)
        assert mc.throttle == 1.0

    def test_throttle_clamped_below_minus_one(self) -> None:
        mc = MotorController()
        mc.set_throttle(-2.0)
        assert mc.throttle == -1.0

    def test_current_speed_mps(self) -> None:
        mc = MotorController(max_speed_mps=4.0)
        mc.set_throttle(0.25)
        assert mc.current_speed_mps == pytest.approx(1.0)

    def test_duty_cycle_pct_positive(self) -> None:
        mc = MotorController()
        mc.set_throttle(0.75)
        assert mc.duty_cycle_pct == pytest.approx(75.0)

    def test_duty_cycle_pct_negative_throttle(self) -> None:
        mc = MotorController()
        mc.set_throttle(-0.5)
        assert mc.duty_cycle_pct == pytest.approx(50.0)

    def test_stop_resets_throttle(self) -> None:
        mc = MotorController()
        mc.set_throttle(0.9)
        mc.stop()
        assert mc.throttle == 0.0

    def test_invalid_max_speed_raises(self) -> None:
        with pytest.raises(ValueError):
            MotorController(max_speed_mps=0.0)

    def test_invalid_negative_max_speed_raises(self) -> None:
        with pytest.raises(ValueError):
            MotorController(max_speed_mps=-1.0)


# ---------------------------------------------------------------------------
# SteeringController
# ---------------------------------------------------------------------------


class TestSteeringController:
    """Tests for :class:`SteeringController`."""

    def test_default_angle_is_zero(self) -> None:
        sc = SteeringController()
        assert sc.angle_rad == 0.0

    def test_set_angle_within_limits(self) -> None:
        sc = SteeringController(max_angle_rad=math.radians(30))
        sc.set_angle(math.radians(15))
        assert sc.angle_rad == pytest.approx(math.radians(15))

    def test_angle_clamped_above_max(self) -> None:
        max_rad = math.radians(30)
        sc = SteeringController(max_angle_rad=max_rad)
        sc.set_angle(math.radians(45))
        assert sc.angle_rad == pytest.approx(max_rad)

    def test_angle_clamped_below_negative_max(self) -> None:
        max_rad = math.radians(30)
        sc = SteeringController(max_angle_rad=max_rad)
        sc.set_angle(-math.radians(45))
        assert sc.angle_rad == pytest.approx(-max_rad)

    def test_angle_deg_conversion(self) -> None:
        sc = SteeringController(max_angle_rad=math.radians(30))
        sc.set_angle(math.radians(20))
        assert sc.angle_deg == pytest.approx(20.0)

    def test_center_resets_angle(self) -> None:
        sc = SteeringController()
        sc.set_angle(math.radians(10))
        sc.center()
        assert sc.angle_rad == 0.0

    def test_invalid_max_angle_raises(self) -> None:
        with pytest.raises(ValueError):
            SteeringController(max_angle_rad=0.0)

    def test_invalid_negative_max_angle_raises(self) -> None:
        with pytest.raises(ValueError):
            SteeringController(max_angle_rad=-math.radians(10))
