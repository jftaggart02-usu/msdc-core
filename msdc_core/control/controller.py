"""Motor and steering controller implementations."""

from __future__ import annotations

import math


class MotorController:
    """Controls the drive motors of the MSDC platform.

    The controller maps a normalised throttle value in ``[-1.0, 1.0]`` to a
    PWM duty-cycle percentage in ``[0, 100]``.  Negative throttle represents
    reverse; zero represents a full stop.

    Parameters
    ----------
    max_speed_mps:
        Maximum forward speed in metres per second.  Used to convert throttle
        fractions to physical speed estimates.

    Examples
    --------
    >>> mc = MotorController(max_speed_mps=2.0)
    >>> mc.set_throttle(0.5)
    >>> mc.current_speed_mps
    1.0
    """

    def __init__(self, max_speed_mps: float = 1.0) -> None:
        if max_speed_mps <= 0:
            raise ValueError("max_speed_mps must be positive")
        self._max_speed_mps: float = max_speed_mps
        self._throttle: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def throttle(self) -> float:
        """Current normalised throttle in ``[-1.0, 1.0]``."""
        return self._throttle

    @property
    def current_speed_mps(self) -> float:
        """Estimated current speed in metres per second (signed)."""
        return self._throttle * self._max_speed_mps

    @property
    def duty_cycle_pct(self) -> float:
        """PWM duty-cycle percentage ``[0, 100]`` derived from throttle magnitude."""
        return abs(self._throttle) * 100.0

    # ------------------------------------------------------------------
    # Control methods
    # ------------------------------------------------------------------

    def set_throttle(self, throttle: float) -> None:
        """Set the motor throttle.

        Parameters
        ----------
        throttle:
            Normalised throttle value clamped to ``[-1.0, 1.0]``.
            Positive values drive forward; negative values drive in reverse.
        """
        self._throttle = max(-1.0, min(1.0, throttle))

    def stop(self) -> None:
        """Immediately stop the motors (set throttle to 0)."""
        self._throttle = 0.0


class SteeringController:
    """Controls the front-wheel steering of the MSDC platform.

    The steering angle is specified in radians and clamped to
    ``[-max_angle_rad, +max_angle_rad]``.  Positive angles steer to the left
    (counter-clockwise when viewed from above), negative angles steer right.

    Parameters
    ----------
    max_angle_rad:
        Maximum absolute steering angle in radians.  Defaults to 30°.

    Examples
    --------
    >>> import math
    >>> sc = SteeringController(max_angle_rad=math.radians(30))
    >>> sc.set_angle(math.radians(15))
    >>> round(math.degrees(sc.angle_rad), 1)
    15.0
    """

    def __init__(self, max_angle_rad: float = math.radians(30)) -> None:
        if max_angle_rad <= 0:
            raise ValueError("max_angle_rad must be positive")
        self._max_angle_rad: float = max_angle_rad
        self._angle_rad: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def angle_rad(self) -> float:
        """Current steering angle in radians."""
        return self._angle_rad

    @property
    def angle_deg(self) -> float:
        """Current steering angle in degrees."""
        return math.degrees(self._angle_rad)

    # ------------------------------------------------------------------
    # Control methods
    # ------------------------------------------------------------------

    def set_angle(self, angle_rad: float) -> None:
        """Set the steering angle.

        Parameters
        ----------
        angle_rad:
            Desired steering angle in radians, clamped to the allowed range.
        """
        self._angle_rad = max(-self._max_angle_rad, min(self._max_angle_rad, angle_rad))

    def center(self) -> None:
        """Return the steering to the straight-ahead (zero-angle) position."""
        self._angle_rad = 0.0
