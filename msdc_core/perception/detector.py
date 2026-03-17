"""Sensor data container and object detector implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SensorReading:
    """A snapshot of raw sensor data captured at a single point in time.

    Parameters
    ----------
    timestamp_s:
        Timestamp in seconds (e.g. from ``time.monotonic()``).
    lidar_distances_m:
        List of range measurements (in metres) from a 2-D LiDAR scan.
        Each entry corresponds to a scan point at a regularly-spaced angle.
    camera_frame:
        Raw image data as a flat list of ``uint8`` pixel values (e.g. RGB or
        greyscale).  An empty list indicates no frame was captured.

    Examples
    --------
    >>> reading = SensorReading(timestamp_s=0.0, lidar_distances_m=[1.5, 2.0])
    >>> reading.timestamp_s
    0.0
    """

    timestamp_s: float
    lidar_distances_m: list[float] = field(default_factory=list)
    camera_frame: list[int] = field(default_factory=list)


@dataclass
class DetectedObject:
    """A single object detected from a :class:`SensorReading`.

    Parameters
    ----------
    label:
        Human-readable category label (e.g. ``"obstacle"``, ``"lane_marker"``).
    distance_m:
        Estimated distance to the object in metres.
    angle_rad:
        Bearing to the object in radians, measured counter-clockwise from the
        vehicle's forward direction.
    confidence:
        Detection confidence score in ``[0.0, 1.0]``.
    """

    label: str
    distance_m: float
    angle_rad: float
    confidence: float = 1.0


class ObjectDetector:
    """Processes :class:`SensorReading` instances to produce detected objects.

    This class provides a simple threshold-based obstacle detection algorithm
    suitable for a LiDAR-equipped miniature vehicle.  The detector flags any
    scan point whose reported range is below a configurable threshold as an
    obstacle.

    Parameters
    ----------
    obstacle_threshold_m:
        LiDAR range (in metres) below which a scan point is classified as an
        obstacle.
    fov_deg:
        Total field-of-view of the LiDAR sensor in degrees.  The scan points
        in ``lidar_distances_m`` are assumed to be evenly distributed across
        this arc, centred on the vehicle's forward direction.

    Examples
    --------
    >>> import math
    >>> detector = ObjectDetector(obstacle_threshold_m=1.0, fov_deg=180.0)
    >>> reading = SensorReading(timestamp_s=0.0, lidar_distances_m=[0.5, 3.0])
    >>> objects = detector.detect(reading)
    >>> len(objects)
    1
    >>> objects[0].label
    'obstacle'
    """

    def __init__(
        self,
        obstacle_threshold_m: float = 1.0,
        fov_deg: float = 180.0,
    ) -> None:
        if obstacle_threshold_m <= 0:
            raise ValueError("obstacle_threshold_m must be positive")
        if fov_deg <= 0 or fov_deg > 360:
            raise ValueError("fov_deg must be in (0, 360]")
        self._obstacle_threshold_m: float = obstacle_threshold_m
        self._fov_deg: float = fov_deg

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, reading: SensorReading) -> list[DetectedObject]:
        """Run obstacle detection on a single sensor reading.

        Parameters
        ----------
        reading:
            The :class:`SensorReading` to process.

        Returns
        -------
        list[DetectedObject]
            Detected obstacles, ordered by ascending distance.
        """
        distances = reading.lidar_distances_m
        if not distances:
            return []

        num_points = len(distances)
        half_fov_rad = math.radians(self._fov_deg / 2.0)
        step_rad = math.radians(self._fov_deg) / max(num_points - 1, 1)
        detected: list[DetectedObject] = []
        for i, dist in enumerate(distances):
            if dist < self._obstacle_threshold_m:
                angle = -half_fov_rad + i * step_rad
                confidence = max(0.0, 1.0 - dist / self._obstacle_threshold_m)
                detected.append(
                    DetectedObject(
                        label="obstacle",
                        distance_m=dist,
                        angle_rad=angle,
                        confidence=confidence,
                    )
                )

        detected.sort(key=lambda obj: obj.distance_m)
        return detected
