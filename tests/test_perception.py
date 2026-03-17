"""Unit tests for the perception submodule."""

import pytest

from msdc_core.perception.detector import DetectedObject, ObjectDetector, SensorReading


# ---------------------------------------------------------------------------
# SensorReading
# ---------------------------------------------------------------------------


class TestSensorReading:
    """Tests for :class:`SensorReading`."""

    def test_construction(self) -> None:
        reading = SensorReading(timestamp_s=1.0, lidar_distances_m=[1.5, 2.0])
        assert reading.timestamp_s == 1.0
        assert reading.lidar_distances_m == [1.5, 2.0]

    def test_defaults(self) -> None:
        reading = SensorReading(timestamp_s=0.0)
        assert reading.lidar_distances_m == []
        assert reading.camera_frame == []

    def test_camera_frame_stored(self) -> None:
        reading = SensorReading(timestamp_s=0.0, camera_frame=[255, 0, 128])
        assert reading.camera_frame == [255, 0, 128]


# ---------------------------------------------------------------------------
# DetectedObject
# ---------------------------------------------------------------------------


class TestDetectedObject:
    """Tests for :class:`DetectedObject`."""

    def test_construction(self) -> None:
        obj = DetectedObject(label="obstacle", distance_m=0.8, angle_rad=0.1)
        assert obj.label == "obstacle"
        assert obj.distance_m == 0.8
        assert obj.angle_rad == 0.1
        assert obj.confidence == 1.0

    def test_custom_confidence(self) -> None:
        obj = DetectedObject(label="cone", distance_m=1.0, angle_rad=0.0, confidence=0.9)
        assert obj.confidence == 0.9


# ---------------------------------------------------------------------------
# ObjectDetector
# ---------------------------------------------------------------------------


class TestObjectDetector:
    """Tests for :class:`ObjectDetector`."""

    def test_no_obstacles_when_all_far(self) -> None:
        detector = ObjectDetector(obstacle_threshold_m=1.0)
        reading = SensorReading(timestamp_s=0.0, lidar_distances_m=[2.0, 3.0, 4.0])
        assert detector.detect(reading) == []

    def test_detects_single_obstacle(self) -> None:
        detector = ObjectDetector(obstacle_threshold_m=1.0)
        reading = SensorReading(timestamp_s=0.0, lidar_distances_m=[0.5])
        objects = detector.detect(reading)
        assert len(objects) == 1
        assert objects[0].label == "obstacle"

    def test_obstacle_distance_correct(self) -> None:
        detector = ObjectDetector(obstacle_threshold_m=2.0)
        reading = SensorReading(timestamp_s=0.0, lidar_distances_m=[1.0, 5.0])
        objects = detector.detect(reading)
        assert len(objects) == 1
        assert objects[0].distance_m == pytest.approx(1.0)

    def test_multiple_obstacles_sorted_by_distance(self) -> None:
        detector = ObjectDetector(obstacle_threshold_m=2.0)
        reading = SensorReading(timestamp_s=0.0, lidar_distances_m=[1.8, 0.5, 1.2])
        objects = detector.detect(reading)
        distances = [o.distance_m for o in objects]
        assert distances == sorted(distances)

    def test_empty_lidar_returns_empty(self) -> None:
        detector = ObjectDetector(obstacle_threshold_m=1.0)
        reading = SensorReading(timestamp_s=0.0, lidar_distances_m=[])
        assert detector.detect(reading) == []

    def test_confidence_decreases_with_distance(self) -> None:
        detector = ObjectDetector(obstacle_threshold_m=1.0)
        reading = SensorReading(timestamp_s=0.0, lidar_distances_m=[0.2, 0.8])
        objects = detector.detect(reading)
        # Closer object should have higher confidence
        assert objects[0].confidence > objects[1].confidence

    def test_confidence_near_zero_for_object_at_threshold(self) -> None:
        threshold = 1.0
        detector = ObjectDetector(obstacle_threshold_m=threshold)
        # Just inside the threshold
        reading = SensorReading(timestamp_s=0.0, lidar_distances_m=[0.999])
        objects = detector.detect(reading)
        assert len(objects) == 1
        assert 0.0 <= objects[0].confidence <= 1.0

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            ObjectDetector(obstacle_threshold_m=0.0)

    def test_invalid_fov_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            ObjectDetector(fov_deg=0.0)

    def test_invalid_fov_over_360_raises(self) -> None:
        with pytest.raises(ValueError):
            ObjectDetector(fov_deg=361.0)
