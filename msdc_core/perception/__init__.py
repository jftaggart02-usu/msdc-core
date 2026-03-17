"""perception – Sensor data and object detection for the MSDC platform.

Public API
----------
SensorReading
    Container for a single sensor snapshot.
ObjectDetector
    Processes :class:`SensorReading` instances and returns detected objects.
"""

from msdc_core.perception.detector import ObjectDetector, SensorReading

__all__ = ["ObjectDetector", "SensorReading"]
