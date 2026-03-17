# msdc-core

**Core Python library for the Mini Self-Driving Car (MSDC) senior design project.**

`msdc-core` provides reusable, well-typed building blocks for autonomous vehicle control,
navigation, and perception. It is designed to run on **Ubuntu 22.04** and integrates
naturally with **ROS 2** workspaces.

---

## Table of Contents

- [Features](#features)
- [Package Structure](#package-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Submodule Reference](#submodule-reference)
  - [control](#control)
  - [navigation](#navigation)
  - [perception](#perception)
- [Development](#development)
  - [Running Tests](#running-tests)
  - [Linting (Pylint)](#linting-pylint)
  - [Type Checking (Mypy)](#type-checking-mypy)
  - [All Checks at Once](#all-checks-at-once)
- [ROS 2 Integration](#ros-2-integration)
- [License](#license)

---

## Features

| Submodule | Highlights |
|-----------|-----------|
| **control** | `MotorController` – normalised throttle → PWM duty-cycle mapping<br>`SteeringController` – angle-clamped front-wheel steering |
| **navigation** | `Waypoint` – lightweight (x, y, heading) dataclass<br>`PathPlanner` – ordered waypoint queue with arrival detection |
| **perception** | `SensorReading` – container for LiDAR and camera snapshots<br>`ObjectDetector` – threshold-based obstacle detection from LiDAR scans |

- 100 % type-annotated (passes **Mypy strict** mode)
- Zero Pylint warnings
- Comprehensive unit-test suite (`pytest` + `pytest-cov`)
- Installable in **editable mode** (`pip install -e .`)

---

## Package Structure

```
msdc-core/
├── msdc_core/                    # Main package
│   ├── __init__.py
│   ├── control/
│   │   ├── __init__.py
│   │   └── controller.py         # MotorController, SteeringController
│   ├── navigation/
│   │   ├── __init__.py
│   │   └── navigator.py          # Waypoint, PathPlanner
│   └── perception/
│       ├── __init__.py
│       └── detector.py           # SensorReading, DetectedObject, ObjectDetector
├── tests/
│   ├── __init__.py
│   ├── test_control.py
│   ├── test_navigation.py
│   └── test_perception.py
├── .github/
│   └── copilot-instructions.md   # Build / test / lint instructions for Copilot
├── Makefile                       # Convenience targets
├── pyproject.toml                 # Package metadata + Pylint/Mypy/Pytest config
├── LICENSE
└── README.md
```

---

## Prerequisites

- **Ubuntu 22.04** (primary target)
- **Python 3.10+**
- `pip` ≥ 22 (comes with Ubuntu 22.04's `python3-pip`)

```bash
sudo apt update && sudo apt install -y python3 python3-pip make
```

---

## Installation

### Editable install (recommended for development)

```bash
# Clone the repository
git clone https://github.com/jftaggart02-usu/msdc-core.git
cd msdc-core

# Install the package + dev dependencies in editable mode
pip install -e ".[dev]"
# or
make install
```

The `[dev]` extra pulls in `pytest`, `pytest-cov`, `pylint`, and `mypy`.

### Production install (no dev tools)

```bash
pip install -e .
```

---

## Quick Start

```python
import math
from msdc_core.control import MotorController, SteeringController
from msdc_core.navigation import PathPlanner, Waypoint
from msdc_core.perception import ObjectDetector, SensorReading

# --- Control ---
motor = MotorController(max_speed_mps=2.0)
steering = SteeringController(max_angle_rad=math.radians(30))

motor.set_throttle(0.6)          # 60 % throttle forward
steering.set_angle(math.radians(10))  # 10° left turn

print(f"Speed: {motor.current_speed_mps:.2f} m/s")
print(f"Steering: {steering.angle_deg:.1f}°")

# --- Navigation ---
planner = PathPlanner(
    waypoints=[Waypoint(0.0, 0.0), Waypoint(1.5, 0.0), Waypoint(1.5, 1.5)],
    arrival_threshold_m=0.1,
)

# Simulate vehicle position
vehicle_x, vehicle_y = 1.45, 0.05
if planner.has_arrived(vehicle_x, vehicle_y):
    planner.advance()

print(f"Current waypoint: {planner.current_waypoint}")
print(f"Remaining: {planner.remaining}")

# --- Perception ---
detector = ObjectDetector(obstacle_threshold_m=1.0, fov_deg=180.0)
reading = SensorReading(
    timestamp_s=0.0,
    lidar_distances_m=[3.0, 0.7, 2.5, 0.4, 1.8],
)
obstacles = detector.detect(reading)
for obs in obstacles:
    print(f"Obstacle at {obs.distance_m:.2f} m, confidence={obs.confidence:.2f}")
```

---

## Submodule Reference

### control

```python
from msdc_core.control import MotorController, SteeringController
```

#### `MotorController(max_speed_mps: float = 1.0)`

| Member | Description |
|--------|-------------|
| `set_throttle(throttle)` | Set normalised throttle in `[-1.0, 1.0]` |
| `stop()` | Immediately set throttle to 0 |
| `throttle` | Current throttle value |
| `current_speed_mps` | Estimated speed = `throttle × max_speed_mps` |
| `duty_cycle_pct` | PWM duty cycle `[0, 100]` derived from throttle magnitude |

#### `SteeringController(max_angle_rad: float = radians(30))`

| Member | Description |
|--------|-------------|
| `set_angle(angle_rad)` | Set steering angle (clamped to `±max_angle_rad`) |
| `center()` | Return to straight-ahead (angle = 0) |
| `angle_rad` | Current angle in radians |
| `angle_deg` | Current angle in degrees |

---

### navigation

```python
from msdc_core.navigation import PathPlanner, Waypoint
```

#### `Waypoint(x, y, heading_rad=0.0)`

Plain dataclass holding a 2-D pose target.

#### `PathPlanner(waypoints=None, arrival_threshold_m=0.1)`

| Member | Description |
|--------|-------------|
| `add_waypoint(wp)` | Append a waypoint to the path |
| `advance()` | Mark current waypoint visited; move to next |
| `reset()` | Restart from the first waypoint |
| `clear()` | Remove all waypoints |
| `distance_to_current(x, y)` | Euclidean distance to current target |
| `has_arrived(x, y)` | `True` if within arrival threshold |
| `current_waypoint` | The waypoint currently being targeted |
| `is_complete` | `True` when all waypoints visited |
| `remaining` | Number of unvisited waypoints |

---

### perception

```python
from msdc_core.perception import ObjectDetector, SensorReading
```

#### `SensorReading(timestamp_s, lidar_distances_m=[], camera_frame=[])`

Dataclass holding a timestamped sensor snapshot.

#### `ObjectDetector(obstacle_threshold_m=1.0, fov_deg=180.0)`

| Member | Description |
|--------|-------------|
| `detect(reading)` | Returns `list[DetectedObject]` sorted by ascending distance |

Each `DetectedObject` has `label`, `distance_m`, `angle_rad`, and `confidence` fields.

---

## Development

All dev commands are available via **`make`**:

| Target | Command | Description |
|--------|---------|-------------|
| `install` | `make install` | Editable install with dev extras |
| `test` | `make test` | Run pytest with coverage |
| `lint` | `make lint` | Run Pylint on package and tests |
| `type-check` | `make type-check` | Run Mypy in strict mode |
| `check` | `make check` | Run lint + type-check + test |
| `clean` | `make clean` | Remove build artefacts |

### Running Tests

```bash
make test
# equivalent:
pytest tests/ -v --tb=short --cov=msdc_core --cov-report=term-missing
```

### Linting (Pylint)

```bash
make lint
# equivalent:
pylint msdc_core tests
```

Configuration is in `pyproject.toml` under `[tool.pylint.*]`.

### Type Checking (Mypy)

```bash
make type-check
# equivalent:
mypy msdc_core tests
```

Mypy runs in **strict** mode. All public symbols require complete annotations.

### All Checks at Once

```bash
make check   # lint → type-check → test
```

---

## ROS 2 Integration

`msdc-core` is designed to be used from within a ROS 2 (Humble / Iron) workspace on
Ubuntu 22.04.

**Option A – install into the active ROS 2 Python environment:**

```bash
source /opt/ros/humble/setup.bash
pip install -e /path/to/msdc-core
```

**Option B – declare as a dependency in your ROS 2 `package.xml`:**

```xml
<exec_depend>python3-msdc-core</exec_depend>
```

**Minimal ROS 2 node example:**

```python
import rclpy
from rclpy.node import Node
from msdc_core.control import MotorController

class DriveNode(Node):
    def __init__(self) -> None:
        super().__init__("drive_node")
        self._motor = MotorController(max_speed_mps=1.5)

def main() -> None:
    rclpy.init()
    rclpy.spin(DriveNode())
    rclpy.shutdown()
```

---

## License

See [LICENSE](LICENSE) for details.
