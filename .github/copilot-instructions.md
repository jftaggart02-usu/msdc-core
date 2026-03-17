# msdc-core – Copilot Instructions

## Overview
`msdc-core` is a Python package that provides core functionality (control, navigation, perception)
for the Mini Self-Driving Car (MSDC) senior design project. The package targets **Ubuntu 22.04**,
is written in **Python 3.10+**, and is designed to be consumed by a **ROS 2** workspace.

---

## Prerequisites

```bash
# Ubuntu 22.04 – ensure Python 3.10+ and pip are available
sudo apt update && sudo apt install -y python3 python3-pip make
```

---

## Bootstrap: Install in editable mode

```bash
# From the repository root
pip install -e ".[dev]"
# or via Makefile
make install
```

This installs the `msdc_core` package and all dev dependencies
(`pytest`, `pytest-cov`, `pylint`, `mypy`).

---

## Running Tests

```bash
# Run all unit tests (with coverage)
make test

# …or directly with pytest
pytest tests/ -v --tb=short --cov=msdc_core --cov-report=term-missing
```

Tests live in `tests/` and are organised by submodule:

| File | Submodule |
|------|-----------|
| `tests/test_control.py` | `msdc_core.control` |
| `tests/test_navigation.py` | `msdc_core.navigation` |
| `tests/test_perception.py` | `msdc_core.perception` |

---

## Running Linting (Pylint)

```bash
make lint
# …or directly
pylint msdc_core tests
```

Pylint is configured in `pyproject.toml` under `[tool.pylint.*]`.
The target is **zero warnings/errors**. Disable specific messages in
`pyproject.toml` rather than inline `# pylint: disable` comments.

---

## Running Type Checking (Mypy)

```bash
make type-check
# …or directly
mypy msdc_core tests
```

Mypy runs in **strict** mode. All public functions and methods must carry
complete type annotations. Configuration lives under `[tool.mypy]` in
`pyproject.toml`.

---

## Running All Checks

```bash
make check   # runs: lint → type-check → test
```

---

## Cleaning Build Artefacts

```bash
make clean
```

---

## Package Structure

```
msdc_core/
├── __init__.py
├── control/
│   ├── __init__.py
│   └── controller.py      # MotorController, SteeringController
├── navigation/
│   ├── __init__.py
│   └── navigator.py       # Waypoint, PathPlanner
└── perception/
    ├── __init__.py
    └── detector.py        # SensorReading, DetectedObject, ObjectDetector
```

---

## Code Style Guidelines

* Max line length: **100 characters** (Pylint enforced).
* All public symbols must have **Google-style docstrings**.
* Use **`from __future__ import annotations`** at the top of every module to
  enable PEP 563 postponed evaluation of annotations.
* Prefer **dataclasses** for plain data containers.
* Keep external dependencies to a minimum; this package is consumed inside a
  ROS 2 workspace where dependency management is constrained.

---

## ROS 2 Integration Notes

When using `msdc-core` inside a ROS 2 package:

1. Add `msdc-core` as a dependency in `package.xml`:
   ```xml
   <depend>python3-msdc-core</depend>
   ```
2. Or install it into the ROS 2 Python environment:
   ```bash
   pip install -e /path/to/msdc-core
   ```
3. Import normally in your ROS 2 node:
   ```python
   from msdc_core.control import MotorController
   ```
