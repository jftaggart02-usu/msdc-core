# Repository Guidelines for Copilot

## Installation

Install prerequisites:
```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv make
```

Make a virtual environment:
```bash
make venv
```

Install the Python package:
```bash
make install
```

## Coding Standards

After modifying code:
1. Run `make lint` and fix errors and warnings.
2. Update the unit tests if necessary. Then run `make test` and ensure all tests pass.
3. Update documentation in `README.md` if necessary.
