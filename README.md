# Peregrin is a py-shiny application for exploratory analysis of cell tracking data

<!-- TODO: Add demonstrational video -->

## Contents
> [About](#about)
> [Set-up](#set-up)
> [User's guide](#users-guide)
> [Exemplary data](#examples)

## About

Application designed for scientists, allowing **exploratory analysis of cell tracking data via an interactive, browser-based UI** built with [Py-Shiny](https://shiny.posit.co/py/).

### Features and flows:
- Import files containing spot information of individual trajectories 
     Support for CSV (recommended), XLSX, XLS
    - Required columns like track_id, t/frame, x, y [, z] and optional metadata
- Computes per-step and per-track metrics (speed, displacement, turning angle, directionality, MSD)
- Provides interactive filtering, grouping, faceting, and condition comparisons
- Exports figures (PNG/SVG) and tables (CSV) for downstream use or publication

Key features:
- Interactive visualizations: trajectory overlays, time series, distributions, MSD plots
- No-code analysis: build views via controls; bookmark/share URLs or save/load session state
- Scales to thousands of tracks; designed for repeatable workflows
- Extensible: modular panels and Python hooks for custom metrics and plots

Tech stack (at a glance):
- UI: Shiny for Python
- Data: pandas/numpy
- Visualization: Plotly/Altair

Status and roadmap:
- Early preview; APIs and UI may change
- Planned: additional file format adapters, richer 3D support, batch reporting, plug‑in examples
<!-- TODO: Describe the problem this project solves, the motivation, and the goals. -->
<!-- TODO: List key features and a brief overview of the architecture/tech stack if helpful. -->
<!-- TODO: Current status/roadmap (optional). -->

## Set-up
### Prerequisites
<!-- TODO: List required tools (e.g., Python version, package manager, system deps). -->

### Installation
<!-- TODO: Steps to clone and install dependencies. Example (adjust as needed): -->
```bash
# Clone the repository
git clone <repo-url>
cd peregrin

# Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Configuration
<!-- TODO: Document environment variables, config files, and secrets. -->
<!-- Example:
- Copy .env.example to .env and fill in values
- Update config.yaml with project-specific settings
-->

### Running locally
<!-- TODO: How to start the app or run the main script. -->
```bash
python -m peregrin
# or
python path\to\main.py
```

### Running tests
<!-- TODO: How to run tests locally/in CI. -->
```bash
pytest -q
```

## User's guide

## Examples
### Minimal usage
<!-- TODO: Show the simplest way to use the project (CLI or library). -->
```bash
# CLI example (if applicable)
peregrin --input sample.txt --output out/
```

```python
# Library example (if applicable)
from peregrin import some_api

result = some_api.do_something("input")
print(result)
```

### Advanced usage
<!-- TODO: Add a more involved example, parameters, and expected outputs. -->