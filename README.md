# Peregrin is a py-shiny application for exploratory analysis of cell tracking data

<!-- TODO: Add demonstrational video -->

## About

Application designed for scientists, allowing **exploratory analysis of cell tracking data via an interactive, browser-based UI** built with [Py-Shiny](https://shiny.posit.co/py/).

### Features and Flows

#### Data input

- Import raw tabular - CSV (recommended) / XLSX / XLS - data files containing information about spots of individual trajectories;
- Imported data must contain *track identificator*, *time point*, *x coordinate* and *y coordinate* columns;
- Optionally, raw input may also contain other information;
- Run Peregrin and compute track statistics;
- Display computed data and download them in CSV format.

#### Filtering

- Availible thresholding methods include *Literal*, *Normalized 0-1*, *Percentile*, and *Relative to (a selected value)*.

#### Visualization

- Generate interactive and highly customizable plots covering:
    - Track reconstruction and animation;
    - Time lag statistics visualizations;
    - Time charts;
    - Superplots.
- Export figures as SVGs and more.

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