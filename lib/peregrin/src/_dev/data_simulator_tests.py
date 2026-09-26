"""
Generate a deterministic, hand-verifiable test dataset.

Structure (2 sets, each with 2 subsets, each subset = 1 CSV file with 10 tracks):

    setA/subsetA1.csv
    setA/subsetA2.csv
    setB/subsetB1.csv
    setB/subsetB2.csv

Each track has exactly 5 time points (frames 0..4) at time step = 1.0 second.
Each track is a PERFECTLY STRAIGHT, CONSTANT-SPEED line so that all metrics
are trivially hand-calculable:

    - a track with per-step distance `d` moving purely along +x:
        * track_length          = 4 * d          (4 steps)
        * track_displacement    = 4 * d
        * straightness_ratio    = 1.0
        * speed (min=max=mean=median) = d / 1.0 = d
        * speed_sd              = 0.0
        * direction             = 0.0 rad (pure +x)
        * directional_change    = 0.0
        * MSD at frame_lag L    = (L * d)^2

Within a CSV file, the 10 tracks use per-step distances d = 1,2,...,10 (µm),
so every file is identical in structure -> outputs per file are identical,
and per-set / per-subset aggregates are identical too. This makes the
expected values easy to write down and verify.

Metadata rows follow the CSV layout expected by `_read_table`:
    row 0 : header (column names)
    row 2 : units row (metadata_row_index=2)
    row 4+: data (skiprows=4)
"""

from __future__ import annotations
import csv
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration (kept in sync with the test module)
# ---------------------------------------------------------------------------
N_TRACKS_PER_FILE = 10
N_FRAMES = 5                 # frames 0..4
TIME_STEP = 1.0              # seconds between consecutive frames
STEP_DISTANCES = list(range(1, N_TRACKS_PER_FILE + 1))  # 1..10 (µm per step)

COLUMNS = ["track_id", "time_point", "x_coordinate", "y_coordinate"]
UNITS_ROW = ["", "second", "micrometer", "micrometer"]

# Set / subset layout -> file paths (relative to this script's dummy dir)
LAYOUT = {
    "setA": {
        "subsetA1": "setA/subsetA1.csv",
        "subsetA2": "setA/subsetA2.csv",
    },
    "setB": {
        "subsetB1": "setB/subsetB1.csv",
        "subsetB2": "setB/subsetB2.csv",
    },
}


def _build_rows() -> list[list]:
    """Build the data rows for one file (10 straight tracks along +x)."""
    rows = []
    for tid, step in enumerate(STEP_DISTANCES, start=1):
        for frame in range(N_FRAMES):
            t = frame * TIME_STEP
            x = frame * step          # moves along +x by `step` each frame
            y = 0.0                   # constant y -> perfectly straight line
            rows.append([tid, t, x, y])
    return rows


def _write_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _build_rows()

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)            # row 0: header
        writer.writerow([""] * len(COLUMNS))  # row 1: blank
        writer.writerow(UNITS_ROW)          # row 2: units (metadata_row_index=2)
        writer.writerow([""] * len(COLUMNS))  # row 3: blank
        writer.writerows(rows)              # row 4+: data (skiprows=4)


def generate(base_dir: Path | None = None) -> Path:
    """
    Generate all test CSV files under `<base_dir>/dummy_data/structured/`.

    Returns the path to the `structured` directory.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    root = base_dir / "dummy_data" / "structured"

    for _set, subsets in LAYOUT.items():
        for _subset, rel in subsets.items():
            _write_csv(root / rel)

    return root


def file_dict(root: Path) -> dict:
    """
    Build the nested dict that `load_data` expects:

        { set: { subset: <csv path> } }
    """
    return {
        _set: {
            _subset: str((root / rel).resolve())
            for _subset, rel in subsets.items()
        }
        for _set, subsets in LAYOUT.items()
    }


if __name__ == "__main__":
    out = generate()
    print(f"Generated test data under: {out}")
    for p in sorted(out.rglob("*.csv")):
        print("  ", p.relative_to(out))