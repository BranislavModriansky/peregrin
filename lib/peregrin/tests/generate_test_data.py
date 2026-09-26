from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import polars as pl


# TrackMate-style header block (4 rows) mirroring the real CSV export.
_HEADER_ROWS = [
    # Row 1: human-readable long names
    ["Label", "Spot ID", "Track ID", "Quality", "X", "Y", "Z", "T", "Frame", "Radius"],
    # Row 2: alt names
    ["Label", "Spot ID", "Track ID", "Quality", "X", "Y", "Z", "T", "Frame", "R"],
    # Row 3: short names
    ["Label", "Spot ID", "Track ID", "Quality", "X", "Y", "Z", "T", "Frame", "R"],
    # Row 4: units
    ["", "", "", "(quality)", "(micron)", "(micron)", "(micron)", "(sec)", "", "(micron)"],
]

_MACHINE_HEADER = [
    "LABEL", "ID", "TRACK_ID", "QUALITY",
    "POSITION_X", "POSITION_Y", "POSITION_Z", "POSITION_T", "FRAME", "RADIUS",
]


def _sample_n_points(rng: np.ndarray, max_points: int, min_points: int = 5) -> int:
    """Draw a point count that leans heavily toward `max_points`.

    Uses a Beta(6, 1) distribution (mass concentrated near 1.0), mapped onto
    [min_points, max_points].
    """
    frac = rng.beta(6.0, 1.0)
    n = int(round(min_points + frac * (max_points - min_points)))
    return max(min_points, min(max_points, n))


def _build_trajectory(
    track_length: float,
    straightness_ratio: float,
    direction_mean: float,
    n_points: int,
    rng: np.random.Generator,
    start_xy: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Generate an (n_points, 2) array of XY positions.

    Guarantees (exactly, up to float precision):
      - summed step length == track_length
      - net_displacement / track_length == straightness_ratio
      - atan2(net_dy, net_dx) == direction_mean
    """
    if not (0.0 <= straightness_ratio <= 1.0):
        raise ValueError("straightness_ratio must be in [0, 1].")

    n_steps = n_points - 1
    if n_steps < 1:
        raise ValueError("n_points must be >= 2.")

    step_len = track_length / n_steps
    net_disp = straightness_ratio * track_length

    # Target net vector.
    target = np.array([net_disp * math.cos(direction_mean),
                       net_disp * math.sin(direction_mean)])

    # Start from equal-direction steps (all pointing along direction_mean),
    # then add random per-step angular perturbations and correct the residual
    # by a small uniform angle so the net vector matches exactly.
    base_angles = np.full(n_steps, direction_mean)

    # Random wiggle whose spread depends on how much slack straightness allows.
    # Perfectly straight (ratio == 1) -> no wiggle possible.
    slack = 1.0 - straightness_ratio
    spread = math.pi * slack  # radians
    wiggle = rng.uniform(-spread, spread, size=n_steps)
    # Zero-mean the wiggle so it does not bias the mean direction.
    wiggle -= wiggle.mean()

    angles = base_angles + wiggle
    steps = step_len * np.column_stack([np.cos(angles), np.sin(angles)])

    # Residual correction: shift endpoints to hit `target` exactly while keeping
    # each step length fixed. We do this by rotating the whole cloud + a final
    # linear correction distributed across steps, then renormalizing lengths.
    net = steps.sum(axis=0)
    residual = target - net
    steps += residual / n_steps  # distribute residual evenly

    # Renormalize each step back to exact step_len (residual distribution perturbs it).
    lengths = np.linalg.norm(steps, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    steps = steps / lengths * step_len

    # After renormalization the net drifts slightly; iterate a few times to converge.
    for _ in range(200):
        net = steps.sum(axis=0)
        residual = target - net
        if np.linalg.norm(residual) < 1e-9:
            break
        steps += residual / n_steps
        lengths = np.linalg.norm(steps, axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0
        steps = steps / lengths * step_len

    positions = np.vstack([np.zeros(2), np.cumsum(steps, axis=0)])
    positions += np.asarray(start_xy)
    return positions


def _simulate_dataframe(
    track_specs: Sequence[dict],
    max_track_points: int,
    time_interval: float,
    rng: np.random.Generator,
    id_offset: int = 0,
) -> pl.DataFrame:
    """Build one long-format DataFrame for a set of tracks."""
    rows = []
    spot_id = id_offset
    for track_id, spec in enumerate(track_specs):
        n_points = _sample_n_points(rng, max_track_points)
        positions = _build_trajectory(
            track_length=spec["track_length"],
            straightness_ratio=spec["straightness_ratio"],
            direction_mean=spec["direction_mean"],
            n_points=n_points,
            rng=rng,
            start_xy=spec.get("start_xy", (0.0, 0.0)),
        )
        for frame, (x, y) in enumerate(positions):
            rows.append({
                "LABEL": f"ID{spot_id}",
                "ID": spot_id,
                "TRACK_ID": track_id,
                "QUALITY": 1.0,
                "POSITION_X": float(x),
                "POSITION_Y": float(y),
                "POSITION_Z": 0.0,
                "POSITION_T": float(frame * time_interval),
                "FRAME": frame,
                "RADIUS": 4.8,
            })
            spot_id += 1

    return pl.DataFrame(rows)


def _write_trackmate_csv(df: pl.DataFrame, path: Path) -> None:
    """Write a DataFrame in TrackMate CSV layout (machine header + 4 meta rows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(_MACHINE_HEADER) + "\n")
        for row in _HEADER_ROWS:
            f.write(",".join(str(c) for c in row) + "\n")
        for record in df.iter_rows(named=True):
            f.write(",".join(str(record[c]) for c in _MACHINE_HEADER) + "\n")


def generate_test_data(
    output_dir: str | Path = "simulated_data/testing_0",
    *,
    track_specs: Optional[Sequence[dict]] = None,
    n_sets: int = 2,
    n_subsets: int = 2,
    tracks_per_df: int = 5,
    max_track_points: int = 20,
    time_interval: float = 60.0,
    seed: int = 0,
) -> list[Path]:
    """Generate a hierarchical test dataset of cell-tracking CSVs.

    Structure produced::

        output_dir/
            set1/subset1.csv
            set1/subset2.csv
            set2/subset1.csv
            set2/subset2.csv

    Parameters
    ----------
    output_dir : str | Path
        Root directory for the generated hierarchy.
    track_specs : sequence of dict, optional
        Per-track metric specifications. Each dict must contain
        ``track_length``, ``straightness_ratio`` and ``direction_mean``
        (and optionally ``start_xy``). If None, `tracks_per_df` specs are
        generated randomly. If provided, its length must equal
        `tracks_per_df` and it is reused for every DataFrame.
    n_sets, n_subsets : int
        Hierarchy dimensions (default 2 x 2 = 4 CSVs).
    tracks_per_df : int
        Number of trajectories per CSV.
    max_track_points : int
        Maximum number of time points per track; the actual count leans
        strongly toward this maximum.
    time_interval : float
        Time step between consecutive frames (seconds).
    seed : int
        Master seed for reproducibility.

    Returns
    -------
    list[Path]
        Paths of the written CSV files.
    """
    root = Path(output_dir)
    master_rng = np.random.default_rng(seed)

    def _default_specs(rng: np.random.Generator) -> list[dict]:
        specs = []
        for _ in range(tracks_per_df):
            specs.append({
                "track_length": float(rng.uniform(20.0, 100.0)),
                "straightness_ratio": float(rng.uniform(0.3, 0.95)),
                "direction_mean": float(rng.uniform(-math.pi, math.pi)),
                "start_xy": (float(rng.uniform(0, 50)), float(rng.uniform(0, 50))),
            })
        return specs

    written: list[Path] = []
    for s in range(1, n_sets + 1):
        for ss in range(1, n_subsets + 1):
            # Deterministic per-file child RNG.
            child_seed = int(master_rng.integers(0, 2**32 - 1))
            rng = np.random.default_rng(child_seed)

            specs = list(track_specs) if track_specs is not None else _default_specs(rng)
            if len(specs) != tracks_per_df:
                raise ValueError(
                    f"track_specs length ({len(specs)}) != tracks_per_df ({tracks_per_df})"
                )

            df = _simulate_dataframe(
                track_specs=specs,
                max_track_points=max_track_points,
                time_interval=time_interval,
                rng=rng,
            )
            path = root / f"set{s}" / f"subset{ss}.csv"
            _write_trackmate_csv(df, path)
            written.append(path)

    return written


if __name__ == "__main__":
    paths = generate_test_data(
        output_dir="lib/peregrin/tests/dummy_data/testing_0", 
        seed=42,
        track_specs=[
            {   # perfectly straight, due East
                "track_length": 100.0,
                "straightness_ratio": 1.0,
                "direction_mean": 0.0,                 # +x
                "start_xy": (0.0, 0.0),
            },
            {   # highly straight, due North
                "track_length": 80.0,
                "straightness_ratio": 0.9,
                "direction_mean": math.pi / 2,         # +y
                "start_xy": (100.0, 0.0),
            },
            {   # moderately meandering, North-East diagonal
                "track_length": 60.0,
                "straightness_ratio": 0.5,
                "direction_mean": math.pi / 4,         # 45 deg
                "start_xy": (200.0, 0.0),
            },
            {   # meandering, due West
                "track_length": 50.0,
                "straightness_ratio": 0.7,
                "direction_mean": math.pi,             # -x
                "start_xy": (300.0, 0.0),
            },
            {   # tortuous, South-East
                "track_length": 40.0,
                "straightness_ratio": 0.3,
                "direction_mean": -math.pi / 4,        # -45 deg
                "start_xy": (400.0, 0.0),
            },
        ],
    )
    for p in paths:
        print("wrote", p)