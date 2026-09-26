from __future__ import annotations

import traceback
import warnings
import numpy as np
import polars as pl
from scipy import stats
from typing import Any, Callable, Literal, Optional, Dict, List

from ..various import Values, is_empty
from ..settings import params
from .data_frames import Calc

from .._pckg_exceptions._pckg_errors import *
from .._pckg_exceptions._pckg_warnings import *



class DataObject(Calc):
    """
    A stateful, callable interface over :class:`Calc`.

    Parameters
    ----------
    inferative_error : bool, optional
        Whether to compute inferative error. Default is False.
    bootstrap_ci : bool, optional
        Whether to compute bootstrap confidence intervals. Default is False.
    confidence_lvl : float, optional
        Confidence level for the bootstrap confidence intervals. Default is 0.95.
    ci_statistic : str, optional
        Statistic to use for the bootstrap confidence intervals. Default is "mean".

    Usage
    -----
    >>> empty_data_object = DataObject()
    >>> loaded_data_object = DataObject()(data)
    >>> loaded_data_object = empty_data_object(data)              # computes & stores Spots_df

    >>> spots_df = loaded_data_object.compute_spots()             # per-spot statistics
    >>> tracks_df = loaded_data_object.compute_tracks()           # per-track statistics
    >>> timepoints_df = loaded_data_object.compute_timepoints()   # per-time-point statistics
    >>> timelags_df = loaded_data_object.compute_timelags()       # per-time-interval statistics

    >>> loaded_data_object.plot_tracks()                          # reconstruct trajectories
    >>> loaded_data_object.plot_msd(band='sem')                   # MSD plot
    """

    inferative_error: Optional[bool] = False
    bootstrap_ci: Optional[bool] = False
    bootstrap_ci_method: Optional[str] = "BCa"
    ci_confidence: Optional[float] = 0.95
    bootstrap_resamples: Optional[int] = 1000
    ci_statistic: Optional[str] = "mean"

    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(
            inferative_error=kwargs.get("inferative_error", self.inferative_error),
            bootstrap_ci=kwargs.get("bootstrap_ci", self.bootstrap_ci),
            ci_confidence=kwargs.get("ci_confidence", self.ci_confidence),
            ci_statistic=kwargs.get("ci_statistic", self.ci_statistic),
            bootstrap_ci_method=kwargs.get("bootstrap_ci_method", self.bootstrap_ci_method),
            bootstrap_resamples=kwargs.get("bootstrap_resamples", self.bootstrap_resamples),
            **kwargs,
        )

        self.spots_df: Optional[pl.DataFrame] = None
        self.tracks_df: Optional[pl.DataFrame] = None
        self.timepoints_df: Optional[pl.DataFrame] = None
        self.timelags_df: Optional[pl.DataFrame] = None

        self._categories: Optional[dict] = None

    # Representation of the DataObject instance
    def __repr__(self) -> str:
        if self.spots_df is None:
            return f"<DataObject: empty, (inferative_error={self.inferative_error}, bootstrap_ci={self.bootstrap_ci}, bootstrap_ci_method={self.bootstrap_ci_method}, ci_confidence={self.ci_confidence}, bootstrap_resamples={self.bootstrap_resamples})>"
        else:
            return f"<DataObject: height={self.spots_df.height}, width={self.spots_df.width}, bytes={self.spots_df.estimated_size()}, (inferative_error={self.inferative_error}, bootstrap_ci={self.bootstrap_ci}, bootstrap_ci_method={self.bootstrap_ci_method}, ci_confidence={self.ci_confidence}, bootstrap_resamples={self.bootstrap_resamples})>"

    # Creates a DataObject instance with pre-computed spot data for intuitive method chaining.
    def __call__(self, df: pl.DataFrame, **kwargs) -> "DataObject":
        """
        Compute per-spot data from raw input and store them in the intance.
        Parameters
        ----------
        df : pl.DataFrame
            Raw input DataFrame containing spot data.
        **kwargs : dict
            Additional keyword arguments passed to the spot computation method.

        Returns
        -------
        self : DataObject
            The instance itself, allowing for method chaining.
        """

        self.spots_df = self.spots(df, **kwargs)

        # Invalidate downstream caches on new input.
        self.tracks_df = None
        self.timepoints_df = None
        self.timelags_df = None

        return self

    def _resolve_spots(self, df: Optional[pl.DataFrame]) -> pl.DataFrame:
        source = df if df is not None else self.spots_df
        return source

    def compute_spots(self, df: Optional[pl.DataFrame] = None, subset: Optional[list[str]] = None, **kwargs) -> pl.DataFrame:
        """Compute (and store) per-spot statistics."""
        if not is_empty(self.spots_df):
            return self.spots_df
        return self.spots(df, subset=subset, **kwargs)

    def compute_tracks(self, df: Optional[pl.DataFrame] = None, subset: Optional[list[str]] = None, **kwargs) -> pl.DataFrame:
        """Compute (and store) per-track statistics from the stored Spots_df."""
        source = self._resolve_spots(df)
        self.tracks_df = self.tracks(source, subset=subset, **kwargs)
        return self.tracks_df

    def compute_timepoints(self, df: Optional[pl.DataFrame] = None, subset: Optional[list[str]] = None, *, grouping_level: Any = 'highest', **kwargs) -> pl.DataFrame:
        """Compute (and store) per-time-point statistics from the stored Spots_df."""
        source = self._resolve_spots(df)
        self.timepoints_df = self.timepoints(source, subset=subset, grouping_level=grouping_level, **kwargs)
        return self.timepoints_df

    def compute_timelags(self, df: Optional[pl.DataFrame] = None, subset: Optional[list[str]] = None, *, grouping_level: Any = 'highest', **kwargs) -> pl.DataFrame:
        """Compute (and store) per-time-interval statistics from the stored Spots_df."""
        source = self._resolve_spots(df)
        self.timelags_df = self.timelags(source, subset=subset, grouping_level=grouping_level, **kwargs)
        return self.timelags_df

    # -----------------------------------------------------------------------
    # Plotting wrappers (lazy imports avoid circular deps)
    # -----------------------------------------------------------------------
    def plot_tracks(self, **kwargs):
        """Reconstruct and plot trajectories from the stored Spots_df."""
        from ..plot.tracks.reconstruct import reconstruct
        return reconstruct(self.spots_df, **kwargs)

    def plot_msd(self, band: Optional[str] = None, *, grouping_level: Any = 'highest', **kwargs):
        """Plot MSD from the stored Spots_df."""
        from ..plot.time.lags import msd
        return msd(self.spots_df, band=band, categories=self._categories, grouping_level=grouping_level, **kwargs)

    def plot_turn_angles(self, *, grouping_level: Any = 'highest', **kwargs):
        """Plot the turning-angle heatmap from the stored Spots_df."""
        from ..plot.time.lags import turn_angles
        return turn_angles(self.spots_df, grouping_level=grouping_level, **kwargs)




create_object = DataObject()