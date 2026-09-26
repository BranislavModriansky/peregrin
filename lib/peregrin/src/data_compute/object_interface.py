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
    """ An intuitive interface built over :class:`Calc`. """

    inferative_error: Optional[bool] = False
    bootstrap_ci: Optional[bool] = False
    bootstrap_ci_method: Optional[str] = "BCa"
    ci_confidence: Optional[float] = 0.95
    bootstrap_resamples: Optional[int] = 1000
    ci_statistic: Optional[str] = "mean"

    def __init__(self) -> None:
        self.spots_df: Optional[pl.DataFrame] = None
        self.tracks_df: Optional[pl.DataFrame] = None
        self.timepoints_df: Optional[pl.DataFrame] = None
        self.timelags_df: Optional[pl.DataFrame] = None


    # Creates a DataObject instance with pre-computed spot data for intuitive method chaining.
    def __call__(self, df: pl.DataFrame, **kwargs) -> "DataObject":
        """
        Compute data from the input DataFrame and store it in the instance.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame acquired with :func:`load_data`.

        inferative_error : bool, optional, default False
            If True, enables inferative error (sem) computation for group statistics in
            :meth:`compute_timepoints` and :meth:`compute_timelags` result DataFrames.

        bootstrap_ci : bool, optional, default False
            If True, enables bootstrap confidence interval computation for group statistics in
            :meth:`compute_timepoints` and :meth:`compute_timelags` result DataFrames.

        bootstrap_ci_method : str, optional, default "BCa"
            Method for bootstrap confidence interval computation.

        ci_confidence : float, optional, default 0.95
            Confidence level for the confidence interval.

        bootstrap_resamples : int, optional, default 1000
            Number of bootstrap resamples.

        ci_statistic : str, optional, default "mean"
            Statistic for confidence interval computation. E.g. "mean", "median".

        Returns
        -------
        self : DataObject
            An instance of :class:`DataObject` allowing method chaining.

        Usage
        -----
        >>> loaded_data_object = create_object(df)
    
        >>> spots_df = loaded_data_object.compute_spots()             # per-trajectory-point statistics
        >>> tracks_df = loaded_data_object.compute_tracks()           # per-whole-trajectory statistics
        >>> timepoints_df = loaded_data_object.compute_timepoints()   # per-time-point statistics
        >>> timelags_df = loaded_data_object.compute_timelags()       # per-time-interval statistics
    
        >>> loaded_data_object.plot_tracks()                          # reconstruct trajectories
        >>> loaded_data_object.plot_msd(band='sem')                   # MSD plot

        Documentation
        -------------
        Please refer to the official documentation for usage examples and more detailed explanations at the official peregrin website: https://peregrin-documentation-url 
        """

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

        self.spots_df = self.spots(df, **kwargs)

        # Invalidate downstream caches on new input.
        self.tracks_df = None
        self.timepoints_df = None
        self.timelags_df = None

        return self


    # Representation of the DataObject instance
    def __repr__(self) -> str:
        if self.spots_df is None:
            return f"<DataObject: empty, (inferative_error={self.inferative_error}, bootstrap_ci={self.bootstrap_ci}, bootstrap_ci_method={self.bootstrap_ci_method}, ci_confidence={self.ci_confidence}, bootstrap_resamples={self.bootstrap_resamples})>"
        else:
            return f"<DataObject: height={self.spots_df.height}, width={self.spots_df.width}, bytes={self.spots_df.estimated_size()}, (inferative_error={self.inferative_error}, bootstrap_ci={self.bootstrap_ci}, bootstrap_ci_method={self.bootstrap_ci_method}, ci_confidence={self.ci_confidence}, bootstrap_resamples={self.bootstrap_resamples})>"


    def _resolve_spots(self, df: Optional[pl.DataFrame]) -> pl.DataFrame:
        source = df if df is not None else self.spots_df
        return source

    def compute_spots(self, df: Optional[pl.DataFrame] = None, **kwargs) -> pl.DataFrame:
        """
        Computes basic per-trajectory-point metrics (previous -> current position) using the :meth:`spots` method.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame <- the result DataFrame of the :func:`load_data` function.
        subset : list[str], optional
            Subset of columns to consider for the computation, by default None.
        **kwargs
            Additional keyword arguments passed to the computation.

        Returns
        -------
        pl.DataFrame
            - `track_id`: Native track identifier included in the input DataFrame.
            - `track_uid`: Unique track identifier assigned to each track.
            - `frame`: Frame number within the track (0-based).
            - `time_point`: Time point of the trajectory point.
            - `x_coordinate`: X coordinate of the trajectory point.
            - `y_coordinate`: Y coordinate of the trajectory point.
            - `distance`: Euclidean distance between the previous and the current position.
            - `direction`: Direction of movement between the previous and the current position.
        """

        if not is_empty(self.spots_df):
            if (kwargs.get('enriched', False) 
                and 'cum_track_length' not in self.spots_df.columns):
                self.spots_df = self.spots(df, **kwargs)
            return self.spots_df
        return self.spots(df, **kwargs)

    def compute_tracks(self, df: Optional[pl.DataFrame] = None, subset: Optional[list[str]] = None, **kwargs) -> pl.DataFrame:
        """
        Computes per-trajectory metrics using the :meth:`tracks` method.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame <- the result DataFrame of the :meth:`spots` method.
        subset : list[str], optional
            List of specific metrics to compute. If None, all available metrics are computed.
        **kwargs
            Additional keyword arguments passed to the computation functions.

        Returns
        -------
        pl.DataFrame
            - `track_id`: Native track identifier.
            - `track_uid`: Unique track identifier.
            - `y_location`: Mean value of the y-coordinates of the track.
            - `x_location`: Mean value of the x-coordinates of the track.
            - `track_length`: Total length of the track.
            - `track_displacement`: Straight-line distance between the start and end points of the track.
            - `straightness_ratio`: Ratio of track displacement to track length.
            - `speed_min`: Minimum speed along the track.
            - `speed_max`: Maximum speed along the track.
            - `speed_mean`: Mean speed along the track.
            - `speed_sd`: Standard deviation of the speed along the track.
            - `speed_median`: Median speed along the track.
            - `mean_straight_line_speed`: Track displacement divided by track duration.
            - `forward_progression_linearity`: Mean straight line speed divided by mean speed. Measures how linearly the track progresses forward.
            - `max_distance_reached`: Maximum distance reached from the starting point of the track.
            - `track_start_frame`: Frame at which the track starts.
            - `track_end_frame`: Frame at which the track ends.
            - `track_points`: Number of points in the track.
            - `direction_mean`: Mean direction of movement along the track.
            - `direction_var`: Variance of the direction of movement along the track.
            - `mean_directional_change`: Mean change in direction between consecutive points along the track.
            - `mean_directional_change_rate`: Mean directional change divided by the time interval. Mean rate of change in direction along the track.
        """
        source = self._resolve_spots(df)
        self.tracks_df = self.tracks(source, subset=subset, **kwargs)
        return self.tracks_df

    def compute_timepoints(self, df: Optional[pl.DataFrame] = None, subset: Optional[list[str]] = None, *, grouping_level: Any = 'highest', **kwargs) -> pl.DataFrame:
        """
        Computes time point statistics for categories (groups) using the :meth:`timepoints` method.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame <- the result DataFrame of the :meth:`spots` method.
        subset : list[str], optional
            List of time point statistics to compute. If None, all available statistics are computed.
        grouping_level : Literal['highest', 'lowest'] | str | int | list, default='highest'
            Level(s) at which to group the data for computing time point statistics.
        **kwargs
            Additional keyword arguments passed to the computation functions.

        Returns
        -------
        pl.DataFrame
            - `time_point`: Time point of the observation.
            - `frame`: Frame number of the observation.

            followed by any of (mean, median, std, var, min, max, sum, count, circular_mean, circular_std, circular_var) for columns
            - `cum_track_length`: Cumulative track length up to the current time point.
            - `cum_track_displacement`: Cumulative track displacement up to the current time point.
            - `cum_straightness_ratio`: Cumulative straightness ratio up to the current time point.
            - `cum_speed_mean`: Cumulative mean speed up to the current time point.
            - `instantaneous_speed`: Instantaneous speed at the current time point.
            - `cum_mean_straight_line_speed`: Cumulative mean straight line speed up to the current time point.
            - `cum_forward_progression_linearity`: Cumulative forward progression linearity up to the current time point.
            - `cum_sum_directional_change`: Cumulative sum of directional changes up to the current time point.
            - `cum_mean_directional_change`: Cumulative mean of directional changes up to the current time point.
        """
        source = self._resolve_spots(df)
        self.timepoints_df = self.timepoints(source, subset=subset, grouping_level=grouping_level, **kwargs)
        return self.timepoints_df

    def compute_timelags(self, df: Optional[pl.DataFrame] = None, subset: Optional[list[str]] = None, *, grouping_level: Any = 'highest', **kwargs) -> pl.DataFrame:
        """
        Computes per-time-interval statistics using the :meth:`timelags` method.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame <- the result DataFrame of the :meth:`spots` method.
        subset : list[str], optional
            Subset of statistics to compute.
        grouping_level : Literal['highest', 'lowest'] | str | int | list | None, optional
            Level at which to group the data.
        **kwargs
            Additional keyword arguments passed to the computation functions.

        Returns
        -------
        pl.DataFrame
            - `time_lag`: The time interval between observations.
            - `frame_lag`: The frame interval between observations.
            - `MSD`: Mean squared displacement for the given time lag.
            - `MSD_sd`: Standard deviation of the mean squared displacement for the given time lag.
            - `tracks_contributing`: Number of tracks contributing to the given time lag.
            - `position_pairs_contributing`: Number of position pairs contributing to the given time lag.
            - `directional_change_mean`: Mean directional change for the given time lag.
            - `directional_change_var`: Variance of the directional change for the given time lag.
        """
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
        return msd(self.spots_df, band=band, categories=None, grouping_level=grouping_level, **kwargs)

    def plot_turn_angles(self, *, grouping_level: Any = 'highest', **kwargs):
        """Plot the turning-angle heatmap from the stored Spots_df."""
        from ..plot.time.lags import turn_angles
        return turn_angles(self.spots_df, grouping_level=grouping_level, **kwargs)




create_object = DataObject()