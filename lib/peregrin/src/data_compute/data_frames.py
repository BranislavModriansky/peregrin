from __future__ import annotations

import traceback
import warnings
import numpy as np
import polars as pl
from dataclasses import dataclass
from scipy import stats
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, cdist
from typing import Any, Callable, Literal, Optional, Dict, List

from ..utils import is_empty
from ..settings import params
from ..utils import ensure_polars

from warnings import warn
from .._pckg_exceptions._pckg_errors import *
from .._pckg_exceptions._pckg_warnings import *



# Per-call statistics options
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StatSettings:
    """
    Resolved error/uncertainty settings for a single computation call.

    Built by :meth:`Calc._stats_configuration`. Keyword arguments passed directly to
    :meth:`Calc.tracks`, :meth:`Calc.timepoints` or :meth:`Calc.timelags`
    take precedence over the instance-level defaults set on :class:`Calc`.
    """

    inferative_error: bool
    bootstrap_ci: bool
    ci_confidence: float
    bootstrap_resamples: int
    bootstrap_ci_method: str
    ci_statistic: Callable[[np.ndarray], float]

    def ci_label(self, hide_ci_statistic: bool = False) -> str:
        """Confidence level as an integer percentage (0.95 -> 95), as used in
        output column names such as `MSD_ci95_low`."""
        cl = int(round(self.ci_confidence * 100)) if self.ci_confidence <= 1 else int(round(self.ci_confidence))

        stat = self.ci_statistic.__name__ if hasattr(self.ci_statistic, '__name__') else str(self.ci_statistic)

        return f'{stat}_ci{cl}' if not hide_ci_statistic else f'ci{cl}'


# Metric registry
# ---------------------------------------------------------------------------

class MetricRegistry:
    """
    Maps user-facing *metric names* -- the values accepted by the `subset`
    parameter of :meth:`Calc.tracks`, :meth:`Calc.timepoints` and
    :meth:`Calc.timelags` -- to builders of output columns.

    A builder is a callable `(context: dict) -> dict[str, pl.Expr | Callable]`.
    One metric may produce several related output columns (e.g. requesting
    `'MSD'` yields `MSD`, `MSD_min`, `MSD_max`, `MSD_sd`, ...). Dict values
    are either:

      - a `pl.Expr` aggregation expression -- all expressions are collected
        into ONE `group_by().agg()` call (single pass over the data), or
      - a post-processing callable `(out_df, context) -> out_df` for statistics
        that cannot be expressed as a polars aggregation (e.g. bootstrap
        confidence intervals).

    Error/uncertainty columns (sem, bootstrap CI bounds) are decided at
    compute time from `context['stat_settings']` (a :class:`StatSettings`), so per-call
    keyword overrides take effect without rebuilding the registry.
    """

    def __init__(self) -> None:
        self._builders: Dict[str, Callable[[dict], Dict[str, Any]]] = {}
        self._order: List[str] = []
        self._derived: Dict[str, dict] = {}

    def add(self, metric: str, builder: Callable[[dict], Dict[str, Any]]) -> None:
        """Register a metric that may produce several output columns."""
        if metric not in self._builders:
            self._order.append(metric)
        self._builders[metric] = builder

    def add_column(self, metric: str, polars_expression: Callable[[dict], pl.Expr]) -> None:
        """Register a metric producing a single column named after itself."""
        self.add(metric, lambda context, m=metric, f=polars_expression: {m: f(context)})

    def add_derived(
        self,
        metric: str,
        dependencies: List[str],
        polars_expression: Callable[[pl.DataFrame], pl.Expr],
    ) -> None:
        """Register a metric *derived* from other metrics' output columns.

        `dependencies` lists the metrics whose aggregated columns this metric
        reads, and `polars_expression(out_df) -> pl.Expr` computes its single column from
        the already-aggregated frame (one row per group). The base aggregates
        are therefore evaluated once in the single `group_by().agg()` pass and
        reused by every derived metric, instead of each ratio re-aggregating the
        same source column (e.g. `pl.col('distance').sum()`) over the per-spot
        data. Dependencies pulled in only to satisfy a derivation are dropped
        from the result, so `subset=` still returns exactly what was asked for.
        """
        self.add(metric, lambda context: {})  # no agg/post output of its own
        self._derived[metric] = {'dependencies': list(dependencies), 'polars_expression': polars_expression}

    def metrics(self) -> List[str]:
        """All registered metric names, in registration order."""
        return list(self._order)

    def resolve(self, subset: Optional[str | List[str]] = None) -> List[str]:
        """
        Validate `subset` and return the metric names to compute.

        subset=None  -> every registered metric.
        subset=[...] -> exactly the requested metrics (in registry order).

        Raises
        ------
        ValueError
            If `subset` contains a name that is not a registered metric.
            Derived statistic names (e.g. 'MSD_sd') are rejected with a hint
            pointing to the base metric.
        """
        if subset is None:
            return self.metrics()
        if isinstance(subset, str):
            subset = [subset]

        unknown_metrics = [m for m in subset if m not in self._builders]
        if unknown_metrics:
            raise ValueError(self._unknown_metrics_message(unknown_metrics))

        requested = set(subset)
        return [m for m in self._order if m in requested]

    def _unknown_metrics_message(self, unknown_metrics: List[str]) -> str:
        lines = [f"Unknown metrics: {unknown_metrics}. Available metrics: {self.metrics()}."]
        for unknown in unknown_metrics:
            base = next(
                (m for m in sorted(self._order, key=len, reverse=True) if unknown.startswith(m)),
                None,
            )
            if base is not None:
                lines.append(
                    f"'{unknown}' looks like a statistic derived from '{base}' -> request '{base}' instead. "
                    "Descriptive statistics (min/max/mean/median/sd) are always included; "
                    "add 'sem' with inferative_error=True and bootstrap CI bounds with bootstrap_ci=True."
                )
        return ' '.join(lines)
        

    def compute(self, requested: List[str], context: dict) -> pl.DataFrame:
        """ Run the requested metric builders against a shared context.

        Parameters
        ----------
        requested : List[str]
            The list of metrics - column names - to compute.
        context : dict
            A shared context dictionary containing the source DataFrame, grouping columns, statistical settings or other.

        `context['data_source']` -> the source pl.DataFrame
        `context['group_by']`     -> grouping column names (list)
        `context['stat_settings']`   -> per-call :class:`StatSettings`
        """
        aggregation_funcs: List[pl.Expr] = []
        callable_funcs: List[Callable] = []

        # Registry order keeps base/post metrics ahead of the derived metrics
        # that read their columns.
        compute_order = [m for m in self._order if m in requested]

        for metric in compute_order:
            for column, item in self._builders[metric](context).items():
                if isinstance(item, pl.Expr):
                    aggregation_funcs.append(item.alias(column))
                elif callable(item):
                    callable_funcs.append(item)

        # List-aggregation helpers requested by post-processors (e.g. the raw
        # per-group values a bootstrap CI resamples from).
        for name, e in context.get('extra_exprs', {}).items():
            aggregation_funcs.append(e.alias(name))

        output = (
            context['data_source']
            .group_by(context['group_by'], maintain_order=True)
            .agg(aggregation_funcs)
        )

        for func in callable_funcs:
            output = func(output, context)

        # Derived metrics: cheap arithmetic on the already-aggregated (one row
        # per group) frame, reusing the base columns computed above.
        for metric in compute_order:
            spec = self._derived.get(metric)
            if spec is not None:
                output = output.with_columns(spec['polars_expression'](output).alias(metric))

        # Drop helper list columns
        helpers = [c for c in output.columns if c.startswith('__list_')]
        if helpers:
            output = output.drop(helpers)

        return output


class Calc:
    """
    A class with methods for computing tracking data statistics:
    spots (per-trajectory-point), tracks (per-whole-trajectory), time points (per-time-point),
    time lags (per-time-lag).

    The parameters below set the *instance defaults* for error/uncertainty
    statistics. Every one of them can also be passed as a keyword argument
    directly to :meth:`tracks`, :meth:`timepoints` or :meth:`timelags`, in
    which case the per-call value takes precedence for that call only.

    Parameters
    ----------
    inferative_error: bool, default False
        If True, sem will be computed for category statistics in the :meth:`timepoints` and :meth:`timelags` result DataFrames.

    bootstrap_ci: bool, default False
        If True, bootstrap confidence intervals will be computed for category statistics in the :meth:`timepoints` and :meth:`timelags` result DataFrames.

    ci_confidence: float, default 0.95
        The confidence level for the confidence intervals.

    bootstrap_resamples: int, default 1000
        The number of bootstrap resamples to use when computing bootstrap confidence intervals.

    bootstrap_ci_method: str, default 'BCa'
        The method to use for computing bootstrap confidence intervals.

    ci_statistic: 'mean' | 'median' | Callable, default 'mean'
        The statistic to use for computing confidence intervals.

    Attributes
    ----------
    - `significant_figures`: Optional[int] -> The number of significant figures to round the numerical results to.
    - `decimal_places`: Optional[int] -> The number of decimal places to keep in the numerical results.
    - `metadata`: Optional[dict] -> Optional metadata associated with the calculation.

    """

    metadata: Optional[dict] = None

    significant_figures: Optional[int] = None
    decimal_places: Optional[int] = None

    _ci_method_used: str = 'BCa'

    DEFAULT_CATEGORIES = ['track_uid', 'subsubgroup', 'subgroup', 'group', 'subset', 'set']

    #: Keyword arguments recognized as per-call statistics options by
    #: :meth:`tracks`, :meth:`timepoints` and :meth:`timelags`.
    STAT_OPTIONS = (
        'inferative_error', 'bootstrap_ci', 'ci_confidence',
        'bootstrap_resamples', 'bootstrap_ci_method', 'ci_statistic',
    )

    COLUMNS = {
        'SPOTS': [
            'track_id', 'track_uid', 'time_point', 'frame', 
            'x_coordinate', 'y_coordinate', 'distance', 'direction',
        ],
        'TRACKS': [
            'track_id', 'track_uid', 'y_location', 'x_location',
            'track_length', 'track_displacement', 'directionality',
            'speed_min', 'speed_max', 'speed_mean', 'speed_sd', 'speed_median',
            'mean_straight_line_speed', 'forward_progression_linearity',
            'greatest_distance', 'straightness',
            'track_start_frame', 'track_end_frame',
            'track_points', 'direction_mean', 'direction_var', 
            'mean_directional_change', 'mean_directional_change_rate'
        ],
        'TIMEPOINTS': [
            'time_point', 'frame', 'tracks_contributing',
            'cum_track_length', 'cum_track_displacement',
            'cum_directionality', 'cum_speed_mean',
            'instantaneous_speed', 'cum_mean_straight_line_speed',
            'cum_forward_progression_linearity',
            'cum_sum_directional_change', 'cum_mean_directional_change',
        ],
        'TIMELAGS': [
            'time_lag', 'frame_lag', 'MSD', 'MSD_min', 'MSD_max', 'MSD_sd', 
            'tracks_contributing', 'position_pairs_contributing', 
            'directional_change_mean', 'directional_change_var',
        ]
    }

    UNIT_TO_SECONDS = {
        "ms": 1e-3,
        "s": 1.0,
        "min": 60.0,
        "h": 3600.0,
        "d": 86400.0,
    }

    UNIT_TO_MICRONS = {
        "nm": 1e-3,
        "μm": 1.0,   # U+03BC (matches InputMetadata.UNIT_ALIASES)
        "µm": 1.0,   # U+00B5 (micro sign, kept for safety)
        "mm": 1e3,
        "cm": 1e4,
        "m": 1e6,
    }

    def __init__(
        self,
        *,
        inferative_error: bool = False,
        bootstrap_ci: bool = False,
        ci_confidence: float = 0.95,
        ci_statistic: Literal['mean', 'median', 'min', 'max'] | Callable[[np.ndarray], float] = 'mean',
        bootstrap_resamples: int = 1000,
        bootstrap_ci_method: str = 'BCa'
    ) -> None:

        self.inferative_error    = inferative_error
        self.bootstrap_ci        = bootstrap_ci
        self.ci_confidence       = ci_confidence
        self.bootstrap_resamples = bootstrap_resamples
        self.bootstrap_ci_method = bootstrap_ci_method
        self.ci_statistic        = self._validate_ci_statistic(ci_statistic)

        # Build registries for tracks, timepoints, and timelags
        self._tracks_registry     = self._build_tracks_registry()
        self._timepoints_registry = self._build_timepoints_registry()
        self._timelags_registry   = self._build_timelags_registry()


    # Per-call statistics options
    # -----------------------------------------------------------------------

    @staticmethod
    def _validate_ci_statistic(statistic: Any) -> Callable[[np.ndarray], float]:
        """Normalize a `ci_statistic` value ('mean', 'median' or a callable) to a callable."""

        match statistic:
            case _ if callable(statistic):
                return statistic
            case 'mean':
                return np.mean
            case 'median':
                return np.median
            case 'min':
                return np.min
            case 'max':
                return np.max
            case _:
                raise ValueError("ci_statistic must be 'mean', 'median', or a callable function.")

    def _stats_configuration(self, **overrides) -> StatSettings:
        """
        Resolve the inferative error (sem) and confidence interval (ci) settings for method calls.

        Passed keywords - explicitly provided - override the corresponding 
        instance defaults (`self.inferative_error`, `self.bootstrap_ci`, ...). 
        
        See :class:`StatSettings`.
        """
        def configure(key: str) -> Any:
            value = overrides.get(key, None)
            if value is None:
                return getattr(self, key)
            return value

        return StatSettings(
            inferative_error    = configure('inferative_error'),
            bootstrap_ci        = configure('bootstrap_ci'),
            ci_confidence       = configure('ci_confidence'),
            bootstrap_resamples = configure('bootstrap_resamples'),
            bootstrap_ci_method = configure('bootstrap_ci_method'),
            ci_statistic        = self._validate_ci_statistic(configure('ci_statistic'))
        )


    
    # DataFrame computation methods
    # -----------------------------------------------------------------------

    def spots(
        self,
        df: pl.DataFrame,
        **kwargs
    ) -> pl.DataFrame:
        """
        Computes basic per-trajectory-point metrics (previous -> current position).

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame <- the result DataFrame of the :func:`load_data` function.
        enriched : bool, optional
            Whether to enrich the DataFrame with cumulative per-trajectory-point metrics, by default False.

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

        if is_empty(df):
            warn("Input DataFrame is empty. No computations performed.")
            return pl.DataFrame(schema = {c: pl.Float64 for c in self.COLUMNS['SPOTS']})

        df = self._guard_df(df)

        grouping_cols = [col for col in self.DEFAULT_CATEGORIES if col in df.columns]

        df = self.assign_track_uid(df)
        df = df.sort(grouping_cols + ['track_uid', 'time_point'])

        uid = 'track_uid'

        # frame: dense rank of time_point per track (0-based)
        df = df.with_columns(
            (pl.col('time_point')
               .rank(method='dense')
               .over(uid) - 1).cast(pl.Int64).alias('frame')
        )

        # Validate per TRACK: one time_point -> exactly one frame within a track.
        impossible_duplicates = (
            df.group_by([uid, 'time_point'])
              .agg(pl.col('frame').n_unique().alias('_n'))
              .select(pl.col('_n').max())
              .item()
        )
        if impossible_duplicates and impossible_duplicates > 1:
            raise TimePointError(
                "Multiple frames assigned to the same track_uid × time_point "
                "combination. Duplicate time_point values within a track. "
                f"Max frames per time point: {impossible_duplicates}."
            )

        # Step deltas, distance and direction
        df = df.with_columns(
            ( pl.col('x_coordinate') - pl.col('x_coordinate').shift(1) ).over(uid).alias('_dx'),
            ( pl.col('y_coordinate') - pl.col('y_coordinate').shift(1) ).over(uid).alias('_dy'),
        ).with_columns(
            ( pl.col('_dx').pow(2) + pl.col('_dy').pow(2) ).sqrt().alias('distance'),
            pl.arctan2( pl.col('_dy'), pl.col('_dx') ).alias('direction'),
        ).drop(['_dx', '_dy'])
        
        keep = [c for c in df.columns
                if c in self.COLUMNS['SPOTS'] 
                or c in grouping_cols]
        df = df.select(keep)

        if kwargs.get('enriched', False):
            df = self._enrich_spots(df, **kwargs)

        if self.significant_figures:
            df = self.signify(df)
        if self.decimal_places:
            df = self.norm_decimals(df)

        return df

    
    def tracks(
        self,
        df: pl.DataFrame,
        subset: Optional[list[str]] = None,
        **kwargs
    ) -> pl.DataFrame:
        """
        Computes per-trajectory metrics.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame <- the result DataFrame of the :meth:`spots` method.
        subset : list[str], optional
            List of metric names to compute. If None, all available metrics
            are computed. Metric names are the base names below -- a metric
            may expand to several output columns (e.g. `'speed'` ->
            `speed_min`, `speed_max`, `speed_mean`, `speed_sd`,
            `speed_median`; `'direction'` -> `direction_mean`,
            `direction_var`).
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
            - `directionality`: Ratio of track displacement to track length (a.k.a. straightness/confinement ratio).
            - `speed_min`: Minimum speed along the track.
            - `speed_max`: Maximum speed along the track.
            - `speed_mean`: Mean speed along the track.
            - `speed_sd`: Standard deviation of the speed along the track.
            - `speed_median`: Median speed along the track.
            - `mean_straight_line_speed`: Track displacement divided by track duration.
            - `forward_progression_linearity`: Mean straight line speed divided by mean speed. Measures how linearly the track progresses forward.
            - `greatest_distance`: Largest Euclidean distance between any two points of the track (its maximum span / "diameter").
            - `straightness`: Ratio of `greatest_distance` to track length.
            - `track_start_frame`: Frame at which the track starts.
            - `track_end_frame`: Frame at which the track ends.
            - `track_points`: Number of points in the track.
            - `direction_mean`: Mean direction of movement along the track.
            - `direction_var`: Variance of the direction of movement along the track.
            - `mean_directional_change`: Mean change in direction between consecutive points along the track.
            - `mean_directional_change_rate`: Mean directional change divided by the time interval. Mean rate of change in direction along the track.
        
        .
        """
        if is_empty(df):
            warn("Input DataFrame is empty. No computation performed.")
            return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['TRACKS']})
        
        df = ensure_polars(df)
        df = df.clone()

        cat_cols = [col for col in self.DEFAULT_CATEGORIES if col in df.columns]

        df = self.assign_track_uid(df)
        df = df.sort(['track_uid', 'time_point'])

        timeinterval = self._resolve_timeinterval(df, metadata = kwargs.get('metadata', None))

        # Derive cumulative per-spot metrics needed by the aggregations
        df = self._enrich_spots(df, timeinterval = timeinterval, **kwargs)

        # Stash categorical identifiers to merge them back into the result
        stash = [c for c in cat_cols if c != 'track_uid']
        stash = df.select(['track_uid'] + stash).unique(subset=['track_uid'], keep='first')

        requested = self._tracks_registry.resolve(subset)

        context = {
            'data_source': df,
            'group_by': ['track_uid'],
            'stat_settings': self._stats_configuration(),
            'timeinterval': timeinterval,
        }
        agg = self._tracks_registry.compute(requested, context)

        # Carry over track_id (first per track)
        original_ids = df.group_by('track_uid', maintain_order=True).agg([pl.col('track_id').first()])
        agg = original_ids.join(agg, on='track_uid', how='right')

        out = stash.join(agg, on='track_uid', how='right')

        # Drop spot-level columns that leaked through
        keep = [c for c in out.columns 
                if c in self.COLUMNS['TRACKS']
                or c in cat_cols]
        
        out = out.select(keep).unique(maintain_order=True)

        if self.significant_figures:
            out = self.signify(out)
        if self.decimal_places:
            out = self.norm_decimals(out)

        return out


    def timepoints(
        self,
        df: pl.DataFrame,
        subset: Optional[list[str]] = None,
        *,
        grouping_level: Literal['highest', 'lowest'] | str | int | list = 'highest',
        **kwargs
    ) -> pl.DataFrame:
        """
        Computes time point statistics for categories (groups).

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame <- the result DataFrame of the :meth:`spots` method.
        subset : list[str], optional
            List of metric names to compute. If None, all available metrics
            are computed. Valid names are the base metrics (`'cum_track_length'`,
            `'cum_track_displacement'`, `'cum_directionality'`,
            `'cum_speed_mean'`, `'instantaneous_speed'`,
            `'cum_mean_straight_line_speed'`, `'cum_forward_progression_linearity'`,
            `'cum_sum_directional_change'`, `'cum_mean_directional_change'`,
            `'tracks_contributing'`, `'instantaneous_direction'`, `'cum_direction'`).
            Each distribution metric expands into its descriptive statistics
            (`_min`, `_max`, `_mean`, `_median`, `_sd`); error statistics
            (`_sem`, `_ciXX_low` / `_ciXX_high`) are controlled by the keyword
            arguments below -- do not request them by name.
        grouping_level : Literal['highest', 'lowest'] | str | int | list, default='highest'
            Level(s) at which to group the data for computing time point statistics.
        **kwargs
            Additional keyword arguments passed to the computation functions.

        Returns
        -------
        pl.DataFrame
            - `time_point`: Time point of the observation.
            - `frame`: Frame number of the observation.

            followed by the descriptive (and, if enabled, error) statistics for the metrics
            - `cum_track_length`: Cumulative track length up to the current time point.
            - `cum_track_displacement`: Cumulative track displacement up to the current time point.
            - `cum_directionality`: Cumulative straightness ratio up to the current time point.
            - `cum_speed_mean`: Cumulative mean speed up to the current time point.
            - `instantaneous_speed`: Instantaneous speed at the current time point.
            - `cum_mean_straight_line_speed`: Cumulative mean straight line speed up to the current time point.
            - `cum_forward_progression_linearity`: Cumulative forward progression linearity up to the current time point.
            - `cum_sum_directional_change`: Cumulative sum of directional changes up to the current time point.
            - `cum_mean_directional_change`: Cumulative mean of directional changes up to the current time point.

        One `group_by().agg()` pass per grouping level; all descriptive,
        error and circular statistics are polars expressions.
        """
        if is_empty(df):
            warn("Input DataFrame is empty. Returning an empty DataFrame with the expected schema.")
            return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['TIMEPOINTS']})
        if df['time_point'].n_unique() < 2:
            warn("Not enough time points available for time interval statistics computations. Returning an empty schema DataFrame.")
            return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['TIMEPOINTS']})

        stat_settings = self._stats_configuration(
            inferative_error    = kwargs.get('inferative_error', None),
            bootstrap_ci        = kwargs.get('bootstrap_ci', None),
            ci_confidence       = kwargs.get('ci_confidence', None),
            bootstrap_resamples = kwargs.get('bootstrap_resamples', None),
            bootstrap_ci_method = kwargs.get('bootstrap_ci_method', None),
            ci_statistic        = kwargs.get('ci_statistic', None),
        )

        df = self.assign_track_uid(df)
        df = self._enrich_spots(df, **kwargs)

        grouping_set = []

        if isinstance(grouping_level, list):
            for g in grouping_level:
                grouping_set.append(self._get_grouping_level(df.columns, g, exclude='track_uid', include=['time_point', 'frame']))
        else:
            grouping_cols = self._get_grouping_level(df.columns, grouping_level, exclude='track_uid', include=['time_point', 'frame'])
            grouping_set = [grouping_cols]

        requested = self._timepoints_registry.resolve(subset)

        level_frames = []
        for group_cols in grouping_set:
            group_lvl = group_cols[0]

            context = {
                'data_source': df,
                'group_by': group_cols,
                'stat_settings': stat_settings,
            }
            level_df = self._timepoints_registry.compute(requested, context)

            level_df = level_df.with_columns(
                pl.lit(group_lvl).alias('grouping_level')
            )

            level_frames.append(level_df)

        if not level_frames:
            warn("No level frames were generated.")
            return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['TIMEPOINTS']})

        out = pl.concat(level_frames, how='diagonal_relaxed')

        # JSON-safe cleanup (no Inf in strict JSON)
        out = out.with_columns([
            pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
            for c, dt in out.schema.items() if dt in (pl.Float32, pl.Float64)
        ])

        if self.significant_figures:
            out = self.signify(out)
        if self.decimal_places:
            out = self.norm_decimals(out)

        return out

    
    def timelags(
        self,
        df: pl.DataFrame,
        subset: Optional[list[str]] = None,
        *,
        grouping_level: Literal['highest', 'lowest'] | str | int | list | None = 'highest',
        **kwargs
    ) -> pl.DataFrame:
        """
        Computes per-time-interval statistics.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame <- the result DataFrame of the :meth:`spots` method.
        subset : list[str], optional
            List of metric names to compute. If None, all available metrics
            are computed. Valid names are `'MSD'`, `'tracks_contributing'`,
            `'position_pairs_contributing'` and `'directional_change'`.
            `'MSD'` expands into `MSD`, `MSD_min`, `MSD_max`, `MSD_sd`; error
            statistics (`MSD_sem`, `MSD_ciXX_low` / `MSD_ciXX_high`) are
            controlled by the keyword arguments below -- do not request them
            by name. `'directional_change'` expands into
            `directional_change_mean` and `directional_change_var`.
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
            - `MSD_sem`: Standard error of the mean squared displacement for the given time lag (with `inferative_error`).
            - `MSD_ciXX_low`: Lower bound of the confidence interval for the mean squared displacement for the given time lag, where `XX` represents the confidence level (with `bootstrap_ci`).
            - `MSD_ciXX_high`: Upper bound of the confidence interval for the mean squared displacement for the given time lag, where `XX` represents the confidence level (with `bootstrap_ci`).
            - `MSD_min`: Minimum value of the mean squared displacement for the given time lag.
            - `MSD_max`: Maximum value of the mean squared displacement for the given time lag.
            - `tracks_contributing`: Number of tracks contributing to the given time lag.
            - `position_pairs_contributing`: Number of position pairs contributing to the given time lag.
            - `directional_change_mean`: Mean directional change for the given time lag.
            - `directional_change_var`: Variance of the directional change for the given time lag.
        
        Pair-building stays numpy-vectorized, all aggregation runs through a
        single polars `group_by().agg()` per grouping level.
        """

        if is_empty(df):
            warn("Input DataFrame is empty. Returning an empty DataFrame with the expected schema.")
            return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['TIMELAGS']})

        stat_settings = self._stats_configuration(
            inferative_error    = kwargs.get('inferative_error', None),
            bootstrap_ci        = kwargs.get('bootstrap_ci', None),
            ci_confidence       = kwargs.get('ci_confidence', None),
            bootstrap_resamples = kwargs.get('bootstrap_resamples', None),
            bootstrap_ci_method = kwargs.get('bootstrap_ci_method', None),
            ci_statistic        = kwargs.get('ci_statistic', None),
        )

        grouping_set = []

        if isinstance(grouping_level, list):
            for g in grouping_level:
                grouping_set.append(self._get_grouping_level(df.columns, g, exclude='track_uid'))
        else:
            grouping_cols = self._get_grouping_level(df.columns, grouping_level, exclude='track_uid')
            grouping_set = [grouping_cols]

        df = self.assign_track_uid(df)

        # Unique time points
        if df['time_point'].n_unique() < 2:
            warn("Not enough time points available for time interval statistics computations. Returning an empty schema DataFrame.")
            return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['TIMELAGS']})

        timeinterval = self._resolve_timeinterval(df, metadata = kwargs.get('metadata', None))

        requested = self._timelags_registry.resolve(subset)

        def _compute_level(source: pl.DataFrame, grouping_cols: str) -> pl.DataFrame:
            """ Compute time-interval stats for a single grouping level. """

            temp = (
                source.select(   
                    grouping_cols + ['track_uid', 'time_point', 'x_coordinate', 'y_coordinate']  # Select needed columns.
                ).with_columns(  
                  ( pl.col('time_point').rank('dense').over('track_uid') - 1 ).cast(pl.Int64).alias('_frame'),  # Assign frame indexes within each track.
                    pl.len().over('track_uid').alias('_size')  # Compute the size of each track.
                ).filter(
                    pl.col('_size') >= 2  # Only keep tracks with at least 2 frames.
                ).with_columns(
                    pl.arctan2(
                      ( pl.col('y_coordinate') - pl.col('y_coordinate').shift(1) ).over('track_uid'),
                      ( pl.col('x_coordinate') - pl.col('x_coordinate').shift(1) ).over('track_uid')
                    ).alias('_theta')  # Radial difference (angle) between consecutive frames.
                )
            )

            if is_empty(temp):
                return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['TIMELAGS']})

            max_lag = int(temp['_frame'].max())
            if max_lag < 1:
                return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['TIMELAGS']})

            uid_arr   = temp['track_uid'].to_numpy()
            frame_arr = temp['_frame'].to_numpy()
            x_arr     = temp['x_coordinate'].to_numpy()
            y_arr     = temp['y_coordinate'].to_numpy()
            theta_arr = temp['_theta'].to_numpy()
            cat_arrs  = {col: temp[col].to_numpy() for col in grouping_cols}

            # (track_uid, frame) -> row lookup so a lag pairs points exactly
            # `lag` frames apart (robust to gaps / non-uniform spacing).
            pos_of = {k: i for i, k in enumerate(zip(uid_arr.tolist(), frame_arr.tolist()))}

            msd_records: List[pl.DataFrame] = []
            turn_records: List[pl.DataFrame] = []

            for lag in range(1, max_lag + 1):

                partner_pos = np.fromiter(
                  ( pos_of.get(k, -1) 
                      for k in zip(uid_arr.tolist(), (frame_arr + lag).tolist()) ), 
                    dtype=np.int64, 
                    count=len(uid_arr)
                )

                valid_mask = partner_pos >= 0
                if not valid_mask.any():
                    continue

                valid_idx   = np.where(valid_mask)[0]
                partner_idx = partner_pos[valid_idx]

                dx = x_arr[partner_idx] - x_arr[valid_idx]
                dy = y_arr[partner_idx] - y_arr[valid_idx]

                msd_records.append(pl.DataFrame({
                    'track_uid': uid_arr[valid_idx],
                    **{col: cat_arrs[col][valid_idx] for col in grouping_cols},
                    'sq_disp':   np.hypot(dx, dy) ** 2,
                    'frame_lag': np.full(valid_idx.size, lag, dtype=np.int64),
                    'time_lag':  np.full(valid_idx.size, lag * timeinterval, dtype=np.float64),
                }))

                # Turning angle: theta defined only where a preceding step exists
                theta_now = theta_arr[valid_idx]
                theta_par = theta_arr[partner_idx]
                turn_valid = (frame_arr[valid_idx] >= 1) & np.isfinite(theta_now) & np.isfinite(theta_par)

                if not turn_valid.any():
                    continue

                ti = valid_idx[turn_valid]
                pi = partner_idx[turn_valid]
                dtheta = theta_arr[pi] - theta_arr[ti]
                dtheta = self.wrap_pi(dtheta)
                turn_records.append(pl.DataFrame({
                    'track_uid': uid_arr[ti],
                    **{col: cat_arrs[col][ti] for col in grouping_cols},
                    'dtheta':    dtheta,
                    'frame_lag': np.full(ti.size, lag, dtype=np.int64),
                    'time_lag':  np.full(ti.size, lag * timeinterval, dtype=np.float64),
                }))

            if not msd_records:
                return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['TIMELAGS']})

            all_msd = pl.concat(msd_records, how='vertical_relaxed')
            all_turn = (
                pl.concat(turn_records, how='vertical_relaxed')
                if turn_records
                else pl.DataFrame(schema={**{col: all_msd.schema[col] for col in grouping_cols},
                                          'track_uid': all_msd.schema['track_uid'],
                                          'dtheta': pl.Float64,
                                          'frame_lag': pl.Int64, 'time_lag': pl.Float64})
            )

            lag_group_cols = grouping_cols + ['frame_lag', 'time_lag']
            context = {
                'data_source': all_msd,
                'group_by': lag_group_cols,
                'stat_settings': stat_settings,
                'turn_src': all_turn,
            }

            lags = self._timelags_registry.compute(requested, context)

            # Drop columns that produced no data
            return self._drop_all_null_columns(lags) if not is_empty(lags) else lags

        level_frames = []

        for group_cols in grouping_set:
            group_lvl = group_cols[0]

            level_df = _compute_level(df, group_cols)

            if is_empty(level_df):
                continue

            level_df = level_df.with_columns(
                pl.lit(group_lvl).alias('grouping_level')
            )

            level_frames.append(level_df)

        if not level_frames:
            return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['TIMELAGS']})

        out = pl.concat(level_frames, how='diagonal_relaxed')

        # JSON-safe cleanup
        out = out.with_columns([
            pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
            for c, dt in out.schema.items() if dt in (pl.Float32, pl.Float64)
        ])

        if self.significant_figures:
            out = self.signify(out)
        if self.decimal_places:
            out = self.norm_decimals(out)

        return out


    # Spots enrichment (internal, used by `tracks` and `timepoints`)
    # -----------------------------------------------------------------------

    def _enrich_spots(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """
        Derive cumulative per-trajectory-point metrics from the base columns of the :meth:`spots` result.
        
        Used internally by `tracks` and `timepoints`. Idempotent: returns the DataFrame unchanged if the derived columns are already present.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame containing trajectory points.
        **kwargs
            Additional keyword arguments, including:
            - `timeinterval`: Time step between consecutive frames.

        Returns
        -------
        pl.DataFrame
            - `cum_track_length`: Cumulative track length for each trajectory point.
            - `cum_track_displacement`: Cumulative track displacement for each trajectory point.
            - `cum_directionality`: Ratio of cumulative displacement to cumulative track length.
            - `cum_speed_mean`: Mean cumulative speed for each trajectory point.
            - `cum_mean_straight_line_speed`: Mean straight-line speed for each trajectory point.
            - `cum_forward_progression_linearity`: Linearity of forward progression for each trajectory point.

        Returns
        -------
        pl.DataFrame
            DataFrame enriched with cumulative per-trajectory-point metrics.
        """
        # Guard if spots already were enriched (useful when used by the timepoints and timelags methods)
        if 'cum_track_length' in df.columns:
            return df

        uid = 'track_uid'
        timeinterval = kwargs.get('timeinterval', self._resolve_timeinterval(df, metadata = kwargs.get('metadata', None)))

        df = df.sort([uid, 'time_point'])
        elapsed = (pl.col('time_point') - pl.col('time_point').first().over(uid))
        
        x = df['x_coordinate'].to_numpy()
        y = df['y_coordinate'].to_numpy()
        lengths = df.group_by(uid, maintain_order=True).len()['len'].to_numpy()

        cum_gd = np.empty(len(df))
        start = 0
        for n in lengths:
            cum_gd[start:start + n] = self._cum_greatest_distance(x[start:start + n], y[start:start + n])
            start += n

        # cum_track_length, cum_track_displacement, cum_track_displacement
        df = df.with_columns(
            pl.col('time_point').cum_count().over(uid).cast(pl.Float64).alias('_cumcount'),
            pl.col('distance').cum_sum().over(uid).alias('cum_track_length'), 
            (   (pl.col('x_coordinate') - pl.col('x_coordinate').first().over(uid)).pow(2) 
              + (pl.col('y_coordinate') - pl.col('y_coordinate').first().over(uid)).pow(2)
            ).sqrt().alias('cum_track_displacement')
        ).with_columns(
            pl.when(
                pl.col('cum_track_displacement') == 0
            ).then(None).otherwise(
                pl.col('cum_track_displacement')
            ).alias('cum_track_displacement')
        )

        # cum_greatest_distance
        df = df.with_columns(pl.Series('cum_greatest_distance', cum_gd, dtype=pl.Float64))

        # cum_straightness, cum_directionality, cum_speed_mean, cum_mean_straight_line_speed, cum_forward_progression_linearity
        df = df.with_columns(
            (   pl.col('cum_greatest_distance')
                / pl.when(pl.col('cum_track_length') == 0).then(None).otherwise(pl.col('cum_track_length'))
            ).alias('cum_straightness'),
            (   pl.col('cum_track_displacement')
                / pl.when(pl.col('cum_track_length') == 0).then(None).otherwise(pl.col('cum_track_length'))
            ).alias('cum_directionality'),
            (   pl.col('cum_track_length')
                / pl.when(elapsed == 0).then(None).otherwise(elapsed)
            ).alias('cum_speed_mean')
        ).with_columns(
            (   pl.col('cum_track_displacement') 
                / (pl.col('_cumcount') * timeinterval)
            ).alias('cum_mean_straight_line_speed'),
        ).with_columns(
            (   pl.col('cum_mean_straight_line_speed') 
                / pl.col('cum_speed_mean')
            ).alias('cum_forward_progression_linearity'),
        )
        # directional_change, cum_mean_directional_change, cum_sum_directional_change, cum_mean_directional_change_rate
        df = df.with_columns((((
                        pl.col('direction') - pl.col('direction').shift(1)
                    ).over(uid) + np.pi
                ).mod(2 * np.pi) - np.pi
            ).abs().degrees().alias('directional_change')
        ).with_columns(
            pl.col('directional_change').cum_sum().over(uid).alias('cum_sum_directional_change'),
            pl.col('directional_change').is_not_null().cum_sum().over(uid).cast(pl.Float64).alias('_valid_count'),
        ).with_columns(
            (   pl.col('cum_sum_directional_change')
                / pl.when(pl.col('_valid_count') == 0).then(None).otherwise(pl.col('_valid_count'))
            ).alias('cum_mean_directional_change')
        ).with_columns(
            pl.when(
                pl.col('directional_change').is_null()
            ).then(None).otherwise(
                pl.col('cum_mean_directional_change')
            ).alias('cum_mean_directional_change'),
        ).with_columns(
            (   pl.col('cum_mean_directional_change') 
                / (pl.col('_cumcount') * timeinterval)
            ).alias('cum_mean_directional_change_rate'),
        )
        # Ccum_direction_var
        df = df.with_columns(
            pl.col('direction').sin().cum_sum().over(uid).alias('_cum_sin'),
            pl.col('direction').cos().cum_sum().over(uid).alias('_cum_cos'),
            (pl.col('_cumcount') - 1).alias('_n_angles'),
        ).with_columns(
            pl.arctan2(pl.col('_cum_sin'), pl.col('_cum_cos')).alias('cum_direction_mean'),
            (   1.0 - (pl.col('_cum_sin').pow(2) + pl.col('_cum_cos').pow(2)).sqrt()
                / pl.when(pl.col('_n_angles') == 0).then(None).otherwise(pl.col('_n_angles'))
            ).alias('cum_direction_var'),
        ).with_columns(
            pl.when(
                pl.col('_n_angles') <= 1
            ).then(None).otherwise(
                pl.col('cum_direction_var')
            ).alias('cum_direction_var'),
        )

        return df.drop(['_cumcount', '_valid_count', '_cum_sin', '_cum_cos', '_n_angles'])


    
    # Metrics registry builders
    # -----------------------------------------------------------------------

    def _distribution_metric(self, src: str, out: str) -> Callable[[dict], Dict[str, Any]]:
        """
        Builder for a distribution-like metric `out` computed from column `src`.

        Always emits the descriptive statistics (cheap, single-pass polars
        aggregations): `_min`, `_max`, `_mean`, `_median`, `_sd`.

        Depending on the per-call options (`context['stat_settings']`):
          - `_sem` when `inferative_error` is enabled,
          - bootstrap `_ciXX_low` / `_ciXX_high` when `bootstrap_ci` is enabled.
        """
        def _build(context: dict) -> Dict[str, Any]:
            stat_settings: StatSettings = context['stat_settings']
            cols: Dict[str, Any] = {
                f'{out}_min':    pl.col(src).min(),
                f'{out}_max':    pl.col(src).max(),
                f'{out}_mean':   pl.col(src).mean(),
                f'{out}_median': pl.col(src).median(),
                f'{out}_sd':     pl.col(src).std(),
            }
            if stat_settings.inferative_error:
                cols[f'{out}_sem'] = self.pl_expr_sem(src)
            if stat_settings.bootstrap_ci:
                cols[f'{out}_ci'] = self._bootstrap_ci_post(context, src=src, out=out)
            return cols
        return _build

    def _bootstrap_ci_post(
        self,
        context: dict,
        *,
        src: str,
        out: str,
        statistic: Optional[Callable[[np.ndarray], float]] = None,
        **kwargs
    ) -> Callable:
        """
        Schedule a per-group bootstrap confidence interval of `src`, emitted
        as the `{out}_ci<level>_low` / `{out}_ci<level>_high` columns.

        The raw group values are collected as a temporary list column during
        the single aggregation pass; the bootstrap itself then runs once per
        group as a post-processing step.
        """
        hide_ci_statistic = kwargs.get('hide_ci_statistic', False)
        helper = f'__list_{out}_ci'
        context.setdefault('extra_exprs', {})[helper] = pl.col(src)

        def _post(out_df: pl.DataFrame, context: dict) -> pl.DataFrame:
            stat_settings: StatSettings = context['stat_settings']
            bounds = [
                self.ci(
                    np.asarray(values, dtype=float),
                    ci_confidence=stat_settings.ci_confidence,
                    bootstrap_resamples=stat_settings.bootstrap_resamples,
                    bootstrap_ci_method=stat_settings.bootstrap_ci_method,
                    ci_statistic=statistic if statistic is not None else stat_settings.ci_statistic,
                )
                for values in out_df[helper].to_list()
            ]
            return out_df.with_columns(
                pl.Series(f'{out}_{stat_settings.ci_label(hide_ci_statistic)}_low',  [b[0] for b in bounds], dtype=pl.Float64),
                pl.Series(f'{out}_{stat_settings.ci_label(hide_ci_statistic)}_high', [b[1] for b in bounds], dtype=pl.Float64),
            )
        return _post


    @staticmethod
    def _cum_greatest_distance(x: np.ndarray, y: np.ndarray, block: int = 1024) -> np.ndarray:
        """Greatest pairwise distance among points 0..k, for every k (NaN for k=0)."""
        n = len(x)
        out = np.full(n, np.nan)
        if n < 2:
            return out

        pts = np.column_stack((x, y))
        far = np.full(n, -np.inf)          # farthest distance from point k to any earlier point

        for s in range(1, n, block):
            e = min(s + block, n)
            d = cdist(pts[s:e], pts[:e - 1])                         # (rows, e-1)
            earlier = np.arange(e - 1)[None, :] < np.arange(s, e)[:, None]   # only i < k
            d = np.where(earlier & np.isfinite(d), d, -np.inf)       # ignore later points / NaN coords
            far[s:e] = d.max(axis=1)

        out = np.maximum.accumulate(far)
        out[~np.isfinite(out)] = np.nan
        return out


    def _build_tracks_registry(self) -> MetricRegistry:
        """
        Per-trajectory metrics registry.

        Representions of all the trajectory metrics are either polars aggregation expressions or post-processing functions.

        `context['timeinterval']` -> the resolved time step
        """ 

        reg = MetricRegistry()  # get a metric registry instance

        def speed(context: dict) -> Dict[str, Any]:
            timeinterval = context['timeinterval']
            return {'speed_min':    pl.col('distance').min()    / timeinterval,
                    'speed_max':    pl.col('distance').max()    / timeinterval,
                    'speed_mean':   pl.col('distance').mean()   / timeinterval,
                    'speed_sd':     pl.col('distance').std()    / timeinterval,
                    'speed_median': pl.col('distance').median() / timeinterval}

        reg.add_column('track_length', lambda context: pl.col('distance').sum())
        reg.add_column('track_displacement', lambda context: pl.col('cum_track_displacement').last())
        reg.add_column('greatest_distance', lambda context: pl.col('cum_greatest_distance').last())

        reg.add_column('mean_straight_line_speed', lambda context: pl.col('cum_mean_straight_line_speed').last())
        reg.add_column('forward_progression_linearity', lambda context: pl.col('cum_forward_progression_linearity').last())

        reg.add_derived(
            'directionality', 
            ['track_displacement', 'track_length'], 
            lambda d: pl.col('track_displacement') / pl.col('track_length')
        )
        reg.add_derived(
            'straightness', 
            ['greatest_distance', 'track_length'],
            lambda d: pl.col('greatest_distance') / pl.col('track_length')
        )

        reg.add_column('direction_mean', lambda context: pl.col('cum_direction_mean').last())
        reg.add_column('direction_var',  lambda context: pl.col('cum_direction_var').last())

        reg.add_column('mean_directional_change',      lambda context: pl.col('cum_mean_directional_change').last())
        reg.add_column('mean_directional_change_rate', lambda context: pl.col('cum_mean_directional_change_rate').last())

        reg.add_column('x_location', lambda context: pl.col('x_coordinate').mean())
        reg.add_column('y_location', lambda context: pl.col('y_coordinate').mean())

        reg.add_column('track_points', lambda context: pl.len())

        reg.add_column('track_start_frame', lambda context: pl.col('frame').min())
        reg.add_column('track_end_frame',   lambda context: pl.col('frame').max())

        reg.add('speed', speed)
        
        return reg
    

    def _build_timepoints_registry(self) -> MetricRegistry:
        """Per-time-point metrics, grouped by category."""
        reg = MetricRegistry()

        # metric name (user-facing) -> source column it is computed from
        metrics = {
            'cum_track_length':                  'cum_track_length',
            'cum_track_displacement':            'cum_track_displacement',
            'cum_greatest_distance':             'cum_greatest_distance',
            'cum_straightness':                  'cum_straightness',
            'cum_directionality':                'cum_directionality',
            'cum_speed_mean':                    'cum_speed_mean',
            'instantaneous_speed':               'distance',
            'cum_mean_straight_line_speed':      'cum_mean_straight_line_speed',
            'cum_forward_progression_linearity': 'cum_forward_progression_linearity',
            'cum_sum_directional_change':        'cum_sum_directional_change',
            'cum_mean_directional_change':       'cum_mean_directional_change',
        }
        for metric, src in metrics.items():
            reg.add(metric, self._distribution_metric(src, metric))

        reg.add_column('tracks_contributing', lambda context: pl.col('track_uid').n_unique().cast(pl.Int64))

        # --- Circular statistics (fully expression-based) -------------------
        reg.add('instantaneous_direction', lambda context: {
            'instantaneous_direction_mean': self.pl_expr_circ_mean('direction'),
            'instantaneous_direction_var':  self.pl_expr_circ_var('direction'),
        })
        reg.add('cum_direction', lambda context: {
            'cum_direction_mean': self.pl_expr_circ_mean('cum_direction_mean'),
            'cum_direction_var':  self.pl_expr_circ_var('cum_direction_mean'),
        })

        return reg
    

    def _build_timelags_registry(self) -> MetricRegistry:
        """Per-time-lag metrics."""
        reg = MetricRegistry()

        def msd(context: dict) -> Dict[str, Any]:
            stat_settings: StatSettings = context['stat_settings']
            cols: Dict[str, Any] = {
                'MSD':     pl.col('sq_disp').mean(),
                'MSD_min': pl.col('sq_disp').min(),
                'MSD_max': pl.col('sq_disp').max(),
                'MSD_sd':  pl.col('sq_disp').std(),
            }
            if stat_settings.inferative_error:
                cols['MSD_sem'] = self.pl_expr_sem('sq_disp')
            if stat_settings.bootstrap_ci:
                # The MSD is a mean, so its CI is always a CI of the mean.
                cols['MSD_ci'] = self._bootstrap_ci_post(context, src='sq_disp', out='MSD', statistic=np.mean, hide_ci_statistic=True)
            return cols
        reg.add('MSD', msd)

        reg.add_column('tracks_contributing',         lambda context: pl.col('track_uid').n_unique().cast(pl.Int64))
        reg.add_column('position_pairs_contributing', lambda context: pl.len().cast(pl.Int64))

        def directional_change(context: dict) -> Dict[str, Any]:
            """Circular mean/variance of the turning angles (`context['turn_src']`),
            joined onto the lag table as a post-processing step."""
            def _post(out_df: pl.DataFrame, _ctx: dict) -> pl.DataFrame:
                turn_src: pl.DataFrame = _ctx['turn_src']
                if is_empty(turn_src):
                    return out_df
                stat = (
                    turn_src
                    .group_by(_ctx['group_by'], maintain_order=True)
                    .agg(
                        self.pl_expr_circ_mean('dtheta').alias('directional_change_mean'),
                        self.pl_expr_circ_var('dtheta').alias('directional_change_var'),
                    )
                )
                return out_df.join(stat, on=_ctx['group_by'], how='left')
            return {'directional_change': _post}
        reg.add('directional_change', directional_change)

        return reg



    # DataFrame handling
    # -----------------------------------------------------------------------

    @staticmethod
    def _drop_all_null_columns(df: pl.DataFrame) -> pl.DataFrame:
        keep = [c for c in df.columns if df[c].null_count() < df.height]
        return df.select(keep)
    
    def _capture_metadata(self, df) -> pl.DataFrame:
        """ Carry over an Input wrapper's `.metadata` onto `self.metadata` (instance),
            `Calc.metadata` and, if applicable, `DataObject.metadata`, then return the DataFrame. """

        try:
            meta = df.metadata.get()
            captured = meta.copy() if meta is not None else None
            captured.pop('columns', None)

        except Exception:
            captured = None

        # instance attribute (covers DataObject instances via inheritance)
        self.metadata = captured

        # class-level "constants"
        Calc.metadata = captured
        type(self).metadata = captured  # e.g. DataObject.metadata when called on a DataObject

        return df

    def _guard_df(self, df) -> pl.DataFrame:
        """ Ensure the DataFrame is a polars DataFrame and capture its metadata. """
        return ensure_polars(self._capture_metadata(df))


    # Time resolution, track UID assignment and grouping helpers
    # -----------------------------------------------------------------------

    def _resolve_timeinterval(self, df: pl.DataFrame, *, metadata: Optional[dict] = None) -> float:
        """Resolve the time step from data (or self.timeinterval if set)."""
        if metadata is not None:
            return metadata['timeinterval']
        
        if self.metadata is not None:
            return self.metadata['timeinterval']

        timeintervals = np.diff(np.sort(df['time_point'].unique().to_numpy()))

        if timeintervals.size == 0:
            return 1.0
        if np.all(timeintervals == timeintervals[0]):
            return float(timeintervals[0])

        timeinterval = float(np.median(timeintervals))
        # warnings.warn(
        #     message=(f"Time points are not uniformly spaced -> this will most probably lead to "
        #                 f"incorrect data computation.\nObserved time steps:\n{timeintervals}\nUsing: {timeinterval}"),
        #     category=TimePointWarning,
        #     stacklevel=3,
        # )
        return timeinterval
    

    def assign_track_uid(self, df: pl.DataFrame) -> pl.DataFrame:
        """ Creates a unique track identifier `track_uid` by combining the category columns present 
            in the DataFrame with `track_id` -> each existing track gets its own unique identifier. """

        if 'track_uid' in df.columns:
            return df

        grouping_cols = [c for c in self.DEFAULT_CATEGORIES
                         if c in df.columns and c != 'track_uid']

        if 'track_id' in df.columns:
            grouping_cols = grouping_cols + ['track_id']

        if not grouping_cols:
            warn("No grouping columns found for 'track_uid' assignment. 'track_uid' assignment will probably fail.")

        keys = (
            df.select(grouping_cols)
              .unique(maintain_order=True)
              .with_row_index('track_uid', 1)
              .with_columns(pl.col('track_uid').cast(pl.Int64))
        )
        return df.join(keys, on=grouping_cols, how='left')


    def _get_grouping_level(
        self,
        df_cols: list[str],
        grouping_level: Literal['highest', 'lowest'] | str | int | list = 'highest',
        *,
        include: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
    ) -> list[str]:
        """ 
        Determine the appropriate grouping columns based on the specified grouping level, inclusion, and exclusion criteria. 
        
        Parameters
        ----------
        df_cols : list
            The columns of the DataFrame to consider for grouping.
        grouping_level : Literal['highest', 'lowest'] | str | int | None, optional
            The desired grouping level. Can be 'highest', 'lowest', a specific column name, an integer index, a list of levels, or None. Default is 'highest'.
        include : str | list[str] | None, optional
            Columns to explicitly include in the grouping, even if not part of the default categories. Default is None.
        exclude : str | list[str] | None, optional
            Columns to explicitly exclude from the grouping. Default is None.

        Returns
        -------
        list[str]
            The determined grouping columns based on the specified criteria.
        
        """

        if not isinstance(include, list):
            include = [include] if include is not None else []
        if not isinstance(exclude, list):
            exclude = [exclude] if exclude is not None else []

        cat_group_cols = [col for col in self.DEFAULT_CATEGORIES if col in df_cols]

        if not is_empty(exclude):
            cat_group_cols = [col for col in cat_group_cols if col not in exclude]
            if len(cat_group_cols) == 0:
                # raise ColumnsNotFoundError(f"All grouping columns have been excluded making data grouping impossible.")
                pass

        if is_empty(cat_group_cols):
            # raise ColumnsNotFoundError(f"No grouping columns found in DataFrame columns: {df_cols}")
            pass

        if isinstance(grouping_level, int):
            if grouping_level < 0 or grouping_level >= len(cat_group_cols):
                raise IndexError(f"Grouping level index {grouping_level} is out of bounds for DataFrame columns: {cat_group_cols}")
            cat_group_cols = cat_group_cols[grouping_level:]
        elif grouping_level == 'highest':
            cat_group_cols = [cat_group_cols[-1]]
        elif grouping_level == 'lowest':
            cat_group_cols = cat_group_cols[1:]
        
        elif isinstance(grouping_level, str):
            idx = cat_group_cols.index(grouping_level)
            cat_group_cols = cat_group_cols[idx:]
        elif not isinstance(grouping_level, list):
            raise InvalidParameterValueError(f"Invalid grouping_level parameter: {grouping_level}. Must be a list of column names, an integer index, 'highest', 'lowest', or None.")

        if isinstance(include, str):
            include = [include]
        for col in include:
            if col not in cat_group_cols:
                cat_group_cols.append(col)

        return cat_group_cols


    # Formatting
    # -----------------------------------------------------------------------

    def format_digits(self, df: pl.DataFrame, *, sig_figs: int = None, decimals: int = None) -> pl.DataFrame:
        """ Formats numeric values according to significant figures / decimals. """
        if sig_figs:
            df = self.signify(df, sig_figs=sig_figs)
        if decimals:
            df = self.norm_decimals(df, decimals=decimals)
        return df


    def signify(self, df: pl.DataFrame, *, sig_figs: int = None) -> pl.DataFrame:
        """ Round numeric values to a number of significant figures. """
        if is_empty(df):
            return df
        return df

        if sig_figs is None:
            sig_figs = self.significant_figures

        # valuer = Values()
        num_cols = [c for c, dt in df.schema.items() if dt.is_numeric()]
        return df.with_columns([
            pl.col(c).map_elements(
                lambda x: valuer.RoundSigFigs(x, sigfigs=sig_figs),
                return_dtype=pl.Float64,
            ).alias(c)
            for c in num_cols
        ])


    def norm_decimals(self, df: pl.DataFrame, decimals: int = None) -> pl.DataFrame:
        """ Normalize decimal places across numeric columns. """

        if is_empty(df):
            return df

        if decimals is None:
            decimals = self.decimal_places

        float_cols = [c for c, dt in df.schema.items() if dt in (pl.Float32, pl.Float64)]
        return df.with_columns([pl.col(c).round(decimals).alias(c) for c in float_cols])


    
    # Aggregation functions
    # -----------------------------------------------------------------------


    def pl_expr_sem(self, col: str) -> pl.Expr:
        """ Polars expression for the standard error of the mean. """
        c = pl.col(col).drop_nans().drop_nulls()
        return c.std() / c.count().cast(pl.Float64).sqrt()
    
    def pl_expr_circ_mean(self, col: str) -> pl.Expr:
        """ Polars expression for the circular mean (radians). """
        c = pl.col(col).filter(pl.col(col).is_finite())
        return pl.arctan2(c.sin().mean(), c.cos().mean())

    def pl_expr_circ_var(self, col: str) -> pl.Expr:
        """ Polars expression for the circular variance defined as 1 - R. """
        c = pl.col(col).filter(pl.col(col).is_finite())
        sin_ = c.sin().mean()
        cos_ = c.cos().mean()
        return 1.0 - (sin_.pow(2) + cos_.pow(2)).sqrt()

    def wrap_pi(self, a: np.ndarray) -> np.ndarray:
        """ Wrap angles in radians to the range [-π, π]. """
        return (a + np.pi) % (2 * np.pi) - np.pi

    def ci(self, a, **kwargs) -> tuple[float, float]:
        """
        Confidence interval via bootstrap. 
        
        Returns
        -------
        tuple[float, float]
            A tuple containing the lower and upper `<low, high>` bounds of the confidence interval.
        """
        seed = kwargs.get('seed', 42)  # Fixed seed for reproducibility

        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]

        if a.size < 2:
            warn("Not enough finite data points to compute confidence interval.")
            return (np.nan, np.nan)

        cl = kwargs.get('ci_confidence', self.ci_confidence)
        if cl > 1:
            cl = cl / 100.0

        ci_statistic = kwargs.get('ci_statistic', self.ci_statistic)
        bootstrap_resamples = kwargs.get('bootstrap_resamples', self.bootstrap_resamples)
        ci_method = kwargs.get('bootstrap_ci_method', self.bootstrap_ci_method)

        try:
            result = stats.bootstrap(
                (a,),
                statistic=ci_statistic,
                n_resamples=bootstrap_resamples,
                confidence_level=cl,
                method=ci_method,
                random_state=seed
            )
            self._ci_method_used = ci_method

        except Exception:
            try:
                result = stats.bootstrap(
                    (a,),
                    statistic=ci_statistic,
                    n_resamples=bootstrap_resamples,
                    confidence_level=cl,
                    method='percentile',
                    random_state=seed
                )
                self._ci_method_used = 'percentile'

            except Exception as e:
                warnings.warn(message=f"Bootstrap confidence interval computation failed for both '{ci_method}' and fallback 'percentile' methods: {e}. Returning (np.nan, np.nan). Traceback:\n{traceback.format_exc()}",
                              category=FailedWarning, stacklevel=2)
                return (np.nan, np.nan)

        if self._ci_method_used != ci_method:
            warnings.warn(message=f"Requested method ('{ci_method}') cannot be used; falling back to '{self._ci_method_used}'.",
                          category=FailedWarning, 
                          stacklevel=2)
            pass

        return (float(result.confidence_interval.low), float(result.confidence_interval.high))


    def units(self, col: str = None, **kwargs) -> dict[str, str] | str:
        """Returns a dictionary of columns with their corresponding units or the corresponding units to the specified column."""

        if self.metadata is not None:
            if 'timeunits' in self.metadata:
                timeunits = self.metadata['timeunits']
            else:
                timeunits = '<timeunits not found>'
            if 'spatialunits' in self.metadata:
                spatialunits = self.metadata['spatialunits']
            else:
                spatialunits = '<spatialunits not found>'
        else:
            timeunits = '<timeunits not found>'
            spatialunits = '<spatialunits not found>'

        units = {
            'x_coordinate': f'{spatialunits}',
            'y_coordinate': f'{spatialunits}',
            'time_point': f'{timeunits}',
            'frame': '',
            'distance': f'{spatialunits}',
            'instantaneous_speed': f'{spatialunits} ⋅ {timeunits}⁻¹',
            'cum_track_length': f'{spatialunits}',
            'cum_track_displacement': f'{spatialunits}',
            'cum_straightness': '',
            'cum_directionality': '',
            'cum_speed_mean': f'{spatialunits} ⋅ {timeunits}⁻¹',
            'cum_mean_straight_line_speed': f'{spatialunits} ⋅ {timeunits}⁻¹',
            'cum_forward_progression_linearity': f'{spatialunits}',
            'direction': 'rad',
            'directional_change': 'rad',
            'cum_sum_directional_change': 'rad',
            'cum_mean_directional_change': 'rad',
            'cum_mean_directional_change_rate': f'rad ⋅ {timeunits}⁻¹',
            'cum_direction_mean': 'rad',
            'cum_direction_var': '',

            'y_location': f'{spatialunits}',
            'x_location': f'{spatialunits}',
            'track_length': f'{spatialunits}',
            'track_displacement': f'{spatialunits}',
            'directionality': '',
            'speed_min': f'{spatialunits} ⋅ {timeunits}⁻¹',
            'speed_max': f'{spatialunits} ⋅ {timeunits}⁻¹',
            'speed_mean': f'{spatialunits} ⋅ {timeunits}⁻¹',
            'speed_sd': f'{spatialunits} ⋅ {timeunits}⁻¹',
            'speed_median': f'{spatialunits} ⋅ {timeunits}⁻¹',
            'mean_straight_line_speed': f'{spatialunits} ⋅ {timeunits}⁻¹',
            'greatest_distance': f'{spatialunits}',
            'straightness': '',
            'direction_mean': 'rad',
            'mean_directional_change': 'rad',
            'mean_directional_change_rate': f'rad ⋅ {timeunits}⁻¹',
            'time_lag': f'{timeunits}',
            'msd': f'{spatialunits}²',
            'directional_change_mean': 'rad',
        }

        if col is not None:
            if col not in units.keys():
                # warnings.warn(f"Column '{col}' not found in units dictionary.", category=FailedWarning, stacklevel=2)
                return ''
            return units[col]
        return units



calc = Calc()