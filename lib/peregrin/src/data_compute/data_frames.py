from __future__ import annotations

import traceback
import warnings
import numpy as np
import polars as pl
from scipy import stats
from typing import Any, Callable, Literal, Optional, Dict, List

from ..utils import is_empty
from ..settings import params
from ..utils import ensure_polars

from warnings import warn
from .._pckg_exceptions._pckg_errors import *
from .._pckg_exceptions._pckg_warnings import *



# Metric registry
# ---------------------------------------------------------------------------

class MetricRegistry:
    """
    Maps an output column name to a callable that builds the column.

    Computers receive a shared context dict and may return either:
      - a `pl.Expr` aggregation expression (collected into ONE group_by().agg()
        call -> single pass over the data)
      - a post-processing callable `(out_df, ctx) -> out_df` for metrics that
        cannot be expressed as a polars aggregation (e.g. bootstrap CIs).

    A `gate` determines whether the column should be included in the default set of computed columns.

    Gate values:
        True  -> always included in the default set
        False -> never included in the default set
        str   -> conditionally included based on the named flag in the settings
    """

    def __init__(self) -> None:
        self._computers: Dict[str, Callable] = {}
        self._gate: Dict[str, bool | str] = {}
        self._order: List[str] = []

    def register(self, column: str, gate: bool = True) -> Callable:
        """Decorator-based registration of a metric column.

        Parameters
        ----------
        column : str
            The name of the output column to register.
        gate : bool, optional
            Whether to actually add the column to the registry, by default True.

        Returns
        -------
        Callable
            The decorator that registers the function.
        """

        def _wrap(fn: Callable) -> Callable:
            if column not in self._computers:
                self._order.append(column)
            self._computers[column] = fn
            self._gate[column] = gate
            return fn
        return _wrap

    def add(self, column: str, fn: Callable, gate: str = True) -> None:
        """
        Imperative registration (non-decorator).

        Parameters
        ----------
        column : str
            The name of the output column to register.
        fn : Callable
            The function that computes the column.
        gate : bool, optional
            Whether to actually add the column to the registry, by default True.
        """
        self.register(column, gate=gate)(fn)

    def all_columns(self) -> List[str]:
        """All columns whose gate is True (default set)."""
        return [c for c in self._order if self._gate[c]]

    def resolve(self, subset: Optional[List[str]] = None) -> List[str]:
        """Determine which registered columns to compute.

        subset=None -> all gate=True columns.
        subset=[..] -> exactly the requested registered columns (gate ignored).
        """
        if subset is None:
            return self.all_columns()
        requested = set(subset)
        return [c for c in self._order if c in requested and c in self._computers]

    def compute(self, wanted: List[str], ctx: dict) -> pl.DataFrame:
        """Run the requested computers against a shared context.

        `ctx['source']` -> the source pl.DataFrame
        `ctx['by']`     -> grouping column names (list)
        """
        exprs: List[pl.Expr] = []
        posts: List[Callable] = []

        for col in wanted:
            result = self._computers[col](ctx)
            if isinstance(result, pl.Expr):
                exprs.append(result.alias(col))
            elif callable(result):
                posts.append(result)

        # include any list-aggregation helpers requested by post-processors
        extra = ctx.get('extra_exprs', {})
        for name, e in extra.items():
            exprs.append(e.alias(name))

        out = (
            ctx['source']
            .group_by(ctx['by'], maintain_order=True)
            .agg(exprs)
        )

        for post in posts:
            out = post(out, ctx)

        # Drop helper list columns
        helper = [c for c in out.columns if c.startswith('__list_')]
        if helper:
            out = out.drop(helper)

        return out


class Calc:
    """
    A class with methods for computing tracking data statistics:
    spots (per-trajectory-point), tracks (per-whole-trajectory), time points (per-time-point),
    time lags (per-time-lag).

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

    ci_statistic: str, default 'mean'
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

    _POLARS_BUILTINS = frozenset({
        'mean', 'median', 'std', 'count', 'sum', 'min', 'max',
        'first', 'last', 'var', 'product', 'len', 'n_unique',
    })

    _EXCLUDE_SUFFIXES = set([
        'track_id', 'track_uid', 'time_point', 'frame', 'time_lag', 'frame_lag', 'sd', 'var', 'sem', 'q25', 'q75'
    ])

    DEFAULTS = set(['min', 'max', 'mean', 'median', 'std'])

    COLUMNS = {
        'SPOTS': [
            'track_id', 'track_uid', 'time_point', 'frame', 
            'x_coordinate', 'y_coordinate', 'distance', 'direction',
        ],
        'TRACKS': [
            'track_id', 'track_uid', 'y_location', 'x_location',
            'track_length', 'track_displacement', 'straightness_ratio',
            'speed_min', 'speed_max', 'speed_mean', 'speed_sd', 'speed_median',
            'mean_straight_line_speed', 'forward_progression_linearity',
            'max_distance_reached', 'track_start_frame', 'track_end_frame',
            'track_points', 'direction_mean', 'direction_var', 
            'mean_directional_change', 'mean_directional_change_rate'
        ],
        'TIMEPOINTS': [
            'time_point', 'frame', 'tracks_contributing',
            'cum_track_length', 'cum_track_displacement',
            'cum_straightness_ratio', 'cum_speed_mean',
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
        bootstrap_resamples: int = 1000,
        bootstrap_ci_method: str = 'BCa',
        ci_statistic: Literal['mean', 'median'] | Callable[[np.ndarray], float] = 'mean',
        **kwargs
    ) -> None:

        self.inferative_error = inferative_error
        self.bootstrap_ci = bootstrap_ci
        self.ci_confidence = ci_confidence
        self.bootstrap_resamples = bootstrap_resamples
        self.bootstrap_ci_method = bootstrap_ci_method

        match ci_statistic:
            case 'mean':
                self.ci_statistic = np.mean
            case 'median':
                self.ci_statistic = np.median
            case _ if callable(ci_statistic):
                self.ci_statistic = ci_statistic
            case _:
                raise ValueError("ci_statistic must be 'mean', 'median', or a callable function.")
        

        # Custom aggregation expression builders (column name -> pl.Expr)
        self.AGG_FUNCTIONS: Dict[str, Callable[[str], pl.Expr]] = {
            'sem':       lambda col: self.pl_expr_sem(col),
            'circ_mean': lambda col: self.pl_expr_circ_mean(col),
            'circ_var':  lambda col: self.pl_expr_circ_var(col),
            'wrap_pi':   lambda col: self.pl_expr_wrap_pi(col),
        }

        self._tracks_registry = self._build_tracks_registry()
        self._timepoints_registry = self._build_timepoints_registry()
        self._timelags_registry = self._build_timelags_registry()

    
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
        subset : list[str], optional
            Subset of columns to consider for the computation, by default None.
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
            # warnings.warn(message="Input DataFrame is empty. No computation performed.",
            #                 category=DataFrameWarning,
            #                 stacklevel=2)
            return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['SPOTS']})

        df = self._guard_df(df)

        grouping_cols = [col for col in self.DEFAULT_CATEGORIES if col in df.columns]

        df = self.assign_track_uid(df)
        df = df.sort(grouping_cols + ['track_uid', 'time_point'])

        uid = 'track_uid'

        # frame: dense rank of time_point per track (0-based)
        df = df.with_columns(
            (pl.col('time_point').rank(method='dense').over(uid) - 1).cast(pl.Int64).alias('frame')
        )

        # Validate per TRACK: one time_point -> exactly one frame within a track.
        bad = (
            df.group_by([uid, 'time_point'])
            .agg(pl.col('frame').n_unique().alias('_n'))
            .select(pl.col('_n').max())
            .item()
        )
        if bad and bad > 1:
            # raise TimePointError(
            #     f"Multiple frames assigned to the same track_uid × time_point "
            #     f"combination. Duplicate time_point values within a track. "
            #     f"Max frames per time point: {bad}."
            # )
            pass

        # Step deltas, distance and direction
        df = df.with_columns(
            (pl.col('x_coordinate') - pl.col('x_coordinate').shift(1)).over(uid).alias('_dx'),
            (pl.col('y_coordinate') - pl.col('y_coordinate').shift(1)).over(uid).alias('_dy'),
        ).with_columns(
            (pl.col('_dx').pow(2) + pl.col('_dy').pow(2)).sqrt().alias('distance'),
            pl.arctan2(pl.col('_dy'), pl.col('_dx')).alias('direction'),
        ).drop(['_dx', '_dy'])

        # Keep only the spot columns (+ categories and color columns)
        # keep = [c for c in df.columns
        #         if c in self.COLUMNS['SPOTS'] or c in grouping_cols or c.endswith('color')]
        # df = df.select(keep)
        
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
        subset: list[str] = None,
        **kwargs
    ) -> pl.DataFrame:
        """
        Computes per-trajectory metrics.

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
        
        All metrics are built as polars aggregation expressions and computed in a
        single `group_by('track_uid').agg(...)` pass. See the original
        documentation for column descriptions.
        """

        if is_empty(df):
            # warnings.warn(message="Input DataFrame is empty. No computation performed.",
            #               category=DataFrameWarning, 
            #               stacklevel=2)
            return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['TRACKS']})

        grouping_cols = [col for col in self.DEFAULT_CATEGORIES if col in df.columns]

        df = self.assign_track_uid(df)
        df = df.sort(['track_uid', 'time_point'])

        timeinterval = self._resolve_timeinterval(df, metadata = kwargs.get('metadata', None))

        # Derive cumulative per-spot metrics needed by the aggregations
        df = self._enrich_spots(df, timeinterval=timeinterval, **kwargs)

        # Stash categorical identifiers to merge them back into the result
        stash_cols = [c for c in grouping_cols if c != 'track_uid']
        stash = df.select(['track_uid'] + stash_cols).unique(subset=['track_uid'], keep='first')

        wanted = self._tracks_registry.resolve(subset)

        ctx = {
            'source': df,
            'by': ['track_uid'],
            'timeinterval': timeinterval,
        }
        agg = self._tracks_registry.compute(wanted, ctx)

        # Carry over color columns and track_id (first per track)
        # carry = [c for c in df.columns if c.endswith('color')]
        if 'track_id' in df.columns:
            # carry = ['track_id'] + carry
            carry = ['track_id']
        if carry:
            firsts = df.group_by('track_uid', maintain_order=True).agg(
                [pl.col(c).first() for c in carry]
            )
            agg = agg.join(firsts, on='track_uid', how='left')

        out = stash.join(agg, on='track_uid', how='right')

        # Drop spot-level columns that leaked through
        drop = [c for c in self.COLUMNS['SPOTS'] if c in out.columns and c not in self.COLUMNS['TRACKS']]
        out = out.drop(drop).unique(maintain_order=True)

        if self.significant_figures:
            out = self.signify(out)
        if self.decimal_places:
            out = self.norm_decimals(out)

        return out


    def timepoints(
        self,
        df: pl.DataFrame,
        subset: list[str] = None,
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

        One `group_by().agg()` pass per grouping level; all descriptive,
        error and circular statistics are polars expressions. See the original
        documentation for details.
        """
        if is_empty(df):
            warn("Input DataFrame is empty. Returning an empty DataFrame with the expected schema.")
            return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['TIMEPOINTS']})
        if df['time_point'].n_unique() < 2:
            warn("Not enough time points available for time interval statistics computations. Returning an empty schema DataFrame.")
            return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['TIMEPOINTS']})

        grouping_set = []

        if isinstance(grouping_level, list):
            for g in grouping_level:
                grouping_set.append(self._get_grouping_level(df.columns, g, exclude='track_uid', include=['time_point', 'frame']))
        else:
            grouping_cols = self._get_grouping_level(df.columns, grouping_level, exclude='track_uid', include=['time_point', 'frame'])
            grouping_set = [grouping_cols]

        df = self.assign_track_uid(df)

        df = self._enrich_spots(df, **kwargs)

        wanted = self._timepoints_registry.resolve(subset)

        level_frames = []
        for group_cols in grouping_set:
            group_lvl = group_cols[0]

            ctx = {
                'source': df,
                'by': group_cols,
            }
            level_df = self._timepoints_registry.compute(wanted, ctx)

            # Split any *_ci tuple columns into low/high (single-pass bootstrap).
            ci_cols = [c for c in level_df.columns if c.endswith('_ci')]
            if ci_cols:
                conf = round(self.ci_confidence * 100)
                exprs = []
                for c in ci_cols:
                    base = c[:-3]  # strip '_ci'
                    exprs.append(pl.col(c).list.get(0).alias(f'{base}_ci{conf}_low'))
                    exprs.append(pl.col(c).list.get(1).alias(f'{base}_ci{conf}_high'))
                level_df = level_df.with_columns(exprs).drop(ci_cols)

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
        subset: list[str] = None,
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
            - `MSD_sem`: Standard error of the mean squared displacement for the given time lag.
            - `MSD_ciXX_low`: Lower bound of the confidence interval for the mean squared displacement for the given time lag, where `XX` represents the confidence level.
            - `MSD_ciXX_high`: Upper bound of the confidence interval for the mean squared displacement for the given time lag, where `XX` represents the confidence level.
            - `MSD_min`: Minimum value of the mean squared displacement for the given time lag.
            - `MSD_max`: Maximum value of the mean squared displacement for the given time lag.
            - `tracks_contributing`: Number of tracks contributing to the given time lag.
            - `position_pairs_contributing`: Number of position pairs contributing to the given time lag.
            - `directional_change_mean`: Mean directional change for the given time lag.
            - `directional_change_var`: Variance of the directional change for the given time lag.
        
        See the original documentation for the full
        description; pair-building stays numpy-vectorized, all aggregation runs
        through a single polars `group_by().agg()` per grouping level.
        """

        if is_empty(df):
            warn("Input DataFrame is empty. Returning an empty DataFrame with the expected schema.")
            return pl.DataFrame(schema={c: pl.Float64 for c in self.COLUMNS['TIMELAGS']})

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

        wanted = self._timelags_registry.resolve(subset)

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
            ctx = {
                'source': all_msd,
                'by': lag_group_cols,
                'turn_src': all_turn,
                'circ_cache': {},
            }

            lags = self._timelags_registry.compute(wanted, ctx)

            if 'MSD_ci' in lags.columns:
                lags = lags.with_columns(
                    pl.col('MSD_ci').list.get(0).alias(f'MSD_ci{round(self.ci_confidence*100)}_low'),
                    pl.col('MSD_ci').list.get(1).alias(f'MSD_ci{round(self.ci_confidence*100)}_high'),
                ).drop('MSD_ci')

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
            - `cum_straightness_ratio`: Ratio of cumulative displacement to cumulative track length.
            - `cum_speed_mean`: Mean cumulative speed for each trajectory point.
            - `cum_mean_straight_line_speed`: Mean straight-line speed for each trajectory point.
            - `cum_forward_progression_linearity`: Linearity of forward progression for each trajectory point.

        Returns
        -------
        pl.DataFrame
            DataFrame enriched with cumulative per-trajectory-point metrics.
        """

        uid = 'track_uid'
        timeinterval = kwargs.get(
            'timeinterval',
            self._resolve_timeinterval(df, metadata = kwargs.get('metadata', None))
        )

        if 'cum_track_length' in df.columns:
            return df

        df = df.sort([uid, 'time_point'])

        # (Re)compute basics if the input was raw
        if 'frame' not in df.columns:
            df = df.with_columns(
                (pl.col('time_point').rank(method='dense').over(uid) - 1).cast(pl.Int64).alias('frame')
            )
        if 'distance' not in df.columns or 'direction' not in df.columns:
            df = df.with_columns(
                (pl.col('x_coordinate') - pl.col('x_coordinate').shift(1)).over(uid).alias('_dx'),
                (pl.col('y_coordinate') - pl.col('y_coordinate').shift(1)).over(uid).alias('_dy'),
            ).with_columns(
                (pl.col('_dx').pow(2) + pl.col('_dy').pow(2)).sqrt().alias('distance'),
                pl.arctan2(pl.col('_dy'), pl.col('_dx')).alias('direction'),
            ).drop(['_dx', '_dy'])

        # Cumulative metrics
        df = df.with_columns(
            pl.col('distance').cum_sum().over(uid).alias('cum_track_length'),
            (
                (pl.col('x_coordinate') - pl.col('x_coordinate').first().over(uid)).pow(2)
                + (pl.col('y_coordinate') - pl.col('y_coordinate').first().over(uid)).pow(2)
            ).sqrt().alias('cum_track_displacement'),
            pl.col('time_point').cum_count().over(uid).cast(pl.Float64).alias('_cumcount'),
        ).with_columns(
            pl.when(pl.col('cum_track_displacement') == 0)
            .then(None).otherwise(pl.col('cum_track_displacement'))
            .alias('cum_track_displacement'),
        )

        elapsed = (pl.col('time_point') - pl.col('time_point').first().over(uid))
        df = df.with_columns(
            (
                pl.col('cum_track_displacement')
                / pl.when(pl.col('cum_track_length') == 0).then(None).otherwise(pl.col('cum_track_length'))
            ).alias('cum_straightness_ratio'),
            (
                pl.col('cum_track_length')
                / pl.when(elapsed == 0).then(None).otherwise(elapsed)
            ).alias('cum_speed_mean'),
        ).with_columns(
            (pl.col('cum_track_displacement') / (pl.col('_cumcount') * timeinterval))
            .alias('cum_mean_straight_line_speed'),
        ).with_columns(
            (pl.col('cum_mean_straight_line_speed') / pl.col('cum_speed_mean'))
            .alias('cum_forward_progression_linearity'),
        )

        # Turning angle (deg, wrapped, abs) and its cumulative statistics
        df = df.with_columns(
            (
                ((pl.col('direction') - pl.col('direction').shift(1)).over(uid) + np.pi)
                .mod(2 * np.pi) - np.pi
            ).abs().degrees().alias('directional_change')
        ).with_columns(
            pl.col('directional_change').cum_sum().over(uid).alias('cum_sum_directional_change'),
            pl.col('directional_change').is_not_null().cum_sum().over(uid)
            .cast(pl.Float64).alias('_valid_count'),
        ).with_columns(
            (
                pl.col('cum_sum_directional_change')
                / pl.when(pl.col('_valid_count') == 0).then(None).otherwise(pl.col('_valid_count'))
            ).alias('cum_mean_directional_change')
        ).with_columns(
            pl.when(pl.col('directional_change').is_null())
            .then(None).otherwise(pl.col('cum_mean_directional_change'))
            .alias('cum_mean_directional_change'),
        ).with_columns(
            (pl.col('cum_mean_directional_change') / (pl.col('_cumcount') * timeinterval))
            .alias('cum_mean_directional_change_rate'),
        )

        # Cumulative circular mean / variance of direction
        df = df.with_columns(
            pl.col('direction').sin().cum_sum().over(uid).alias('_cum_sin'),
            pl.col('direction').cos().cum_sum().over(uid).alias('_cum_cos'),
            (pl.col('_cumcount') - 1).alias('_n_angles'),
        ).with_columns(
            pl.arctan2(pl.col('_cum_sin'), pl.col('_cum_cos')).alias('cum_direction_mean'),
            (
                1.0 - (pl.col('_cum_sin').pow(2) + pl.col('_cum_cos').pow(2)).sqrt()
                / pl.when(pl.col('_n_angles') == 0).then(None).otherwise(pl.col('_n_angles'))
            ).alias('cum_direction_var'),
        ).with_columns(
            pl.when(pl.col('_n_angles') == 0).then(None)
            .when(pl.col('_n_angles') == 1).then(0.0)
            .otherwise(pl.col('cum_direction_var'))
            .alias('cum_direction_var'),
        )

        return df.drop(['_cumcount', '_valid_count', '_cum_sin', '_cum_cos', '_n_angles'])


    
    # Metrics registry builders
    # -----------------------------------------------------------------------

    def _build_tracks_registry(self) -> MetricRegistry:
        """ 
        One aggregation expression per TRACKS output column.

        ctx['timeinterval'] -> the resolved time step
        """
        reg = MetricRegistry()

        reg.add('speed_min',    lambda ctx: pl.col('distance').min()    / ctx['timeinterval'])
        reg.add('speed_max',    lambda ctx: pl.col('distance').max()    / ctx['timeinterval'])
        reg.add('speed_mean',   lambda ctx: pl.col('distance').mean()   / ctx['timeinterval'])
        reg.add('speed_sd',     lambda ctx: pl.col('distance').std()    / ctx['timeinterval'])
        reg.add('speed_median', lambda ctx: pl.col('distance').median() / ctx['timeinterval'])

        reg.add('track_length', lambda ctx: pl.col('distance').sum())

        reg.add('x_location', lambda ctx: pl.col('x_coordinate').mean())
        reg.add('y_location', lambda ctx: pl.col('y_coordinate').mean())

        reg.add('max_distance_reached', lambda ctx: pl.col('cum_track_displacement').max())

        reg.add('track_start_frame', lambda ctx: pl.col('frame').min())
        reg.add('track_end_frame',   lambda ctx: pl.col('frame').max())

        reg.add('mean_straight_line_speed',      lambda ctx: pl.col('cum_mean_straight_line_speed').last())
        reg.add('forward_progression_linearity', lambda ctx: pl.col('cum_forward_progression_linearity').last())

        reg.add('direction_mean', lambda ctx: pl.col('cum_direction_mean').last())
        reg.add('direction_var',  lambda ctx: pl.col('cum_direction_var').last())
        reg.add('mean_directional_change',      lambda ctx: pl.col('cum_mean_directional_change').last())
        reg.add('mean_directional_change_rate', lambda ctx: pl.col('cum_mean_directional_change_rate').last())

        reg.add('track_points', lambda ctx: pl.len())

        reg.add('track_displacement', lambda ctx: pl.col('cum_track_displacement').last())
        reg.add('straightness_ratio', lambda ctx: pl.col('cum_track_displacement').last() / pl.col('distance').sum())

        return reg
    

    def _build_timepoints_registry(self) -> MetricRegistry:
        """ One calculation per given timepoints columns. """
        reg = MetricRegistry()  # Initialize a new metric registry for timepoints calculations

        metric_out = {   # Mapping of metric names to their corresponding timepoints columns
            'cum_track_length': 'cum_track_length',
            'cum_track_displacement': 'cum_track_displacement',
            'cum_straightness_ratio': 'cum_straightness_ratio',
            'cum_speed_mean': 'cum_speed_mean',
            'distance': 'instantaneous_speed',
            'cum_mean_straight_line_speed': 'cum_mean_straight_line_speed',
            'cum_forward_progression_linearity': 'cum_forward_progression_linearity',
            'cum_sum_directional_change': 'cum_sum_directional_change',
            'cum_mean_directional_change': 'cum_mean_directional_change',
        }

        def _ci_post(src: str, ci_col: str) -> Callable:
            """Post-processor: bootstrap CI of `src` -> temporary list column `ci_col`.

            Runs the bootstrap ONCE per group; the `timepoints` method splits
            the resulting (low, high) tuple into `_low` / `_high` columns.
            """
            def _computer(ctx: dict) -> Callable:
                # Collect the source values per group in the single agg pass.
                ctx.setdefault('extra_exprs', {})[f'__list_{ci_col}'] = pl.col(src)

                def _post(out: pl.DataFrame, _ctx: dict) -> pl.DataFrame:
                    bounds = [
                        self.ci(np.asarray(v, dtype=float))
                        for v in out[f'__list_{ci_col}'].to_list()
                    ]
                    return out.with_columns(
                        pl.Series(ci_col, bounds, dtype=pl.List(pl.Float64))
                    )
                return _post
            return _computer

        for src, mout in metric_out.items():
            reg.add(f'{mout}_min',    lambda ctx, s = src: pl.col(s).min())
            reg.add(f'{mout}_max',    lambda ctx, s = src: pl.col(s).max())
            reg.add(f'{mout}_mean',   lambda ctx, s = src: pl.col(s).mean())
            reg.add(f'{mout}_median', lambda ctx, s = src: pl.col(s).median())
            reg.add(f'{mout}_sd',     lambda ctx, s = src: pl.col(s).std())

            reg.add(f'{mout}_sem',    lambda ctx, s = src: self.AGG_FUNCTIONS['sem'](s), gate=self.inferative_error)

            reg.add(f'{mout}_ci', _ci_post(src, f'{mout}_ci'))


        reg.add('tracks_contributing', lambda ctx: pl.col('track_uid').n_unique().cast(pl.Int64))

        # --- Circular statistics (fully expression-based) -------------------
        reg.add('instantaneous_direction_mean',
                lambda ctx: self.AGG_FUNCTIONS['circ_mean']('direction'))
        reg.add('instantaneous_direction_var',
                lambda ctx: self.AGG_FUNCTIONS['circ_var']('direction'))
        reg.add('cum_direction_mean',
                lambda ctx: self.AGG_FUNCTIONS['circ_mean']('cum_direction_mean'))
        reg.add('cum_direction_var',
                lambda ctx: self.AGG_FUNCTIONS['circ_var']('cum_direction_mean'))
        reg.add('cum_mean_directional_change_mean',
                lambda ctx: pl.col('cum_mean_directional_change').mean())

        return reg
    

    def _build_timelags_registry(self) -> MetricRegistry:
        """One computer per TIMELAGS output column."""
        reg = MetricRegistry()

        def _circ_post(col: str, expr_key: str) -> Callable:
            def _computer(ctx: dict) -> Callable:
                def _post(out: pl.DataFrame, _ctx: dict) -> pl.DataFrame:
                    turn_src: pl.DataFrame = _ctx['turn_src']
                    if is_empty(turn_src):
                        return out
                    stat = (
                        turn_src
                        .group_by(_ctx['by'], maintain_order=True)
                        .agg(self.AGG_FUNCTIONS[expr_key]('dtheta').alias(col))
                    )
                    return out.join(stat, on=_ctx['by'], how='left')
                return _post
            return _computer

        def _msd_ci(ctx: dict) -> Callable:
            """Post-processor: bootstrap CI of MSD (mean statistic).

            Runs the bootstrap ONCE per group and stores the resulting
            (low, high) tuple in a temporary `MSD_ci` column, which the
            `timelags` method splits into `_low` / `_high` columns.
            """
            # Collect sq_disp values per group in the single agg pass.
            ctx.setdefault('extra_exprs', {})['__list_MSD_ci'] = pl.col('sq_disp')

            def _post(out: pl.DataFrame, _ctx: dict) -> pl.DataFrame:
                bounds = [
                    self.ci(np.asarray(v, dtype=float), ci_statistic='mean')
                    for v in out['__list_MSD_ci'].to_list()
                ]
                return out.with_columns(
                    pl.Series('MSD_ci', bounds, dtype=pl.List(pl.Float64))
                )
            return _post

        reg.add('MSD',     lambda ctx: pl.col('sq_disp').mean())
        reg.add('MSD_min', lambda ctx: pl.col('sq_disp').min())
        reg.add('MSD_max', lambda ctx: pl.col('sq_disp').max())
        reg.add('MSD_sd',  lambda ctx: pl.col('sq_disp').std())
        reg.add('MSD_sem', lambda ctx: self.AGG_FUNCTIONS['sem']('sq_disp'), gate=self.inferative_error)
        reg.add('MSD_ci',  _msd_ci, gate=self.bootstrap_ci)

        reg.add('tracks_contributing',         lambda ctx: pl.col('track_uid').n_unique().cast(pl.Int64))
        reg.add('position_pairs_contributing', lambda ctx: pl.len().cast(pl.Int64))

        reg.add('directional_change_mean', _circ_post('directional_change_mean', 'circ_mean'))
        reg.add('directional_change_var',  _circ_post('directional_change_var',  'circ_var'))

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
            # raise ColumnsNotFoundError(
            #     "Cannot create track_uid -> missing category or track_id columns."
            # )
            pass

        keys = (
            df.select(grouping_cols)
            .unique(maintain_order=True)
            .with_row_index('track_uid')
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
            pass
        
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


    def _general_agg_stats(self, df: pl.DataFrame, exclude: list[str], *, group_by: list[str] = ['track_uid']) -> pl.DataFrame:
        """ Compute general aggregate statistics (min, max, mean, sd, sem, median)
            for numeric columns, grouped by `group_by`, excluding given columns. """

        exclude = [col for col in (exclude or []) if col != 'track_uid']

        try:
            num_cols = [
                c for c, dt in df.schema.items()
                if dt.is_numeric() and c not in exclude and c not in group_by and c != 'track_uid'
            ]

            if not num_cols:
                return df.select(group_by).unique(maintain_order=True)

            exprs = []
            for col in num_cols:
                exprs += [
                    pl.col(col).min().alias(f"{col} min"),
                    pl.col(col).max().alias(f"{col} max"),
                    pl.col(col).mean().alias(f"{col} mean"),
                    pl.col(col).std(ddof=1).alias(f"{col} sd"),
                    (pl.col(col).std(ddof=1) / pl.col(col).count().cast(pl.Float64).sqrt()).alias(f"{col} sem"),
                    pl.col(col).median().alias(f"{col} median"),
                ]

            return df.group_by(group_by, maintain_order=True).agg(exprs)

        except Exception as e:
            # warnings.warn(message=f"Stats._general_agg_stats() encountered an error: {e}. Returning empty DataFrame. Traceback:\n{traceback.format_exc()}",
            #               category=FailedWarning, stacklevel=2)
            return pl.DataFrame()


    def resolve(self, agg_spec: dict[str, str] | list[str]) -> dict[str, Callable[[str], pl.Expr]]:
        """Resolves a list/dict of aggregation specs into a mapping of output
        labels to polars aggregation-expression builders."""

        def _builder(func_name: str) -> Callable[[str], pl.Expr]:
            if func_name in self._POLARS_BUILTINS:
                return lambda c, f=func_name: getattr(pl.col(c), f)()
            if func_name in self.AGG_FUNCTIONS:
                return self.AGG_FUNCTIONS[func_name]
            if func_name == 'ci':
                return 'ci'  # sentinel handled by callers
            raise ValueError(
                f"Unknown aggregation '{func_name}'. "
                f"Available: {sorted(self._POLARS_BUILTINS | set(self.AGG_FUNCTIONS) | {'ci'})}"
            )

        resolved = {}
        if isinstance(agg_spec, list):
            for func_name in agg_spec:
                resolved[func_name] = _builder(func_name)
        elif isinstance(agg_spec, dict):
            for label, func_name in agg_spec.items():
                resolved[label] = _builder(func_name)
        return resolved


    def _insert_at_position(self, d: dict, key: Any, value: Any = None, *, where: int | str = 0) -> dict:
        """Insert a (key: value) pair into a dictionary at a specific position."""
        items = list(d.items())

        if isinstance(where, int):
            index = where
        elif isinstance(where, str):
            keys = [k for k, _ in items]
            if where not in keys:
                raise ValueError(f"Key '{where}' not found in dictionary.")
            index = keys.index(where) + 1
        else:
            raise ValueError("Parameter 'where' must be an integer index or a string key.")

        items.insert(index, (key, value))
        return dict(items)


    
    # Aggregation functions
    # -----------------------------------------------------------------------


    def pl_expr_sem(self, col: str) -> pl.Expr:
        """ Polars expression for the standard error of the mean. """
        c = pl.col(col).drop_nans().drop_nulls()
        return c.std() / c.count().cast(pl.Float64).sqrt()
    
    def pl_expr_circ_mean(self, col: str) -> float:
        """ Polars expression for the circular mean (radians). """
        c = pl.col(col).filter(pl.col(col).is_finite())
        return pl.arctan2(c.sin().mean(), c.cos().mean())

    def pl_expr_circ_var(self, col: str) -> float:
        """ Polars expression for the circular variance defined as 1 - R. """
        c = pl.col(col).filter(pl.col(col).is_finite())
        sin_ = c.sin().mean()
        cos_ = c.cos().mean()
        return 1.0 - (sin_.pow(2) + cos_.pow(2)).sqrt()

    def pl_expr_wrap_pi(self, col: str) -> np.ndarray:
        """ Wrap angles in radians to the range [-π, π]. """
        c = pl.col(col).filter(pl.col(col).is_finite())
        return (c + np.pi) % (2 * np.pi) - np.pi

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
            warn("Not enough finite data points to compute confidence interval.", stacklevel=2)
            return (np.nan, np.nan)

        cl = kwargs.get('ci_confidence', self.ci_confidence)
        if cl > 1:
            cl = cl / 100.0

        bootstrap_resamples = kwargs.get('bootstrap_resamples', self.bootstrap_resamples)
        ci_method = kwargs.get('bootstrap_ci_method', self.bootstrap_ci_method)

        try:
            result = stats.bootstrap(
                (a,),
                statistic=kwargs.get('ci_statistic', self.ci_statistic),
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
                    statistic=kwargs.get('ci_statistic', self.ci_statistic),
                    n_resamples=bootstrap_resamples,
                    confidence_level=cl,
                    method='percentile',
                    random_state=seed
                )
                self._ci_method_used = 'percentile'

            except Exception as e:
                # warnings.warn(message=f"Bootstrap confidence interval computation failed for both '{ci_method}' and fallback 'percentile' methods: {e}. Returning (np.nan, np.nan). Traceback:\n{traceback.format_exc()}",
                #               category=FailedWarning, stacklevel=2)
                return (np.nan, np.nan)

        if self._ci_method_used != ci_method:
            # warnings.warn(message=f"Requested method ('{ci_method}') cannot be used; falling back to '{self._ci_method_used}'.",
            #               category=FailedWarning, 
            #               stacklevel=2)
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
            'cum_straightness_ratio': f'{spatialunits}',
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
            'speed_min': f'{spatialunits} ⋅ {timeunits}⁻¹',
            'speed_max': f'{spatialunits} ⋅ {timeunits}⁻¹',
            'speed_mean': f'{spatialunits} ⋅ {timeunits}⁻¹',
            'speed_sd': f'{spatialunits} ⋅ {timeunits}⁻¹',
            'speed_median': f'{spatialunits} ⋅ {timeunits}⁻¹',
            'mean_straight_line_speed': f'{spatialunits} ⋅ {timeunits}⁻¹',
            'max_distance_reached': f'{spatialunits}',
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