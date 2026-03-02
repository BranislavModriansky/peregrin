from __future__ import annotations
import traceback
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
from typing import *

from .._general import Values
from .._handlers._reports import Level, Reporter


@dataclass
class BaseDataInventory:
    """
    #### *Data inventory of trajectory statistics Dataframes.*

    This class serves as a centralized inventory for the DataFrames that store trajectory statistics at different planes of aggregation. 
    Each attribute is intended to store a DataFrame (one of Spots, Tracks, Frames, TimeIntervals) after any of the computation methods in the Stats class are called.
    """

    Spots: pd.DataFrame
    Tracks: pd.DataFrame
    Frames: pd.DataFrame
    TimeIntervals: pd.DataFrame



class Stats:
    """
    #### *Class for computing trajectory statistics at various levels of aggregation.*

    This class provides methods to compute trajectory statistics at multiple levels of aggregation: 
    per-spot (Spots), per-track (Tracks), per-frame/time point (Frames), and per-time interval/lag (TimeIntervals). 
    Each method processes the input DataFrame and updates the corresponding DataFrame in the BaseDataInventory.

    Parameters
    ----------
    pool_replicates : bool, optional
        *If True, replicates will be pooled together when calculating statistics; if False, statistics will be calculated separately for each replicate. Default is True.*

    Attributes
    ----------
    SIGNIFICANT_FIGURES : int, optional
        *If specified, during computations all values are going to be rounded to this number of significant figures.*

    DECIMALS_PLACES : int, optional
        *If specified, during computations all floating-point values are going to be normalized (rounded) to this number of decimal places.*

    BOOTSTRAP_RESAMPLES : int, optional
        *Number of resamples to perform when calculating bootstrap confidence intervals. Default is 1000.*

    CONFIDENCE_LEVEL : float, optional
        *Confidence level to use when calculating confidence intervals (e.g. 95 for 95% confidence intervals). Default is 95.*
    
    CI_STATISTIC : str, optional
        *Statistic to calculate confidence intervals for (e.g. 'mean', 'median'). Default is 'mean'.*

    Methods
    -------
    GetAllData(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """

    
    SIGNIFICANT_FIGURES: None | int = None
    DECIMALS_PLACES: None | int = None
    BOOTSTRAP_RESAMPLES: int = 1000
    CONFIDENCE_LEVEL: float = 95
    CI_STATISTIC: str = 'mean'
    _ci_method_used: str = 'BCa'

    _PANDAS_BUILTINS = frozenset({
        'mean', 'median', 'std', 'count', 'sum', 'min', 'max',
        'first', 'last', 'var', 'prod', 'size', 'nunique',
    })

    _EXCLUDE_SUFFIXES = ['per', 'Condition', 'Replicate', 'Track ID', 'Track UID', 'Time point', 'Frame', 'Time lag', 'Frame lag', 'sd', 'var', 'sem', 'q25', 'q75']
    _DESCR     = ['min', 'max', 'mean', 'median', 'q25', 'q75']
    _DESCR_ERR = ['std']
    _INFER_ERR = ['sem']

    _COLUMNS = {
        'SPOTS': [
            'X coordinate','Y coordinate',
            'Time point','Frame','Track ID','Condition','Replicate','Track UID',
            'Distance','Cumulative track length','Cumulative track displacement',
            'Cumulative straightness index','Cumulative speed','Direction',
            'Cumulative direction mean','Cumulative direction var'
        ],
        'TRACKS': [
            'Condition','Replicate','Track ID', 'Track points',
            'Track length','Track displacement','Straightness index',
            'Speed min','Speed max','Speed mean','Speed sd','Speed sem',
            'Speed median','Speed q25','Speed q75',
            f'Speed ci{CONFIDENCE_LEVEL} low',f'Speed ci{CONFIDENCE_LEVEL} high',
            'Direction mean','Direction var'
        ],
        'FRAMES': [
            'Condition','Replicate','Time point','Frame',
            'Mean speed','Median speed','Speed sd','Speed sem',
            'Speed q25','Speed q75',
            f'Speed ci{CONFIDENCE_LEVEL} low',f'Speed ci{CONFIDENCE_LEVEL} high',
            'Direction mean','Direction var'
        ],
        'TIMEINTERVALS': [
            'Condition','Replicate','Time lag','Frame lag',
            'Mean speed','Median speed','Speed sd','Speed sem',
            'Speed q25','Speed q75',
            f'Speed ci{CONFIDENCE_LEVEL} low',f'Speed ci{CONFIDENCE_LEVEL} high',
            'Direction mean','Direction var'
        ]
    }


    def __init__(self, pool_replicates: bool = False, *, cat_descr: bool = True, cat_descr_err: bool = False, cat_infer_err: bool = False, bootstrap: bool = False, **kwargs) -> None:

        self.tier = ['Condition'] if pool_replicates else ['Condition', 'Replicate']
        self.noticequeue = kwargs.get('noticequeue', None)

        self.cat_descr = cat_descr
        self.cat_descr_err = cat_descr_err
        self.cat_infer_err = cat_infer_err

        # Build infer-stat list first (bootstrap controls whether CI is included)
        base_infer = ['sem', 'ci'] if bootstrap else ['sem']

        # Always keep these as lists (never None) to avoid list-concat bugs later
        self.DESCR     = list(self._DESCR) if cat_descr else []
        self.DESCR_ERR = list(self._DESCR_ERR) if cat_descr_err else []
        self.INFER_ERR = list(base_infer) if cat_infer_err else []

        self.CUSTOM_AGG_FUNCTIONS = {
            'q25':        self._q25,
            'q75':        self._q75,
            'ci':         self._ci,
            'sem':        self._sem,
            'circ_mean':  self._circ_mean,
            'circ_var':   self._circ_var,
        }


    def GetAllData(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        #### *Computes all trajectory statistics (Spots, Tracks, Frames, TimeIntervals) from raw spot data.*

        Parameters
        ----------
        df : pd.DataFrame
            ***Input DataFrame must contain (at minimum) these columns:***
            - ``Condition``
            - ``Replicate``
            - ``Track ID``
            - ``X coordinate``
            - ``Y coordinate``
            - ``Time point``

        significant_figures : int, optional
            *If provided, rounds all values in the results to the specified number of significant figures.*
        
        normalize_decimals : int, optional
            *If provided, formats all floating-point values in the results to the specified number of decimal places.*

        Returns
        -------
        ``Spots``, ``Tracks``, ``Frames``, ``TimeIntervals`` : pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
            ***Updates and returns BaseDataInventory items -> DataFrames of trajectory statistics at various planes of aggregation.***
        """

        self.Spots(df)
        self.Tracks(BaseDataInventory.Spots)
        self.Frames(BaseDataInventory.Spots)
        self.TimeIntervals(BaseDataInventory.Spots)

        return BaseDataInventory.Spots, BaseDataInventory.Tracks, BaseDataInventory.Frames, BaseDataInventory.TimeIntervals


    def Spots(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        #### *Computes per-point trajectory statistics, both local (previous -> current position) and cumulative (start -> current position).*

        Parameters
        ----------
        df : pd.DataFrame
            ***Input DataFrame must contain (at minimum) these columns:***
            - ``Condition``
            - ``Replicate``
            - ``Track ID``
            - ``X coordinate``
            - ``Y coordinate``
            - ``Time point``

        significant_figures : int, optional
            *If provided, rounds all values in the results to the specified number of significant figures.*
        
        normalize_decimals : int, optional
            *If provided, formats all floating-point values in the results to the specified number of decimal places.*
        
        Returns
        -------
        pd.DataFrame
            ***The input DataFrame, formatted, with additional columns:***

            - **``Distance``**:
            Euclidean distance between consecutive positions (step length).

            - **``Cumulative track length``**:
            Cumulative sum of ``Distances`` along the track up to the current position.

            - **``Cumulative track displacement``**:
            Euclidean distance from the first position in the track to the current position.

            - **``Cumulative straightness index``**:
            ``Cumulative track displacement`` / ``Cumulative track length``.

            - **``Cumulative speed``**:
            Mean speed from start to current position: ``Cumulative track length`` / ``Frame``.

            - **``Direction``**:
            Direction of motion in radians ``np.arctan2(Δy, Δx)``.

            - **``Cumulative direction mean``**:
            Mean of motion directions from start to current position.

            - **``Cumulative direction var``**:
            Cumulative direction variance from start to current position.
        """
        
        if df.empty:
            Reporter(Level.warning, f"Input DataFrame to Spots method is empty; no computations performed.", noticequeue=self.noticequeue)
            return pd.DataFrame(columns=self._COLUMNS['SPOTS'])

        df.sort_values(self.tier + ['Track ID', 'Time point'], inplace=True)

        grp = df.groupby(self.tier + ['Track ID'], sort=False)

        # Provides a completely unique identifier for each track (Track UID) that is consistent across all methods and is used for merging.
        df['Track UID'] = grp.ngroup()
        df.set_index(['Track UID'], drop=False, append=False, inplace=True, verify_integrity=False)

        # Assigns frame numbers within each data subset based on the order of time points; starts at 0 for the first point in each track.
        df['Frame'] = df.groupby(self.tier, sort=False)['Time point'].rank(method='dense').astype('Int64') - 1

        # Optional sanity warning (kept lightweight)
        try:
            bad = df.groupby(self.tier + ['Time point'], sort=False)['Frame'].nunique(dropna=True).max()

            if bad and bad > 1:
                Reporter(Level.warning, f"Multiple frames assigned to the same {self.tier} × Time point combination; this may indicate time point multiplicates within the data or other data issues. Max frames per time point: {bad}.", noticequeue=self.noticequeue)
                pass

        except Exception:
            pass

        # Distance between current and next position
        df['Distance'] = np.hypot(
            grp['X coordinate'].diff(),
            grp['Y coordinate'].diff()
        ).fillna(np.nan)

        # Cumulative track length
        df['Cumulative track length'] = grp['Distance'].cumsum()

        # Net (straight-line) distance: start -> current last position
        start = grp[['X coordinate', 'Y coordinate']].transform('first')
        df['Cumulative track displacement'] = np.hypot(
            (df['X coordinate'] - start['X coordinate']),
            (df['Y coordinate'] - start['Y coordinate'])
        ).replace(0, np.nan)

        # Straightness index: Track displacement vs. actual path length
        # Avoid division by zero by replacing zeros with NaN, then fill
        df['Cumulative straightness index'] = (df['Cumulative track displacement'] / df['Cumulative track length'].replace(0, np.nan)).fillna(np.nan)

        # Cumulative speed: mean speed from start to current position
        df['Cumulative speed'] = df['Cumulative track length'] / df['Frame'].replace(0, np.nan)
        # Direction of travel (radians) based on diff to previous point
        df['Direction'] = np.arctan2(
            grp['Y coordinate'].diff(),
            grp['X coordinate'].diff()
        ).fillna(np.nan)

        # Cumulative direction 
        theta_from_start = np.arctan2(
            df['Y coordinate'] - start['Y coordinate'],
            df['X coordinate'] - start['X coordinate'],
        )

        # Exclude the first frame from cumulative direction stats (no angle data there)
        valid = df['Frame'].gt(0) & theta_from_start.notna()

        # Define grouping keys
        gkeys = self.tier + ['Track ID']

        # Calculate sin and cos of the cumulated direction mean for each point excluding first frame
        sinv = pd.Series(np.where(valid, np.sin(theta_from_start), np.nan), index=df.index)
        cosv = pd.Series(np.where(valid, np.cos(theta_from_start), np.nan), index=df.index)

        gkeys_arrays = [df[k] for k in gkeys]

        # Cumulative sums of sin and cos components
        cum_sin = sinv.groupby(gkeys_arrays, sort=False).cumsum()
        cum_cos = cosv.groupby(gkeys_arrays, sort=False).cumsum()

        # Number of contributing angles so far (starts at 0 on frame 1)
        n_angles = valid.astype(int).groupby(gkeys_arrays, sort=False).cumsum().replace(0, np.nan)

        # Cumulative direction mean and variance
        df['Cumulative direction mean'] = np.arctan2(cum_sin, cum_cos)
        R = (np.hypot(cum_sin, cum_cos) / n_angles)
        df['Cumulative direction var'] = (1.0 - R)

        # Drop (if any) present all nan columns  
        df.dropna(how='all', axis='columns', inplace=True)

        # descr = self.DESCR_ERR + self.DESCR

        # if 'Replicate' in self.tier:
        #     df = self._describe_infer(df, group_cols=['Condition', 'Replicate'], stats=descr)

        # combined_stats = descr + self.INFER_ERR
        # df = self._describe_infer(df, group_cols=['Condition'], stats=combined_stats)

        if self.SIGNIFICANT_FIGURES:
            df = self.Signify(df)
        if self.DECIMALS_PLACES:
            df = self.NormDecimals(df)

        BaseDataInventory.Spots = df

        return df


    def Tracks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        #### *Computes a comprehensive DataFrame of track-level statistics for each trajectory of the input Spots DataFrame.*

        Parameters
        ----------
        df : pd.DataFrame
            ***(Spots) Input DataFrame must contain (at minimum) these columns:***
            - ``Condition``
            - ``Replicate``
            - ``Track ID``
            - ``Distance``
            - ``X coordinate``
            - ``Y coordinate``
            - ``Direction``

        Returns
        -------
        pd.DataFrame
            ***A DataFrame with one row per unique track, containing the following columns:***

            - **``Track length``**:
            Total length of the track (sum of ``Distance``).

            - **``Track displacement``**:
            Euclidean distance from the start to the end of the track.

            - **``Straightness index``**:
            ``Track displacement`` / ``Track length``.

            - **Speed statistics**:
            Minimum, maximum, mean, standard deviation, standard error of the mean, median, interquartile range, 95% confidence interval.

            - **Direction statistics**:
            Circular mean, circular variance.
        """

        if df.empty:
            return pd.DataFrame(columns=self._COLUMNS['TRACKS'])

        stash = df[self.tier + ['Track ID']].drop_duplicates()

        df = df.set_index('Track UID', drop=True, verify_integrity=False)
        gcols = ['Track UID']

        grp = df.groupby(gcols, sort=False)

        # Resolve speed aggregation spec through the resolver
        speed_agg_spec = self.resolve({
            'Speed min':    'min',
            'Speed max':    'max',
            'Speed mean':   'mean',
            'Speed sd':     'std',
            'Speed median': 'median',
            'Speed q25':    'q25',
            'Speed q75':    'q75',
        })

        # Build the named agg dict: each entry is (column, func)
        agg_spec = {
            'Track length': ('Distance', 'sum'),
        }
        for label, func in speed_agg_spec.items():
            agg_spec[label] = ('Distance', func)

        # Add coordinate first/last for displacement calculation
        agg_spec['start_x'] = ('X coordinate', 'first')
        agg_spec['end_x']   = ('X coordinate', 'last')
        agg_spec['start_y'] = ('Y coordinate', 'first')
        agg_spec['end_y']   = ('Y coordinate', 'last')

        agg_spec['Max distance reached'] = ('Cumulative track displacement', 'max')

        agg = grp.agg(**agg_spec)

        # If colors were assigned, carry them over
        if 'Replicate color' in df.columns:
            colors = grp['Replicate color'].first()
            agg = agg.merge(colors, left_index=True, right_index=True)
        if 'Condition color' in df.columns:
            colors = grp['Condition color'].first()
            agg = agg.merge(colors, left_index=True, right_index=True)

        # Other general stats
        other = self._general_agg_stats(df, exclude=self._COLUMNS['SPOTS'])

        # Displacement and straightness
        agg['Track displacement'] = np.hypot(agg['end_x'] - agg['start_x'], agg['end_y'] - agg['start_y'])
        agg['Straightness index'] = (agg['Track displacement'] / agg['Track length'])
        agg = agg.drop(columns=['start_x','end_x','start_y','end_y'])

        # Points per track
        n = grp.size().rename('Track points')
        agg = agg.merge(n, left_index=True, right_index=True)

        # Sin/Cos components for circular statistics
        sin_cos = df.assign(
            _sin=np.sin(df['Direction']), 
            _cos=np.cos(df['Direction'])
        )
        # Group and aggregate sin/cos components along tracks
        dir_agg = sin_cos.groupby(gcols, sort=False).agg(
            mean_sin=('_sin','mean'),
            mean_cos=('_cos','mean')
        )

        # Get circular mean from the aggregated sin/cos
        dir_agg['Direction mean'] = np.arctan2(dir_agg['mean_sin'], dir_agg['mean_cos'])

        # Circular variance
        R = np.hypot(dir_agg['mean_sin'], dir_agg['mean_cos'])
        dir_agg['Direction var'] = (1.0 - R)

        # Remove temporary columns
        dir_agg = dir_agg.drop(columns=['mean_sin','mean_cos'])

        # Merge circular statistics into main agg DataFrame
        df = agg.merge(dir_agg, left_index=True, right_index=True)

        df = df.merge(other, left_index=True, right_index=True, how='right')

        df = stash.merge(df, left_index=True, right_index=True, how='right')

        df.drop_duplicates(inplace=True)
        
        # If 'Replicate' is not present in self.tier, recover it from Spots
        if 'Replicate' not in df.columns:
            rep_map = (
                BaseDataInventory.Spots[['Track UID', 'Replicate']]
                .drop_duplicates(subset=['Track UID'])
                .set_index('Track UID')
            )
            df = df.merge(rep_map, left_on='Track UID', right_index=True, how='left')


        # descr = self.DESCR_ERR + self.DESCR

        # if 'Replicate' in self.tier:
        #     df = self._describe_infer(df, group_cols=['Condition', 'Replicate'], stats=descr)

        # combined_stats = descr + self.INFER_ERR
        # if combined_stats:
        #     df = self._describe_infer(df, group_cols=['Condition'], stats=combined_stats)
        
        if self.SIGNIFICANT_FIGURES:
            df = self.Signify(df)
        if self.DECIMALS_PLACES:
            df = self.NormDecimals(df)
        
        BaseDataInventory.Tracks = df

        return df
    

    def Frames(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        #### *Computes time point statistics aggregated across tracks for each tier × Time point.*

        Supports category-labeled outputs:
        - `{per replicate}` when working with replicates
        - `{per condition}` when pooled or when optional condition-level stats are attached
        """

        reps = 'Replicate' in self.tier
        base_prefix = '{per replicate}' if reps else '{per condition}'
        add_conds = reps and (self.cat_descr or self.cat_descr_err or self.cat_infer_err)

        group_cols = self.tier + ['Time point', 'Frame']
        cond_group_cols = ['Condition', 'Time point', 'Frame']

        metrics = [
            'Cumulative track length',
            'Cumulative track displacement',
            'Cumulative straightness index',
            'Cumulative speed',
            'Distance',
        ]
        metric_out = {m: m for m in metrics}
        metric_out['Distance'] = 'Instantaneous speed'

        # Stats are driven by configured category lists
        requested_stats = list(dict.fromkeys(self.DESCR + self.DESCR_ERR + self.INFER_ERR))
        resolved_stats = self.resolve(requested_stats) if requested_stats else {}

        def _stat_label(stat_name: str) -> str:
            return 'sd' if stat_name == 'std' else stat_name

        def _build_expected_cols(prefixes: list[str]) -> list[str]:
            cols = []
            for pfx in prefixes:
                for m in metrics:
                    mout = metric_out[m]
                    for s in requested_stats:
                        if s == 'ci':
                            cols.append(f'{pfx} {mout} {self.CI_STATISTIC} ci{self.CONFIDENCE_LEVEL} low')
                            cols.append(f'{pfx} {mout} {self.CI_STATISTIC} ci{self.CONFIDENCE_LEVEL} high')
                        else:
                            cols.append(f'{pfx} {mout} {_stat_label(s)}')

            cols.extend([
                f'{pfx} Instantaneous direction mean',
                f'{pfx} Instantaneous direction var',
                f'{pfx} Cumulative direction global mean',
                f'{pfx} Cumulative direction global var',
            ])
            return cols

        if df is None or df.empty:
            Reporter(Level.warning, "Input DataFrame to Frames method is empty; no computations performed.", noticequeue=self.noticequeue)
            prefixes = [base_prefix] + (['{per condition}'] if add_conds else [])
            cols = group_cols + _build_expected_cols(prefixes)
            return pd.DataFrame(columns=cols)

        def _compute_level(source: pd.DataFrame, by_cols: list[str], prefix: str) -> pd.DataFrame:
            g = source.groupby(by_cols, sort=False, observed=True)
            out = pd.DataFrame(index=g.size().index)

            # Scalar stats
            for m in metrics:
                mout = metric_out[m]
                sgroup = g[m]

                for s, func in resolved_stats.items():
                    if s == 'ci':
                        ci = sgroup.agg(func)
                        out[f'{prefix} {mout} ci{self.CONFIDENCE_LEVEL} low'] = ci.apply(
                            lambda x: x[0] if isinstance(x, tuple) and len(x) == 2 else np.nan
                        )
                        out[f'{prefix} {mout} ci{self.CONFIDENCE_LEVEL} high'] = ci.apply(
                            lambda x: x[1] if isinstance(x, tuple) and len(x) == 2 else np.nan
                        )
                    else:
                        out[f'{prefix} {mout} {_stat_label(s)}'] = sgroup.agg(func)

            # Circular stats
            tmp = source[by_cols + ['Direction', 'Cumulative direction mean']].copy()
            dir_vals = pd.to_numeric(tmp['Direction'], errors='coerce').to_numpy(dtype=float)
            cum_vals = pd.to_numeric(tmp['Cumulative direction mean'], errors='coerce').to_numpy(dtype=float)

            tmp['_sin_dir'] = np.sin(dir_vals)
            tmp['_cos_dir'] = np.cos(dir_vals)
            tmp['_sin_cum'] = np.sin(cum_vals)
            tmp['_cos_cum'] = np.cos(cum_vals)

            circ = tmp.groupby(by_cols, sort=False, observed=True).agg(
                sin_dir=('_sin_dir', 'mean'),
                cos_dir=('_cos_dir', 'mean'),
                sin_cum=('_sin_cum', 'mean'),
                cos_cum=('_cos_cum', 'mean'),
            )

            out[f'{prefix} Instantaneous direction mean'] = np.arctan2(circ['sin_dir'], circ['cos_dir'])
            out[f'{prefix} Instantaneous direction var'] = 1.0 - np.hypot(circ['sin_dir'], circ['cos_dir'])

            out[f'{prefix} Cumulative direction global mean'] = np.arctan2(circ['sin_cum'], circ['cos_cum'])
            out[f'{prefix} Cumulative direction global var'] = 1.0 - np.hypot(circ['sin_cum'], circ['cos_cum'])

            return out.reset_index()

        # Base level (per replicate or per condition)
        base_df = _compute_level(df, group_cols, base_prefix)

        # Optional per-condition stats attached to replicate rows (TimeIntervals-like behavior)
        if add_conds:
            cond_df = _compute_level(df, cond_group_cols, '{per condition}')
            base_df = base_df.merge(cond_df, on=cond_group_cols, how='left')

        if self.SIGNIFICANT_FIGURES:
            base_df = self.Signify(base_df)
        if self.DECIMALS_PLACES:
            base_df = self.NormDecimals(base_df)

        BaseDataInventory.Frames = base_df
        return base_df
    

    def TimeIntervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        #### *Computes time interval (time lag) statistics across tracks for each tier × Time lag.*
        
        Parameters
        ----------
        df : pd.DataFrame
            ***(Spots) Input DataFrame must contain (at minimum) these columns:***
            - ``Condition``
            - ``Replicate``
            - ``Track ID``
            - ``X coordinate``
            - ``Y coordinate``
            - ``Time point``

        Returns
        -------
        pd.DataFrame
            ***A DataFrame with one row per unique tier × Time lag. Contains following columns:***

            - **``Frame lag``**: 
            Time lag in frames (integer).

            - **``Time lag``**:
            Time lag in actual time units, computed from the median time step of the input data.

            - **``Tracks contributing``**: 
            Number of tracks that contributed to the statistics at that time lag.

            **``MSD`` *(mean squared displacement)*** statistics across trajectories at specific time intervals/lags:
            - **``min``**: minimal MSD.
            - **``max``**: maximal MSD.
            - **``mean``**: mean MSD.
            - **``sd``**: standard deviation of MSD.
            - **``median``**: median MSD.
            - **``q25``**: 25th percentile of MSD.
            - **``q75``**: 75th percentile of MSD.

            **``Turn``** statistics across circular mean of turning angles for trajectories at specific time intervals/lags:
            - **``mean``**: circular mean of turning angles.
            - **``var``**: circular variance of turning angles.
        """

        if df.empty: 
            Reporter(Level.warning, f"Input DataFrame to TimeIntervals method is empty; no computations performed.", noticequeue=self.noticequeue)
            return pd.DataFrame(columns=self._COLUMNS['TIMEINTERVALS'])

        # Ensure Track UID is the index (it may be a column if called outside of the normal pipeline)
        if df.index.name != 'Track UID' and 'Track UID' in df.columns:
            df = df.set_index('Track UID', drop=False, verify_integrity=False)

        # Unique time point differences -> time steps; use median to resist irregular sampling
        t_unique = np.sort(df['Time point'].unique())

        # <2 unique time points returns an empty DataFrame.
        if t_unique.size < 2:
            return pd.DataFrame(columns=self._COLUMNS['TIMEINTERVALS'])
        
        t_step = float(np.diff(t_unique)[0])

        # Collect per-track metrics at at lag
        per_track_msd = defaultdict(list)
        per_track_turn_mean = defaultdict(list)

        reps = 'Replicate' in self.tier
        add_conds = reps and (self.cat_descr or self.cat_descr_err or self.cat_infer_err)

        # Iterate through tracks -> accepts Track UID either as an ordinary column or an index
        for _, g in df.groupby(level='Track UID', sort=False):

            # Number of points in this track
            n = len(g)

            # Skips tracks with <2 points
            if n < 2:
                continue

            # Sort frames within track
            g = g.sort_values('Time point')

            # Get coordinates and categories
            x = g['X coordinate'].to_numpy()
            y = g['Y coordinate'].to_numpy()
            cond = g['Condition'].iloc[0]
            rep  = g['Replicate'].iloc[0] if reps else None


            # Displacements between consecutive points -> Angles
            dx1 = x[1:] - x[:-1]
            dy1 = y[1:] - y[:-1]
            theta = np.arctan2(dy1, dx1)

            # Compute per-track time lag metrics for lags 1 to n-1
            for lag in range(1, n):

                # X coord and Y coord displacements
                dx = x[lag:] - x[:-lag]
                dy = y[lag:] - y[:-lag]

                # Mean squared displacement
                if dx.size > 0:
                    msd_track = float((dx**2 + dy**2).mean())
                    if reps:
                        per_track_msd[(cond, rep, lag)].append(msd_track)
                        if add_conds:
                            per_track_msd[(cond, lag)].append(msd_track)
                    else:
                        per_track_msd[(cond, lag)].append(msd_track)

                # Turning angles for this lag
                if theta.size > lag:
                    dtheta = self._wrap_pi(theta[lag:] - theta[:-lag])
                    if dtheta.size > 0:
                        if reps:
                            per_track_turn_mean[(cond, rep, lag)].append(self._circ_mean(dtheta))
                            if add_conds:
                                per_track_turn_mean[(cond, lag)].append(self._circ_mean(dtheta))
                        else:
                            per_track_turn_mean[(cond, lag)].append(self._circ_mean(dtheta))

        # If no tracks had enough points to compute any metrics, return an empty DataFrame with the correct columns.
        if not per_track_msd and not per_track_turn_mean:
            return pd.DataFrame(columns=self._COLUMNS['TIMEINTERVALS'])

        # Summarize computed mean squared displacements and turning angles across tracks
        keys = set(per_track_msd.keys()) | set(per_track_turn_mean.keys())

        # If working per-replicate, only create base rows for (cond, rep, lag).
        # Condition-level stats are then attached to these rows when add_conds=True.
        if reps:
            keys = {k for k in keys if len(k) == 3}

        rows = []
        cat = '{per replicate}' if reps else '{per condition}'

        if not self.cat_descr or not self.cat_descr_err:
            Reporter(Level.info, f"Exception -> Descriptive, per-category statistics are going to be computed for TimeIntervals even when disabled.", noticequeue=self.noticequeue)

        for key in keys:
            if len(key) == 3:
                cond, rep, lag = key
                is_rep_row = True
            else:
                cond, lag = key
                rep = None
                is_rep_row = False

            if is_rep_row:
                arr = np.asarray(per_track_msd.get((cond, rep, lag), []), dtype=float)
                turn_means = np.asarray(per_track_turn_mean.get((cond, rep, lag), []), dtype=float)
            else:
                arr = np.asarray(per_track_msd.get((cond, lag), []), dtype=float)
                turn_means = np.asarray(per_track_turn_mean.get((cond, lag), []), dtype=float)

            cat_n_tracks = arr.size
            turn_mean_agg = self._circ_mean(turn_means) if turn_means.size else np.nan
            turn_var_agg = self._circ_var(turn_means) if turn_means.size else np.nan

            if add_conds and is_rep_row:
                cond_arr = np.asarray(per_track_msd.get((cond, lag), []), dtype=float)
                cond_n_tracks = cond_arr.size
                turn_cond_means = np.asarray(per_track_turn_mean.get((cond, lag), []), dtype=float)
                cond_turn_mean_agg = self._circ_mean(turn_cond_means) if turn_cond_means.size else np.nan
                cond_turn_var_agg = self._circ_var(turn_cond_means) if turn_cond_means.size else np.nan

            row = {
                'Condition': cond,
                'Frame lag': lag,
                'Time lag': lag * t_step,
                f'{cat} Tracks contributing': int(max(cat_n_tracks, turn_means.size)),

                f'{cat} MSD min':    float(arr.min()) if cat_n_tracks else None,
                f'{cat} MSD max':    float(arr.max()) if cat_n_tracks else None,
                f'{cat} MSD mean':   float(arr.mean()) if cat_n_tracks else None,
                f'{cat} MSD median': float(np.median(arr)) if cat_n_tracks else None,
                f'{cat} MSD q25':    float(self._q25(arr)) if cat_n_tracks else None,
                f'{cat} MSD q75':    float(self._q75(arr)) if cat_n_tracks else None,

                f'{cat} Turn mean':  np.rad2deg(np.abs(turn_mean_agg)) if not np.isnan(turn_mean_agg) else None,
            }

            if self.cat_descr_err:
                row.update({
                    f'{cat} MSD sd':   float(arr.std(ddof=1)) if cat_n_tracks > 1 else None,
                    f'{cat} Turn var': turn_var_agg if not np.isnan(turn_var_agg) else None,
                })

            if self.cat_infer_err:
                ci_low, ci_high = self._ci(arr) if cat_n_tracks > 1 else (None, None)
                row.update({
                    f'{cat} MSD sem':    float(self._sem(arr)) if cat_n_tracks > 1 else None,
                    f'{cat} MSD {self.CI_STATISTIC} ci{self.CONFIDENCE_LEVEL} low':  ci_low if ci_low is not None and not np.isnan(ci_low) else None,
                    f'{cat} MSD {self.CI_STATISTIC} ci{self.CONFIDENCE_LEVEL} high': ci_high if ci_high is not None and not np.isnan(ci_high) else None,
                })
            

            if add_conds and is_rep_row:
                if self.cat_descr:
                    row.update({
                        '{per condition} MSD min':    float(cond_arr.min()) if cond_n_tracks else None,
                        '{per condition} MSD max':    float(cond_arr.max()) if cond_n_tracks else None,
                        '{per condition} MSD mean':   float(cond_arr.mean()) if cond_n_tracks else None,
                        '{per condition} MSD median': float(np.median(cond_arr)) if cond_n_tracks else None,
                        '{per condition} MSD q25':    float(self._q25(cond_arr)) if cond_n_tracks else None,
                        '{per condition} MSD q75':    float(self._q75(cond_arr)) if cond_n_tracks else None,
                        '{per condition} Turn mean':  np.rad2deg(np.abs(cond_turn_mean_agg)) if not np.isnan(cond_turn_mean_agg) else None,
                    })
    
                if self.cat_descr_err:
                    row.update({
                        '{per condition} MSD sd':     float(cond_arr.std(ddof=1)) if cond_n_tracks > 1 else None,
                        '{per condition} Turn var':   cond_turn_var_agg if not np.isnan(cond_turn_var_agg) else None,
                    })

                if self.cat_infer_err:
                    ci_low, ci_high = self._ci(cond_arr) if cond_n_tracks > 1 else (None, None)
                    row.update({
                        '{per condition} MSD sem':    float(self._sem(cond_arr)) if cond_n_tracks > 1 else None,
                        f'{{per condition}} MSD {self.CI_STATISTIC} ci{self.CONFIDENCE_LEVEL} low':  ci_low if ci_low is not None and not np.isnan(ci_low) else None,
                        f'{{per condition}} MSD {self.CI_STATISTIC} ci{self.CONFIDENCE_LEVEL} high': ci_high if ci_high is not None and not np.isnan(ci_high) else None,
                    })

                if self.cat_descr or self.cat_infer_err or self.cat_descr_err:
                    row.update({'{per condition} Tracks contributing': int(cond_n_tracks)})

            if rep is not None:
                row = self._insert_at_position(row, 'Replicate', rep, where=1)

            rows.append(row)

        df = pd.DataFrame(rows).sort_values(
            self.tier + ['Frame lag'], ignore_index=True
        )

        # JSON-safe cleanup for Shiny/front-end serializers (no NaN/Inf in strict JSON)
        df = df.replace([np.inf, -np.inf], np.nan)

        if self.SIGNIFICANT_FIGURES:
            df = self.Signify(df)
        if self.DECIMALS_PLACES:
            df = self.NormDecimals(df)

        BaseDataInventory.TimeIntervals = df
        return df


    def FormatDigits(self, df: pd.DataFrame, *, sig_figs: int = None, decimals: int = None) -> pd.DataFrame:
        
        if sig_figs:
            df = self.Signify(df, sig_figs=sig_figs)

        if decimals:
            df = self.NormDecimals(df, decimals=decimals)

        return df


    def Signify(self, df: pd.DataFrame, *, sig_figs: int = None) -> float:
        """
        #### *Round all numeric values in a DataFrame to the specified number of significant figures.*

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with numeric values to be rounded
        sig_figs : int
            Number of significant figures to round to.

        Returns
        -------
        pd.DataFrame
            DataFrame with all numeric values rounded to the specified number of significant figures.
        """

        if df.empty:
            return df.copy()

        if sig_figs is None:
            sig_figs = self.SIGNIFICANT_FIGURES

        valuer = Values()

        df_rounded = df.copy()
        for col in df_rounded.select_dtypes(include=[np.number]).columns:
            df_rounded[col] = df_rounded[col].apply(lambda x: valuer.RoundSigFigs(x, sigfigs=sig_figs))

        return df_rounded
    

    def NormDecimals(self, df: pd.DataFrame, decimals: int = None) -> pd.DataFrame:
        """
        #### *Normalize numeric values per column in a DataFrame, so that each column's values have a consistent amount of decimals.*

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with numeric values to be normalized.

        Returns
        -------
        pd.DataFrame
            DataFrame with all numeric values normalized to consistent decimals.
        """

        if df.empty:
            return df.copy()
        
        if decimals is None:
            decimals = self.DECIMALS_PLACES

        # Ensure all values have the same number of decimals: (round, fill)
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].apply(lambda x: round(x, decimals) if pd.notnull(x) else x)

        return df


    def _general_agg_stats(self, df: pd.DataFrame, exclude: list[str], *, group_by: list[str] = ['Track UID']) -> pd.DataFrame:
        """
        #### *Computes basic statistics (min, max, mean, sd, sem, median) for all numeric columns in the input DataFrame, excluding core columns.*

        Parameters
        ----------
        df : pd.DataFrame
            ***Unstripped Spots DataFrame** containing columns, other than the core <- ``self._COLUMNS['SPOTS']`` columns, with numeric values to be aggregated.*

        exclude : list[str]
            *List of column names to be excluded from aggregation.*

        group_by : list[str], optional
            *List of column or index names to group by. Default is ['Track UID'].*

        Returns
        -------
        pd.DataFrame
            *DataFrame with basic statistics for each additional numeric column which is not found in the core <- ``self._COLUMNS['SPOTS']`` columns.*
        """

        if exclude is None:
            return pd.DataFrame()

        # Keep only numeric columns and exclude core columns
        df = df.select_dtypes(include=[np.number]).drop(columns=exclude, errors='ignore')

        # Stash leftover columns
        other_cols = df.columns.tolist()

        # Group by Track UID
        grp = df.groupby(level=group_by, sort=False)

        # For each bonus column, compute basic statistics, rename columns, and merge back
        for col in other_cols:
            agg = grp[col].agg(['min','max','mean','std','sem','median'])

            agg.columns = [f"{col} min", f"{col} max", f"{col} mean", f"{col} sd", f"{col} sem", f"{col} median"]

            df = df.merge(agg, left_index=True, right_index=True)

        # Drop original columns
        df.drop(columns=other_cols, inplace=True)

        # drop multiplicates if present
        df = df.drop_duplicates()
        
        return df


    def _describe_infer(self, df: pd.DataFrame, group_cols: list[str], *, stats: dict[str, str] | list[str] = None, **kwargs) -> pd.Series:
        if not stats:
            return df

        # only numeric columns, excluding id/category-like columns
        value_cols = [c for c in df.columns
                      if c not in group_cols
                      and pd.api.types.is_numeric_dtype(df[c])
                      and not any(s in c for s in kwargs.get('exclude', self._EXCLUDE_SUFFIXES))]

        resolving = self.resolve(stats)
        resolving_circular = self.resolve(kwargs.get('circular_stats', {'mean': 'circ_mean', 'var': 'circ_var'}))

        # named aggregation so output columns are already flat
        named_agg = {}
        for col in value_cols:
            
            if any(t in col for t in ['Direction', 'direction', 'Turn', 'turn']) and not col.endswith('var'):
                for stat_name, func in resolving_circular.items():
                    named_agg[f"{'{'}per {group_cols[-1].lower()}{'}'} {col} {stat_name}"] = (col, func)
            else:
                for stat_name, func in resolving.items():
                    if stat_name != 'ci':
                        named_agg[f"{'{'}per {group_cols[-1].lower()}{'}'} {col} {stat_name}"] = (col, func)
                    else:
                        named_agg[f"{'{'}per {group_cols[-1].lower()}{'}'} {col} {self.CI_STATISTIC} ci{self.CONFIDENCE_LEVEL}"] = (col, func)

        grp_stats = (
            df.groupby(group_cols, observed=True, sort=False)
            .agg(**named_agg)
            .reset_index())

        return df.merge(grp_stats, on=group_cols, how='left')
    
    
    def resolve(self, agg_spec: dict[str, str] | list[str]) -> dict[str, str | callable]:
        """
        ***Resolve an aggregation specification dict so that custom names
        (e.g. 'iqr', 'ci95') are replaced with their callable implementations
        while built-in pandas names are kept as strings.***

        Parameters
        ----------
        agg_spec : dict[str, str] | list[str]
            *Either a list of aggregation function names (e.g. ['mean', 'iqr', 'ci95']) or a dict mapping output labels to function names (e.g. {'Speed mean': 'mean', 'Speed iqr': 'iqr', 'Speed ci95': 'ci95'}).*

        Returns
        -------
        dict[str, str | callable]
            Ready to pass to ``pdpd.groupby().agg()``.
        """
        
        resolved = {}
        if isinstance(agg_spec, list):
            for func_name in agg_spec:
                if func_name in self._PANDAS_BUILTINS:
                    resolved[func_name] = func_name
                elif func_name in self.CUSTOM_AGG_FUNCTIONS:
                    resolved[func_name] = self.CUSTOM_AGG_FUNCTIONS[func_name]
                else:
                    raise ValueError(
                        f"Unknown aggregation '{func_name}'. "
                        f"Available: {sorted(self._PANDAS_BUILTINS | set(self.CUSTOM_AGG_FUNCTIONS))}"
                    )

        elif isinstance(agg_spec, dict):
            for label, func_name in agg_spec.items():
                if func_name in self._PANDAS_BUILTINS:
                    resolved[label] = func_name
                elif func_name in self.CUSTOM_AGG_FUNCTIONS:
                    resolved[label] = self.CUSTOM_AGG_FUNCTIONS[func_name]
                else:
                    raise ValueError(
                        f"Unknown aggregation '{func_name}'. "
                        f"Available: {sorted(self._PANDAS_BUILTINS | set(self.CUSTOM_AGG_FUNCTIONS))}"
                    )
        return resolved
        

    def _insert_at_position(self, d: dict, key: Any, value: Any = None, *, where: int | str = 0) -> dict:
        """
        Insert a key-value pair into a dictionary at a specific position.

        Parameters
        ----------
        d : dict
            *The original dictionary.*

        insert : tuple
            *The key-value pair to insert.*

        where : int | str, optional (default=0)
            *The position at which to insert the new key-value pair. If an integer, it is treated as an index. If a string, it is treated as a key name.*
        """

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


    def _wrap_pi(self, a: np.ndarray) -> np.ndarray:
        """Wrap angles in radians to the range [-π, π]."""
        return (a + np.pi) % (2*np.pi) - np.pi


    def _circ_mean(self, a: np.ndarray) -> float:
        """Circular mean of angles in radians."""
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return np.nan
        
        s = np.nanmean(np.sin(a))
        c = np.nanmean(np.cos(a))
        if np.isnan(s) or np.isnan(c):
            return np.nan
        
        return float(np.arctan2(s, c))


    def _circ_var(self, a: np.ndarray) -> float:
        """Circular variance defined as 1 - R, where R is the mean resultant length of the angles."""
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return np.nan
        
        s = np.nanmean(np.sin(a))
        c = np.nanmean(np.cos(a))
        if np.isnan(s) or np.isnan(c):
            return np.nan
        
        R = np.hypot(s, c)
        return float(1.0 - R)
    

    def _q25(self, a: np.ndarray) -> float:
        """Lower bound of the interquartile range = Q1."""
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]  # drop NaN/Inf
        if a.size == 0:
            return np.nan
        return float(np.percentile(a, 25))
    

    def _q75(self, a: np.ndarray) -> float:
        """Upper bound of the interquartile range = Q3."""
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]  # drop NaN/Inf
        if a.size == 0:
            return np.nan
        return float(np.percentile(a, 75))
    

    def _ci(self, a, *, n_resamples: int | None = None, confidence_level: float | None = None, **kwargs) -> tuple[float, float]:
        """
        ***Confidence interval via bootstrap.***

        
        Parameters
        ----------
        a : array-like
            *1D array of values to compute the confidence interval for.*
        
        n_resamples : int, optional = ``self.BOOTSTRAP_RESAMPLES``
            *Number of bootstrap resamples to perform. Default is ``1000``.*

        confidence_level : float, optional = ``self.CONFIDENCE_LEVEL``
            *Confidence level for the interval. Default is ``95`` (%).*

        statistic : callable, optional
            *Function for which the confidence interval is computed (e.g. ``np.mean``, ``np.median``). Default is ``np.mean``.*
        
        method : str, optional
            *Method for confidence interval calculation. Default is ``'BCa'`` (bias-corrected and accelerated). 
            If ``'BCa'`` fails, the method falls back to ``'percentile'``. The used method is stored in ``self._ci_method_used`` for reference.*
            
        Returns
        -------
        tuple[float, float]
            *A tuple containing the lower and upper bounds of the confidence interval. If computation fails, returns ``(np.nan, np.nan)``.*
        """

        method = kwargs.get('method', 'BCa')
        seed = 42 # Fixed seed for reproducibility
        
        # Ensure input is a numpy array of floats for the bootstrap compatibility 
        a = np.asarray(a, dtype=float)

        # Drop NaN values, for bootstrap cannot handle them
        a = a[~np.isnan(a)]

        if a.size < 2:
            return (np.nan, np.nan)

        # Convert percentage (e.g. 95) to fraction (0.95) for scipy
        cl = self.CONFIDENCE_LEVEL if confidence_level is None else confidence_level
        if cl > 1:
            cl = cl / 100.0

        try:
            result = stats.bootstrap(
                (a,),
                statistic=kwargs.get('statistic', getattr(np, self.CI_STATISTIC)),
                n_resamples=self.BOOTSTRAP_RESAMPLES if n_resamples is None else n_resamples,
                confidence_level=cl,
                method=method,
                random_state=seed
            )
            self._ci_method_used = method
            return (float(result.confidence_interval.low), float(result.confidence_interval.high))
        
        except Exception:
            # Fallback to the percentile method if previous the method fails
            try:
                result = stats.bootstrap(
                    (a,),
                    statistic=kwargs.get('statistic', getattr(np, self.CI_STATISTIC)),
                    n_resamples=self.BOOTSTRAP_RESAMPLES if n_resamples is None else n_resamples,
                    confidence_level=cl,
                    method='percentile',
                    random_state=seed
                )
                self._ci_method_used = 'percentile'
                return (float(result.confidence_interval.low), float(result.confidence_interval.high))
            
            except Exception as e:
                Reporter(Level.error, f"Confidence interval computation failed: {e}", trace=traceback.format_exc(), noticequeue=self.noticequeue)
                return (np.nan, np.nan)
    
    def _sem(self, x: np.ndarray | pd.Series) -> float:
        """Standard error of the mean."""
        if isinstance(x, np.ndarray):
            # x = x[~np.isnan(x)]
            n = len(x)
            if n < 2:
                return np.nan
            return np.std(x, ddof=1) / np.sqrt(n)
        else:
            n = x.count()
            if n < 2:
                return np.nan
            return x.std(ddof=1) / np.sqrt(n)

    
    


class Summarize:

    @staticmethod
    def dataframe_summary(df: pd.DataFrame) -> dict:
        return {
            "rows": len(df),
            "columns": df.shape[1],
            "missing_cells": int(df.isna().sum().sum()),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        }

    @staticmethod
    def column_summary(series: pd.Series) -> dict:
        # Robust handling of pandas nullable dtypes (pd.NA) and mixed types
        if pd.api.types.is_numeric_dtype(series):
            s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

            # If there is at least one real numeric value, treat as numeric summary
            if s.notna().any():
                mode = s.mode(dropna=True)
                return {
                    "type": "type_one",
                    "missing": int(series.isna().sum()),
                    "distinct": int(series.nunique(dropna=True)),
                    "min": s.min(skipna=True),
                    "max": s.max(skipna=True),
                    "mean": s.mean(skipna=True),
                    "median": s.median(skipna=True),
                    "mode": float(mode.iloc[0]) if not mode.empty else None,
                    "sd": s.std(ddof=1, skipna=True),
                    "variance": s.var(skipna=True),
                }

        value_counts = series.value_counts(dropna=True, normalize=True).head(3)
        return {
            "type": "type_zero",
            "missing": int(series.isna().sum()),
            "distinct": int(series.nunique(dropna=True)),
            "top": [(idx, round(val * 100, 1)) for idx, val in value_counts.items()],
        }

