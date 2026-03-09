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
    """ Data inventory

    Serves as an inventory for the DataFrames computed via the `(class) Stats`, storing trajectory statistics at different planes of aggregation. 

    Attributes
    ----------
    Spots : pd.DataFrame
        *Contains per-trajectory-point statistics, including both local (previous -> current position) and cumulative (start -> current position) metrics.*

    Tracks : pd.DataFrame
        *Contains whole-trajectory statistics. Single row per unique trajectory.*

    Frames : pd.DataFrame
        *Contains per-time-point statistics, including local and cumulative metrics.*

    TimeIntervals : pd.DataFrame
        *Contains per-time-interval statistics.*

    See also
    --------
    `(class) Stats` - a class with methods for computing the mentioned DataFrames.
    """

    Spots: pd.DataFrame
    Tracks: pd.DataFrame
    Frames: pd.DataFrame
    TimeIntervals: pd.DataFrame


class Stats:
    """ A class providing methods for computing trajectory statistics at various levels of aggregation: \n
    Spots (per-trajectory-point), Tracks (per-whole-trajectory), Frames (per-time-point), Time intervals (per-time-interval). \n
    Calling this method initializes its statistical configuration.

    Parameters
    ----------
    cat_descr : bool, default True
        *If True, descriptive statistics (min, max, mean, median, q25, q75) will be computed for the.*
    
    cat_descr_err : bool, default True
        *If True, descriptive error statistics (std) will be computed.*

    cat_infer_err : bool, default False
        *If True, inferative statistics (sem, ?ci) will be computed.

    bootstrap_ci : bool, default False
        *If True, ci will be computed when the `cat_infer_err` is set to True.*


    Attributes
    ----------
    SIGNIFICANT_FIGURES : int, optional
        *If specified, all values are going to be rounded to the given number of significant figures.*

    DECIMALS_PLACES : int, optional
        *If specified, all floating-point values are going to be rounded to the given number of decimal places.*

    BOOTSTRAP_RESAMPLES : int, default 1000
        *A number of resamples to perform when calculating bootstrap confidence intervals.*

    CONFIDENCE_LEVEL : int, default 95
        *Confidence level (%) to use when calculating confidence intervals.*
    
    CI_STATISTIC : str, default 'mean'
        *Statistic to calculate confidence intervals for (e.g. 'mean', 'median').*

    
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
            'Condition','Replicate','Track ID','Track UID','Time point','Frame',
            'X coordinate','Y coordinate','Distance','Cumulative track length','Cumulative track displacement',
            'Cumulative straightness index','Cumulative speed','Direction','Cumulative direction mean','Cumulative direction var'
        ],
        'TRACKS': [
            'Condition','Replicate','Track ID','Track UID',
            'Track length','Track displacement','Straightness index',
            'Speed min','Speed max','Speed mean','Speed sd','Speed median','Speed q25','Speed q75',
            'Max distance reached','Track start frame','Track end frame',
            'Direction mean','Direction var'
        ],
        'FRAMES':        ['Condition','Replicate','Time point','Frame'],
        'TIMEINTERVALS': ['Condition','Replicate','Time lag','Frame lag']
    }


    def __init__(self, *, cat_descr: bool = True, cat_descr_err: bool = True, cat_infer_err: bool = False, bootstrap_ci: bool = False, **kwargs) -> None:

        self.tier = ['Condition', 'Replicate']
        self.noticequeue = kwargs.get('noticequeue', None)

        self.cat_descr = cat_descr
        self.cat_descr_err = cat_descr_err
        self.cat_infer_err = cat_infer_err

        base_infer = ['sem', 'ci'] if bootstrap_ci else ['sem']

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


    def get_all_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """ Computes all trajectory statistics (Spots, Tracks, Frames, TimeIntervals) from raw trajectory spot (track point) data.

        Parameters
        ----------
        df : pd.DataFrame
            ***Input DataFrame must contain these columns:***
            - `Condition`
            - `Replicate`
            - `Track ID`
            - `X coordinate`
            - `Y coordinate`
            - `Time point`

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            `Stats.Spots()`, `Stats.Tracks()`, `Stats.Frames()` and `Stats.TimeIntervals()` DataFrames.

            \n *Sets `BaseDataInventory.Spots`, `BaseDataInventory.Tracks`, `BaseDataInventory.Frames`, `BaseDataInventory.TimeIntervals` to the computed DataFrames.*

        See also
        --------
        `Stats.Spots()`- 
        computes per-trajectory-point statistics, both local (previous -> current position) and cumulative (start -> current position).

        `Stats.Tracks()`- 
        computes per-whole-trajectory statistics from the Spots DataFrame.

        `Stats.Frames()`- 
        computes per-time-point statistics from the Spots DataFrame.

        `Stats.TimeIntervals()`- 
        computes per-time-interval statistics from the Spots DataFrame.

        `(dataclass) BaseDataInventory`- 
        serves as an inventory, storing the computed DataFrames computed via the `(class) Stats`.

        """

        self.Spots(df)
        self.Tracks(BaseDataInventory.Spots)
        self.Frames(BaseDataInventory.Spots)
        self.TimeIntervals(BaseDataInventory.Spots)

        return BaseDataInventory.Spots, BaseDataInventory.Tracks, BaseDataInventory.Frames, BaseDataInventory.TimeIntervals


    def Spots(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes per-trajectory-point statistics, both local (previous -> current position) and cumulative (start -> current position).

        Parameters
        ----------
        df : pd.DataFrame
            ***The input DataFrame must contain these columns:***
            - `Condition`
            - `Replicate`
            - `Track ID`
            - `X coordinate`
            - `Y coordinate`
            - `Time point`

        Returns
        -------
        pd.DataFrame
            *The computed DataFrame containing these columns:*

            - **`Condition`**
            - **`Replicate`**
            - **`Track ID`**
            - **`Track UID`**
            - **`Time point`**
            - **`Frame`**

            - **`X coordinate`**
            - **`Y coordinate`**

            - **`Distance`**- 
            Euclidean distance between consecutive (previous -> current) positions (step length).

            - **`Cumulative track length`**- 
            Cumulative sum of `Distance` along the track up to the current position.

            - **`Cumulative track displacement`**- 
            Euclidean distance from the starting position of the track to the current position.

            - **`Cumulative straightness index`**- 
            Track's straigtness calculated as `Cumulative track displacement` / `Cumulative track length`.

            - **`Cumulative speed`**- 
            Mean speed (`Distance`) from the starting to the current position <- `Cumulative track length` / `Frame`.

            - **`Direction`**- 
            Instantaneous direction of motion in radians `np.arctan2(Δy, Δx)`. Calculated between the previous and current positions.

            - **`Cumulative direction mean`**- 
            Mean of directions of motion from the starting to the current position.

            - **`Cumulative direction var`**- 
            Cumulative direction variance from the starting to the current position.

            - **`Other`**- 
            *any additional columns from the input DataFrame that are not part of the above list will be retained in the output if they contain any non-NA values; otherwise, they will be dropped.*

            \n Sets `BaseDataInventory.Spots` to the computed DataFrame.

        See also
        --------
        `Stats.get_all_data()`- 
        computes all DataFrames (Spots, Tracks, Frames, TimeIntervals) from raw spot data in one call.

        `Stats.Tracks()`- 
        computes per-whole-trajectory statistics from the Spots DataFrame.

        `Stats.Frames()`- 
        computes per-time-point statistics from the Spots DataFrame.

        `Stats.TimeIntervals()`- 
        computes per-time-interval statistics from the Spots DataFrame.

        `(dataclass) BaseDataInventory`- 
        serves as an inventory, storing the computed DataFrames computed via the `(class) Stats`.

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

        if self.SIGNIFICANT_FIGURES:
            df = self.Signify(df)
        if self.DECIMALS_PLACES:
            df = self.NormDecimals(df)

        BaseDataInventory.Spots = df

        return df


    def Tracks(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes a comprehensive DataFrame of track-level statistics for each trajectory of the input Spots DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            *This method expects the dataframe acquired by `Stats.Spots()`. **The input DataFrame must contain these columns:***
            - `Condition`
            - `Replicate`
            - `Track ID`
            - `Track UID`
            - `Frame`
            - `X coordinate`
            - `Y coordinate`
            - `Distance`
            - `Cumulative track displacement`
            - `Direction`

        Returns
        -------
        pd.DataFrame
            ***A DataFrame with one row per unique track, containing the following columns:***

            - **`Condition`**
            - **`Replicate`**
            - **`Track ID`**
            - **`Track UID`**
            
            - **`Track length`**- 
            Total length of the track (sum of `Distance`).

            - **`Speed`** **`min`**, **`max`**, **`mean`**, **`sd`**, **`median`** and **`q25`**, **`q75`** (iqr) - 
            of the `Distances` between consecutive points (step lengths).

            - **`Max distance reached`**- 
            Maximum Euclidean distance from the starting position reached at any point along the track.

            - **`Track start frame`**- 
            Frame number of the first point in the track.

            - **`Track end frame`**- 
            Frame number of the last point in the track.

            - **`Track displacement`**- 
            Euclidean distance from the starting position to the end position of the track.

            - **`Straightness index`**- 
            Track's straigtness calculated as `Track displacement` / `Track length`.

            - **`Track points`**- 
            The number of points the trajectory is comprised of.

            - **`Direction mean`**- 
            Circular mean of the `Direction` values.

            - **`Direction var`**- 
            Circular variance of the `Direction` values.

            - **`Other`**- 
            *any additional columns from the input DataFrame that are not part of the above list will be retained in the output if they contain any non-NA values; otherwise, they will be dropped.*

        See also
        --------
        `Stats.get_all_data()`- 
        computes all DataFrames (Spots, Tracks, Frames, TimeIntervals) from raw spot data in one call.

        `Stats.Tracks()`- 
        computes per-whole-trajectory statistics from the Spots DataFrame.

        `Stats.Frames()`- 
        computes per-time-point statistics from the Spots DataFrame.

        `Stats.TimeIntervals()`- 
        computes per-time-interval statistics from the Spots DataFrame.

        `(dataclass) BaseDataInventory`- 
        serves as an inventory, storing the computed DataFrames computed via the `(class) Stats`.

        """

        df = df.copy()

        if df.empty:
            return pd.DataFrame(columns=self._COLUMNS['TRACKS'])

        stash = df[self.tier + ['Track ID']].drop_duplicates()

        df = df.set_index('Track UID', drop=True, verify_integrity=False)
        gcols = ['Track UID']

        grp = df.groupby(level=gcols, sort=False)

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

        agg_spec['Track start frame'] = ('Frame', 'min')
        agg_spec['Track end frame'] = ('Frame', 'max')

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
        
        df.insert(df.columns.get_loc('Track ID') + 1, 'Track UID', df.index)
        
        if self.SIGNIFICANT_FIGURES:
            df = self.Signify(df)
        if self.DECIMALS_PLACES:
            df = self.NormDecimals(df)
        
        BaseDataInventory.Tracks = df

        return df
    

    def Frames(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes time point statistics:

        - `{per replicate}`- aggregated across all tracks of the same `Replicate`
        - `{per condition}`- aggregated across all tracks of the same `Condition`

        Parameters
        ----------
        df : pd.DataFrame
            *This method expects the dataframe acquired by `Stats.Spots()`. **The input DataFrame must contain these columns:***
            - `Condition`
            - `Replicate`
            - `Time point`
            - `Frame`
            - `Distance`
            - `Cumulative track length`
            - `Cumulative track displacement`
            - `Cumulative straightness index`
            - `Cumulative speed`
            - `Direction`
            - `Cumulative direction mean`

        Returns
        -------
        pd.DataFrame
            *A DataFrame with one row per unique combination of `Condition` × `Replicate` × `Time point`, containing the following columns:*

            - **`Condition`**
            - **`Replicate`**
            - **`Time point`**
            - **`Frame`**

            \n **`{per category}`*****`{metric}`***
                - ***descriptive base statistics:***  **`min`**, **`max`**, **`mean`**, **`median`**, **`q25`**, **`q75`** (iqr) if `cat_descr` is set to `True` when initializing the Stats class
                - ***descriptive error statistics:*** **`std`** if `descr_descr_err` is set to True when initializing the Stats class
                - ***inferative error statistics:***  **`sem`**, if `descr_infer_err` is set to True when initializing the Stats class, 
                ***`{CI_STATISTIC}`*****`ci`*****`{CONFIDENCE_LEVEL}`*****`low`** and ***`{CI_STATISTIC}`*****`ci`*****`{CONFIDENCE_LEVEL}`*****`high`** (confidence interval) if both `descr_infer_err` and `bootstrap_ci` are set to True
                - ***circular statistics:*** **`mean`** (circular mean) and **`var`** (circular variance) calculated for each of `Direction` and `Cumulative direction`.
            
            - for each of these metrics
                - **`Cumulative track length`**
                - **`Cumulative track displacement`**
                - **`Cumulative straightness index`**
                - **`Cumulative speed`**
                - **`Instantaneous speed`**
                - **`Instantaneous direction`**
                - **`Cumulative direction global`**

        See also
        --------
        `Stats.get_all_data()`- 
        computes all DataFrames (Spots, Tracks, Frames, TimeIntervals) from raw spot data in one call.

        `Stats.Tracks()`- 
        computes per-whole-trajectory statistics from the Spots DataFrame.

        `Stats.Frames()`- 
        computes per-time-point statistics from the Spots DataFrame.

        `Stats.TimeIntervals()`- 
        computes per-time-interval statistics from the Spots DataFrame.

        `(dataclass) BaseDataInventory`- 
        serves as an inventory, storing the computed DataFrames computed via the `(class) Stats`.

        """

        df = df.copy()

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
            cols = self._COLUMNS['FRAMES'] + _build_expected_cols(prefixes)
            return pd.DataFrame(columns=cols)

        # Separate resolved stats into CI vs. non-CI for batching
        non_ci_stats = {s: func for s, func in resolved_stats.items() if s != 'ci'}
        ci_func = resolved_stats.get('ci', None)

        def _compute_level(source: pd.DataFrame, by_cols: list[str], prefix: str) -> pd.DataFrame:
            g = source.groupby(by_cols, sort=False, observed=True)

            # --- Batch scalar stats via a single named-agg call ---
            named_agg = {}
            for m in metrics:
                mout = metric_out[m]
                for s, func in non_ci_stats.items():
                    named_agg[f'{prefix} {mout} {_stat_label(s)}'] = (m, func)

            out = g.agg(**named_agg) if named_agg else pd.DataFrame(index=g.keys if hasattr(g, 'keys') else g.size().index)

            # --- CI columns (these require tuple unpacking, handled in bulk per metric) ---
            if ci_func is not None:
                for m in metrics:
                    mout = metric_out[m]
                    ci_series = g[m].agg(ci_func)
                    # Unpack tuples in bulk instead of row-by-row lambda
                    ci_unpacked = pd.DataFrame(
                        ci_series.tolist(),
                        index=ci_series.index,
                        columns=[
                            f'{prefix} {mout} {self.CI_STATISTIC} ci{self.CONFIDENCE_LEVEL} low',
                            f'{prefix} {mout} {self.CI_STATISTIC} ci{self.CONFIDENCE_LEVEL} high',
                        ],
                    )
                    # Replace non-tuple results (e.g. single NaN) with NaN
                    out = out.join(ci_unpacked)

            # --- Circular stats: compute sin/cos in-place on source, single groupby ---
            dir_vals = pd.to_numeric(source['Direction'], errors='coerce').to_numpy(dtype=float)
            cum_vals = pd.to_numeric(source['Cumulative direction mean'], errors='coerce').to_numpy(dtype=float)

            sin_dir = np.sin(dir_vals)
            cos_dir = np.cos(dir_vals)
            sin_cum = np.sin(cum_vals)
            cos_cum = np.cos(cum_vals)

            # Build a lightweight frame with only what we need (avoids copying the whole source)
            circ_df = pd.DataFrame({
                '_sin_dir': sin_dir,
                '_cos_dir': cos_dir,
                '_sin_cum': sin_cum,
                '_cos_cum': cos_cum,
            }, index=source.index)
            for col in by_cols:
                circ_df[col] = source[col].values

            circ = circ_df.groupby(by_cols, sort=False, observed=True).agg(
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

        # Optional per-condition stats attached to replicate rows
        if add_conds:
            cond_df = _compute_level(df, cond_group_cols, '{per condition}')
            base_df = base_df.merge(cond_df, on=cond_group_cols, how='left')

        if self.SIGNIFICANT_FIGURES:
            base_df = self.Signify(base_df)
        if self.DECIMALS_PLACES:
            base_df = self.NormDecimals(base_df)

        BaseDataInventory.Frames = base_df

        print(base_df.columns.tolist())

        return base_df
    
    
    def TimeIntervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes per-time-interval statistics, including mean squared displacement (MSD) and turning angles:

        For each frame lag (1, 2, …, maximum), squared displacements and turning angles are computed across trajectories.

        - `{per replicate}`- aggregated across all tracks of the same `Replicate`
        - `{per condition}`- aggregated across all tracks of the same `Condition`


        Example
        -------

        Trajectories A, B, and C comprised of consecutive points and their positions:
        ```
        pa1 ─ pa2 ─ pa3 ─ pa4 ─ pa5 ─ pa6 ─ pa7
              pb1 ─ pb2 ─ pb3 ─ pb4 ─ pb5 ─ pb6
              pc1 ─ pc2 ─ pc3 ─ pc4 ─ pc5
        ```

        Valid position pairs for the interval (lag) of three frames:
        ```
        pa1 ───────────── pa4
              pa2 ───────────── pa5
              pb1 ───────────── pb4
              pc1 ───────────── pc4
                    pa3 ───────────── pa6
                    pb2 ───────────── pb5
                    pc2 ───────────── pc5
                          pa4 ───────────── pa7
                          pb3 ───────────── pb6
        ```

        MSD formula for a given time lag *k* for a given trajactory *i* with trajectory point positions *p* at a time position *t* :
        ```
        MSDᵢ(k) = ||pᵢ(t+k) - pᵢ(t)||²
        ```
        \n The per-track MSD values are then aggregated across tracks within each of unique `Time lag` × `Replicate` and `Time lag` × `Condition`.
        
            
        Turning angle formula for a given time lag *k* for a trajectory *i* with trajectory point positions *y* and *x* at a time position *t* :
        ```
        Δθᵢ(k) = ||θᵢ(Δy(t+k), Δx(t+k)) - θᵢ(Δy(t), Δx(t))||
        ```
        \n The angular difference in the direction of motion is then wrapped to [−π, π]. 
        Per-track turning angles values are then aggregated (circular mean and variance) across tracks within each of unique `Time lag` × `Replicate` and `Time lag` × `Condition` .



        Parameters
        ----------
        df : pd.DataFrame
            *This method expects the dataframe acquired by `Stats.Spots()`. **The input DataFrame must contain these columns:***
            - `Condition`
            - `Replicate`
            - `Track UID`
            - `Time point`
            - `X coordinate`
            - `Y coordinate`

        Returns
        -------
        pd.DataFrame
            *A DataFrame with one row per unique combination of `Condition` × `Replicate` × `Frame lag`, containing the following columns:*

            - **`Condition`**
            - **`Replicate`**
            - **`Frame lag`**- Integer lag in frames (1, 2, 3, …).
            - **`Time lag`**- Corresponding time lag computed as `Frame lag` × time step.

            \n **`{per category}`**
                - **`Tracks contributing`**- the number of tracks that contributed data at each lag
                - **`Turn mean`**- mean absolute turning angle in degrees
                - **`Turn var`**- circular variance of turning angles
                - **`MSD`** ***descriptive base statistics:*** **`min`**, **`max`**, **`mean`**, **`median`**, **`q25`**, **`q75`** (iqr) if `cat_descr` is set to `True` when initializing the Stats class
                - **`MSD`** ***descriptive error statistics:*** **`sd`** if `cat_descr_err` is set to `True` when initializing the Stats class
                - **`MSD`** ***inferative error statistics:*** **`sem`** if `cat_infer_err` is set to `True` when initializing the Stats class,
                ***`{CI_STATISTIC}`*****`ci`*****`{CONFIDENCE_LEVEL}`*****`low`** and ***`{CI_STATISTIC}`*****`ci`*****`{CONFIDENCE_LEVEL}`*****`high`** (confidence interval) if both `cat_infer_err` and `bootstrap_ci` are set to `True`


            \n Sets `BaseDataInventory.TimeIntervals` to the computed DataFrame.

        Notes
        -----
        - Tracks with fewer than 2 points are excluded from computation.
        - If fewer than 2 unique time points exist in the input, an empty DataFrame is returned.

        See also
        --------
        `Stats.get_all_data()`- 
        computes all DataFrames (Spots, Tracks, Frames, TimeIntervals) from raw spot data in one call.

        `Stats.Spots()`- 
        computes per-trajectory-point statistics, both local (previous -> current position) and cumulative (start -> current position).

        `Stats.Tracks()`- 
        computes per-whole-trajectory statistics from the Spots DataFrame.

        `Stats.Frames()`- 
        computes per-time-point statistics from the Spots DataFrame.

        `(dataclass) BaseDataInventory`- 
        serves as an inventory, storing the computed DataFrames computed via the `(class) Stats`.

        """

        if df.empty: 
            Reporter(Level.warning, f"Input DataFrame to TimeIntervals method is empty; no computations performed.", noticequeue=self.noticequeue)
            return pd.DataFrame(columns=self._COLUMNS['TIMEINTERVALS'])

        # Work on a copy to avoid mutating the caller's DataFrame (e.g. BaseDataInventory.Spots)
        df = df.copy()

        # Ensure Track UID is the index (it may be a column if called outside of the normal pipeline)
        if df.index.name != 'Track UID' and 'Track UID' in df.columns:
            df = df.set_index('Track UID', drop=False, verify_integrity=False)

        # Unique time point differences -> time steps; use median to resist irregular sampling
        t_unique = np.sort(df['Time point'].unique())

        # <2 unique time points returns an empty DataFrame.
        if t_unique.size < 2:
            return pd.DataFrame(columns=self._COLUMNS['TIMEINTERVALS'])
        
        t_step = float(np.diff(t_unique)[0])

        reps = 'Replicate' in self.tier
        add_conds = reps

        df.reset_index(drop=True, inplace=True)

        # ---- Vectorized per-track, per-lag computation ----
        # Sort once; assign a within-track sequential position for shift-based lag computation
        work = (
            df[['Condition'] + (['Replicate'] if reps else []) + ['Track UID', 'Time point', 'X coordinate', 'Y coordinate']]
            .copy()
            .sort_values(['Track UID', 'Time point'])
        )
        uid_col = work['Track UID'].values if 'Track UID' in work.columns else work.index.get_level_values('Track UID').values

        # Within-track sequential position (0-based)
        work['_pos'] = work.groupby('Track UID', sort=False).cumcount()

        # Track length (number of points)
        track_sizes = work.groupby('Track UID', sort=False).size()
        work['_n'] = work['Track UID'].map(track_sizes).values if 'Track UID' in work.columns else uid_col

        # Filter out tracks with <2 points early
        work = work[work['_n'] >= 2].copy()

        if work.empty:
            return pd.DataFrame(columns=self._COLUMNS['TIMEINTERVALS'])

        # Pre-compute consecutive angles (theta) for turning angle computation
        # theta[i] = arctan2(y[i]-y[i-1], x[i]-x[i-1])
        grp_work = work.groupby('Track UID', sort=False)
        work['_dx1'] = grp_work['X coordinate'].diff()
        work['_dy1'] = grp_work['Y coordinate'].diff()
        work['_theta'] = np.arctan2(work['_dy1'].values, work['_dx1'].values)

        max_lag = int(track_sizes.max()) - 1

        # Collect per-track per-lag records in bulk using vectorized shifts
        msd_records = []
        turn_records = []

        # We group once and iterate by lag (not by track) — much fewer iterations
        # For each lag, compute shifted coordinates within each track
        x_arr = work['X coordinate'].values
        y_arr = work['Y coordinate'].values
        theta_arr = work['_theta'].values
        pos_arr = work['_pos'].values
        n_arr = work['_n'].values
        cond_arr = work['Condition'].values
        rep_arr = work['Replicate'].values if reps else None
        uid_arr = work['Track UID'].values if 'Track UID' in work.columns else work.index.get_level_values('Track UID').values

        # Build a mapping: for each row, the row index within the sorted work frame
        # We'll use groupby + shift approach on the sorted dataframe
        work_reset = work.reset_index(drop=True)

        for lag in range(1, max_lag + 1):
            # Only rows where pos + lag < n (i.e., the shifted partner exists)
            valid_mask = (pos_arr + lag) < n_arr

            if not valid_mask.any():
                break

            # For MSD: we need x[pos+lag] - x[pos] and y[pos+lag] - y[pos]
            # Use groupby shift on the sorted frame
            shifted_x = grp_work['X coordinate'].shift(-lag)
            shifted_y = grp_work['Y coordinate'].shift(-lag)

            dx = shifted_x.values - x_arr
            dy = shifted_y.values - y_arr
            sq_disp = dx**2 + dy**2

            # Per-track mean MSD at this lag: group by Track UID, take mean of sq_disp
            # But only for valid rows
            valid_idx = np.where(valid_mask)[0]
            if valid_idx.size == 0:
                continue

            # Build a small frame for this lag's valid entries
            lag_df = pd.DataFrame({
                'Track UID': uid_arr[valid_idx],
                'Condition': cond_arr[valid_idx],
                'sq_disp': sq_disp[valid_idx],
            })
            if reps:
                lag_df['Replicate'] = rep_arr[valid_idx]

            # Per-track mean MSD at this lag
            track_msd = lag_df.groupby('Track UID', sort=False).agg(
                msd=('sq_disp', 'mean'),
                Condition=('Condition', 'first'),
                **({'Replicate': ('Replicate', 'first')} if reps else {}),
            )
            track_msd['lag'] = lag
            msd_records.append(track_msd)

            # Turning angles: theta[pos+lag] - theta[pos] for consecutive-step angles
            # theta is defined for pos >= 1 (diff-based), so valid turning angles
            # require both theta[pos] and theta[pos+lag] to be valid
            shifted_theta = grp_work['_theta'].shift(-lag)
            dtheta_all = shifted_theta.values - theta_arr

            # Wrap to [-pi, pi]
            dtheta_all = (dtheta_all + np.pi) % (2 * np.pi) - np.pi

            # Valid turning angles: pos >= 1 (theta exists) AND pos+lag < n AND both thetas not NaN
            turn_valid = valid_mask & (pos_arr >= 1) & np.isfinite(theta_arr) & np.isfinite(shifted_theta.values)
            turn_idx = np.where(turn_valid)[0]

            if turn_idx.size > 0:
                turn_df = pd.DataFrame({
                    'Track UID': uid_arr[turn_idx],
                    'Condition': cond_arr[turn_idx],
                    'dtheta': dtheta_all[turn_idx],
                })
                if reps:
                    turn_df['Replicate'] = rep_arr[turn_idx]

                # Per-track circular mean of turning angles at this lag
                turn_df['_sin'] = np.sin(turn_df['dtheta'].values)
                turn_df['_cos'] = np.cos(turn_df['dtheta'].values)

                track_turn = turn_df.groupby('Track UID', sort=False).agg(
                    mean_sin=('_sin', 'mean'),
                    mean_cos=('_cos', 'mean'),
                    Condition=('Condition', 'first'),
                    **({'Replicate': ('Replicate', 'first')} if reps else {}),
                )
                track_turn['turn_mean'] = np.arctan2(track_turn['mean_sin'].values, track_turn['mean_cos'].values)
                track_turn['lag'] = lag
                turn_records.append(track_turn[['Condition'] + (['Replicate'] if reps else []) + ['turn_mean', 'lag']])

        # If no records produced, return empty
        if not msd_records:
            return pd.DataFrame(columns=self._COLUMNS['TIMEINTERVALS'])

        # Concatenate all lag results
        all_msd = pd.concat(msd_records, ignore_index=True)
        all_turn = pd.concat(turn_records, ignore_index=True) if turn_records else pd.DataFrame(columns=['Condition'] + (['Replicate'] if reps else []) + ['turn_mean', 'lag'])

        # ---- Aggregate across tracks per tier × lag ----
        cat = '{per replicate}' if reps else '{per condition}'
        tier_lag_cols = self.tier + ['lag']
        cond_lag_cols = ['Condition', 'lag']

        def _agg_msd_turn(msd_src: pd.DataFrame, turn_src: pd.DataFrame, by_cols: list[str], prefix: str) -> pd.DataFrame:
            """Aggregate MSD and turn stats across tracks for given grouping."""
            # MSD aggregation
            msd_grp = msd_src.groupby(by_cols, sort=False)['msd']

            agg_dict = {}
            if self.cat_descr:
                agg_dict[f'{prefix} MSD min'] = msd_grp.min()
                agg_dict[f'{prefix} MSD max'] = msd_grp.max()
                agg_dict[f'{prefix} MSD mean'] = msd_grp.mean()
                agg_dict[f'{prefix} MSD median'] = msd_grp.median()
                agg_dict[f'{prefix} MSD q25'] = msd_grp.agg(self._q25)
                agg_dict[f'{prefix} MSD q75'] = msd_grp.agg(self._q75)

            if self.cat_descr_err:
                agg_dict[f'{prefix} MSD sd'] = msd_grp.std(ddof=1)

            if self.cat_infer_err:
                agg_dict[f'{prefix} MSD sem'] = msd_grp.agg(self._sem)
                ci_series = msd_grp.agg(self._ci)
                ci_unpacked = pd.DataFrame(
                    ci_series.tolist(),
                    index=ci_series.index,
                    columns=[
                        f'{prefix} MSD {self.CI_STATISTIC} ci{self.CONFIDENCE_LEVEL} low',
                        f'{prefix} MSD {self.CI_STATISTIC} ci{self.CONFIDENCE_LEVEL} high',
                    ],
                )
                for c in ci_unpacked.columns:
                    agg_dict[c] = ci_unpacked[c]

            # Track count
            agg_dict[f'{prefix} Tracks contributing'] = msd_grp.size().astype(int)

            result = pd.DataFrame(agg_dict)

            # Turn aggregation (circular stats across per-track circular means)
            if not turn_src.empty:
                turn_grp = turn_src.groupby(by_cols, sort=False)['turn_mean']

                turn_sin = turn_src.assign(_s=np.sin(turn_src['turn_mean'].values), _c=np.cos(turn_src['turn_mean'].values))
                turn_circ = turn_sin.groupby(by_cols, sort=False).agg(
                    ms=('_s', 'mean'),
                    mc=('_c', 'mean'),
                )
                circ_mean = np.arctan2(turn_circ['ms'].values, turn_circ['mc'].values)
                circ_R = np.hypot(turn_circ['ms'].values, turn_circ['mc'].values)
                circ_var = 1.0 - circ_R

                if self.cat_descr or self.cat_descr_err or self.cat_infer_err:
                    result[f'{prefix} Turn mean'] = pd.Series(np.rad2deg(np.abs(circ_mean)), index=turn_circ.index)

                if self.cat_descr_err:
                    result[f'{prefix} Turn var'] = pd.Series(circ_var, index=turn_circ.index)

            return result

        # Base level aggregation
        base_result = _agg_msd_turn(all_msd, all_turn, tier_lag_cols, cat)
        base_result = base_result.reset_index()

        # Condition-level stats attached when working per-replicate
        if add_conds:
            cond_result = _agg_msd_turn(all_msd, all_turn, cond_lag_cols, '{per condition}')
            cond_result = cond_result.reset_index()
            base_result = base_result.merge(cond_result, on=cond_lag_cols, how='left')

        # Rename 'lag' -> 'Frame lag' and add 'Time lag'
        base_result = base_result.rename(columns={'lag': 'Frame lag'})
        base_result['Time lag'] = base_result['Frame lag'] * t_step

        # Reorder: tier columns first, then Frame lag, Time lag, then stats
        front_cols = self.tier + ['Frame lag', 'Time lag']
        other_cols = [c for c in base_result.columns if c not in front_cols]
        base_result = base_result[front_cols + other_cols]

        # Sort
        base_result = base_result.sort_values(self.tier + ['Frame lag'], ignore_index=True)

        # JSON-safe cleanup for Shiny/front-end serializers (no NaN/Inf in strict JSON)
        base_result = base_result.replace([np.inf, -np.inf], np.nan)

        if self.SIGNIFICANT_FIGURES:
            base_result = self.Signify(base_result)
        if self.DECIMALS_PLACES:
            base_result = self.NormDecimals(base_result)

        BaseDataInventory.TimeIntervals = base_result

        print(base_result.columns.tolist())
        
        return base_result


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
            Reporter(Level.warning, "No columns specified for exclusion in Stats._general_agg_stats(); all numeric columns will be aggregated.", noticequeue=self.noticequeue)

        exclude = [col for col in exclude if col != 'Track UID']

        # Keep only numeric columns and exclude core columns
        additional = df.copy()

        try:
            additional = additional.drop(columns=exclude, errors='ignore')

            if not additional.shape[1]:
                return pd.DataFrame(index=additional.index)

            additional = additional.select_dtypes(include=[np.number])

            if additional.empty or additional.shape[1] == 0:
                return pd.DataFrame(index=df.index if group_by == ['Track UID'] else df.groupby(level=group_by, sort=False).ngroup().index)

            # Stash leftover columns (exclude 'Track UID' if it's among them)
            other_cols = [c for c in additional.columns.tolist() if c != 'Track UID']

            if not other_cols:
                return pd.DataFrame(index=additional.index)

            # Group by Track UID
            grp = additional.groupby(level=group_by, sort=False)

            # For each bonus column, compute basic statistics, rename columns, and merge back
            for col in other_cols:
                agg = grp[col].agg(['min','max','mean','std','sem','median'])

                agg.columns = [f"{col} min", f"{col} max", f"{col} mean", f"{col} sd", f"{col} sem", f"{col} median"]

                additional = additional.merge(agg, left_index=True, right_index=True)

            # Drop original columns
            additional.drop(columns=other_cols, inplace=True)

            # drop multiplicates if present
            additional = additional.drop_duplicates()
            
            return additional
        
        except Exception as e:
            Reporter(Level.error, f"Error in _general_agg_stats while preparing DataFrame for aggregation: {e}", trace=traceback.format_exc(), noticequeue=self.noticequeue)
            return pd.DataFrame(index=df.index)


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

