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

    RawInput: pd.DataFrame
    Spots: pd.DataFrame
    Tracks: pd.DataFrame
    Frames: pd.DataFrame
    TimeIntervals: pd.DataFrame

    def __post_init__(self):
        self.RawInput = pd.DataFrame()
        self.Spots = pd.DataFrame()
        self.Tracks = pd.DataFrame()
        self.Frames = pd.DataFrame()
        self.TimeIntervals = pd.DataFrame()



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
            'Condition','Replicate','Track ID','Track UID',
            'Time point','Frame','X coordinate','Y coordinate','Distance',
            'Cumulative track length','Cumulative track displacement','Cumulative straightness ratio', 
            'Cumulative speed','Cumulative mean straight line speed',
            'Cumulative forward progression linearity','Direction','Directional change',
            'Cumulative sum directional change','Cumulative mean directional change',
            'Cumulative direction mean','Cumulative direction var'
        ],
        'TRACKS': [
            'Condition','Replicate','Track ID','Track UID',
            'Y location', 'X location', 
            'Track length','Track displacement','Straightness ratio',
            'Speed min','Speed max','Speed mean','Speed sd','Speed median',
            'Mean straight line speed', 'Forward progression linearity',  # https://imagej.net/plugins/trackmate/analyzers/
            'Max distance reached','Track start frame','Track end frame',
            'Direction mean','Direction var','Mean directional change'
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

        self.DESCR     = self._DESCR if cat_descr else []
        self.DESCR_ERR = self._DESCR_ERR if cat_descr_err else []
        if cat_infer_err:
            self.INFER_ERR = self._INFER_ERR
            if bootstrap_ci:
                self.INFER_ERR.append('ci')
        else:
            self.INFER_ERR = []
        

        self.CUSTOM_AGG_FUNCTIONS = {
            'q25':        self._q25,
            'q75':        self._q75,
            'ci':         self.ci,
            'sem':        self.sem,
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

        self.RawInput = df
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

            - **`Cumulative straightness ratio`**- 
            Track's straigtness calculated as `Cumulative track displacement` / `Cumulative track length`.

            - **`Cumulative speed`**- 
            Mean speed (`Distance`) from the starting to the current position <- `Cumulative track length` / `Frame`.

            - **`Cumulative mean straight line speed`**-
            Calculated as `Cumulative track displacement` / Track point count.

            - **`Cumulative forward progression linearity`**-
            Calculated as `Cumulative mean straight line speed` / `Cumulative speed`

            - **`Direction`**- 
            Instantaneous direction of motion in radians `np.arctan2(Δy, Δx)`. Calculated between the previous and current positions.

            - **`Directional change`**- 
            Absolute turning angle (degrees) between consecutive directions, calculated as the angular difference between the current and previous `Direction` values, wrapped to the range [-180°, 180°].

            - **`Cumulative sum directional change`**-
            Cumulative sum of `Directional change` along the track up to the current position.

            - **`Cumulative mean directional change`**- 
            Mean of all absolute `Directional change` values along the track up to the current position

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

        BaseDataInventory.RawInput = df.copy()

        if df.empty:
            Reporter(Level.warning, f"Input DataFrame to Spots method is empty; no computations performed.", noticequeue=self.noticequeue)
            return pd.DataFrame(columns=self._COLUMNS['SPOTS'])

        df.sort_values(self.tier + ['Track ID', 'Time point'], inplace=True)

        # Define grouping keys
        gkeys = self.tier + ['Track ID']

        grp = df.groupby(gkeys, sort=False)

        # Provides a unique trajectory identifier (Track UID) as index that is consistent throughout dataflow and is used for grouping, iterating, filtering, and merging.
        df['Track UID'] = grp.ngroup()
        df.set_index(['Track UID'], drop=False, append=False, inplace=True, verify_integrity=False)

        # Assigns frame numbers within each data subset based on the order of time points; starts at 0 for the first point in each track.
        df['Frame'] = df.groupby(self.tier, sort=False)['Time point'].rank(method='dense').astype('Int64') - 1

        # Sanity guard, checking for multiple frames assigned to the same tier × time point combination
        try:
            bad = df.groupby(self.tier + ['Time point'], sort=False)['Frame'].nunique(dropna=True).max()

            if bad and bad > 1:
                Reporter(Level.warning, f"Multiple frames assigned to the same {self.tier} × Time point combination; this may indicate time point multiplicates within the data or other data issues. Max frames per time point: {bad}.", noticequeue=self.noticequeue)
                pass

        except Exception:
            pass

        # Distance between the previous and the current position
        df['Distance'] = np.hypot(
            grp['X coordinate'].diff(),
            grp['Y coordinate'].diff()
        ).fillna(np.nan)

        _temp = grp['Track UID'].transform('count')

        # Cumulative track length
        df['Cumulative track length'] = grp['Distance'].cumsum()

        # Cumulative track displacement -> straight-line distance (start -> current position)
        start = grp[['X coordinate', 'Y coordinate']].transform('first')
        df['Cumulative track displacement'] = np.hypot(
            (df['X coordinate'] - start['X coordinate']),
            (df['Y coordinate'] - start['Y coordinate'])
        ).replace(0, np.nan)

        # Straightness index: Track displacement vs. actual trajectory length ratio
        # Avoid division by zero by replacing zeros with NaN, then fill
        df['Cumulative straightness ratio'] = (df['Cumulative track displacement'] / df['Cumulative track length'].replace(0, np.nan)).fillna(np.nan)

        # Cumulative speed -> mean speed from the starting to the current position
        df['Cumulative speed'] = df['Cumulative track length'] / df['Frame'].replace(0, np.nan)

        df['Cumulative straight line speed'] = df['Cumulative track displacement'] / _temp.replace(0, np.nan)
        df['Cumulative forward progression linearity'] = df['Cumulative straight line speed'] / df['Cumulative speed']

        # Instantaneous direction of motion (rad) -> difference between the previous and current position
        df['Direction'] = np.arctan2(
            grp['Y coordinate'].diff(),
            grp['X coordinate'].diff()
        ).fillna(np.nan)

        # Directional change (turning angle) -> angular difference between consecutive directions, wrapped to [-π, π], then absolute, converted to degrees
        raw_dir_change = grp['Direction'].diff()
        # Wrap to [-π, π] first, then take absolute value, then convert to degrees
        wrapped_dir_change = (raw_dir_change + np.pi) % (2 * np.pi) - np.pi
        df['Directional change'] = np.rad2deg(wrapped_dir_change.abs()).fillna(np.nan)

        # Mean directional change -> mean of absolute directional changes (degrees) along the track up to the current position
        _temp = df.groupby(gkeys, sort=False)['Directional change'].expanding()

        df['Cumulative sum directional change'], df['Cumulative mean directional change'] = (
            _temp.sum().droplevel(list(range(len(gkeys)))),
            _temp.mean().droplevel(list(range(len(gkeys))))
        )

        # First two points of each track have no directional change; set them to NaN
        df.loc[df['Directional change'].isna(), ['Cumulative mean directional change']] = np.nan

        # Cumulative direction of motion (circular mean and variance) calculations
        theta_from_start = np.arctan2(
            df['Y coordinate'] - start['Y coordinate'],
            df['X coordinate'] - start['X coordinate'],
        )

        # Exclude the first frame from cumulative direction stats (no angle data there)
        valid = df['Frame'].gt(0) & theta_from_start.notna()

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

        # Drop all-NaN columns (if any are present)
        df.dropna(how='all', axis='columns', inplace=True)

        if self.SIGNIFICANT_FIGURES:
            df = self.signify(df)
        if self.DECIMALS_PLACES:
            df = self.norm_decimals(df)

        BaseDataInventory.Spots = df.copy()

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

            - **`Y location`**-
            The mean Y position of the track's starting position.

            - **`X location`**-
            The mean X position of the track's starting position.
            
            - **`Track length`**- 
            Total length of the track (sum of `Distance`).

            - **`Speed`** **`min`**, **`max`**, **`mean`**, **`sd`**, **`median`** - 
            of the `Distances` between consecutive points (step lengths).

            - **`Max distance reached`**- 
            Maximum Euclidean distance from the starting position reached at any point along the track.

            - **`Track start frame`**- 
            Frame number of the first point in the track.

            - **`Track end frame`**- 
            Frame number of the last point in the track.

            - **`Track displacement`**- 
            Euclidean distance from the starting position to the end position of the track.

            - **`Track straightness ratio`**- 
            Track's straigtness calculated as `Track displacement` / `Track length`.

            - **`Mean straight line speed`**-
            Calculated as `Track displacement` / `Track points`

            - **`Forward progression linearity`**-
            Calculated as `Mean straight line speed` / `Speed mean`

            - **`Track points`**- 
            The number of points the trajectory is comprised of.

            - **`Direction mean`**- 
            Circular mean of the `Direction` values.

            - **`Mean directional change`**- 
            Mean of absolute directional changes per track (degrees).

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

        # Work on a copy to avoid mutating the caller's DataFrame
        df = df.copy()

        if df.empty:
            return pd.DataFrame(columns=self._COLUMNS['TRACKS'])

        # Stash the categorical identifiers for merging them back into the aggregated result DataFrame
        stash = df[self.tier + ['Track ID']].drop_duplicates()

        df = df.set_index('Track UID', drop=True, verify_integrity=False)
        uid = ['Track UID']

        grp = df.groupby(level=uid, sort=False)

        # Resolve the aggregation functions for speed statistics
        speed_agg_spec = self.resolve({
            'Speed min':    'min',
            'Speed max':    'max',
            'Speed mean':   'mean',
            'Speed sd':     'std',
            'Speed median': 'median'
        })

        # Build the named agg dict: each entry is (column, func)
        agg_spec = {
            'Track length': ('Distance', 'sum'),
        }
        for label, func in speed_agg_spec.items():
            agg_spec[label] = ('Distance', func)

        # Add coordinates first/last for displacement calculation
        agg_spec['start_x'] = ('X coordinate', 'first')
        agg_spec['end_x']   = ('X coordinate', 'last')
        agg_spec['start_y'] = ('Y coordinate', 'first')
        agg_spec['end_y']   = ('Y coordinate', 'last')

        agg_spec['X location'] = ('X coordinate', 'mean')
        agg_spec['Y location'] = ('Y coordinate', 'mean')

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

        # Other general stats if the input data were not stripped
        other = self._general_agg_stats(df, exclude=self._COLUMNS['SPOTS'])

        # Displacement and straightness
        agg['Track displacement'] = np.hypot(agg['end_x'] - agg['start_x'], agg['end_y'] - agg['start_y'])
        agg['Track straightness ratio'] = (agg['Track displacement'] / agg['Track length'])
        agg = agg.drop(columns=['start_x','end_x','start_y','end_y'])

        # Points/ per track
        n = grp.size().rename('Track points')
        agg = agg.merge(n, left_index=True, right_index=True)

        agg['Mean straight line speed'] = agg['Track displacement'] / agg['Track points']
        agg['Forward progression linearity'] = agg['Mean straight line speed'] / agg['Speed mean']

        # Sin/Cos values for circular statistics
        sin_cos = df.assign(
            _sin=np.sin(df['Direction']), 
            _cos=np.cos(df['Direction'])
        )
        # Group and aggregate sin/cos components along tracks, getting mean sin and cos for each track
        dir_agg = sin_cos.groupby(uid, sort=False).agg(
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

        # Mean directional change: mean of absolute directional changes per track (degrees) — constant per track, take any value
        mean_dir_change = df['Cumulative mean directional change'].groupby(uid, sort=False).last()
        dir_agg['Mean directional change'] = mean_dir_change

        # Merge results
        df = agg.merge(dir_agg, left_index=True, right_index=True)
        df = df.merge(other, left_index=True, right_index=True, how='right')
        df = stash.merge(df, left_index=True, right_index=True, how='right')

        df.drop_duplicates(inplace=True)
        
        # Insert Track UID as a column right after Track ID
        df.insert(df.columns.get_loc('Track ID') + 1, 'Track UID', df.index)
        
        if self.SIGNIFICANT_FIGURES:
            df = self.signify(df)
        if self.DECIMALS_PLACES:
            df = self.norm_decimals(df)
        
        BaseDataInventory.Tracks = df.copy()

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
            - `Cumulative straightness ratio`
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
                - **`Cumulative straightness ratio`**
                - **`Instantaneous speed`**
                - **`Cumulative speed`**
                - **`Cumulative straight line speed`**
                - **`Cumulative forward progression linearity`**
                - **`Instantaneous direction`**
                - **`Cumulative direction`**
                - **`Cumulative sum directional change`**
                - **`Cumulative mean directional change`**

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

        # Work on a copy to avoid mutating the caller's DataFrame
        df = df.copy()

        # Stash color columns if present, to carry them over to the output
        _color_cols = [c for c in ('Replicate color', 'Condition color') if c in df.columns]
        _color_stash = None
        if _color_cols:
            # Build a lookup keyed by the replicate/condition grouping columns
            _stash_keys = self.tier[:]  # ['Condition', 'Replicate']
            _color_stash = df[_stash_keys + _color_cols].drop_duplicates(subset=_stash_keys)

        rep_group_cols  =  self.tier + ['Time point', 'Frame']
        cond_group_cols = ['Condition', 'Time point', 'Frame']

        # Expected metrics to compute stats for (their input df labels)
        metrics = [
            'Cumulative track length',
            'Cumulative track displacement',
            'Cumulative straightness ratio',
            'Cumulative speed',
            'Distance',
            'Cumulative straight line speed',
            'Cumulative forward progression linearity',
            'Cumulative sum directional change',
            'Cumulative mean directional change',
        ]
        # (output df labels)
        metric_out = {m: m for m in metrics}
        metric_out['Distance'] = 'Instantaneous speed'

        # Stats are driven by configured category lists
        requested_stats = self.DESCR + self.DESCR_ERR + self.INFER_ERR
        print(f"Requested stats: {requested_stats}")
        resolved_stats = self.resolve(requested_stats) if requested_stats else {}

        # helper, converting 'std' to 'sd'
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
                    f'{pfx} Cumulative direction mean',
                    f'{pfx} Cumulative direction var',
                    f'{pfx} Cumulative mean directional change mean',
                ])
            return cols

        if df is None or df.empty:
            prefixes = ['{per replicate}', '{per condition}']
            cols = self._COLUMNS['FRAMES'] + _build_expected_cols(prefixes)
            return pd.DataFrame(columns=cols)
        

        def _compute_level(source: pd.DataFrame, by_cols: list[str], prefix: str) -> pd.DataFrame:
            grp = source.groupby(by_cols, sort=False, observed=True)
            out = pd.DataFrame(index=grp.size().index)

            # Batch scalar statistics via a single named-agg call
            for m in metrics:
                mout = metric_out[m]
                sgrp = grp[m]

                if self.cat_descr:
                    out[f'{prefix} {mout} min'] = sgrp.min()
                    out[f'{prefix} {mout} max'] = sgrp.max()
                    out[f'{prefix} {mout} mean'] = sgrp.mean()
                    out[f'{prefix} {mout} median'] = sgrp.median()
                    out[f'{prefix} {mout} q25'] = sgrp.agg(self._q25)
                    out[f'{prefix} {mout} q75'] = sgrp.agg(self._q75)

                if self.cat_descr_err:
                    out[f'{prefix} {mout} sd'] = sgrp.std(ddof=1)

                if self.cat_infer_err:
                    out[f'{prefix} {mout} sem'] = sgrp.agg(self.sem)
                    
                    if 'ci' in self.INFER_ERR:
                        ci_series = sgrp.agg(self.ci)
                        ci_unpacked = pd.DataFrame(
                            ci_series.tolist(),
                            index=ci_series.index,
                            columns=[
                                f'{prefix} {mout} {self.CI_STATISTIC} ci{self.CONFIDENCE_LEVEL} low',
                                f'{prefix} {mout} {self.CI_STATISTIC} ci{self.CONFIDENCE_LEVEL} high',
                            ],
                        )
                        for c in ci_unpacked.columns:
                            out[c] = ci_unpacked[c]

            # Circular statistics: compute sin/cos
            dir_vals = pd.to_numeric(source['Direction'], errors='coerce').to_numpy(dtype=float)
            cum_vals = pd.to_numeric(source['Cumulative direction mean'], errors='coerce').to_numpy(dtype=float)

            sin_dir = np.sin(dir_vals)
            cos_dir = np.cos(dir_vals)
            sin_cum = np.sin(cum_vals)
            cos_cum = np.cos(cum_vals)

            # Build a temporary DataFrame for circular stats aggregation
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
            out[f'{prefix} Cumulative direction mean'] = np.arctan2(circ['sin_cum'], circ['cos_cum'])
            out[f'{prefix} Cumulative direction var'] = 1.0 - np.hypot(circ['sin_cum'], circ['cos_cum'])

            # Cumulative mean directional change mean (regular mean, not circular)
            out[f'{prefix} Cumulative mean directional change mean'] = source.groupby(by_cols, sort=False)['Cumulative mean directional change'].mean().values


            # Reset index to turn grouping keys back into columns
            return out.reset_index()

        # Base level (per replicate or per condition)
        reps  = _compute_level(df, rep_group_cols, '{per replicate}')
        conds = _compute_level(df, cond_group_cols, '{per condition}')
        df = reps.merge(conds, on=cond_group_cols, how='left')

        # Re-attach color columns if they were present on input
        if _color_stash is not None:
            df = df.merge(_color_stash, on=_stash_keys, how='left')

        if self.SIGNIFICANT_FIGURES:
            df = self.signify(df)
        if self.DECIMALS_PLACES:
            df = self.norm_decimals(df)

        BaseDataInventory.Frames = df.copy()

        return df
    
    
    def TimeIntervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Computes per-time-interval statistics, including mean squared displacement (MSD) and directional change:

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
        \n The per-track MSD values are then aggregated across tracks within each of unique `Time lag` × `Condition` × `Replicate` and `Time lag` × `Condition`.
        
            
        Turning angle formula for a given time lag *k* for a trajectory *i* with trajectory point positions *y* and *x* at a time position *t* :
        ```
        Δθᵢ(k) = ||θᵢ(Δy(t+k), Δx(t+k)) - θᵢ(Δy(t), Δx(t))||
        ```
        \n The angular difference in the direction of motion is then wrapped to [−π, π]. 
        Per-track turning angles values are then aggregated (circular mean and variance) across tracks within each of unique `Time lag` × `Condition` × `Replicate` and `Time lag` × `Condition`.



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
                - **`Tracks contributing`**- the number of tracks that contributed data at a given time lag for a given category
                - **`Position pairs contributing`**- the number of position pairs that contributed data at a given time lag for a given category
                - **`Directional change mean`**- mean absolute turning angle in degrees
                - **`Directional change var`**- circular variance of turning angles
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
            return pd.DataFrame(columns=self._COLUMNS['TIMEINTERVALS'])

        # Work on a copy to avoid mutating the caller's DataFrame
        df = df.copy()

        # Stash color columns if present, to carry them over to the output
        _color_cols = [c for c in ('Replicate color', 'Condition color') if c in df.columns]
        _color_stash = None
        if _color_cols:
            # Build a lookup keyed by the replicate/condition grouping columns
            _stash_keys = self.tier[:]  # ['Condition', 'Replicate']
            _color_stash = df[_stash_keys + _color_cols].drop_duplicates(subset=_stash_keys)

        # Ensure Track UID is used as index
        if df.index.name != 'Track UID' and 'Track UID' in df.columns:
            df = df.set_index('Track UID', drop=False, verify_integrity=False)

        # Unique time points
        t_unique = np.sort(df['Time point'].unique())

        # <2 unique time points per interval returns an empty DataFrame.
        if t_unique.size < 2:
            return pd.DataFrame(columns=self._COLUMNS['TIMEINTERVALS'])

        # Unique time steps (time interval)
        t_step = float(np.diff(t_unique)[0])

        df.reset_index(drop=True, inplace=True)

        # Vectorized per-track, per-lag computation
        # Sort once; assign within-a-track sequential position for shift-based lag computation
        temp = (
            df[['Condition', 'Replicate', 'Track UID', 'Time point', 'X coordinate', 'Y coordinate']]
            .copy()
            .sort_values(['Track UID', 'Time point'])
        )
        uid_col = temp['Track UID'].values if 'Track UID' in temp.columns else temp.index.get_level_values('Track UID').values

        # Within-a-track sequential position (0, 1, 2, …) for shift-based lag computations
        temp['_pos'] = temp.groupby('Track UID', sort=False).cumcount()

        # Number of track points per track for filtering and lag validation
        track_sizes = temp.groupby('Track UID', sort=False).size()
        temp['_n'] = temp['Track UID'].map(track_sizes).values if 'Track UID' in temp.columns else uid_col

        # Filter out tracks with <2 points early
        temp = temp[temp['_n'] >= 2].copy()

        if temp.empty:
            Reporter(Level.error, f"No tracks with 2 or more points available for TimeIntervals computation; returning empty DataFrame.", noticequeue=self.noticequeue)
            return pd.DataFrame(columns=self._COLUMNS['TIMEINTERVALS'])

        # Pre-compute consecutive angles (theta) for turning angle computation
        # theta[i] = arctan2(y[i]-y[i-1], x[i]-x[i-1])
        grp_temp = temp.groupby('Track UID', sort=False)
        temp['_dx1']   = grp_temp['X coordinate'].diff()
        temp['_dy1']   = grp_temp['Y coordinate'].diff()
        temp['_theta'] = np.arctan2(temp['_dy1'].values, temp['_dx1'].values)

        max_lag = int(track_sizes.max()) - 1

        # Collect raw per-pair per-lag records; aggregate later at replicate/condition levels
        msd_records = []
        turn_records = []

        # We group once and iterate by lag
        # For each lag, compute shifted coordinates within trajectories
        x_arr     = temp['X coordinate'].values
        y_arr     = temp['Y coordinate'].values
        theta_arr = temp['_theta'].values
        pos_arr   = temp['_pos'].values
        n_arr     = temp['_n'].values
        cond_arr  = temp['Condition'].values
        rep_arr   = temp['Replicate'].values
        uid_arr   = temp['Track UID'].values if 'Track UID' in temp.columns else temp.index.get_level_values('Track UID').values

        for lag in range(1, max_lag + 1):
            # Only rows where pos + lag < n (i.e., the shifted partner exists)
            valid_mask = (pos_arr + lag) < n_arr

            if not valid_mask.any():
                break

            # MSD: x[pos+lag] - x[pos] and y[pos+lag] - y[pos]
            shifted_x = grp_temp['X coordinate'].shift(-lag)
            shifted_y = grp_temp['Y coordinate'].shift(-lag)

            dx = shifted_x.values - x_arr
            dy = shifted_y.values - y_arr
            sq_disp = dx**2 + dy**2

            valid_idx = np.where(valid_mask)[0]
            if valid_idx.size == 0:
                continue

            # Keep raw squared-displacement pairs for pooled aggregation later
            lag_df = pd.DataFrame({
                'Track UID': uid_arr[valid_idx],
                'Condition': cond_arr[valid_idx],
                'Replicate': rep_arr[valid_idx],
                'sq_disp':   sq_disp[valid_idx],
                'lag':       lag,
            })
            msd_records.append(lag_df)

            # Turning angles: theta[pos+lag] - theta[pos]
            shifted_theta = grp_temp['_theta'].shift(-lag)
            dtheta_all = shifted_theta.values - theta_arr

            # Wrap to [-pi, pi]
            dtheta_all = (dtheta_all + np.pi) % (2 * np.pi) - np.pi

            # Valid directional changes: pos >= 1 AND pos+lag < n AND both thetas finite
            turn_valid = valid_mask & (pos_arr >= 1) & np.isfinite(theta_arr) & np.isfinite(shifted_theta.values)
            turn_idx = np.where(turn_valid)[0]

            if turn_idx.size > 0:
                turn_df = pd.DataFrame({
                    'Track UID': uid_arr[turn_idx],
                    'Condition': cond_arr[turn_idx],
                    'Replicate': rep_arr[turn_idx],
                    'dtheta':    dtheta_all[turn_idx],
                    'lag':       lag,
                })
                turn_records.append(turn_df)

        # If no records produced, return empty
        if not msd_records:
            return pd.DataFrame(columns=self._COLUMNS['TIMEINTERVALS'])

        # Concatenate all lag results
        all_msd  = pd.concat(msd_records, ignore_index=True)
        all_turn = (
            pd.concat(turn_records, ignore_index=True)
            if turn_records
            else pd.DataFrame(columns=['Condition', 'Replicate', 'Track UID', 'dtheta', 'lag'])
        )

        # Aggregate pooled pairs per tier × lag
        tier_lag_cols =  self.tier + ['lag']
        cond_lag_cols = ['Condition', 'lag']

        def _agg_msd_turn(msd_src: pd.DataFrame, turn_src: pd.DataFrame, by_cols: list[str], prefix: str) -> pd.DataFrame:
            """Aggregate pooled MSD pairs and turning-angle pairs for given grouping."""
            msd_grp = msd_src.groupby(by_cols, sort=False)['sq_disp']
            msd_keys = msd_src.groupby(by_cols, sort=False)

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
                agg_dict[f'{prefix} MSD sem'] = msd_grp.agg(self.sem)
                if 'ci' in self.INFER_ERR:
                    ci_series = msd_grp.agg(self.ci)
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

            # Counts based on raw MSD pairs
            agg_dict[f'{prefix} Tracks contributing'] = msd_keys['Track UID'].nunique().astype(int)
            agg_dict[f'{prefix} Position pairs contributing'] = msd_grp.size().astype(int)

            result = pd.DataFrame(agg_dict)

            # Turning-angle aggregation from raw angle pairs
            if not turn_src.empty:
                turn_sin = turn_src.assign(
                    _s=np.sin(turn_src['dtheta'].values),
                    _c=np.cos(turn_src['dtheta'].values),
                )
                turn_circ = turn_sin.groupby(by_cols, sort=False).agg(
                    ms=('_s', 'mean'),
                    mc=('_c', 'mean'),
                )
                circ_mean = np.arctan2(turn_circ['ms'].values, turn_circ['mc'].values)
                circ_R = np.hypot(turn_circ['ms'].values, turn_circ['mc'].values)
                circ_var = 1.0 - circ_R

                if self.cat_descr or self.cat_descr_err or self.cat_infer_err:
                    result[f'{prefix} Directional change mean'] = pd.Series(np.rad2deg(np.abs(circ_mean)), index=turn_circ.index)

                if self.cat_descr_err:
                    result[f'{prefix} Directional change var'] = pd.Series(circ_var, index=turn_circ.index)

            return result

        # Base level aggregation
        reps  = _agg_msd_turn(all_msd, all_turn, tier_lag_cols, '{per replicate}'); reps.reset_index(inplace=True)
        conds = _agg_msd_turn(all_msd, all_turn, cond_lag_cols, '{per condition}'); conds.reset_index(inplace=True)
        df = reps.merge(conds, on=cond_lag_cols, how='left')

        # Rename 'lag' -> 'Frame lag' and add 'Time lag'
        df = df.rename(columns={'lag': 'Frame lag'})
        df['Time lag'] = df['Frame lag'] * t_step

        # Reorder: tier columns first, then Frame lag, Time lag, then stats
        front_cols = self.tier + ['Frame lag', 'Time lag']
        other_cols = [c for c in df.columns if c not in front_cols]
        df = df[front_cols + other_cols]

        # Sort
        df = df.sort_values(self.tier + ['Frame lag'], ignore_index=True)

        # JSON-safe cleanup for Shiny/front-end serializers (no NaN/Inf in strict JSON)
        df = df.replace([np.inf, -np.inf], np.nan)

        # Re-attach color columns if they were present on input
        if _color_stash is not None:
            df = df.merge(_color_stash, on=_stash_keys, how='left')

        if self.SIGNIFICANT_FIGURES:
            df = self.signify(df)
        if self.DECIMALS_PLACES:
            df = self.norm_decimals(df)

        BaseDataInventory.TimeIntervals = df.copy()

        return df


    def format_digits(self, df: pd.DataFrame, *, sig_figs: int = None, decimals: int = None) -> pd.DataFrame:
        """ Formats numeric values in the DataFrame according to specified significant figures and decimal places. """

        if sig_figs:
            df = self.signify(df, sig_figs=sig_figs)

        if decimals:
            df = self.norm_decimals(df, decimals=decimals)

        return df


    def signify(self, df: pd.DataFrame, *, sig_figs: int = None) -> float:
        """ Round numeric values in a DataFrame to a specified number of significant figures. """

        if df.empty:
            return df.copy()

        if sig_figs is None:
            sig_figs = self.SIGNIFICANT_FIGURES

        valuer = Values()

        df_rounded = df.copy()
        for col in df_rounded.select_dtypes(include=[np.number]).columns:
            df_rounded[col] = df_rounded[col].apply(lambda x: valuer.RoundSigFigs(x, sigfigs=sig_figs))

        return df_rounded
    

    def norm_decimals(self, df: pd.DataFrame, decimals: int = None) -> pd.DataFrame:
        """ Normalize decimal places across numeric columns in a DataFrame by rounding to a specified number of decimal places. """

        if df.empty:
            return df.copy()
        
        if decimals is None:
            decimals = self.DECIMALS_PLACES

        # Ensure all values have the same number of decimals: (round, fill)
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].apply(lambda x: round(x, decimals) if pd.notnull(x) else x)

        return df


    def _general_agg_stats(self, df: pd.DataFrame, exclude: list[str], *, group_by: list[str] = ['Track UID']) -> pd.DataFrame:
        """ Compute general aggregate statistics (min, max, mean, sd, sem, median) for numeric columns in the DataFrame, grouped by specified columns. Exclude specified columns from aggregation. """

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
            
            if any(t in col for t in ['Direction', 'direction', 'Directional', 'directional', 'Turn', 'turn']) and not col.endswith('var'):
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
        """ Resolves a list or dictionary of aggregation specs into a mapping of output labels to aggregation functions. """
        
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
        """ Insert a (key: value) pair into a dictionary at a specific position.

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
        """ Wrap angles in radians to the range [-π, π]. """
        return (a + np.pi) % (2*np.pi) - np.pi


    def _circ_mean(self, a: np.ndarray) -> float:
        """ Circular mean of angles in radians. """
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return np.nan
        
        s = np.nanmean(np.sin(a))
        c = np.nanmean(np.cos(a))
        if np.isnan(s) or np.isnan(c):
            return np.nan
        
        return float(np.arctan2(s, c))


    def _circ_var(self, a: np.ndarray) -> float:
        """ Circular variance defined as 1 - R, where R is the mean resultant length of the angles. """
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
        """ Lower bound of the interquartile range = Q1. """
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]  # drop NaN/Inf
        if a.size == 0:
            return np.nan
        return float(np.percentile(a, 25))
    

    def _q75(self, a: np.ndarray) -> float:
        """ Upper bound of the interquartile range = Q3. """
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]  # drop NaN/Inf
        if a.size == 0:
            return np.nan
        return float(np.percentile(a, 75))
    

    def ci(self, a, *, n_resamples: int | None = None, confidence_level: float | None = None, **kwargs) -> tuple[float, float]:
        """ Confidence interval via bootstrap.
        
        Parameters
        ----------
        a : array-like
            *1D array of values to compute the confidence interval for.*
        
        n_resamples : int, (default self.BOOTSTRAP_RESAMPLES = 1000)
            *Number of bootstrap resamples to perform. Can be set through Stats.BOOTSTRAP_RESAMPLES*

        confidence_level : float, (default self.CONFIDENCE_LEVEL = 95)
            *Confidence level for the interval (%). Can be set through Stats.CONFIDENCE_LEVEL*

        statistic : callable, (default `np.mean`)
            *Function for which the confidence interval is computed (e.g. `np.mean`, `np.median`).*
        
        method : str, (default 'BCa')
            *Confidence interval computation method. Default is 'BCa' (bias-corrected and accelerated). 
            If 'BCa' fails, the method falls back to 'percentile'. Used method is stored in and can be acquired through `Stats._ci_method_used`.*
            
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
    
    def sem(self, x: np.ndarray | pd.Series) -> float:
        """ Standard error of the mean. """
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
    """ Contains static methods utilized in the Peregrin Shiny App """

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

