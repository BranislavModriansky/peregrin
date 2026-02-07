from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

from .._general import Values
from .._handlers._reports import Level


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

    Attributes
    ----------
    SIGNIFICANT_FIGURES : int, optional
        *If specified, during computations all values are going to be rounded to this number of significant figures.*

    DECIMALS_PLACES : int, optional
        *If specified, during computations all floating-point values are going to be normalized (rounded) to this number of decimal places.*
    """

    SIGNIFICANT_FIGURES: None | int = None
    DECIMALS_PLACES: None | int = None

    COLUMNS = {
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
            'Speed CI95 low','Speed CI95 high',
            'Direction mean','Direction var'
        ]
    }


    def __init__(self, **kwargs) -> None:
        self.noticequeue = kwargs.get('noticequeue', None)


    def GetAllStats(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
            self.noticequeue.Report(Level.warning, f"Input DataFrame to Spots method is empty; no computations performed.")
            return pd.DataFrame(columns=self.COLUMNS['SPOTS'])

        df.sort_values(by=['Condition', 'Replicate', 'Track ID', 'Time point'], inplace=True)

        grp = df.groupby(['Condition', 'Replicate', 'Track ID'], sort=False)

        # Provides a completely unique identifier for each track (Track UID) that is consistent across all methods and is used for merging.
        df['Track UID'] = grp.ngroup()
        df.set_index(['Track UID'], drop=False, append=False, inplace=True, verify_integrity=False)

        # Assigns frame numbers within each data subset based on the order of time points; starts at 1 for the first point in each track.
        df['Frame'] = df.groupby(['Condition', 'Replicate'], sort=False)['Time point'].rank(method='dense').astype('Int64')

        # Optional sanity warning (kept lightweight)
        try:
            bad = df.groupby(['Condition', 'Replicate', 'Time point'], sort=False)['Frame'].nunique(dropna=True).max()
            if bad and bad > 1:
                self.noticequeue.Report(Level.warning, f"Multiple frames assigned to the same Condition × Replicate × Time point combination; this may indicate time point multiplicates within the data or other data issues. Max frames per time point: {bad}.")
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
        valid = df['Frame'].gt(1) & theta_from_start.notna()

        # Define grouping keys
        gkeys = [df['Condition'], df['Replicate'], df['Track ID']]

        # Calculate sin and cos of the cumulated direction mean for each point excluding first frame
        sinv = pd.Series(np.where(valid, np.sin(theta_from_start), np.nan), index=df.index)
        cosv = pd.Series(np.where(valid, np.cos(theta_from_start), np.nan), index=df.index)

        # Cumulative sums of sin and cos components
        cum_sin = sinv.groupby(gkeys, sort=False).cumsum()
        cum_cos = cosv.groupby(gkeys, sort=False).cumsum()

        # Number of contributing angles so far (starts at 0 on frame 1)
        n_angles = valid.astype(int).groupby(gkeys, sort=False).cumsum().replace(0, np.nan)

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
            return pd.DataFrame(columns=self.COLUMNS['TRACKS'])

        stash = df[['Condition','Replicate','Track ID']].drop_duplicates()
        print(stash)

        df = df.set_index('Track UID', drop=True, verify_integrity=False)
        gcols = ['Track UID']

        
        grp = df.groupby(gcols, sort=False)

        # choose the quantiles you want; edit as needed
        agg_spec = {
            'Track length':     ('Distance', 'sum'),
            'Speed min':        ('Distance', 'min'),
            'Speed max':        ('Distance', 'max'),
            'Speed mean':       ('Distance', 'mean'),
            'Speed sd':         ('Distance', 'std'),
            'Speed sem':        ('Distance', lambda x: stats.sem(x)),
            'Speed median':     ('Distance', 'median'),
            'Speed q25':        ('Distance', lambda x: np.nanquantile(x, 0.25)),
            'Speed q75':        ('Distance', lambda x: np.nanquantile(x, 0.75)),
            'Speed CI95':       ('Distance', lambda x: stats.t.interval(0.95, len(x)-1, loc=np.mean(x), scale=stats.sem(x)) if len(x) > 1 else (np.nan, np.nan)),
            'start_x':          ('X coordinate', 'first'),
            'end_x':            ('X coordinate', 'last'),
            'start_y':          ('Y coordinate', 'first'),
            'end_y':            ('Y coordinate', 'last'),
        }
        
        agg = grp.agg(**agg_spec)

        # Split CI95 into two columns
        agg['Speed CI95 low'] = agg['Speed CI95'].apply(lambda x: x[0])
        agg['Speed CI95 high'] = agg['Speed CI95'].apply(lambda x: x[1])
        agg = agg.drop(columns=['Speed CI95'])

        # If colors were assigned, carry them over
        if 'Replicate color' in df.columns:
            colors = grp['Replicate color'].first()
            agg = agg.merge(colors, left_index=True, right_index=True)
        if 'Condition color' in df.columns:
            colors = grp['Condition color'].first()
            agg = agg.merge(colors, left_index=True, right_index=True)

        # Other general stats
        other = self._general_agg_stats(df, exclude=self.COLUMNS['SPOTS'])
        # agg = agg.merge(other, left_index=True, right_index=True)

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
        
        
        if self.SIGNIFICANT_FIGURES:
            df = self.Signify(df)
            
        if self.DECIMALS_PLACES:
            df = self.NormDecimals(df)
        
        BaseDataInventory.Tracks = df

        return df
    

    def Frames(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        #### *Computes time point statistics aggregated across tracks for each Condition × Replicate × Time point.*
        
        Parameters
        ----------
        df : pd.DataFrame
            ***(Spots) Input DataFrame must contain (at minimum) these columns:***
            - ``Condition``
            - ``Replicate``
            - ``Time point``
            - ``Frame``
            - ``Cumulative track length``
            - ``Cumulative track displacement``
            - ``Cumulative straightness index``
            - ``Distance``
            - ``Direction``

        Returns
        -------
        pd.DataFrame
            ***A DataFrame with one row per unique Condition × Replicate × Time point. Contains following columns:***

            For each of **``Cumulative track length``**, **``Cumulative track displacement``**, **``Cumulative straightness index``**, **``Cumulative speed``**, **``Distance => Instantaneous speed``** ***metrics***:
            - ***metric*``min``**:        *minimum value of a given metric across tracks at that time point.*
            - ***metric*``max``**:        *maximum value of a given metric across tracks at that time point.*
            - ***metric*``mean``**:       *mean value of a given metric across tracks at that time point.*
            - ***metric*``sd``**:         *standard deviation of a given metric across tracks at that time point.*
            - ***metric*``sem``**:        *standard error of the mean of a given metric across tracks at that time point.*
            - ***metric*``median``**:     *median value of a given metric across tracks at that time point.*
            - ***metric*``q25``**:        *25th percentile of a given metric across tracks at that time point.*
            - ***metric*``q75``**:        *75th percentile of a given metric across tracks at that time point.*
            - ***metric*``CI95 low``**:   *lower bound of the 95% confidence interval of a given metric across tracks at that time point.*
            - ***metric*``CI95 high``**:  *upper bound of the 95% confidence interval of a given metric across tracks at that time point.*
            \n
            Circular statistics:

            - **``Instantaneous direction mean``**:
            Mean of **``Direction => Instantaneous direction``** across tracks at that time point.

            - **``Instantaneous direction var``**:
            Circular variance of **``Direction => Instantaneous direction``** across tracks at that time point.

            - **``Cumulative direction global mean``**:
            Mean of **``Cumulative direction mean``** across tracks at that time point.

            - **``Cumulative direction global var``**:
            Circular variance of **``Cumulative direction mean``** across tracks at that time point.
        """

        # Keep output labels consistent with your docstring
        group_cols = ['Condition','Replicate','Time point','Frame']

        # Base metric names (Distance later rewritten to Instantaneous speed)
        metrics = [
            'Cumulative track length',
            'Cumulative track displacement',
            'Cumulative straightness index',
            'Cumulative speed',
            'Distance',
        ]

        # Map metrics to output labels
        metric_out = {m: m for m in metrics}
        # Rename 'Distance' to 'Instantaneous speed'
        metric_out['Distance'] = 'Instantaneous speed'

        # An empty DataFrame equivalent
        if df is None or df.empty:
            stats_cols = ['min','max','mean','sd','sem','median','q25','q75','CI95 low','CI95 high']
            cols = (
                group_cols
                + [f"{metric_out[m]} {s}" for m in metrics for s in stats_cols]
                + ['Instantaneous direction mean', 'Instantaneous direction var',
                   'Cumulative direction global mean', 'Cumulative direction global var']
            )
            return pd.DataFrame(columns=cols)

        # One groupby for basic aggregation statistics
        g = df.groupby(group_cols, sort=False, observed=True)[metrics]
        basic = g.agg(['min', 'max', 'mean', 'std', 'median', 'count'])  # std is sample sd (ddof=1)

        # Group (Condition × Replicate × Time point / Frame) quantiles
        q25 = g.quantile(0.25)
        q75 = g.quantile(0.75)

        # Assemble the DataFrame
        out = pd.DataFrame(index=basic.index)

        # Per-metric calculation of [min, max, mean, sd, std, median, IQR, CI95] statistics
        for m in metrics:
            # Output metric name assembly
            mout = metric_out[m]

            # Extract basic per-group stats
            vmin = basic[(m, 'min')]
            vmax = basic[(m, 'max')]
            vmean = basic[(m, 'mean')]
            vsd = basic[(m, 'std')]
            vmed = basic[(m, 'median')]
            # Counts non-nan values only (used in mean/std calculation); <2 if all values are the same or if there are nans
            vn = basic[(m, 'count')].astype(float)

            # sem = sd / sqrt(n), requires n>=2, else is undefined
            vsem = vsd / np.sqrt(vn)
            vsem = vsem.where(vn >= 2, np.nan)

            # CI95 = mean ± tcrit(df=n-1) * sem, requires n>=2, else is undefined
            tcrit = stats.t.ppf(0.975, vn - 1)
            ci_low = (vmean - tcrit * vsem).where(vn >= 2, np.nan)
            ci_high = (vmean + tcrit * vsem).where(vn >= 2, np.nan)

            out[f'{mout} min'] = vmin
            out[f'{mout} max'] = vmax
            out[f'{mout} mean'] = vmean
            out[f'{mout} sd'] = vsd
            out[f'{mout} sem'] = vsem
            out[f'{mout} median'] = vmed
            out[f'{mout} q25'] = q25[m]
            out[f'{mout} q75'] = q75[m]
            out[f'{mout} CI95 low'] = ci_low
            out[f'{mout} CI95 high'] = ci_high

        # Computes both instantaneous direction (from 'Direction') and cumulative direction (from 'Cumulative direction mean') stats
        tmp = df[group_cols + ['Direction', 'Cumulative direction mean']].copy()

        # Convert to numeric, coercing errors to NaN, then to numpy arrays (for sin/cos calculations)
        dir_vals = pd.to_numeric(tmp['Direction'], errors='coerce').to_numpy(dtype=float)
        cum_vals = pd.to_numeric(tmp['Cumulative direction mean'], errors='coerce').to_numpy(dtype=float)

        # Calculate sin and cos for both directions and store in temporary columns
        tmp['_sin_dir'] = np.sin(dir_vals)
        tmp['_cos_dir'] = np.cos(dir_vals)
        tmp['_sin_cum'] = np.sin(cum_vals)
        tmp['_cos_cum'] = np.cos(cum_vals)

        # Group by the same keys and calculate mean of sin and cos components for both directions
        circ = tmp.groupby(group_cols, sort=False, observed=True).agg(
            sin_dir=('_sin_dir', 'mean'),
            cos_dir=('_cos_dir', 'mean'),
            sin_cum=('_sin_cum', 'mean'),
            cos_cum=('_cos_cum', 'mean'),
        )

        # Calculate circular mean and variance for Instantaneous direction
        out['Instantaneous direction mean'] = np.arctan2(circ['sin_dir'], circ['cos_dir'])
        R_dir = np.hypot(circ['sin_dir'], circ['cos_dir'])
        out['Instantaneous direction var'] = (1.0 - R_dir)

        # Calculate global circular mean and variance for Cumulative direction mean
        out['Cumulative direction global mean'] = np.arctan2(circ['sin_cum'], circ['cos_cum'])
        R_cum = np.hypot(circ['sin_cum'], circ['cos_cum'])
        out['Cumulative direction global var'] = (1.0 - R_cum)

        # Reset index to columns
        df = out.reset_index()

        if self.SIGNIFICANT_FIGURES:
            df = self.Signify(df)

        if self.DECIMALS_PLACES:
            df = self.NormDecimals(df)

        BaseDataInventory.Frames = df

        return df


    def TimeIntervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        #### *Computes time interval (time lag) statistics across tracks for each Condition × Replicate × Time lag.*
        
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
            ***A DataFrame with one row per unique Condition × Replicate × Time lag. Contains following columns:***

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
            - **``sem``**: standard error of the mean of MSD.
            - **``sd``**: standard deviation of MSD.
            - **``median``**: median MSD.
            - **``q25``**: 25th percentile of MSD.
            - **``q75``**: 75th percentile of MSD.
            - **``CI95 low``**: lower bound of the 95% confidence interval of MSD.
            - **``CI95 high``**: upper bound of the 95% confidence interval of MSD.

            **``Turn``** statistics across circular mean of turning angles for trajectories at specific time intervals/lags:
            - **``mean``**: circular mean of turning angles.
            - **``var``**: circular variance of turning angles.
        """

        # Output columns
        cols = [
            'Condition','Replicate','Frame lag','Time lag','Tracks contributing',
            *[f'MSD {s}' for s in ['min','max','mean','sem','sd','median','q25','q75','CI95 low','CI95 high']],
            'Turn mean','Turn var'
        ]

        if df.empty: 
            return pd.DataFrame(
                columns=cols
            )

        # Unique time point differences -> time steps; use median to resist irregular sampling
        t_unique = np.sort(df['Time point'].unique())

        # <2 unique time points returns an empty DataFrame.
        if t_unique.size < 2:
            return pd.DataFrame(columns=cols)
        
        t_step = float(np.diff(t_unique)[0])

        # Collect per-track metrics at each lag
        per_track_msd = defaultdict(list)
        per_track_turn_mean = defaultdict(list)

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
            rep  = g['Replicate'].iloc[0]

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
                    per_track_msd[(cond, rep, lag)].append(msd_track)

                # Turning angles for this lag
                if theta.size > lag:
                    dtheta = self._wrap_pi(theta[lag:] - theta[:-lag])
                    if dtheta.size > 0:
                        per_track_turn_mean[(cond, rep, lag)].append(self._circ_mean(dtheta))

        # If no tracks had enough points to compute any metrics, return an empty DataFrame with the correct columns.
        if not per_track_msd and not per_track_turn_mean:
            return pd.DataFrame(columns=cols)

        # Summarize computed mean squared displacements and turning angles across tracks
        keys = set(per_track_msd.keys()) | set(per_track_turn_mean.keys())

        # Initialize list of rows for the output DataFrame
        rows = []

        # Iterate through each unique Condition × Replicate × Time lag combination and compute summary statistics
        for (cond, rep, lag) in sorted(keys, key=lambda k: (k[0], k[1], k[2])):

            # Mean squared displacement statistics across tracks
            # Get the array of mean squared displacement values
            arr = np.asarray(per_track_msd.get((cond, rep, lag), []), dtype=float)
            n_tracks = arr.size

            # Turning angle statistics across tracks
            turn_means = np.asarray(per_track_turn_mean.get((cond, rep, lag), []), dtype=float)
            turn_mean_agg = self._circ_mean(turn_means) if turn_means.size else np.nan
            turn_var_agg = self._circ_var(turn_means) if turn_means.size else np.nan

            rows.append({
                'Condition': cond,
                'Replicate': rep,
                'Frame lag': lag,
                'Time lag': lag * t_step,
                'Tracks contributing': int(max(n_tracks, turn_means.size)),

                'MSD min': float(arr.min()) if n_tracks else np.nan,
                'MSD max': float(arr.max()) if n_tracks else np.nan,
                'MSD mean': float(arr.mean()) if n_tracks else np.nan,
                'MSD sem': float(stats.sem(arr)) if n_tracks > 1 else np.nan,
                'MSD sd': float(arr.std(ddof=1)) if n_tracks > 1 else np.nan,
                'MSD median': float(np.median(arr)) if n_tracks else np.nan,
                'MSD q25': float(np.nanquantile(arr, 0.25)) if n_tracks else np.nan,
                'MSD q75': float(np.nanquantile(arr, 0.75)) if n_tracks else np.nan,
                'MSD CI95': stats.t.interval(0.95, n_tracks - 1, loc=arr.mean(), scale=stats.sem(arr)) if n_tracks > 1 else (np.nan, np.nan),

                'Turn mean': np.rad2deg(np.abs(turn_mean_agg)),
                'Turn var': turn_var_agg,
            })

        df = pd.DataFrame(rows).sort_values(
            ['Condition', 'Replicate', 'Frame lag'], ignore_index=True
        )

        # Split MSD CI95 into two columns
        df['MSD CI95 low'] = df['MSD CI95'].apply(lambda x: x[0])
        df['MSD CI95 high'] = df['MSD CI95'].apply(lambda x: x[1])
        df = df.drop(columns=['MSD CI95'])

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
            Input DataFrame with numeric values to be rounded.
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
        

    def _wrap_pi(self, a: np.ndarray) -> np.ndarray:
        return (a + np.pi) % (2*np.pi) - np.pi


    def _circ_mean(self, a: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return np.nan
        s = np.nanmean(np.sin(a))
        c = np.nanmean(np.cos(a))
        if np.isnan(s) or np.isnan(c):
            return np.nan
        return float(np.arctan2(s, c))


    def _circ_var(self, a: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return np.nan
        s = np.nanmean(np.sin(a))
        c = np.nanmean(np.cos(a))
        if np.isnan(s) or np.isnan(c):
            return np.nan
        R = np.hypot(s, c)
        return float(1.0 - R)

    
    def _general_agg_stats(self, df: pd.DataFrame, exclude: list[str], *, group_by: list[str] = ['Track UID']) -> pd.DataFrame:
        """
        #### *Computes basic statistics (min, max, mean, sd, sem, median) for all numeric columns in the input DataFrame, excluding core columns.*

        Parameters
        ----------
        df : pd.DataFrame
            ***Unstripped Spots DataFrame** containing columns, other than the core <- ``self.COLUMNS['SPOTS']`` columns, with numeric values to be aggregated.*

        exclude : list[str]
            *List of column names to be excluded from aggregation.*

        group_by : list[str], optional
            *List of column or index names to group by. Default is ['Track UID'].*

        Returns
        -------
        pd.DataFrame
            *DataFrame with basic statistics for each additional numeric column which is not found in the core <- ``self.COLUMNS['SPOTS']`` columns.*
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
                    "sd": s.std(skipna=True),
                    "variance": s.var(skipna=True),
                }

        value_counts = series.value_counts(dropna=True, normalize=True).head(3)
        return {
            "type": "type_zero",
            "missing": int(series.isna().sum()),
            "distinct": int(series.nunique(dropna=True)),
            "top": [(idx, round(val * 100, 1)) for idx, val in value_counts.items()],
        }

