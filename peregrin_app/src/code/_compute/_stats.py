from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict



@staticmethod
def Spots(df: pd.DataFrame) -> pd.DataFrame:

    """
    Compute per-frame tracking metrics for each cell track in the DataFrame:
    - Distance: Euclidean distance between consecutive positions
    - Direction: direction of travel in radians
    - Track length: cumulative distance along the track
    - Track displacement: straight-line distance from track start
    - Confinement ratio: Track displacement / Track length

    Expects columns: Condition, Replicate, Track ID, X coordinate, Y coordinate, Time point
    Returns a DataFrame sorted by Condition, Replicate, Track ID, Time point with new metric columns.
    """
    if df.empty:
        return df.copy()

    df.sort_values(by=['Condition', 'Replicate', 'Track ID', 'Time point'], inplace=True)

    # Sort and work on a copy
    # df = df.sort_values(['Condition', 'Replicate', 'Track ID', 'Time point']).copy()
    grp = df.groupby(['Condition', 'Replicate', 'Track ID'], sort=False)

    # ---- Add unique per-track index (1-based) ----
    df['Track UID'] = grp.ngroup()
    df.set_index(['Track UID'], drop=False, append=False, inplace=True, verify_integrity=False)

    # Distance between current and next position
    df['Distance'] = np.hypot(
        grp['X coordinate'].diff(),
        grp['Y coordinate'].diff()
    ).fillna(np.nan)

    # Direction of travel (radians) based on diff to previous point
    df['Direction'] = np.arctan2(
        grp['Y coordinate'].diff(),
        grp['X coordinate'].diff()
        ).fillna(np.nan)

    # Cumulative track length
    df['Cumulative track length'] = grp['Distance'].cumsum()

    # Net (straight-line) distance from the start of the track
    start = grp[['X coordinate', 'Y coordinate']].transform('first')
    df['Cumulative track displacement'] = np.hypot(
        (df['X coordinate'] - start['X coordinate']),
        (df['Y coordinate'] - start['Y coordinate'])
    ).replace(0, np.nan)

    # Confinement ratio: Track displacement vs. actual path length
    # Avoid division by zero by replacing zeros with NaN, then fill
    df['Cumulative confinement ratio'] = (df['Cumulative track displacement'] / df['Cumulative track length'].replace(0, np.nan)).fillna(np.nan)

    df['Frame'] = grp['Time point'].rank(method='dense').astype(int)

    return df


@staticmethod
def Tracks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive track-level metrics for each cell track in the DataFrame, including:
    - Track length, displacement, confinement ratio
    - Speed stats: min, max, mean, std, var, median, quantiles, IQR, SEM, CI95
    - Circular direction stats (mean/median in rad/deg; 'std' via resultant length proxy)
    Expects columns: Condition, Replicate, Track ID, Distance, X coordinate, Y coordinate, Direction
    """
    if df.empty:
        cols = [
            'Condition','Replicate','Track ID',
            'Track length','Track displacement','Confinement ratio',
            'Speed min','Speed max','Speed mean','Speed std','Speed median',
            'Speed q10','Speed q25','Speed q50','Speed q75','Speed q95','Speed IQR',
            'Speed SEM','Speed CI95 low','Speed CI95 high',
            'Direction mean','Direction sd','Direction median',
            'Track points'
        ]
        return pd.DataFrame(columns=cols)

    grp = df.groupby(['Condition','Replicate','Track ID'], sort=False)

    # choose the quantiles you want; edit as needed
    agg_spec = {
        'Track length':     ('Distance', 'sum'),
        'Speed min':        ('Distance', 'min'),
        'Speed max':        ('Distance', 'max'),
        'Speed mean':       ('Distance', 'mean'),
        'Speed sd':         ('Distance', 'std'),
        'Speed median':     ('Distance', 'median'),
        'start_x':          ('X coordinate', 'first'),
        'end_x':            ('X coordinate', 'last'),
        'start_y':          ('Y coordinate', 'first'),
        'end_y':            ('Y coordinate', 'last'),
    }

    agg = grp.agg(**agg_spec)

    if 'Replicate color' in df.columns:
        colors = grp['Replicate color'].first()
        agg = agg.merge(colors, left_index=True, right_index=True)

    # Displacement and confinement
    agg['Track displacement'] = np.hypot(agg['end_x'] - agg['start_x'], agg['end_y'] - agg['start_y'])
    agg['Confinement ratio'] = (agg['Track displacement'] / agg['Track length'].replace(0, np.nan)).fillna(0)
    agg = agg.drop(columns=['start_x','end_x','start_y','end_y'])

    # Points per track
    n = grp.size().rename('Track points')
    agg = agg.merge(n, left_index=True, right_index=True)

    # Circular direction stats
    sin_cos = df.assign(
        _sin=np.sin(df['Direction']), 
        _cos=np.cos(df['Direction'])
    )
    
    dir_agg = sin_cos.groupby(['Condition','Replicate','Track ID'], sort=False).agg(
        mean_sin=('_sin','mean'),
        mean_cos=('_cos','mean')
    )

    dir_aggs = {
        'Direction mean': lambda row: np.arctan2(row['mean_sin'], row['mean_cos']),
        'Direction sd': lambda row: np.sqrt(-2 * np.log(np.sqrt(row['mean_sin']**2 + row['mean_cos']**2))),
    }

    for col, func in dir_aggs.items():
        dir_agg[col] = dir_agg.apply(func, axis=1)

    dir_agg = dir_agg.drop(columns=['mean_sin','mean_cos'])

    result = agg.merge(dir_agg, left_index=True, right_index=True).reset_index()
    result['Track UID'] = np.arange(len(result))
    result.set_index('Track UID', drop=True, inplace=True, verify_integrity=True)
    return result


@staticmethod
def Frames(df: pd.DataFrame) -> pd.DataFrame:
    
    if df.empty:
        # define columns
        cols = ['Condition','Replicate','Time point','Frame'] + \
            [f'{metric} {stat}' for metric in ['Track length','Track displacement','Confinement ratio'] for stat in ['min','max','mean','sd','median']] + \
            [f'Speed {stat}' for stat in ['min','max','mean','sd','median']] + \
            ['Direction mean','Direction sd']
        return pd.DataFrame(columns=cols)

    group_cols = ['Condition','Replicate','Time point','Frame']

    # 1) stats on track metrics per frame
    metrics = ['Cumulative track length','Cumulative track displacement','Cumulative confinement ratio']
    agg_funcs = ['min','max','mean','std','median']
    # build agg dict
    agg_dict = {m: agg_funcs for m in metrics}
    frame_agg = df.groupby(group_cols).agg(agg_dict)
    # flatten columns
    frame_agg.columns = [f'{metric} {stat}' for metric, stat in frame_agg.columns]

    # 2) speed stats (Distance distributions)
    speed_agg = df.groupby(group_cols)['Distance'].agg(['min','max','mean','std','median'])
    speed_agg.columns = [f'Speed {stat}' for stat in speed_agg.columns]

    # 3) circular direction stats per frame
    # compute sin/cos columns
    tmp = df.assign(_sin=np.sin(df['Direction']), _cos=np.cos(df['Direction']))
    dir_frame = tmp.groupby(group_cols).agg({'_sin':'mean','_cos':'mean','Direction':'count'})
    # mean direction
    dir_frame['Direction mean'] = np.arctan2(dir_frame['_sin'], dir_frame['_cos'])
    # circular std: R = sqrt(mean_sin^2+mean_cos^2)
    dir_frame['Direction sd'] = np.hypot(dir_frame['_sin'], dir_frame['_cos'])
    
    dir_frame = dir_frame.drop(columns=['_sin','_cos','Direction'], errors='ignore')

    # merge all
    time_stats = frame_agg.merge(speed_agg, left_index=True, right_index=True)
    time_stats = time_stats.merge(dir_frame, left_index=True, right_index=True)
    time_stats = time_stats.rename(columns={
        'Cumulative track length min': 'Track length min',
        'Cumulative track length max': 'Track length max',
        'Cumulative track length mean': 'Track length mean',
        'Cumulative track length sd': 'Track length sd',
        'Cumulative track length median': 'Track length median',
        'Cumulative track displacement min': 'Track displacement min',
        'Cumulative track displacement max': 'Track displacement max',
        'Cumulative track displacement mean': 'Track displacement mean',
        'Cumulative track displacement sd': 'Track displacement sd',
        'Cumulative track displacement median': 'Track displacement median',
        'Cumulative confinement ratio min': 'Confinement ratio min',
        'Cumulative confinement ratio max': 'Confinement ratio max',
        'Cumulative confinement ratio mean': 'Confinement ratio mean',
        'Cumulative confinement ratio sd': 'Confinement ratio sd',
        'Cumulative confinement ratio median': 'Confinement ratio median',
    })
    time_stats = time_stats.reset_index()

    return time_stats


from collections import defaultdict

class TimeIntervals:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    @staticmethod
    def _wrap_pi(a: np.ndarray) -> np.ndarray:
        # Wrap to (-pi, pi]
        return (a + np.pi) % (2*np.pi) - np.pi

    @staticmethod
    def _circ_mean(a: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return np.nan
        s = np.nanmean(np.sin(a))
        c = np.nanmean(np.cos(a))
        if np.isnan(s) or np.isnan(c):
            return np.nan
        return float(np.arctan2(s, c))

    # @staticmethod
    def _circ_median(self, a: np.ndarray) -> float:
        # Approximate circular median by unwrapping around the circular mean
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return np.nan
        mu = self._circ_mean(a)
        if np.isnan(mu):
            return np.nan
        shifted = self._wrap_pi(a - mu)
        med = np.median(shifted)
        return float(self._wrap_pi(mu + med))

    # @staticmethod
    def ComputeTimeIntervals(self) -> pd.DataFrame:
        """
        MSD across lags for each Condition×Replicate.
        Unit of analysis = track. Each track contributes one MSD value per lag.
        Statistics across tracks:
        - MSD mean, MSD sd (ddof=1), MSD sem = sd/sqrt(N), MSD median
        - Turn angle mean (rad), Turn angle median (rad) using circular statistics
        - Tracks contributing = number of tracks with at least one pair for that lag

        Turning-angle definition for lag L:
        - Compute step headings from consecutive frames: theta[t] = atan2(y[t+1]-y[t], x[t+1]-x[t]).
        - Turning angles at lag L are wrapped differences: wrap(theta[t+L] - theta[t]).
        - Per-track circular mean/median are computed from those differences, then aggregated across tracks
            with circular mean/median so each track has equal weight.
        """

        cols = [
            'Condition', 'Replicate', 'Frame lag', 'Time lag', 'Tracks contributing',
            'MSD mean', 'MSD sem', 'MSD sd', 'MSD median',
            'Turn mean (rad)', 'Turn median (rad)'
        ]

        if self.df.empty: 
            return pd.DataFrame(columns=cols)

        df = self.df.copy()

        # Ensure numeric coords
        df['X coordinate'] = pd.to_numeric(df['X coordinate'], errors='coerce')
        df['Y coordinate'] = pd.to_numeric(df['Y coordinate'], errors='coerce')

        # Time step from unique diffs; use median to resist irregular sampling
        t_unique = np.sort(df['Time point'].unique())
        if t_unique.size < 2:
            return pd.DataFrame(columns=cols)
        t_step = float(np.median(np.diff(t_unique)))

        # Collect per-track metrics at each lag
        per_track_msd = defaultdict(list)       # (cond, rep, lag) -> [msd_track, ...]
        per_track_turn_mean = defaultdict(list) # (cond, rep, lag) -> [circ_mean_track, ...]
        per_track_turn_med  = defaultdict(list) # (cond, rep, lag) -> [circ_median_track, ...]

        # Iterate tracks
        # Accept either ordinary column or index; group by column to avoid index requirements
        # if 'Track UID' not in df.columns:
        #     raise KeyError("Lags expects a 'Track UID' column.")
        for track_uid, g in df.groupby(level='Track UID', sort=False):
            if len(g) < 2:
                continue

            # Sort frames within track
            g = g.sort_values('Time point')

            x = g['X coordinate'].to_numpy()
            y = g['Y coordinate'].to_numpy()
            cond = g['Condition'].iloc[0]
            rep  = g['Replicate'].iloc[0]

            n = len(g)

            # Headings from lag-1 steps for turning angles
            dx1 = x[1:] - x[:-1]
            dy1 = y[1:] - y[:-1]
            theta = np.arctan2(dy1, dx1)  # length n-1

            for lag in range(1, n):
                # MSD per-track at this lag
                dx = x[lag:] - x[:-lag]
                dy = y[lag:] - y[:-lag]
                if dx.size > 0:
                    msd_track = float((dx*dx + dy*dy).mean())
                    per_track_msd[(cond, rep, lag)].append(msd_track)

                # Turning angles for this lag, using headings separated by 'lag'
                if theta.size > lag:
                    dtheta = self._wrap_pi(theta[lag:] - theta[:-lag])  # length (n-1 - lag)
                    if dtheta.size > 0:
                        per_track_turn_mean[(cond, rep, lag)].append(self._circ_mean(dtheta))
                        per_track_turn_med[(cond, rep, lag)].append(self._circ_median(dtheta))

        if not per_track_msd and not per_track_turn_mean and not per_track_turn_med:
            return pd.DataFrame(columns=cols)

        # Summarize across tracks
        keys = set(per_track_msd.keys()) | set(per_track_turn_mean.keys()) | set(per_track_turn_med.keys())

        rows = []
        for (cond, rep, lag) in sorted(keys, key=lambda k: (k[0], k[1], k[2])):
            # MSD across tracks (linear)
            arr = np.asarray(per_track_msd.get((cond, rep, lag), []), dtype=float)
            n_tracks = arr.size
            mean = float(arr.mean()) if n_tracks else np.nan
            sd = float(arr.std(ddof=1)) if n_tracks > 1 else np.nan
            sem = float(sd / np.sqrt(n_tracks)) if n_tracks > 1 else np.nan
            median = float(np.median(arr)) if n_tracks else np.nan

            # Turning angle across tracks (circular)
            turn_means = np.asarray(per_track_turn_mean.get((cond, rep, lag), []), dtype=float)
            turn_meds  = np.asarray(per_track_turn_med.get((cond, rep, lag), []), dtype=float)
            turn_mean_agg = self._circ_mean(turn_means) if turn_means.size else np.nan
            turn_med_agg  = self._circ_median(turn_meds) if turn_meds.size else np.nan

            rows.append({
                'Condition': cond,
                'Replicate': rep,
                'Frame lag': lag,
                'Time lag': lag * t_step,
                'Tracks contributing': int(max(n_tracks, turn_means.size, turn_meds.size)),

                'MSD mean': mean,
                'MSD sem': sem,
                'MSD sd': sd,
                'MSD median': median,

                'Turn mean (degrees)': np.rad2deg(np.abs(turn_mean_agg)),
                'Turn median (degrees)': np.rad2deg(np.abs(turn_med_agg)),
            })

        out = pd.DataFrame(rows).sort_values(
            ['Condition', 'Replicate', 'Frame lag'], ignore_index=True
        )
        return out
    
    def __call__(self) -> pd.DataFrame:
        return self.ComputeTimeIntervals()
