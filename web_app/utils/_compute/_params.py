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
            'Direction mean','Direction std','Direction median',
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
        mean_cos=('_cos','mean'),
        median_sin=('_sin','median'),
        median_cos=('_cos','median')
    )

    dir_aggs = {
        'Direction mean': lambda row: np.arctan2(row['mean_sin'], row['mean_cos']),
        'Direction sd': lambda row: np.sqrt(-2 * np.log(np.sqrt(row['mean_sin']**2 + row['mean_cos']**2))),
        'Direction median': lambda row: np.arctan2(row['median_sin'], row['median_cos']),
    }

    for col, func in dir_aggs.items():
        dir_agg[col] = dir_agg.apply(func, axis=1)

    dir_agg = dir_agg.drop(columns=['mean_sin','mean_cos','median_sin','median_cos'])

    result = agg.merge(dir_agg, left_index=True, right_index=True).reset_index()
    result['Track UID'] = np.arange(len(result))
    result.set_index('Track UID', drop=True, inplace=True, verify_integrity=True)
    return result


@staticmethod
def Frames(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-frame (time point) summary metrics grouped by Condition, Replicate, Time point:
    - Track length, Track displacement, Confinement ratio distributions: min, max, mean, std, median
    - Speed (Distance) distributions as Speed min, Speed max, Speed mean, Speed std, Speed median
    - Direction distributions (circular): Direction mean (rad), Direction std (rad), Direction median (rad)
        and corresponding degrees

    Expects columns: Condition, Replicate, Time point, Track length, Track displacement,
                    Confinement ratio, Distance, Direction
    Returns a DataFrame indexed by Condition, Replicate, Time point with all time-point metrics.
    """
    if df.empty:
        # define columns
        cols = ['Condition','Replicate','Time point','Frame'] + \
            [f'{metric} {stat}' for metric in ['Track length','Track displacement','Confinement ratio'] for stat in ['min','max','mean','std','median']] + \
            [f'Speed {stat}' for stat in ['min','max','mean','std','median']] + \
            ['Direction mean (rad)','Direction std (rad)','Direction median (rad)',
                'Direction mean (deg)','Direction std (deg)','Direction median (deg)']
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
    dir_frame['Direction std'] = np.hypot(dir_frame['_sin'], dir_frame['_cos'])
    # median direction: use groupby apply median sin/cos
    median = tmp.groupby(group_cols).agg({'_sin':'median','_cos':'median'})
    dir_frame['Direction median'] = np.arctan2(median['_sin'], median['_cos'])
    
    dir_frame = dir_frame.drop(columns=['_sin','_cos','Direction'], errors='ignore')

    # merge all
    time_stats = frame_agg.merge(speed_agg, left_index=True, right_index=True)
    time_stats = time_stats.merge(dir_frame, left_index=True, right_index=True)
    time_stats = time_stats.rename(columns={
        'Cumulative track length min': 'Track length min',
        'Cumulative track length max': 'Track length max',
        'Cumulative track length mean': 'Track length mean',
        'Cumulative track length std': 'Track length std',
        'Cumulative track length median': 'Track length median',
        'Cumulative track displacement min': 'Track displacement min',
        'Cumulative track displacement max': 'Track displacement max',
        'Cumulative track displacement mean': 'Track displacement mean',
        'Cumulative track displacement std': 'Track displacement std',
        'Cumulative track displacement median': 'Track displacement median',
        'Cumulative confinement ratio min': 'Confinement ratio min',
        'Cumulative confinement ratio max': 'Confinement ratio max',
        'Cumulative confinement ratio mean': 'Confinement ratio mean',
        'Cumulative confinement ratio std': 'Confinement ratio std',
        'Cumulative confinement ratio median': 'Confinement ratio median',
    })
    time_stats = time_stats.reset_index()

    return time_stats


@staticmethod
def TimeIntervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean displacement (your current 'MSD') for each Condition, Replicate, and time interval.
    Exact outcome preserved:
      - lag==1 uses mean of 'Distance'
      - lag>1 uses mean of np.hypot(dx, dy)
      - 'Tracks contributing' is the global count per lag (not per group)
    Expects columns:
      'Condition','Replicate','Track UID','Frame','Time point','X coordinate','Y coordinate','Distance' (for lag==1).
    """
    cols = [
        'Condition', 'Replicate', 'Tracks contributing', 'Time lag',
        'MSD mean', 'MSD sem', 'MSD sd', 'MSD median', 'Frame lag'
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)

    df = df.copy()
    # Ensure track-ordering by frame
    df.set_index('Track UID', inplace=True, drop=False, verify_integrity=False)
    df.sort_index(inplace=True)
    # Time step from unique time points
    t_unique = np.sort(df['Time point'].unique())
    if t_unique.size < 2:
        return pd.DataFrame(columns=cols)
    t_step = (t_unique[-1] - t_unique[0]) / (t_unique.size - 1)

    # Accumulators
    values_by_key = defaultdict(list)   # key = (Condition, Replicate, lag)
    contrib_global = defaultdict(int)   # key = lag -> number of tracks with length > lag

    # Iterate tracks once
    for track_uid, g in df.groupby(level='Track UID'):
        n = len(g)
        if n < 2:
            continue

        x = g['X coordinate'].to_numpy()
        y = g['Y coordinate'].to_numpy()
        cond = g['Condition'].iloc[0]
        rep  = g['Replicate'].iloc[0]

        # Precompute for lag==1 from provided Distance to match your output exactly
        dist_col = g['Distance'].to_numpy() if 'Distance' in g else None

        # For each lag, vectorized slice diffs; no inner per-step Python loop
        for lag in range(1, n):
            # track contributes at this lag
            contrib_global[lag] += 1

            # if lag == 1 and dist_col is not None:
            #     msd_track = dist_col.mean()
            # else:
            dx = x[:-lag] - x[lag:]
            dy = y[:-lag] - y[lag:]
            # mean of Euclidean displacement, matching your code
            msd_track = np.hypot(dx, dy).mean()

            values_by_key[(cond, rep, lag)].append(msd_track)

    # Build result once
    out_rows = []
    for (cond, rep, lag), vals in values_by_key.items():
        arr = np.asarray(vals, dtype=float)
        mean = arr.mean()
        sd = arr.std(ddof=0)
        sem = stats.sem(arr) if arr.size > 1 else np.nan
        median = np.median(arr)
        out_rows.append({
            'Condition': cond,
            'Replicate': rep,
            'Tracks contributing': contrib_global.get(lag, 0),  # global count per lag to mirror original
            'Time lag': lag * t_step,
            'MSD mean': mean,
            'MSD sem': sem,
            'MSD sd': sd,
            'MSD median': median,
            'Frame lag': lag,
        })

    if not out_rows:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(out_rows)
    # Optional stable sort
    out.sort_values(['Condition', 'Replicate', 'Frame lag'], inplace=True, ignore_index=True)
    return out
