import numpy as np
import pandas as pd

@staticmethod
def Spots(df: pd.DataFrame) -> pd.DataFrame:

    """
    Compute per-frame tracking metrics for each cell track in the DataFrame:
    - Distance: Euclidean distance between consecutive positions
    - Direction (rad): direction of travel in radians
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
    df['Distance'] = np.sqrt(
        (grp['X coordinate'].shift(-1) - df['X coordinate'])**2 +
        (grp['Y coordinate'].shift(-1) - df['Y coordinate'])**2
        ).fillna(0)

    # Direction of travel (radians) based on diff to previous point
    df['Direction (rad)'] = np.arctan2(
        grp['Y coordinate'].diff(),
        grp['X coordinate'].diff()
        ).fillna(0)

    # Cumulative track length
    df['Cumulative track length'] = grp['Distance'].cumsum()

    # Net (straight-line) distance from the start of the track
    start = grp[['X coordinate', 'Y coordinate']].transform('first')
    df['Cumulative track displacement'] = np.sqrt(
        (df['X coordinate'] - start['X coordinate'])**2 +
        (df['Y coordinate'] - start['Y coordinate'])**2
        )

    # Confinement ratio: Track displacement vs. actual path length
    # Avoid division by zero by replacing zeros with NaN, then fill
    df['Cumulative confinement ratio'] = (df['Cumulative track displacement'] / df['Cumulative track length'].replace(0, np.nan)).fillna(0)

    df['Frame'] = grp['Time point'].rank(method='dense').astype(int)

    return df


@staticmethod
def Tracks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive track-level metrics for each cell track in the DataFrame, including:
    - Track length: sum of Distance
    - Track displacement: straight-line from first to last position
    - Confinement ratio: Track displacement / Track length
    - Min speed, Max speed, Mean speed, Std speed, Median speed (per-track on Distance)
    - Mean direction (rad/deg), Std deviation (rad/deg), Median direction (rad/deg) (circular stats)

    Expects columns: Condition, Replicate, Track ID, Distance, X coordinate, Y coordinate, Direction (rad)
    Returns a single DataFrame indexed by Condition, Replicate, Track ID with all metrics.
    """
    if df.empty:
        cols = [
            'Condition','Replicate','Track ID',
            'Track length','Track displacement','Confinement ratio',
            'Speed min','Speed max','Speed mean','Speed std','Speed median',
            'Direction mean (rad)','Direction std (rad)','Direction median (rad)',
            'Direction mean (deg)','Direction std (deg)','Direction median (deg)',
        ]
        return pd.DataFrame(columns=cols)

    # Group by track
    grp = df.groupby(['Condition','Replicate','Track ID'], sort=False)

    agg = grp.agg(
        **{
            'Track length': ('Distance', 'sum'),
            'Speed mean':  ('Distance', 'mean'),
            'Speed median':('Distance', 'median'),
            'Speed min':   ('Distance', 'min'),
            'Speed max':   ('Distance', 'max'),
            'Speed std':   ('Distance', 'std'),
            'start_x':     ('X coordinate', 'first'),
            'end_x':       ('X coordinate', 'last'),
            'start_y':     ('Y coordinate', 'first'),
            'end_y':       ('Y coordinate', 'last')
        }
    )

    if 'Replicate color' in df.columns:
        colors = grp['Replicate color'].first()
        agg = agg.merge(colors, left_index=True, right_index=True)

    # Compute net displacement and confinement ratio
    agg['Track displacement'] = np.hypot(agg['end_x'] - agg['start_x'], agg['end_y'] - agg['start_y'])
    agg['Confinement ratio'] = (agg['Track displacement'] / agg['Track length'].replace(0, np.nan)).fillna(0)
    agg = agg.drop(columns=['start_x','end_x','start_y','end_y'])

    # Circular direction statistics: need sin & cos per observation
    sin_cos = df.assign(_sin=np.sin(df['Direction (rad)']), _cos=np.cos(df['Direction (rad)']))
    dir_agg = sin_cos.groupby(['Condition','Replicate','Track ID'], sort=False).agg(
        mean_sin=('_sin','mean'), mean_cos=('_cos','mean'),
        median_sin=('_sin','median'), median_cos=('_cos','median')
    )
    # derive circular metrics
    dir_agg['Direction mean (rad)'] = np.arctan2(dir_agg['mean_sin'], dir_agg['mean_cos'])
    dir_agg['Direction std (rad)'] = np.hypot(dir_agg['mean_sin'], dir_agg['mean_cos'])
    dir_agg['Direction median (rad)'] = np.arctan2(dir_agg['median_sin'], dir_agg['median_cos'])
    dir_agg['Direction mean (deg)'] = np.degrees(dir_agg['Direction mean (rad)']) % 360
    dir_agg['Direction std (deg)'] = np.degrees(dir_agg['Direction std (rad)']) % 360
    dir_agg['Direction median (deg)'] = np.degrees(dir_agg['Direction median (rad)']) % 360
    dir_agg = dir_agg.drop(columns=['mean_sin','mean_cos','median_sin','median_cos'])

        # Count points per track
    # number of rows (frames) per track
    point_counts = grp.size().rename('Track points')

    # Merge all metrics into one DataFrame
    result = agg.merge(dir_agg, left_index=True, right_index=True)
    # Merge point counts
    result = result.merge(point_counts, left_index=True, right_index=True).reset_index()
    result['Track UID'] = np.arange(len(result))  # starts at 0
    result.set_index('Track UID', drop=True, inplace=True, verify_integrity=True)

    return result


@staticmethod
def Frames(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-frame (time point) summary metrics grouped by Condition, Replicate, Time point:
    - Track length, Track displacement, Confinement ratio distributions: min, max, mean, std, median
    - Speed (Distance) distributions as Speed min, Speed max, Speed mean, Speed std, Speed median
    - Direction (rad) distributions (circular): Direction mean (rad), Direction std (rad), Direction median (rad)
        and corresponding degrees

    Expects columns: Condition, Replicate, Time point, Track length, Track displacement,
                    Confinement ratio, Distance, Direction (rad)
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
    tmp = df.assign(_sin=np.sin(df['Direction (rad)']), _cos=np.cos(df['Direction (rad)']))
    dir_frame = tmp.groupby(group_cols).agg({'_sin':'mean','_cos':'mean','Direction (rad)':'count'})
    # mean direction
    dir_frame['Direction mean (rad)'] = np.arctan2(dir_frame['_sin'], dir_frame['_cos'])
    # circular std: R = sqrt(mean_sin^2+mean_cos^2)
    dir_frame['Direction std (rad)'] = np.hypot(dir_frame['_sin'], dir_frame['_cos'])
    # median direction: use groupby apply median sin/cos
    median = tmp.groupby(group_cols).agg({'_sin':'median','_cos':'median'})
    dir_frame['Direction median (rad)'] = np.arctan2(median['_sin'], median['_cos'])
    # degrees
    dir_frame['Direction mean (deg)'] = np.degrees(dir_frame['Direction mean (rad)']) % 360
    dir_frame['Direction std (deg)'] = np.degrees(dir_frame['Direction std (rad)']) % 360
    dir_frame['Direction median (deg)'] = np.degrees(dir_frame['Direction median (rad)']) % 360
    dir_frame = dir_frame.drop(columns=['_sin','_cos','Direction (rad)'], errors='ignore')

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