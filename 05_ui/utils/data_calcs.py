import numpy as np
import pandas as pd
from math import floor, ceil



def dir_round(value, digits=3, direction='down'):
    if direction == 'up':
        return ceil(value * 10**digits) / 10**digits
    elif direction == 'down':
        return floor(value * 10**digits) / 10**digits
    elif direction == None:
        return round(value, digits)


def percentile_thresholding(df, column_name: str, values: tuple):
    try:
        lower_percentile, upper_percentile = values
        lower_threshold = np.percentile(df[column_name], lower_percentile)
        upper_threshold = np.percentile(df[column_name], upper_percentile)
        return df[(df[column_name] >= lower_threshold) & (df[column_name] <= upper_threshold)]
    except ValueError:
        return df

def literal_thresholding(df, column_name: str, values: tuple):
    lower_threshold, upper_threshold = values
    return df[(df[column_name] >= lower_threshold) & (df[column_name] <= upper_threshold)]

def dataframe_filter(df, df_filter):
    return df[df["TRACK_ID"].isin(df_filter["TRACK_ID"])]

def values_for_a_metric(df, metric):
    df.dropna()
    min_value = floor(df[metric].min())
    max_value = ceil(df[metric].max())
    return min_value, max_value


def try_convert_numeric(x):
    try:
        # Only process strings
        if isinstance(x, str):
            x_stripped = x.strip()
            num = float(x_stripped)
            if num.is_integer():
                return int(num)
            else:
                return num
        else:
            return x
    except ValueError:
        return x

def merge_dfs(dataframes, on):

    # Initialize the first DataFrame as the base for merging
    merged_df = dataframes[0].map(str)

    # Use a for loop to merge each subsequent DataFrame
    for df in dataframes[1:]:

        df = df.reset_index(drop=True)
        df = df.map(str)
        merge_columns = [col for col in df.columns if col not in merged_df.columns or col in on]
        merged_df = pd.merge(
            merged_df,
            df[merge_columns],  # Select only necessary columns from df
            on=on,
            how='outer'
        )
    
    merged_df = merged_df.map(try_convert_numeric)
    return merged_df

def butter(df):                                                                                      # Smoothing the raw dataframe

    float_columns = [ # Definition of unneccesary float columns in the df which are to be convertet to integers
    'ID', 
    'TRACK_ID', 
    'POSITION_T', 
    'FRAME'
    ]

    df = pd.DataFrame(df)  
    df = df.apply(pd.to_numeric, errors='coerce').dropna(subset=['POSITION_X', 'POSITION_Y', 'POSITION_T'])                                 # Gets rid of the multiple index rows by converting the values to a numeric type and then dropping the NaN values


    # For some reason, the y coordinates extracted from trackmate are mirrored. That ofcourse would not affect the statistical tests, only the data visualization. However, to not get mindfucked..
    # Reflect y-coordinates around the midpoint for the directionality to be accurate, according to the microscope videos.
    y_mid = (df['POSITION_Y'].min() + df['POSITION_Y'].max()) / 2
    df['POSITION_Y'] = 2 * y_mid - df['POSITION_Y']

    columns_list = df.columns.tolist()
    columns_list.remove('LABEL')

    df = df[columns_list]

    # Here we convert the unnecessary floats (from the list in which we defined them) to integers
    df[float_columns] = df[float_columns].astype(int)

    return df





def Spots(df: pd.DataFrame) -> pd.DataFrame:

    """
    Compute per-frame tracking metrics for each cell track in the DataFrame:
      - DISTANCE: Euclidean distance between consecutive positions
      - DIRECTION_RAD: direction of travel in radians
      - TRACK_LENGTH: cumulative distance along the track
      - NET_DISTANCE: straight-line distance from track start
      - CONFINEMENT_RATIO: NET_DISTANCE / TRACK_LENGTH

    Expects columns: CONDITION, REPLICATE, TRACK_ID, POSITION_X, POSITION_Y, POSITION_T
    Returns a DataFrame sorted by CONDITION, REPLICATE, TRACK_ID, POSITION_T with new metric columns.
    """
    if df.empty:
        return df.copy()
    
    df.sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T'], inplace=True)

    # Sort and work on a copy
    # df = df.sort_values(['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T']).copy()
    grp = df.groupby(['CONDITION', 'REPLICATE', 'TRACK_ID'], sort=False)

    # Distance between current and next position
    df['DISTANCE'] = np.sqrt(
        (grp['POSITION_X'].shift(-1) - df['POSITION_X'])**2 +
        (grp['POSITION_Y'].shift(-1) - df['POSITION_Y'])**2
        ).fillna(0)

    # Direction of travel (radians) based on diff to previous point
    df['DIRECTION_RAD'] = np.arctan2(
        grp['POSITION_Y'].diff(),
        grp['POSITION_X'].diff()
        ).fillna(0)

    # Cumulative track length
    df['TRACK_LENGTH'] = grp['DISTANCE'].cumsum()

    # Net (straight-line) distance from the start of the track
    start = grp[['POSITION_X', 'POSITION_Y']].transform('first')
    df['NET_DISTANCE'] = np.sqrt(
        (df['POSITION_X'] - start['POSITION_X'])**2 +
        (df['POSITION_Y'] - start['POSITION_Y'])**2
        )

    # Confinement ratio: net distance vs. actual path length
    # Avoid division by zero by replacing zeros with NaN, then fill
    df['CONFINEMENT_RATIO'] = (df['NET_DISTANCE'] / df['TRACK_LENGTH'].replace(0, np.nan)).fillna(0)

    return df

def Tracks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive track-level metrics for each cell track in the DataFrame, including:
      - TRACK_LENGTH: sum of DISTANCE
      - NET_DISTANCE: straight-line from first to last position
      - CONFINEMENT_RATIO: NET_DISTANCE / TRACK_LENGTH
      - MIN_SPEED, MAX_SPEED, MEAN_SPEED, STD_SPEED, MEDIAN_SPEED (per-track on DISTANCE)
      - MEAN_DIRECTION_RAD/DEG, STD_DEVIATION_RAD/DEG, MEDIAN_DIRECTION_RAD/DEG (circular stats)

    Expects columns: CONDITION, REPLICATE, TRACK_ID, DISTANCE, POSITION_X, POSITION_Y, DIRECTION_RAD
    Returns a single DataFrame indexed by CONDITION, REPLICATE, TRACK_ID with all metrics.
    """
    if df.empty:
        cols = [
            'CONDITION','REPLICATE','TRACK_ID',
            'TRACK_LENGTH','NET_DISTANCE','CONFINEMENT_RATIO',
            'MIN_SPEED','MAX_SPEED','MEAN_SPEED','STD_SPEED','MEDIAN_SPEED',
            'MEAN_DIRECTION_RAD','STD_DEVIATION_RAD','MEDIAN_DIRECTION_RAD',
            'MEAN_DIRECTION_DEG','STD_DEVIATION_DEG','MEDIAN_DIRECTION_DEG'
        ]
        return pd.DataFrame(columns=cols)

    # Group by track
    grp = df.groupby(['CONDITION','REPLICATE','TRACK_ID'], sort=False)

    # Aggregate distance and speed metrics, and capture start/end positions
    agg = grp.agg(
        TRACK_LENGTH=('DISTANCE','sum'),
        MIN_SPEED=('DISTANCE','min'),
        MAX_SPEED=('DISTANCE','max'),
        MEAN_SPEED=('DISTANCE','mean'),
        STD_SPEED=('DISTANCE','std'),
        MEDIAN_SPEED=('DISTANCE','median'),
        start_x=('POSITION_X','first'), end_x=('POSITION_X','last'),
        start_y=('POSITION_Y','first'), end_y=('POSITION_Y','last')
    )

    # Compute net displacement and confinement ratio
    agg['NET_DISTANCE'] = np.hypot(agg['end_x'] - agg['start_x'], agg['end_y'] - agg['start_y'])
    agg['CONFINEMENT_RATIO'] = (agg['NET_DISTANCE'] / agg['TRACK_LENGTH'].replace(0, np.nan)).fillna(0)
    agg = agg.drop(columns=['start_x','end_x','start_y','end_y'])

    # Circular direction statistics: need sin & cos per observation
    sin_cos = df.assign(_sin=np.sin(df['DIRECTION_RAD']), _cos=np.cos(df['DIRECTION_RAD']))
    dir_agg = sin_cos.groupby(['CONDITION','REPLICATE','TRACK_ID'], sort=False).agg(
        mean_sin=('_sin','mean'), mean_cos=('_cos','mean'),
        median_sin=('_sin','median'), median_cos=('_cos','median')
    )
    # derive circular metrics
    dir_agg['MEAN_DIRECTION_RAD'] = np.arctan2(dir_agg['mean_sin'], dir_agg['mean_cos'])
    dir_agg['STD_DEVIATION_RAD'] = np.hypot(dir_agg['mean_sin'], dir_agg['mean_cos'])
    dir_agg['MEDIAN_DIRECTION_RAD'] = np.arctan2(dir_agg['median_sin'], dir_agg['median_cos'])
    dir_agg['MEAN_DIRECTION_DEG'] = np.degrees(dir_agg['MEAN_DIRECTION_RAD']) % 360
    dir_agg['STD_DEVIATION_DEG'] = np.degrees(dir_agg['STD_DEVIATION_RAD']) % 360
    dir_agg['MEDIAN_DIRECTION_DEG'] = np.degrees(dir_agg['MEDIAN_DIRECTION_RAD']) % 360
    dir_agg = dir_agg.drop(columns=['mean_sin','mean_cos','median_sin','median_cos'])

        # Count points per track
    # number of rows (frames) per track
    point_counts = grp.size().rename('TRACK_POINTS')

    # Merge all metrics into one DataFrame
    result = agg.merge(dir_agg, left_index=True, right_index=True)
    # Merge point counts
    result = result.merge(point_counts, left_index=True, right_index=True)
    result = result.reset_index()
    return result

def Time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-frame (time point) summary metrics grouped by CONDITION, REPLICATE, POSITION_T:
      - TRACK_LENGTH, NET_DISTANCE, CONFINEMENT_RATIO distributions: MIN, MAX, MEAN, STD, MEDIAN
      - SPEED (DISTANCE) distributions as MIN_SPEED, MAX_SPEED, MEAN_SPEED, STD_SPEED, MEDIAN_SPEED
      - DIRECTION_RAD distributions (circular): MEAN_DIRECTION_RAD, STD_DIRECTION_RAD, MEDIAN_DIRECTION_RAD
        and corresponding degrees

    Expects columns: CONDITION, REPLICATE, POSITION_T, TRACK_LENGTH, NET_DISTANCE,
                     CONFINEMENT_RATIO, DISTANCE, DIRECTION_RAD
    Returns a DataFrame indexed by CONDITION, REPLICATE, POSITION_T with all time-point metrics.
    """
    if df.empty:
        # define columns
        cols = ['CONDITION','REPLICATE','POSITION_T'] + \
               [f'{stat}_{metric}' for metric in ['TRACK_LENGTH','NET_DISTANCE','CONFINEMENT_RATIO'] for stat in ['MIN','MAX','MEAN','STD','MEDIAN']] + \
               [f'{stat}_SPEED' for stat in ['MIN','MAX','MEAN','STD','MEDIAN']] + \
               ['MEAN_DIRECTION_RAD','STD_DIRECTION_RAD','MEDIAN_DIRECTION_RAD',
                'MEAN_DIRECTION_DEG','STD_DIRECTION_DEG','MEDIAN_DIRECTION_DEG']
        return pd.DataFrame(columns=cols)

    group_cols = ['CONDITION','REPLICATE','POSITION_T']

    # 1) stats on track metrics per frame
    metrics = ['TRACK_LENGTH','NET_DISTANCE','CONFINEMENT_RATIO']
    agg_funcs = ['min','max','mean','std','median']
    # build agg dict
    agg_dict = {m: agg_funcs for m in metrics}
    frame_agg = df.groupby(group_cols).agg(agg_dict)
    # flatten columns
    frame_agg.columns = [f'{stat.upper()}_{metric}' for metric, stat in frame_agg.columns]

    # 2) speed stats (DISTANCE distributions)
    speed_agg = df.groupby(group_cols)['DISTANCE'].agg(['min','max','mean','std','median'])
    speed_agg.columns = [f'{stat.upper()}_SPEED' for stat in speed_agg.columns]

    # 3) circular direction stats per frame
    # compute sin/cos columns
    tmp = df.assign(_sin=np.sin(df['DIRECTION_RAD']), _cos=np.cos(df['DIRECTION_RAD']))
    dir_frame = tmp.groupby(group_cols).agg({'_sin':'mean','_cos':'mean','DIRECTION_RAD':'count'})
    # mean direction
    dir_frame['MEAN_DIRECTION_RAD'] = np.arctan2(dir_frame['_sin'], dir_frame['_cos'])
    # circular std: R = sqrt(mean_sin^2+mean_cos^2)
    dir_frame['STD_DIRECTION_RAD'] = np.hypot(dir_frame['_sin'], dir_frame['_cos'])
    # median direction: use groupby apply median sin/cos
    median = tmp.groupby(group_cols).agg({'_sin':'median','_cos':'median'})
    dir_frame['MEDIAN_DIRECTION_RAD'] = np.arctan2(median['_sin'], median['_cos'])
    # degrees
    dir_frame['MEAN_DIRECTION_DEG'] = np.degrees(dir_frame['MEAN_DIRECTION_RAD']) % 360
    dir_frame['STD_DIRECTION_DEG'] = np.degrees(dir_frame['STD_DIRECTION_RAD']) % 360
    dir_frame['MEDIAN_DIRECTION_DEG'] = np.degrees(dir_frame['MEDIAN_DIRECTION_RAD']) % 360
    dir_frame = dir_frame.drop(columns=['_sin','_cos','DIRECTION_RAD'], errors='ignore')

    # merge all
    time_stats = frame_agg.merge(speed_agg, left_index=True, right_index=True)
    time_stats = time_stats.merge(dir_frame, left_index=True, right_index=True)
    time_stats = time_stats.reset_index()

    return time_stats


def get_cond_repl(df):
    # Get unique conditions from the DataFrame
    dictionary = {'all': ['all']}
    

    conditions = df['CONDITION'].unique()
    for condition in conditions:
        # Get unique replicates for each condition
        replicates_list = ['all']
        replicates = df[df['CONDITION'] == condition]['REPLICATE'].unique()
        for replicate in replicates.tolist():
            replicate = str(replicate)
            replicates_list.append(replicate)
            
        dictionary.update({str(condition): replicates_list})
    return dictionary