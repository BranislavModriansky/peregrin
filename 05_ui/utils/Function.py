import numpy as np
import pandas as pd
from math import floor, ceil
import os.path as op
from typing import List, Any
from pandas.api.types import is_object_dtype
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import seaborn as sns
from itertools import chain
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def _get_cmap(c_mode):
    """
    Get a colormap according to the selected color mode.

    """

    if c_mode == 'greyscale LUT':
        return plt.cm.gist_yarg
    elif c_mode == 'jet LUT':
        return plt.cm.jet
    elif c_mode == 'brg LUT':
        return plt.cm.brg
    elif c_mode == 'hot LUT':
        return plt.cm.hot
    elif c_mode == 'gnuplot LUT':
        return plt.cm.gnuplot
    elif c_mode == 'viridis LUT':
        return plt.cm.viridis
    elif c_mode == 'rainbow LUT':
        return plt.cm.rainbow
    elif c_mode == 'turbo LUT':
        return plt.cm.turbo
    elif c_mode == 'nipy-spectral LUT':
        return plt.cm.nipy_spectral
    elif c_mode == 'gist-ncar LUT':
        return plt.cm.gist_ncar
    else:
        return None


def _pick_encoding(path, encodings=("utf-8", "cp1252", "latin1", "iso8859_15")):
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue

def _has_strings(s: pd.Series) -> bool:
    # pandas "string" dtype (pyarrow/python)
    if isinstance(s.dtype, pd.StringDtype):
        return s.notna().any()
    # categorical of strings?
    if isinstance(s.dtype, pd.CategoricalDtype):
        return isinstance(s.dtype.categories.dtype, pd.StringDtype) and s.notna().any()
    # numeric, datetime, bool, etc.
    if not is_object_dtype(s.dtype):
        return False
    # Fallback for object-dtype (mixed types): minimal Python loop over NumPy array
    arr = s.to_numpy(dtype=object, copy=False)
    return any(isinstance(v, (str, np.str_)) for v in arr)

def _build_replicate_palette(df, palette_fallback):
    reps = df['Replicate'].unique().tolist()
    mp = {}
    if 'Replicate color' in df.columns:
        mp = (df[['Replicate', 'Replicate color']]
                .dropna()
                .drop_duplicates('Replicate')
        )
        mp = mp.set_index('Replicate')['Replicate color'].to_dict()

    missing = [r for r in reps if r not in mp]
    if missing:
        cyc = sns.color_palette(palette_fallback, n_colors=len(missing))
        mp.update({r: cyc[i] for i, r in enumerate(missing)})

    print(f"Replicate colors: {mp}")

    return mp

def SetUnits(t: str) -> dict:
    return {
        "Track length": "(µm)",
        "Track displacement": "(µm)",
        "Confinement ratio": "",
        "Track points": "",
        "Speed mean": f"(µm·{t}⁻¹)",
        "Speed median": f"(µm·{t}⁻¹)",
        "Speed max": f"(µm·{t}⁻¹)",
        "Speed min": f"(µm·{t}⁻¹)",
        "Speed std": f"(µm·{t}⁻¹)",
        "Direction mean (deg)": "",
        "Direction mean (rad)": "",
        "Direction std (deg)": "",
        "Direction std (rad)": "",
    }


def _rand_hex_colors(n, rng):
    return [mcolors.to_hex(rng.random(3)) for _ in range(n)]

def _rand_hex_greys(n, rng):
    greys = rng.random(n)
    return [mcolors.to_hex((float(g), float(g), float(g))) for g in greys]



class DataLoader:
    
    @staticmethod
    def GetDataFrame(filepath: str) -> pd.DataFrame:
        """
        Loads a DataFrame from a file based on its extension.
        Supported formats: CSV, Excel, Feather, Parquet, HDF5, JSON.
        """
        _, ext = op.splitext(filepath.lower())

        try:
            if ext == '.csv':
                return _pick_encoding(filepath)
            elif ext in ['.xls', '.xlsx']:
                return pd.read_excel(filepath)
            elif ext == '.feather':
                return pd.read_feather(filepath)
            elif ext == '.parquet':
                return pd.read_parquet(filepath)
            elif ext in ['.h5', '.hdf5']:
                return pd.read_hdf(filepath)
            elif ext == '.json':
                return pd.read_json(filepath)
        except ValueError as e:
            raise e(f"{ext} is not a supported file format.")
        except Exception as e:
            raise e(f"Failed to load file '{filepath}': {e}")
    

    @staticmethod
    def Extract(df: pd.DataFrame, id_col: str, t_col: str, x_col: str, y_col: str, mirror_y: bool = True) -> pd.DataFrame:
        # Keep only relevant columns and convert to numeric
        df = df[[id_col, t_col, x_col, y_col]].apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

        # Mirror Y if needed
        if mirror_y:
            """
            TrackMate may export y-coordinates mirrored.
            This does not affect statistics but leads to incorrect visualization.
            Here Y data is reflected across its midpoint for accurate directionality.
            """
            y_mid = (df[y_col].min() + df[y_col].max()) / 2
            df[y_col] = 2 * y_mid - df[y_col]

        # Standardize column names
        return df.rename(columns={id_col: 'Track ID', t_col: 'Time point', x_col: 'X coordinate', y_col: 'Y coordinate'})


    @staticmethod
    def GetColumns(path: str) -> List[str]:
        """
        Returns a list of column names from the DataFrame.
        """
        df = DataLoader.GetDataFrame(path)  # or pd.read_excel(path), depending on file type
        return df.columns.tolist()
    
    @staticmethod
    def FindMatchingColumn(columns: List[str], lookfor: List[str]) -> str:
        """
        Looks for matches with any of the provided strings.
        - First tries exact matches.
        - Then checks if the column starts with any of given terms.
        - Finally checks if any term is a substring of the column name.
        If no match is found, returns None.
        """

        # Normalize columns for matching
        normalized_columns = [
            (col, str(col).replace('_', ' ').strip().lower() if col is not None else '') for col in columns
        ]
        # Try exact matches first
        for col, norm_col in normalized_columns:
            for look in lookfor:
                if norm_col == look.lower():
                    return col
        # Then try startswith
        for col, norm_col in normalized_columns:
            for look in lookfor:
                if norm_col.startswith(look.lower()):
                    return col
        # Then try substring
        for col, norm_col in normalized_columns:
            for look in lookfor:
                if look.lower() in norm_col:
                    return col
        return None
        



class Process:

    @staticmethod
    def TryConvertNumeric(x: Any) -> Any:
        """
        Try to convert a string to an int or float, otherwise return the original value.
        """
        try:
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
        
    @staticmethod
    def TryFloat(x: Any) -> Any:
        """
        Try to convert a string to an int or float, otherwise return the original value.
        """
        try:
            if isinstance(x, str):
                x_stripped = x.strip()
                num = float(x_stripped)
                if num.is_integer():
                    return float(num)
                else:
                    return num
            else:
                return x
        except ValueError:
            return x

    @staticmethod
    def MergeDFs(dataframes: List[pd.DataFrame], on: List[str]) -> pd.DataFrame:
        """
        Merges a list of DataFrames on the specified columns using an outer join.
        All values are coerced to string before merging, then converted back to numeric where possible.

        Parameters:
            dataframes: List of DataFrames to merge.
            on: List of column names to merge on.

        Returns:
            pd.DataFrame: The merged DataFrame with numerics restored where possible.
        """
        if not dataframes:
            raise ValueError("No dataframes provided for merging.")

        # Initialize the first DataFrame as the base for merging (all values as string)
        merged_df = dataframes[0].map(str)

        for df in dataframes[1:]:
            df = df.map(str)
            # Ensure all key columns are present for merging
            merge_columns = [col for col in df.columns if col not in merged_df.columns or col in on]
            merged_df = pd.merge(
                merged_df,
                df[merge_columns],
                on=on,
                how='outer'
            )

        # Use the static method for numeric conversion
        merged_df = merged_df.applymap(Process.TryConvertNumeric)
        return merged_df


    def Round(value, step, round_method="nearest"):
        """
        Rounds value to the nearest multiple of step.
        """
        if round_method == "nearest":
            return round(value)
        elif round_method == "floor":
            return floor(value)
        elif round_method == "ceil":
            return ceil(value)
        else:
            raise ValueError(f"Unknown round method: {round_method}")



class Calc:
    
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
            cols = ['Condition','Replicate','Time point'] + \
                [f'{metric} {stat}' for metric in ['Track length','Track displacement','Confinement ratio'] for stat in ['min','max','mean','std','median']] + \
                [f'Speed {stat}' for stat in ['min','max','mean','std','median']] + \
                ['Direction mean (rad)','Direction std (rad)','Direction median (rad)',
                    'Direction mean (deg)','Direction std (deg)','Direction median (deg)']
            return pd.DataFrame(columns=cols)

        group_cols = ['Condition','Replicate','Time point']

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



class Threshold:

    @staticmethod
    def Normalize_01(df, col) -> pd.Series:
        """
        Normalize a column to the [0, 1] range.
        """
        # s = pd.to_numeric(df[col], errors='coerce')
        try:
            s = pd.Series(Process.TryFloat(df[col]), dtype=float)
            if _has_strings(s):
                normalized = pd.Series(0.0, index=s.index, name=col)
            lo, hi = s.min(), s.max()
            if lo == hi:
                normalized = pd.Series(0.0, index=s.index, name=col)
            else:
                normalized = pd.Series((s - lo) / (hi - lo), index=s.index, name=col)

        except Exception:
            normalized = pd.Series(0.0, index=df.index, name=col)

        return normalized  # <-- keeps index

    @staticmethod
    def JoinByIndex(a: pd.Series, b: pd.Series) -> pd.DataFrame:
        """
        Join two Series of potentially different lengths into a DataFrame.
        """

        if b.index.is_unique and not a.index.is_unique:
            df = a.rename(a.name).to_frame().set_index(a.index)
            df[b.name] = b.reindex(df.index)
        else:
            df = b.rename(b.name).to_frame().set_index(b.index)
            df[a.name] = a.reindex(df.index)

        return df

    @staticmethod
    def GetInfo(total_tracks: int, filtered_tracks: int, width: int = 160, height: int = 180, txt_color: str = "#000000") -> str:
        """
        Generate an SVG info panel summarizing filter info.

        Parameters:
            total_tracks: Total number of tracks before filtering.
            filtered_tracks: Number of tracks after filtering.
            width: Width of the SVG panel in pixels.
            height: Height of the SVG panel in pixels.

        Returns:
            str: SVG file.
        """
        
        if total_tracks < 0:
            return ''
        if filtered_tracks < 0:
            filtered_tracks = total_tracks

        percent = 0 if total_tracks == 0 else round((filtered_tracks / total_tracks) * 100)
        
        # Layout metrics
        pad = 16
        title_size = 18
        body_size = 14
        line_gap = 8
        section_gap = 14
        
        # y cursor helper
        y = pad + title_size  # baseline for title
        
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Info panel">
            
            <!-- Title -->
            <text x="{pad}" y="{y}" font-family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif" font-size="{title_size}" font-weight="700" fill="{txt_color}">Info</text>
            
            <!-- Body -->
            <g font-family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif" font-size="{body_size}" fill="{txt_color}">
                <!-- Cells in total -->
                <text x="{pad}" y="{y + section_gap + body_size}" fill="{txt_color}">Cells in total:</text>
                <text x="{pad}" y="{y + section_gap + body_size*2 + line_gap}" font-weight="700">{total_tracks}</text>
                
                <!-- In focus -->
                <text x="{pad}" y="{y + section_gap*2 + body_size*3 + line_gap}" fill="{txt_color}">In focus:</text>
                <text x="{pad}" y="{y + section_gap*2 + body_size*4 + line_gap*2}" font-weight="700">{filtered_tracks} ({percent}%)</text>
            </g>
            </svg>
        '''
        
        return svg

    from math import floor, ceil

    @staticmethod
    def GetInfoSVG(
        *,
        total_tracks: int,
        filtered_tracks: int,
        threshold_list: list[int],
        threshold_dimension: str,               # "1D" or "2D"
        thresholds_state: dict,                 # dict of threshold index -> state with "tracks"/"spots"
        props: dict = None,                     # for 1D: {t: property_name}
        ftypes: dict = None,                    # for 1D: {t: filter_type}
        refs: dict = None,                      # for 1D: {t: reference_label}
        ref_vals: dict = None,                  # for 1D: {t: reference_value}
        values: dict = None,                    # for 1D: {t: (min,max)}
        propsX: dict = None,                    # for 2D: {t: propX}
        propsY: dict = None,                    # for 2D: {t: propY}
        txt_color: str = "#000000",
        width: int = 180,
        font_family: str = "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif",
    ) -> str:
        """
        Generate an SVG info panel similar to the Shiny UI.

        Arguments:
        - total_tracks: int
        - filtered_tracks: int
        - threshold_list: list of ints (active thresholds)
        - threshold_dimension: "1D" or "2D"
        - thresholds_state: dict of t -> state {"tracks": ..., "spots": ...}
        - props, ftypes, refs, ref_vals, values: dicts keyed by threshold index (for 1D mode)
        - propsX, propsY: dicts keyed by threshold index (for 2D mode)
        """
        if total_tracks < 0:
            return ""
        if filtered_tracks < 0:
            filtered_tracks = total_tracks

        in_focus_pct = 0 if total_tracks == 0 else round(filtered_tracks / total_tracks * 100)

        pad = 16
        title_size = 20
        h2_size = 16
        body_size = 14
        hr_thickness = 1
        gap_line = 6
        gap_section = 14
        gap_rule = 14

        def lh(size): return size + 4

        y = pad + lh(title_size) - 4
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="1000" viewBox="0 0 {width} 1000">',
            f'<text x="{pad}" y="{y}" font-family="{font_family}" font-size="{title_size}" font-weight="700" fill="{txt_color}">Info</text>'
        ]

        # Cells in total
        y += gap_section + lh(body_size)
        svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" fill="{txt_color}" font-family="{font_family}">Cells in total:</text>')
        y += lh(body_size)
        svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-weight="700" fill="{txt_color}" font-family="{font_family}">{total_tracks}</text>')

        # In focus
        y += gap_section + lh(body_size)
        svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" fill="{txt_color}" font-family="{font_family}">In focus:</text>')
        y += lh(body_size)
        svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-weight="700" fill="{txt_color}" font-family="{font_family}">{filtered_tracks} ({in_focus_pct}%)</text>')

        # Threshold blocks
        for t in sorted(thresholds_state.keys()):
            if t not in threshold_list:
                continue

            t_state = thresholds_state.get(t)
            t_state_after = thresholds_state.get(t + 1)
            data = len(t_state.get("tracks", []))
            data_after = len(t_state_after.get("tracks", [])) if t_state_after else data
            out = data - data_after
            out_percent = round(out / data * 100) if data else 0

            # Divider
            y += gap_rule
            svg.append(f'<line x1="{pad}" x2="{width-pad}" y1="{y}" y2="{y}" stroke="{txt_color}" stroke-opacity="0.4" stroke-width="{hr_thickness}" />')
            y += gap_rule + lh(h2_size) - 4

            svg.append(f'<text x="{pad}" y="{y}" font-size="{h2_size}" font-weight="700" font-family="{font_family}" fill="{txt_color}">Threshold {t+1}</text>')

            # Filtered out
            y += gap_section + lh(body_size)
            svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-family="{font_family}" fill="{txt_color}">Filtered out:</text>')
            y += lh(body_size)
            svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-weight="700" font-family="{font_family}" fill="{txt_color}">{out} ({out_percent}%)</text>')

            y += gap_section

            if threshold_dimension == "1D":
                prop = props[t]
                ftype = ftypes[t]
                val_min, val_max = values[t]
                ref = refs.get(t) if refs else None
                ref_val = ref_vals.get(t) if ref_vals else None

                # Property
                svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-family="{font_family}" fill="{txt_color}">Property:</text>')
                y += lh(body_size)
                svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-style="italic" font-weight="600" font-family="{font_family}" fill="{txt_color}">{prop}</text>')

                # Filter
                y += lh(body_size)
                svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-family="{font_family}" fill="{txt_color}">Filter:</text>')
                y += lh(body_size)
                svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-style="italic" font-weight="600" font-family="{font_family}" fill="{txt_color}">{ftype}</text>')

                # Range
                y += lh(body_size)
                svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-family="{font_family}" fill="{txt_color}">Range:</text>')
                y += lh(body_size)
                svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-style="italic" font-weight="600" font-family="{font_family}" fill="{txt_color}">{val_min} - {val_max}</text>')

                # Reference if available
                if ftype == "Relative to..." and ref:
                    y += lh(body_size)
                    svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-family="{font_family}" fill="{txt_color}">Reference:</text>')
                    y += lh(body_size)
                    ref_text = f"{ref} ({ref_val})" if ref_val is not None else ref
                    svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-style="italic" font-weight="600" font-family="{font_family}" fill="{txt_color}">{ref_text}</text>')

            elif threshold_dimension == "2D":
                propX = propsX[t]
                propY = propsY[t]

                try:
                    track_data = t_state_after.get("tracks")
                    spot_data = t_state_after.get("spots")
                except Exception:
                    track_data = t_state.get("tracks")
                    spot_data = t_state.get("spots")

                dataX = track_data.get(propX, []) if isinstance(track_data, dict) else []
                dataY = track_data.get(propY, []) if isinstance(track_data, dict) else []

                if propX == "Confinement ratio":
                    minX, maxX = f"{min(dataX):.2f}", f"{ceil(max(dataX)):.2f}"
                else:
                    minX, maxX = floor(min(dataX)), ceil(max(dataX))
                if propY == "Confinement ratio":
                    minY, maxY = f"{min(dataY):.2f}", f"{ceil(max(dataY)):.2f}"
                else:
                    minY, maxY = floor(min(dataY)), ceil(max(dataY))

                # Properties
                svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-family="{font_family}" fill="{txt_color}">Properties:</text>')

                # X
                y += lh(body_size)
                svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-style="italic" font-weight="600" font-family="{font_family}" fill="{txt_color}">{propX}</text>')
                y += lh(body_size)
                svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-family="{font_family}" fill="{txt_color}">Range:</text>')
                svg.append(f'<text x="{pad+60}" y="{y}" font-size="{body_size}" font-family="{font_family}" fill="{txt_color}">{minX} - {maxX}</text>')

                # Y
                y += gap_line + lh(body_size)
                svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-style="italic" font-weight="600" font-family="{font_family}" fill="{txt_color}">{propY}</text>')
                y += lh(body_size)
                svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-family="{font_family}" fill="{txt_color}">Range:</text>')
                svg.append(f'<text x="{pad+60}" y="{y}" font-size="{body_size}" font-family="{font_family}" fill="{txt_color}">{minY} - {maxY}</text>')

        svg.append("</svg>")
        return "\n".join(svg)




class Plot:

    _cell = '🦠'

    _random = ['👨🏽‍🦽‍➡️', '🌪️', '🚀', '👨‍🦯‍➡️', '🏇🏼', '🛩️', '🚁', '🚂', '✈️', '🛳️', '🪂', '🚡', '💩', '🚕', '🚓', '🛒', '🐌', '💸']

    _farm = ['🐂', '🐃', '🐄', '🐏', '🐖', '🐎', '🦃', '🫏', '🐇']

    _safari = ['🦍', '🐅', '🐆', '🦒', '🦘', '🦓', '🐂']

    _insects = ['🪰', '🦗', '🦟', '🐞', '🐜', '🐝', '🦋', '🐛']

    _birds = ['🐦', '🦜', '🦆', '🦅', '🦉', '🦩', '🦚', '🦃']

    _forest = ['🦌', '🦫', '🦦', '🦔', '🦇', '🦉', '🦅', '🦆', '🐢', '🐍', '🦎', '🐌']

    _aquarium = ['🐠', '🐟', '🐢', '🏊🏻‍♀️', '🐡', '🦈', '🐙', '🐬', '🦭', '🐋', '🪼', '🦑', '🦐']

    _scaled = {
        '0-5': '🪦',
        '5-10': '🌳',
        '10-20': '🐌',
        '20-30': '👨🏽‍🦽‍➡️',
        '30-40': '👩🏼‍🦼‍➡️',
        '40-50': '⛵',
        '50-60': '🚎',
        '60-70': '🐆',
        '70-80': '🌪️',
        '80-90': '🚀',
        '90-100': '🦸🏼'
    }

    _trains = {
        '0-20': '🚂',
        '20-40': '🚝',
        '40-60': '🚈',
        '60-80': '🚄',
        '80-100': '🚅'
        }


    def _generate_random_color():
        """
        Generate a random color in hexadecimal format.

        """

        r = np.random.randint(0, 255)   # Red LED intensity
        g = np.random.randint(0, 255)   # Green LED intensity
        b = np.random.randint(0, 255)   # Blue LED intensity

        return '#{:02x}{:02x}{:02x}'.format(r, g, b)

    def _generate_random_grey():
        """
        Generate a random grey color in hexadecimal format.

        """

        n = np.random.randint(0, 240)  # All LED intensities

        return '#{:02x}{:02x}{:02x}'.format(n, n, n)

    def _make_cmap(elements, cmap):
        """
        Generate a qualitative colormap for a given list of elements.

        """

        n = len(elements)   # Number of elements in the dictionary
        if n == 0:          # Return an empty list if there are no elements
            return []       
        
        cmap = plt.get_cmap(cmap)                                   # Get the colormap
        colors = [mcolors.to_hex(cmap(i / n)) for i in range(n)]    # Generate a color for each element

        return colors

    

    def _assign_marker(value, markers):
        """
        Qualitatively map a metric's percentile value to a symbol.

        """

        lut = []    # Initialize a list to store the ranges and corresponding symbols

        for key, val in markers.items():                # Iterate through the markers dictionary
            low, high = map(float, key.split('-'))      # Split the key into low and high values
            lut.append((low, high, val))                # Append the range and symbol to the list

        for low, high, symbol in lut:               # Return the symbol for the range that contains the given value
            if low <= value < high:                  # Check if the value falls within the range
                return symbol
        
        return list(markers.items())[-1][-1]            # Return the last symbol for thr 100th percentile (which is not included in the ranges)


    class Superplots:

        @staticmethod
        def SwarmPlot(
            df: pd.DataFrame,
            metric: str,
            *args,
            title: str = '',
            palette: str = 'tab10',

            show_swarm: bool = True,
            swarm_size: int = 2,
            swarm_outline_color: str = 'black',
            swarm_alpha: float = 0.75,

            show_violin: bool = True,
            violin_fill_color: str = 'whitesmoke',
            violin_edge_color: str = 'lightgrey',
            violin_alpha: float = 0.5,
            violin_outline_width: float = 1,

            show_mean: bool = True,
            mean_span: float = 0.16,
            mean_color: str = 'black',
            mean_ls: str = '-',
            show_median: bool = True,
            median_span: float = 0.12,
            median_color: str = 'black',
            median_ls: str = '--',
            line_width: float = 2,

            show_error_bars: bool = True,
            errorbar_capsize: int = 4,
            errorbar_color: str = 'black',
            errorbar_lw: int = 2,
            errorbar_alpha: float = 0.8,

            show_mean_balls: bool = True,
            mean_ball_size: int = 90,
            mean_ball_outline_color: str = 'black',
            mean_ball_outline_width: float = 0.75,
            mean_ball_alpha: int = 1,
            show_median_balls: bool = False,
            median_ball_size: int = 70,
            median_ball_outline_color: str = 'black',
            median_ball_outline_width: float = 0.75,
            median_ball_alpha: int = 1,

            show_kde: bool = False,
            kde_inset_width: float = 0.5,
            kde_outline: float = 1,
            kde_alpha: float = 0.5,
            kde_fill: bool = False,

            show_legend: bool = True,
            show_grid: bool = False,
            open_spine: bool = True,

            plot_width: int = 15,
            plot_height: int = 9,
        ):
            """
            **Swarmplot plotting function.**

            ## Parameters:
                **df**:
                Track DataFrame;
                **metric**:
                Column name of the desired metric;
                **palette**:
                Qualitative color palette differentiating replicates (default: 'tab10');
                **show_swarm**:
                Show individual tracks as swarm points (default: True);
                **swarm_size**:
                Size of the swarm points (default: 5); *Swarm point size is automatically adjusted if the points are overcrowded*;
                **swarm_outline_color**:
                (default: 'black');
                **swarm_alpha**:
                Swarm points transparency (default: 0.5);
                **show_violin**:
                (default: True);
                **violin_fill_color**:
                (default: 'whitesmoke');
                **violin_edge_color**:
                (default: 'lightgrey');
                **violin_alpha**:
                Violins transparency (default: 0.5);
                **violin_outline_width**:
                (default: 1);
                **show_mean**:
                Show condition mean as a line (default: True);
                **mean_span**:
                Span length of the mean line (default: 0.12);
                **mean_color**:
                (default: 'black');
                **mean_ls**:
                Condition mean line style (default: '-');
                **show_median**:
                Show condition median as a line (default: True);
                **median_span**:
                Span length of the median line (default: 0.08);
                **median_color**:
                (default: 'black');
                **median_ls**:
                Condition median line style (default: '--');
                **line_width**:
                Line width of mean and median lines (default: 1);
                **set_main_line**:
                Set whether to show mean or median as a full line, while showing the other as a dashed line (default: 'mean');
                **show_error_bars**:
                Show standard deviation error bars around the mean (default: True);
                **errorbar_capsize**:
                Span length of the errorbar caps (default: 4);
                **errorbar_color**:
                (default: 'black');
                **errorbar_lw**:
                Line width of the error bars (default: 1);
                **errorbar_alpha**:
                Transparency of the error bars (default: 0.5);
                **show_mean_balls**:
                Show replicate means (default: True);
                **mean_ball_size**:
                (default: 5);
                **mean_ball_outline_color**:
                (default: 'black');
                **mean_ball_outline_width**:
                (default: 0.75);
                **mean_ball_alpha**:
                (default: 1);
                **show_median_balls**:
                Show replicate medians (default: False);
                **median_ball_size**:
                (default: 5);
                **median_ball_outline_color**:
                (default: 'black');
                **median_ball_outline_width**:
                (default: 0.75);
                **median_ball_alpha**:
                (default: 1);
                **show_kde**:
                Show inset KDE plotted next to each condition for each replicate (default: False);
                **kde_inset_width**:
                Height of the inset KDE (default: 0.5);
                **kde_outline**:
                Line width of the KDE outline (default: 1);
                **kde_alpha**:
                Transparency of the KDE (default: 0.5);
                **kde_fill**:
                Fill the KDE plots (default: False);
                **show_legend**:
                Show legend (default: True);
                **show_grid**:
                Show grid (default: False);
                **open_spine**:
                Don't show the top and right axes spines (default: True);
            """

            plt.figure(figsize=(plot_width, plot_height))
            ax = plt.gca()

            if df is None or df.empty:
                return

            _df = df.copy()

            # === condition order & optional spacers (for KDE layout) ===
            conditions = _df['Condition'].unique().tolist()

            if show_kde: # TODO - make kde work for datasets with one cond only
                spaced_conditions = ["spacer_0"] + list(
                    chain.from_iterable(
                        (cond, f"spacer_{i+1}") if i < len(conditions) - 1 else (cond,)
                        for i, cond in enumerate(conditions)
                    )
                )
                # Categorical with unobserved categories retained (order matters for aligning x)
                _df['Condition'] = pd.Categorical(_df['Condition'],
                                                categories=spaced_conditions,
                                                ordered=True)
                categories_for_stats = spaced_conditions
            else:
                _df['Condition'] = pd.Categorical(_df['Condition'],
                                                categories=conditions,
                                                ordered=True)
                categories_for_stats = conditions

            _palette = _build_replicate_palette(_df, palette_fallback=palette)

            # === stats (single pass) ===
            # keep all categories (observed=False) so spacers appear with NaNs
            _cond_stats = (
                _df.groupby('Condition', observed=False)[metric]
                .agg(mean='mean', median='median', std='std', count='count')
                .reindex(categories_for_stats)        # align to category order
                .reset_index()
            )
            _rep_stats = (
                _df.groupby(['Condition', 'Replicate'], observed=False)[metric]
                .agg(mean='mean', median='median')
                .reset_index()
            )

            # === base layers (seaborn handles packing/jitter efficiently) ===
            if show_swarm:
                sp = sns.swarmplot(
                    data=_df,
                    x="Condition",
                    y=metric,
                    hue='Replicate',
                    palette=_palette,
                    size=swarm_size,
                    edgecolor=swarm_outline_color,
                    dodge=False,
                    alpha=swarm_alpha,
                    legend=False,
                    zorder=1,
                    ax=ax,
                )

            if show_violin:
                sns.violinplot(
                    data=_df,
                    x='Condition',
                    y=metric,
                    color=violin_fill_color,
                    edgecolor=violin_edge_color if violin_edge_color else None,
                    linewidth=violin_outline_width,   # (seaborn uses 'linewidth')
                    inner=None,
                    gap=0.1,
                    alpha=violin_alpha,
                    zorder=0,
                    ax=ax,
                )

            # === replicate mean/median markers (single scatter each) ===
            if show_mean_balls:
                sns.scatterplot(
                    data=_rep_stats,
                    x='Condition', y='mean',
                    hue='Replicate',
                    palette=_palette,
                    edgecolor=mean_ball_outline_color,
                    s=mean_ball_size,
                    legend=False,
                    alpha=mean_ball_alpha,
                    linewidth=mean_ball_outline_width,
                    zorder=4,
                    ax=ax,
                )

            if show_median_balls:
                sns.scatterplot(
                    data=_rep_stats,
                    x='Condition', y='median',
                    hue='Replicate',
                    palette=_palette,
                    edgecolor=median_ball_outline_color,
                    s=median_ball_size,
                    legend=False,
                    alpha=median_ball_alpha,
                    linewidth=median_ball_outline_width,
                    zorder=4,
                    ax=ax,
                )

            # === vectorized lines & error bars (big win) ===
            # Build x centers as 0..N-1 in category order (seaborn does the same internally)
            n = len(categories_for_stats)
            x_centers = np.arange(n)

            # Mask out spacers so we don't draw stats on them
            is_spacer = _cond_stats['Condition'].astype(str).str.startswith('spacer')
            valid = ~is_spacer

            y_mean = _cond_stats.loc[valid, 'mean'].to_numpy()
            y_median = _cond_stats.loc[valid, 'median'].to_numpy()
            y_std = _cond_stats.loc[valid, 'std'].to_numpy()
            x_valid = x_centers[valid.to_numpy()]

            # Mean & median short spans using vectorized hlines
            if show_mean and y_mean.size:
                xmin = x_valid - mean_span
                xmax = x_valid + mean_span
                ax.hlines(y_mean, xmin, xmax,
                        colors=mean_color, linestyles=mean_ls,
                        linewidths=line_width, zorder=3, label='Mean')

            if show_median and y_median.size:
                xmin_m = x_valid - median_span
                xmax_m = x_valid + median_span
                # If both present, only label once (matplotlib de-dupes identical labels later)
                ax.hlines(y_median, xmin_m, xmax_m,
                        colors=median_color, linestyles=median_ls,
                        linewidths=line_width, zorder=3,
                        label=('Median' if not show_mean else None))

            if show_error_bars and y_mean.size:
                ax.errorbar(
                    x_valid, y_mean, yerr=y_std,
                    fmt='none',
                    color=errorbar_color,
                    alpha=errorbar_alpha,
                    linewidth=errorbar_lw,
                    capsize=errorbar_capsize,
                    zorder=3,
                    label=('Mean ± SD' if (show_mean or show_median) else 'SD')
                )

            # === KDE insets (loop only once per actual condition) ===
            if show_kde:
                # After base layers, use established y-limits to size insets
                y_ax_min, y_ax_max = ax.get_ylim()

                # Iterate over actual conditions (even positions in spaced layout)
                for i, cond in enumerate(conditions):
                    group_df = _df[_df['Condition'] == cond]
                    if group_df.empty:
                        continue

                    # inset geometry in data coords
                    x_pos = 2 * i + 2  # even positions: 0,2,4...
                    offset_x = 0.31
                    inset_height = y_ax_max - (y_ax_max - group_df[metric].max()) + abs(y_ax_min * 2)

                    inset_ax = ax.inset_axes(
                        [x_pos - offset_x, y_ax_min, kde_inset_width, inset_height],
                        transform=ax.transData, zorder=0, clip_on=True
                    )
                    sns.kdeplot(
                        data=group_df, y=metric, hue='Replicate',
                        fill=kde_fill, alpha=kde_alpha, lw=kde_outline,
                        palette=_palette, ax=inset_ax, legend=False,
                        zorder=0, clip=(y_ax_min, y_ax_max),
                    )
                    # inset_ax.invert_xaxis()
                    inset_ax.set_xticks([]); inset_ax.set_yticks([])
                    inset_ax.set_xlabel(''); inset_ax.set_ylabel('')
                    sns.despine(ax=inset_ax, left=True, bottom=True, top=True, right=True)

                # Ticks: show only real conditions
                ticks = [i for i, lbl in enumerate(categories_for_stats) if not str(lbl).startswith('spacer')]
                labels = [categories_for_stats[i] for i in ticks]
                plt.xticks(ticks=ticks, labels=labels)

            # === axes cosmetics (unchanged) ===
            plt.title(title)
            plt.xlabel("Condition")
            plt.ylabel(f"{metric} {SetUnits(t='s').get(metric)}")

            if show_legend:
                handles, labels = [], []

                # Replicate entries
                for r in _df['Replicate'].astype(str).unique().tolist():
                    c = _palette.get(r, 'grey')
                    handles.append(mlines.Line2D([], [], linestyle='None',
                                                marker='o', markersize=8,
                                                markerfacecolor=c,
                                                markeredgecolor='black',
                                                label=(str(r) + " median")))
                    labels.append(str(r) + " median")

                # Stats entries (mirror your original logic)
                if show_mean and not show_error_bars:
                    handles.append(mlines.Line2D([], [], color=mean_color,
                                                linestyle=mean_ls, linewidth=line_width,
                                                label='Mean'))
                    labels.append('Mean')
                elif show_error_bars and not show_mean:
                    handles.append(mlines.Line2D([], [], color=errorbar_color,
                                                linestyle='-', linewidth=errorbar_lw,
                                                marker='_', markersize=10,
                                                label='SD'))
                    labels.append('SD')
                elif show_mean and show_error_bars:
                    handles.append(mlines.Line2D([], [], color=errorbar_color,
                                                linestyle='-', linewidth=errorbar_lw,
                                                marker='_', markersize=10,
                                                label='Mean ± SD'))
                    labels.append('Mean ± SD')

                if show_median:
                    handles.append(mlines.Line2D([], [], color=median_color,
                                                linestyle=median_ls, linewidth=line_width,
                                                label='Median'))
                    labels.append('Median')

                leg = ax.legend(handles, labels, title='Legend',
                                title_fontsize=12, fontsize=10,
                                loc='upper right', bbox_to_anchor=(1.15, 1),
                                frameon=True)
                try:
                    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1))
                except Exception:
                    pass
            else:
                try:
                    plt.legend().remove()
                except Exception:
                    pass

            sns.despine(top=open_spine, right=open_spine, bottom=False, left=False)
            plt.tick_params(axis='y', which='major', length=7, width=1.5, direction='out', color='black')
            plt.tick_params(axis='x', which='major', length=5, width=1.5, direction='out', color='black', rotation=345)
            if show_grid:
                plt.grid(show_grid, axis='y', color='lightgrey', linewidth=1.5, alpha=0.2)
            else:
                plt.grid(False)

            # Keep your legend move safeguard
            try:
                if plt.gca().get_legend() is not None:
                    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1))
            except Exception:
                try:
                    ax = plt.gca()
                    if plt.gca().get_legend() is not None:
                        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1))
                except Exception:
                    pass

            return plt.gcf()


    class Tracks:
        
        @staticmethod
        def VisualizeTracksRealistics(
            Spots_df: pd.DataFrame,
            Tracks_df: pd.DataFrame,
            condition: str,
            *args,
            replicate: str = 'all',
            c_mode: str = 'differentiate replicates',
            only_one_color: str = 'blue',
            lut_scaling_metric: str = 'Track displacement',
            background: str = 'dark',
            smoothing_index: int | float = 0,
            lw: float = 1.0,
            grid: bool = True,
            mark_heads: bool = False,
            marker: dict = {"symbol": "o", "fill": True},
            markersize: float = 5.0,
            title: str = 'Track Visualization',
        ):
            # --- Early outs / guards -------------------------------------------------

            Spots = Spots_df.copy()
            Tracks = Tracks_df.copy()
            
            # --- Filter & sort once ---------------------------------------------------
            required = ['Condition', 'Replicate', 'Track ID', 'Time point', 'X coordinate', 'Y coordinate']
            if any(col not in Spots.columns for col in required):
                return plt.gcf()

            if replicate == 'all':
                Spots = Spots.loc[Spots['Condition'] == condition]
                Tracks = Tracks.loc[Tracks['Condition'] == condition]
            elif replicate != 'all':
                Spots = Spots.loc[Spots['Replicate'] == replicate]
                Tracks = Tracks.loc[Tracks['Replicate'] == replicate]

            Spots = Spots.sort_values(['Condition', 'Replicate', 'Track ID', 'Time point'])
            Tracks = Tracks.sort_values(['Condition', 'Replicate', 'Track ID'])

            # Ensure we can group efficiently
            key_cols = ['Condition', 'Replicate', 'Track ID']

            # Ensure keys exist only as index (no duplicate columns)
            Spots = Spots.set_index(key_cols, drop=True)      # drop=True is the default; keeps keys out of columns
            Tracks = Tracks.set_index(key_cols, drop=True)

            # --- Optional smoothing (vectorized) --------------------------------------
            if isinstance(smoothing_index, (int, float)) and smoothing_index > 1:
                win = int(smoothing_index)
                Spots['X coordinate'] = (
                    Spots.groupby(level=key_cols)['X coordinate'].transform(lambda s: s.rolling(win, min_periods=1).mean())
                )
                Spots['Y coordinate'] = (
                    Spots.groupby(level=key_cols)['Y coordinate'].transform(lambda s: s.rolling(win, min_periods=1).mean())
                )

            # --- Colors: compute once, map to each track ------------------------------
            rng = np.random.default_rng(42)

            def rand_color():
                return mcolors.to_hex(rng.random(3))
            def rand_grey():
                g = float(rng.random())
                return mcolors.to_hex((g, g, g))

            colormap = None
            if c_mode in ['random colors', 'random greys', 'only-one-color']:
                # one color per *track*
                unique_tracks = Tracks.index.unique()
                if c_mode == 'random colors':
                    colors = [rand_color() for _ in range(len(unique_tracks))]
                elif c_mode == 'random greys':
                    colors = [rand_grey() for _ in range(len(unique_tracks))]
                else:
                    colors = [only_one_color] * len(unique_tracks)
                track_to_color = dict(zip(unique_tracks, colors))
                Tracks['Track color'] = [track_to_color[idx] for idx in Tracks.index]

            elif c_mode == 'differentiate replicates':
                # Color by replicate if available, else fall back
                Tracks['Track color'] = Tracks['Replicate color'] if 'Replicate color' in Tracks.columns else "red"

            else:
                # interpret c_mode as a matplotlib cmap name
                use_instantaneous = (lut_scaling_metric == 'Speed instantaneous')

                if lut_scaling_metric in Tracks.columns and not use_instantaneous:
                    colormap = _get_cmap(c_mode)
                    vmin = float(Tracks[lut_scaling_metric].min())
                    vmax = float(Tracks[lut_scaling_metric].max())
                    norm = plt.Normalize(vmin, vmax if np.isfinite(vmax) and vmax > vmin else vmin + 1.0)
                    Tracks['Track color'] = [mcolors.to_hex(colormap(norm(v))) for v in Tracks[lut_scaling_metric].to_numpy()]

                elif use_instantaneous:
                    # Color segments by the most-recent (ending) spot of each segment
                    # Compute per-spot speed for the segment that ENDS at this spot:
                    # speed_end[i] = distance from spot i-1 -> i
                    colormap = _get_cmap(c_mode)
                    if 'Distance' not in Spots.columns:
                        # Fallback: compute instantaneous distances if missing
                        g = Spots.groupby(level=key_cols)
                        d = np.sqrt(
                            (g['X coordinate'].diff())**2 +
                            (g['Y coordinate'].diff())**2
                        )
                        speed_end = d
                    else:
                        # Distance in Calc.Spots is from current -> next; shift to align with segment end
                        speed_end = Spots.groupby(level=key_cols)['Distance'].shift(1)

                    vmax = float(np.nanmax(speed_end.to_numpy())) if np.isfinite(speed_end.to_numpy()).any() else 1.0
                    vmin = 0.0
                    norm = plt.Normalize(vmin, vmax if vmax > 0 else 1.0)

                    # Color each spot by the speed of the segment that ends at this spot
                    Spots['Spot color'] = [
                        mcolors.to_hex(colormap(norm(v))) if np.isfinite(v) else mcolors.to_hex(colormap(0.0))
                        for v in speed_end.to_numpy()
                    ]

                # else: no-op; colors may have been assigned above

            # Map per-track color down to Spots only if not using instantaneous coloring
            if not (lut_scaling_metric == 'Speed instantaneous'):
                Spots = Spots.join(
                    Tracks[['Track color']],
                    on=['Condition', 'Replicate', 'Track ID'],
                    how='left',
                    validate='many_to_one',
                )

            # --- Build line segments for LineCollection -------------------------------
            segments = []
            seg_colors = []

            if lut_scaling_metric == 'Speed instantaneous':
                # One segment per consecutive pair, colored by the ending spot's color
                for (cond, repl, tid), g in Spots.groupby(level=key_cols, sort=False):
                    xy = g[['X coordinate', 'Y coordinate']].to_numpy(dtype=float, copy=False)
                    if xy.shape[0] >= 2:
                        cols = g['Spot color'].astype(str).to_numpy()
                        # Build pairwise segments: [i-1 -> i], colored by cols[i]
                        for i in range(1, xy.shape[0]):
                            segments.append(xy[i-1:i+1])
                            seg_colors.append(cols[i])
            else:
                # One polyline per track, colored by its track color
                for (cond, repl, tid), g in Spots.groupby(level=key_cols, sort=False):
                    xy = g[['X coordinate', 'Y coordinate']].to_numpy(dtype=float, copy=False)
                    if xy.shape[0] >= 2:
                        segments.append(xy)
                        seg_colors.append(g['Track color'].iloc[0])

            # --- Figure / axes setup ---------------------------------------------------
            if background == 'white':
                grid_color, face_color, grid_alpha, grid_ls = 'gainsboro', 'white', 0.5, '-.' if grid else 'None'
            elif background == 'light':
                grid_color, face_color, grid_alpha, grid_ls = 'silver', 'lightgrey', 0.5, '-.' if grid else 'None'
            elif background == 'mid':
                grid_color, face_color, grid_alpha, grid_ls = 'silver', 'darkgrey', 0.5, '-.' if grid else 'None'
            elif background == 'dark':
                grid_color, face_color, grid_alpha, grid_ls = 'grey', 'dimgrey', 0.5, '-.' if grid else 'None'
            elif background == 'black':
                grid_color, face_color, grid_alpha, grid_ls = 'dimgrey', 'black', 0.5, '-.' if grid else 'None'

            fig, ax = plt.subplots(figsize=(13, 10))
            if len(Spots):
                x = Spots['X coordinate'].to_numpy()
                y = Spots['Y coordinate'].to_numpy()
                ax.set_xlim(np.nanmin(x), np.nanmax(x))
                ax.set_ylim(np.nanmin(y), np.nanmax(y))

            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('X coordinate [microns]')
            ax.set_ylabel('Y coordinate [microns]')
            ax.set_title(title, fontsize=12)
            ax.set_facecolor(face_color)
            ax.grid(grid, which='both', axis='both', color=grid_color, linestyle=grid_ls, linewidth=1, alpha=grid_alpha)

            # Ticks
            ax.xaxis.set_major_locator(MultipleLocator(200))
            ax.yaxis.set_major_locator(MultipleLocator(200))
            ax.xaxis.set_minor_locator(MultipleLocator(50))
            ax.yaxis.set_minor_locator(MultipleLocator(50))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax.tick_params(axis='both', which='major', labelsize=8)

            # --- Draw all tracks at once ----------------------------------------------
            if segments:
                lc = LineCollection(segments, colors=seg_colors, linewidths=lw, zorder=10)
                ax.add_collection(lc)

            # --- Optional markers at track heads -------------------------------------
            if mark_heads:
                ends = Spots.groupby(level=key_cols, sort=False).tail(1)
                if len(ends):
                    xe = ends['X coordinate'].to_numpy(dtype=float, copy=False)
                    ye = ends['Y coordinate'].to_numpy(dtype=float, copy=False)
                    if lut_scaling_metric == 'Speed instantaneous':
                        cols = ends['Spot color'].astype(str).to_numpy()
                    else:
                        cols = ends['Track color'].astype(str).to_numpy()
                    m = np.isfinite(xe) & np.isfinite(ye)
                    if m.any():
                        ax.scatter(
                            xe[m],
                            ye[m],
                            marker=marker["symbol"],
                            s=markersize,
                            edgecolor=cols[m],
                            facecolor=cols[m] if marker["fill"] else "none",
                            linewidths=lw,
                            zorder=12,
                        )

            return plt.gcf()


        @staticmethod
        def VisualizeTracksNormalized(
            Spots_df: pd.DataFrame,
            Tracks_df: pd.DataFrame,
            condition: str,
            *args,
            replicate: str = 'all',
            c_mode: str = 'differentiate replicates',
            only_one_color: str = 'blue',
            lut_scaling_metric: str = 'Track displacement',
            smoothing_index: float = 0,
            lw: float = 1.0,
            show_tracks: bool = True,
            grid: bool = True,
            arrows: bool = False,
            arrowsize: int = 5,
        ):
            # ----------------- copies / guards -----------------
            Spots_all = Spots_df.copy()
            Spots = Spots_df.copy()
            Tracks = Tracks_df.copy()

            required_spots = ['Condition','Replicate','Track ID','Time point','X coordinate','Y coordinate']
            if any(col not in Spots.columns for col in required_spots):
                return plt.gcf()

            # ----------------- filter subset to draw -----------------
            if condition == 'all':
                if replicate != 'all':
                    Spots = Spots.loc[Spots['Replicate'] == replicate]
                    Tracks = Tracks.loc[Tracks['Replicate'] == replicate]
            else:
                Spots = Spots.loc[Spots['Condition'] == condition]
                Tracks = Tracks.loc[Tracks['Condition'] == condition]
                if replicate != 'all':
                    Spots = Spots.loc[Spots['Replicate'] == replicate]
                    Tracks = Tracks.loc[Tracks['Replicate'] == replicate]

            sort_cols = ['Condition','Replicate','Track ID','Time point']
            key_cols = ['Condition','Replicate','Track ID']
            Spots = Spots.sort_values(sort_cols).set_index(key_cols, drop=True)
            Tracks = Tracks.sort_values(['Condition','Replicate','Track ID']).set_index(key_cols, drop=True)

            # ----------------- smoothing (subset) -----------------
            if isinstance(smoothing_index, (int, float)) and smoothing_index > 1:
                win = int(smoothing_index)
                Spots['X coordinate'] = Spots.groupby(level=key_cols)['X coordinate'] \
                    .transform(lambda s: s.rolling(win, min_periods=1).mean())
                Spots['Y coordinate'] = Spots.groupby(level=key_cols)['Y coordinate'] \
                    .transform(lambda s: s.rolling(win, min_periods=1).mean())

            # ----------------- normalize (subset) -----------------
            x0 = Spots.groupby(level=key_cols)['X coordinate'].transform('first')
            y0 = Spots.groupby(level=key_cols)['Y coordinate'].transform('first')
            Spots['Xn'] = Spots['X coordinate'] - x0
            Spots['Yn'] = Spots['Y coordinate'] - y0

            # ----------------- colors on Tracks, join to Spots -----------------
            rng = np.random.default_rng(42)
            track_index = Tracks.index.unique()

            if c_mode in ['random colors', 'random greys', 'only-one-color']:
                if c_mode == 'random colors':
                    cols = _rand_hex_colors(len(track_index), rng)
                elif c_mode == 'random greys':
                    cols = _rand_hex_greys(len(track_index), rng)
                else:
                    cols = [mcolors.to_hex(only_one_color)] * len(track_index)
                Tracks['Track color'] = [dict(zip(track_index, cols))[idx] for idx in Tracks.index]

            elif c_mode == 'differentiate replicates':
                if 'Replicate color' in Tracks.columns:
                    Tracks['Track color'] = Tracks['Replicate color'].astype(str)
                else:
                    reps = Tracks.reset_index()['Replicate'].unique().tolist()
                    rep_cols = _rand_hex_colors(len(reps), rng)
                    rep2col = dict(zip(reps, rep_cols))
                    Tracks = Tracks.reset_index()
                    Tracks['Track color'] = Tracks['Replicate'].map(rep2col)
                    Tracks = Tracks.set_index(key_cols)

            else:
                cmap = _get_cmap(c_mode)
                vmin = float(Tracks[lut_scaling_metric].min()) if lut_scaling_metric in Tracks.columns else 0.0
                vmax = float(Tracks[lut_scaling_metric].max()) if lut_scaling_metric in Tracks.columns else 1.0
                norm = plt.Normalize(vmin, vmax)
                vals = Tracks[lut_scaling_metric].to_numpy() if lut_scaling_metric in Tracks.columns else np.zeros(len(Tracks))
                Tracks['Track color'] = [mcolors.to_hex(cmap(norm(v))) for v in vals]

            Spots = Spots.join(Tracks[['Track color']], on=key_cols, how='left', validate='many_to_one')

            # ----------------- polar conversion (subset) -----------------
            Spots['r'] = np.sqrt(Spots['Xn']**2 + Spots['Yn']**2)
            Spots['theta'] = np.arctan2(Spots['Yn'], Spots['Xn'])

            # ----------------- GLOBAL y_max from full dataset -----------------
            All = Spots_all.sort_values(sort_cols).set_index(key_cols, drop=True)
            if isinstance(smoothing_index, (int, float)) and smoothing_index > 1:
                win = int(smoothing_index)
                AllX = All.groupby(level=key_cols)['X coordinate'].transform(lambda s: s.rolling(win, min_periods=1).mean())
                AllY = All.groupby(level=key_cols)['Y coordinate'].transform(lambda s: s.rolling(win, min_periods=1).mean())
            else:
                AllX, AllY = All['X coordinate'], All['Y coordinate']
            AllX0 = AllX.groupby(level=key_cols).transform('first')
            AllY0 = AllY.groupby(level=key_cols).transform('first')
            All_r = np.sqrt((AllX - AllX0)**2 + (AllY - AllY0)**2)

            y_max_global = float(np.nanmax(All_r.to_numpy())) if len(All_r) else 1.0
            if not np.isfinite(y_max_global) or y_max_global <= 0:
                y_max_global = 1.0
            y_max = y_max_global * 1.1  # headroom

            # ----------------- segments (subset) -----------------
            segments, seg_colors = [], []
            for (cond, repl, tid), g in Spots.groupby(level=key_cols, sort=False):
                th = g['theta'].to_numpy(dtype=float, copy=False)
                rr = g['r'].to_numpy(dtype=float, copy=False)
                if th.size >= 2:
                    segments.append(np.column_stack([th, rr]))
                    seg_colors.append(g['Track color'].iloc[0])

            # ----------------- figure / axes -----------------
            fig, ax = plt.subplots(figsize=(12.5, 9.5), subplot_kw={'projection': 'polar'})
            ax.set_facecolor('white')
            ax.set_title('Normalized Tracks', fontsize=12)
            ax.set_ylim(0, y_max)        # <- global, consistent across subsets
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.spines['polar'].set_visible(False)
            ax.grid(grid)

            # ----------------- draw tracks -----------------
            if segments:
                lc = LineCollection(segments, colors=seg_colors, linewidths=lw if show_tracks else 0, zorder=10)
                lc.set_transform(ax.transData)
                ax.add_collection(lc)

            # ----------------- arrows (optional) -----------------
            if arrows and 'Direction mean (rad)' in Tracks.columns:
                arrow_length = 1.0
                last_pts = Spots.groupby(level=key_cols, sort=False).tail(1).reset_index()
                mean_dir = Tracks.reset_index()[key_cols + ['Direction mean (rad)','Track color']]
                merged = pd.merge(last_pts, mean_dir, on=key_cols, how='left', validate='one_to_one')
                xe, ye = merged['Xn'].to_numpy(), merged['Yn'].to_numpy()
                md = merged['Direction mean (rad)'].to_numpy()
                cols = merged['Track color'].astype(str).to_numpy()
                xt, yt = xe + arrow_length*np.cos(md), ye + arrow_length*np.sin(md)
                tail_r = np.sqrt(xe**2 + ye**2); tail_th = np.arctan2(ye, xe)
                tip_r  = np.sqrt(xt**2 + yt**2); tip_th  = np.arctan2(yt, xt)
                if tail_r.size:
                    for col in np.unique(cols):
                        m = (cols == col)
                        for th0, r0, th1, r1 in zip(tail_th[m], tail_r[m], tip_th[m], tip_r[m]):
                            ax.annotate('', xy=(th1, r1), xytext=(th0, r0),
                                        arrowprops=dict(arrowstyle='-|>', color=str(col),
                                                        lw=lw if show_tracks else 0, mutation_scale=arrowsize),
                                        annotation_clip=False)

            # ----------------- subtle grid cosmetics -----------------
            for i, line in enumerate(ax.get_xgridlines()):
                if i % 2 == 0:
                    line.set_linestyle('--'); line.set_color('grey'); line.set_linewidth(0.5)
            for line in ax.get_ygridlines():
                line.set_linestyle('-.'); line.set_color('lightgrey'); line.set_linewidth(0.5)

            # ----------------- μm label on the side (no line) -----------------
            # Show the global radius (rounded) just outside the right edge, centered vertically.
            label_um = f"{int(np.round(y_max_global))} μm"
            ax.text(1.03, 0.5, label_um,
                    transform=ax.transAxes, ha='left', va='center',
                    fontsize=10, color='dimgray', clip_on=False)

            return plt.gcf()



    class TimeCharts:
        pass