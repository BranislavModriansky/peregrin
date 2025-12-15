from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from .._common import Colors
from io import BytesIO
from ..._handlers._reports import Level


# FIXME: 'differentiate conditions' has no color mapping in most places
# FIXME: 'differentiate replicates' if column 'Replicate color' does not exist colors all tracks red
#        add random colors per replicate as a fallback and insert warnings
# FIXME: turning functions into classes and making data flow smoother would be way better


class ReconstructTracks:
    
    def __init__(self, Spots_df: pd.DataFrame, Tracks_df: pd.DataFrame, *args,
                 conditions: list, replicates: list, 
                 c_mode: str, lut_scaling_metric: str, only_one_color: str,
                 smoothing_index: float, lw: float,
                 mark_heads: bool, marker: dict, markersize: float,
                 background: str, grid: bool,
                 title: str, **kwargs):
        
        self.Spots_df = Spots_df
        self.Tracks_df = Tracks_df
        self.conditions = conditions
        self.replicates = replicates
        self.c_mode = c_mode
        self.lut_scaling_metric = lut_scaling_metric
        self.only_one_color = only_one_color
        self.smoothing_index = smoothing_index
        self.lw = lw
        self.mark_heads = mark_heads
        self.marker = marker
        self.markersize = markersize
        self.background = background
        self.grid = grid
        self.title = title
        self.noticequeue = kwargs.get('noticequeue', None)
        
        
    def Realistic():



@staticmethod
def VisualizeTracksRealistics(
    Spots_df: pd.DataFrame,
    Tracks_df: pd.DataFrame,
    conditions: list,
    replicates: list,
    *args,
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
    **kwargs
):
    
    noticequeue = kwargs.get('noticequeue', None) if 'noticequeue' in kwargs else None

    Spots = Spots_df.copy()
    Tracks = Tracks_df.copy()
    
    # --- Filter & sort once ---------------------------------------------------
    required = ['Condition', 'Replicate', 'Track ID', 'Time point', 'X coordinate', 'Y coordinate']
    if any(col not in Spots.columns for col in required):
        return plt.gcf()
    
    if not conditions:
        noticequeue.Report("warning", "No conditions selected! At at least one condition must be selected.")
        return plt.gcf()
    if not replicates:
        noticequeue.Report("warning", "No replicates selected! At at least one replicate must be selected.")
        return plt.gcf()

    Spots = Spots.loc[Spots['Condition'].isin(conditions)].loc[Spots['Replicate'].isin(replicates)]
    Tracks = Tracks.loc[Tracks['Condition'].isin(conditions)].loc[Tracks['Replicate'].isin(replicates)]

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
        Tracks['Track color'] = Tracks['Replicate color'] if 'Replicate color' in Tracks.columns else "red"
    elif c_mode == 'differentiate conditions':
        Tracks['Track color'] = Tracks['Condition color'] if 'Condition color' in Tracks.columns else "red"

    else:
        # interpret c_mode as a matplotlib cmap name
        use_instantaneous = (lut_scaling_metric == 'Speed instantaneous')

        if lut_scaling_metric in Tracks.columns and not use_instantaneous:
            colormap = Colors.GetCmap(c_mode)
            vmin = float(Tracks[lut_scaling_metric].min())
            vmax = float(Tracks[lut_scaling_metric].max())
            norm = plt.Normalize(vmin, vmax if np.isfinite(vmax) and vmax > vmin else vmin + 1.0)
            Tracks['Track color'] = [mcolors.to_hex(colormap(norm(v))) for v in Tracks[lut_scaling_metric].to_numpy()]

        elif use_instantaneous:
            # Color segments by the most-recent (ending) spot of each segment
            # Compute per-spot speed for the segment that ENDS at this spot:
            # speed_end[i] = distance from spot i-1 -> i
            colormap = Colors.GetCmap(c_mode)
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


    # Map per-track color down to Spots only if not using instantaneous coloring
    if c_mode in ['random colors', 'random greys', 'only-one-color'] or not (lut_scaling_metric == 'Speed instantaneous'): 
        Spots = Spots.join(
            Tracks[['Track color']],
            on=['Condition', 'Replicate', 'Track ID'],
            how='left',
            validate='many_to_one',
        )

    # --- Build line segments for LineCollection -------------------------------
    segments = []
    seg_colors = []

    if c_mode not in ['random colors', 'random greys', 'only-one-color'] and lut_scaling_metric == 'Speed instantaneous':
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
    ax.grid(grid, which='both', axis='both', color=grid_color, linestyle=grid_ls, linewidth=1, alpha=grid_alpha) if grid else ax.grid(False)

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
            if c_mode not in ['random colors', 'random greys', 'only-one-color'] and lut_scaling_metric == 'Speed instantaneous':
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
    conditions: list,
    replicates: list,
    *args,
    c_mode: str = 'differentiate replicates',
    only_one_color: str = 'blue',
    lut_scaling_metric: str = 'Track displacement',
    smoothing_index: float = 0,
    lw: float = 1.0,
    background: str = 'white',
    grid: bool = True,
    grid_style: str = 'alternating 1',
    mark_heads: bool = False,
    marker: dict = {"symbol": "o", "fill": True},
    markersize: int = 5,
    title: str = 'Normalized Track Visualization',
    **kwargs
):
    
    noticequeue = kwargs.get('noticequeue', None) if 'noticequeue' in kwargs else None
    
    # ----------------- copies / guards -----------------
    Spots_all = Spots_df.copy()
    Spots = Spots_df.copy()
    Tracks = Tracks_df.copy()

    if not conditions:
        noticequeue.Report("warning", "No conditions selected! At at least one condition must be selected.")
        return plt.gcf()
    if not replicates:
        noticequeue.Report("warning", "No replicates selected! At at least one replicate must be selected.")
        return plt.gcf()

    Spots = Spots.loc[Spots['Condition'].isin(conditions)].loc[Spots['Replicate'].isin(replicates)]
    Tracks = Tracks.loc[Tracks['Condition'].isin(conditions)].loc[Tracks['Replicate'].isin(replicates)]

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
            cols = [mcolors.to_hex(rng.random(3)) for _ in range(len(track_index))]
        elif c_mode == 'random greys':
            cols = [mcolors.to_hex((float(g), float(g), float(g))) for g in rng.random(len(track_index))]
        else:
            cols = [mcolors.to_hex(only_one_color)] * len(track_index)
        Tracks['Track color'] = [dict(zip(track_index, cols))[idx] for idx in Tracks.index]

    elif c_mode == 'differentiate replicates':
        if 'Replicate color' in Tracks.columns:
            Tracks['Track color'] = Tracks['Replicate color'].astype(str)
        else:
            reps = Tracks.reset_index()['Replicate'].unique().tolist()
            rep_cols = [mcolors.to_hex(rng.random(3)) for _ in range(len(reps))]
            rep2col = dict(zip(reps, rep_cols))
            Tracks = Tracks.reset_index()
            Tracks['Track color'] = Tracks['Replicate'].map(rep2col)
            Tracks = Tracks.set_index(key_cols)

    else:
        # interpret c_mode as a matplotlib cmap name
        use_instantaneous = (lut_scaling_metric == 'Speed instantaneous')

        if lut_scaling_metric in Tracks.columns and not use_instantaneous:
            cmap = Colors.GetCmap(c_mode)
            vmin = float(Tracks[lut_scaling_metric].min()) if lut_scaling_metric in Tracks.columns else 0.0
            vmax = float(Tracks[lut_scaling_metric].max()) if lut_scaling_metric in Tracks.columns else 1.0
            # guard against degenerate range
            if not np.isfinite(vmin): vmin = 0.0
            if not np.isfinite(vmax) or vmax <= vmin: vmax = vmin + 1.0
            norm = plt.Normalize(vmin, vmax)
            vals = Tracks[lut_scaling_metric].to_numpy() if lut_scaling_metric in Tracks.columns else np.zeros(len(Tracks))
            Tracks['Track color'] = [mcolors.to_hex(cmap(norm(v))) for v in vals]

        elif use_instantaneous:
            # Color by instantaneous speed at each spot (ending segment)
            colormap = Colors.GetCmap(c_mode)
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
            vmax = vmax if vmax > 0 else 1.0
            norm = plt.Normalize(vmin, vmax)

            # Color each spot by the speed of the segment that ends at this spot
            Spots['Spot color'] = [
                mcolors.to_hex(colormap(norm(v))) if np.isfinite(v) else mcolors.to_hex(colormap(0.0))
                for v in speed_end.to_numpy()
            ]

    # Only join per-track colors when not using instantaneous LUT
    if lut_scaling_metric != 'Speed instantaneous':
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
    if lut_scaling_metric == 'Speed instantaneous':
        # One segment per consecutive pair in polar coords, colored by ending spot color
        for (cond, repl, tid), g in Spots.groupby(level=key_cols, sort=False):
            th = g['theta'].to_numpy(dtype=float, copy=False)
            rr = g['r'].to_numpy(dtype=float, copy=False)
            cols = g.get('Spot color', pd.Series(index=g.index, dtype=object)).astype(str).to_numpy()
            n = th.size
            if n >= 2:
                for i in range(1, n):
                    # segment from i-1 -> i
                    segments.append(np.array([[th[i-1], rr[i-1]], [th[i], rr[i]]], dtype=float))
                    # color by ending spot
                    seg_colors.append(cols[i] if i < len(cols) else cols[-1] if len(cols) else 'black')
    else:
        # One polyline per track in polar coords, colored by track color
        for (cond, repl, tid), g in Spots.groupby(level=key_cols, sort=False):
            th = g['theta'].to_numpy(dtype=float, copy=False)
            rr = g['r'].to_numpy(dtype=float, copy=False)
            if th.size >= 2:
                segments.append(np.column_stack([th, rr]))
                seg_colors.append(g['Track color'].iloc[0])

    # ----------------- figure / axes -----------------
    fig, ax = plt.subplots(figsize=(12.5, 9.5), subplot_kw={'projection': 'polar'})
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, y_max)        # <- global, consistent across subsets
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines['polar'].set_visible(False)
    # ax.grid(grid)

    # ----------------- draw tracks -----------------
    if segments:
        lc = LineCollection(segments, colors=seg_colors, linewidths=lw, zorder=10)
        lc.set_transform(ax.transData)
        ax.add_collection(lc)

    # ----------------- arrows (optional) -----------------
    if mark_heads:
        ends = Spots.groupby(level=key_cols, sort=False).tail(1)
        if len(ends):
            th_e = ends['theta'].to_numpy(dtype=float, copy=False)
            r_e = ends['r'].to_numpy(dtype=float, copy=False)
            if lut_scaling_metric == 'Speed instantaneous':
                cols = ends['Spot color'].astype(str).to_numpy()
            else:
                cols = ends['Track color'].astype(str).to_numpy()
            m = np.isfinite(th_e) & np.isfinite(r_e)
            if m.any():
                ax.scatter(
                    th_e[m],
                    r_e[m],
                    marker=marker["symbol"],
                    s=markersize,
                    edgecolor=cols[m],
                    facecolor=cols[m] if marker["fill"] else "none",
                    linewidths=lw,
                    zorder=12,
                )

    # ----------------- subtle grid cosmetics -----------------
    if background == 'white':
        ax.set_facecolor('white')
        _color, _alpha1, _alpha2 = 'lightgrey', 0.7, 0.8
    elif background == 'light':
        ax.set_facecolor('lightgrey')
        _color, _alpha1, _alpha2 = 'darkgrey', 0.7, 0.6
    elif background == 'mid':
        ax.set_facecolor('darkgrey')
        _color, _alpha1, _alpha2 = 'dimgrey', 0.7, 0.5
    elif background == 'dark':
        ax.set_facecolor('dimgrey')
        _color, _alpha1, _alpha2 = 'grey', 0.7, 0.6
    elif background == 'black':
        ax.set_facecolor('black')
        _color, _alpha1, _alpha2 = 'dimgrey', 0.5, 0.4

    if grid:
        if grid_style in ['simple-1', 'simple-2']:
            ax.xaxis.grid(True, color=_color, linestyle='-', linewidth=1, alpha=_alpha1)
            ax.yaxis.grid(False)

            if grid_style == 'simple-1':
                for i, line in enumerate(ax.get_xgridlines()):
                    if i % 2 != 0:
                        line.set_color('none')
            if grid_style == 'simple-2':
                for i, line in enumerate(ax.get_xgridlines()):
                    if i % 2 == 0:
                        line.set_color('none')

        if grid_style in ['dartboard-1', 'dartboard-2']:
            ax.grid(True, lw=0.75, color=_color, alpha=_alpha1)
            if grid_style == 'dartboard-1':
                for i, line in enumerate(ax.get_xgridlines()):
                    if i % 2 == 0:
                        line.set_linestyle('-.'); line.set_color(_color); line.set_linewidth(0.75), line.set_alpha(_alpha1)
                for line in ax.get_ygridlines():
                    line.set_linestyle('--'); line.set_color(_color); line.set_linewidth(0.75), line.set_alpha(_alpha2)

            if grid_style == 'dartboard-2':
                for i, line in enumerate(ax.get_xgridlines()):
                    if i % 2 != 0:
                        line.set_linestyle('-.'); line.set_color(_color); line.set_linewidth(0.75), line.set_alpha(_alpha1)
                for line in ax.get_ygridlines():
                    line.set_linestyle('--'); line.set_color(_color); line.set_linewidth(0.75), line.set_alpha(_alpha2)
        elif grid_style == 'spindle':
            ax.xaxis.grid(True, color=_color, linestyle='-', linewidth=1, alpha=_alpha1)
            ax.yaxis.grid(False)
        elif grid_style == 'radial':
            ax.xaxis.grid(False)
            ax.yaxis.grid(True, color=_color, linestyle='-', linewidth=1, alpha=_alpha1)

    elif not grid:
        ax.grid(False)

    # ----------------- μm label on the side (no line) -----------------
    # Show the global radius (rounded) just outside the right edge, centered vertically.
    label_um = f"{int(np.round(y_max_global))} μm"
    ax.text(1.03, 0.5, label_um,
            transform=ax.transAxes, ha='left', va='center',
            fontsize=10, color='dimgray', clip_on=False)

    return plt.gcf()

@staticmethod
def GetLutMap(Tracks_df: pd.DataFrame, Spots_df: pd.DataFrame, c_mode: str, lut_scaling_metric: str, units: dict, *args, _extend: bool = True):

    if c_mode not in ['random colors', 'random greys', 'only-one-color', 'diferentiate replicates']:

        if lut_scaling_metric != 'Speed instantaneous':
            lut_norm_df = Tracks_df[[lut_scaling_metric]].drop_duplicates()
            _lut_scaling_metric = lut_scaling_metric
        if lut_scaling_metric == 'Speed instantaneous':
            lut_norm_df = Spots_df[['Distance']].drop_duplicates()
            _lut_scaling_metric = 'Distance'

        # Normalize the Net distance to a 0-1 range
        lut_min = lut_norm_df[_lut_scaling_metric].min()
        lut_max = lut_norm_df[_lut_scaling_metric].max()
        norm = plt.Normalize(vmin=lut_min, vmax=lut_max)

        # Get the colormap based on the selected mode
        colormap = Colors.GetCmap(c_mode)
    
        # Add a colorbar to show the LUT map
        sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        # Create a separate figure for the LUT map (colorbar)
        fig_lut, ax_lut = plt.subplots(figsize=(2, 6))
        ax_lut.axis('off')
        cbar = fig_lut.colorbar(sm, ax=ax_lut, orientation='vertical', extend='both' if _extend else 'neither')
        print(units.get(lut_scaling_metric))
        cbar.set_label(f"{lut_scaling_metric} {units.get(lut_scaling_metric)}", fontsize=10)

        return plt.gcf()

    else:
        pass



def _rgba_over_background(rgba: np.ndarray, bg: tuple[int, int, int]) -> np.ndarray:
        """Composite uint8 RGBA over an RGB background. Returns uint8 RGB."""
        if rgba.shape[-1] != 4:
            raise ValueError("expected RGBA input")
        rgb = rgba[..., :3].astype(np.float32)
        a = rgba[..., 3:4].astype(np.float32) / 255.0
        bg_rgb = np.array(bg, dtype=np.float32).reshape(1, 1, 1, 3)
        out = rgb * a + bg_rgb * (1.0 - a)
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)


class Animated:

    def create_image_stack(
        Spots_df: pd.DataFrame,
        Tracks_df: pd.DataFrame,
        conditions: list,
        replicates: list,
        *args,
        c_mode: str = "differentiate replicates",
        only_one_color: str = "blue",
        lut_scaling_metric: str = "Track displacement",
        background: str = "dark",
        smoothing_index: int | float = 0,
        lw: float = 1.0,
        units_time: str = "s",
        grid: bool = True,
        mark_heads: bool = False,
        marker: dict = {"symbol": "o", "fill": True},
        markersize: float = 5.0,
        title: str = "Track Visualization",
        frames_mode: str = "cumulative",  # 'cumulative' | 'per_frame'
        dpi: int = 100,
        units: str = "μm",
        size: tuple[int, int] = (975, 750),
        **kwargs
    ) -> np.ndarray:
        """
        Build a stack of frames from tracks, returning uint8 RGBA of shape (N, H, W, 4).
        If Spots_df/Tracks_df are not provided, falls back to a simple demo sine path.
        """
        noticequeue = kwargs.get('noticequeue', None) if 'noticequeue' in kwargs else None

        W, H = size

        # ---- Real data path (modified ImageStackTracksRealistics) ----
        Spots = Spots_df.copy()
        Tracks = Tracks_df.copy()

        required = ["Condition", "Replicate", "Track ID", "Time point", "X coordinate", "Y coordinate"]
        if any(col not in Spots.columns for col in required):
            # Shape-safe empty
            return
        
        if not conditions:
            noticequeue.Report("warning", "No conditions selected! At at least one condition must be selected.")
            return
        if not replicates:
            noticequeue.Report("warning", "No replicates selected! At at least one replicate must be selected.")
            return

        Spots = Spots.loc[Spots["Condition"].isin(conditions)].loc[Spots["Replicate"].isin(replicates)]
        Tracks = Tracks.loc[Tracks["Condition"].isin(conditions)].loc[Tracks["Replicate"].isin(replicates)]

        if Spots.empty:
            return

        # Sort and index
        key_cols = ["Condition", "Replicate", "Track ID"]
        Spots = Spots.sort_values(key_cols + ["Time point"]).set_index(key_cols, drop=True)
        Tracks = Tracks.sort_values(key_cols).set_index(key_cols, drop=True)

        # Optional smoothing
        if isinstance(smoothing_index, (int, float)) and smoothing_index > 1:
            win = int(smoothing_index)
            Spots["X coordinate"] = Spots.groupby(level=key_cols)["X coordinate"].transform(
                lambda s: s.rolling(win, min_periods=1).mean()
            )
            Spots["Y coordinate"] = Spots.groupby(level=key_cols)["Y coordinate"].transform(
                lambda s: s.rolling(win, min_periods=1).mean()
            )

        # Colors
        rng = np.random.default_rng(42)

        def _rand_color_hex():
            return mcolors.to_hex(rng.random(3))

        def _rand_grey_hex():
            g = float(rng.random())
            return mcolors.to_hex((g, g, g))

        if c_mode in ["random colors", "random greys", "only-one-color"]:
            unique_tracks = Tracks.index.unique()
            if c_mode == "random colors":
                colors = [_rand_color_hex() for _ in range(len(unique_tracks))]
            elif c_mode == "random greys":
                colors = [_rand_grey_hex() for _ in range(len(unique_tracks))]
            else:
                colors = [only_one_color] * len(unique_tracks)
            track_to_color = dict(zip(unique_tracks, colors))
            Tracks["Track color"] = [track_to_color[idx] for idx in Tracks.index]

        elif c_mode == "differentiate replicates":
            Tracks["Track color"] = Tracks["Replicate color"] if "Replicate color" in Tracks.columns else "red"

        else:
            use_instantaneous = lut_scaling_metric == "Speed instantaneous"
            if lut_scaling_metric in Tracks.columns and not use_instantaneous:
                cmap = Colors.GetCmap(c_mode)
                vmin = float(Tracks[lut_scaling_metric].min())
                vmax = float(Tracks[lut_scaling_metric].max())
                vmax = vmax if np.isfinite(vmax) and vmax > vmin else vmin + 1.0
                norm = plt.Normalize(vmin, vmax)
                Tracks["Track color"] = [mcolors.to_hex(cmap(norm(v))) for v in Tracks[lut_scaling_metric].to_numpy()]
            elif use_instantaneous:
                cmap = Colors.GetCmap(c_mode)
                g = Spots.groupby(level=key_cols)
                d = np.sqrt((g["X coordinate"].diff()) ** 2 + (g["Y coordinate"].diff()) ** 2)
                speed_end = d  # distance per step; you can scale by time externally if needed
                vmax = float(np.nanmax(speed_end.to_numpy())) if np.isfinite(speed_end.to_numpy()).any() else 1.0
                norm = plt.Normalize(0.0, vmax if vmax > 0 else 1.0)
                Spots["Spot color"] = [
                    mcolors.to_hex(Colors.GetCmap(c_mode)(norm(v))) if np.isfinite(v) else mcolors.to_hex(Colors.GetCmap(c_mode)(0.0))
                    for v in speed_end.to_numpy()
                ]

        if not (lut_scaling_metric == "Speed instantaneous"):
            Spots = Spots.join(
                Tracks[["Track color"]],
                on=key_cols,
                how="left",
                validate="many_to_one",
            )

        # Axes limits fixed over time
        x_all = Spots["X coordinate"].to_numpy(dtype=float, copy=False)
        y_all = Spots["Y coordinate"].to_numpy(dtype=float, copy=False)
        xlim = (np.nanmin(x_all), np.nanmax(x_all))
        ylim = (np.nanmin(y_all), np.nanmax(y_all))

        # Background presets
        if background == "white":
            grid_color, face_color, grid_alpha, grid_ls = "gainsboro", "white", 0.5, "-." if grid else "None"
        elif background == "light":
            grid_color, face_color, grid_alpha, grid_ls = "silver", "lightgrey", 0.5, "-." if grid else "None"
        elif background == "mid":
            grid_color, face_color, grid_alpha, grid_ls = "silver", "darkgrey", 0.5, "-." if grid else "None"
        elif background == "dark":
            grid_color, face_color, grid_alpha, grid_ls = "grey", "dimgrey", 0.5, "-." if grid else "None"
        elif background == "black":
            grid_color, face_color, grid_alpha, grid_ls = "dimgrey", "black", 0.5, "-." if grid else "None"
        else:
            grid_color, face_color, grid_alpha, grid_ls = "gainsboro", "white", 0.5, "-." if grid else "None"

        # Time points
        time_points = np.unique(Spots["Time point"].to_numpy())
        time_points.sort()

        
        frames = Spots_df["Frame"].unique()
        frames_size = frames.size

        if frames is not None and len(time_points) > frames_size:
            time_points = time_points[:frames_size]


        image_stack: list[np.ndarray] = []

        for t in time_points:
            if frames_mode == "per_frame":
                Spots_t = Spots.loc[Spots["Time point"] == t]
            else:  # cumulative
                Spots_t = Spots.loc[Spots["Time point"] <= t]

            # Build line segments
            segments = []
            seg_colors = []

            if lut_scaling_metric == "Speed instantaneous" and "Spot color" in Spots_t.columns:
                for _, g in Spots_t.groupby(level=key_cols, sort=False):
                    xy = g[["X coordinate", "Y coordinate"]].to_numpy(dtype=float, copy=False)
                    if xy.shape[0] >= 2:
                        cols = g["Spot color"].astype(str).to_numpy()
                        for i in range(1, xy.shape[0]):
                            segments.append(xy[i - 1 : i + 1])
                            seg_colors.append(cols[i])
            else:
                for _, g in Spots_t.groupby(level=key_cols, sort=False):
                    xy = g[["X coordinate", "Y coordinate"]].to_numpy(dtype=float, copy=False)
                    if xy.shape[0] >= 2:
                        segments.append(xy)
                        seg_colors.append(g["Track color"].iloc[0] if "Track color" in g.columns else "red")

            # Render
            fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xlabel(f"X coordinate {units}")
            ax.set_ylabel(f"Y coordinate {units}")
            ax.set_title(f"{title} | Time point: {t} {units_time}" if title else f"Time point: {t} {units_time}")
            ax.set_facecolor(face_color)
            if grid:
                ax.grid(True, which="both", axis="both", color=grid_color, linestyle=grid_ls, linewidth=1, alpha=grid_alpha)
            else:
                ax.grid(False)

            ax.xaxis.set_major_locator(MultipleLocator(200))
            ax.yaxis.set_major_locator(MultipleLocator(200))
            ax.xaxis.set_minor_locator(MultipleLocator(50))
            ax.yaxis.set_minor_locator(MultipleLocator(50))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            ax.tick_params(axis="both", which="major", labelsize=8)

            if segments:
                lc = LineCollection(segments, colors=seg_colors, linewidths=lw, zorder=10)
                ax.add_collection(lc)

            if mark_heads:
                ends = Spots_t.groupby(level=key_cols, sort=False).tail(1)
                if len(ends):
                    xe = ends["X coordinate"].to_numpy(dtype=float, copy=False)
                    ye = ends["Y coordinate"].to_numpy(dtype=float, copy=False)
                    if lut_scaling_metric == "Speed instantaneous" and "Spot color" in ends.columns:
                        cols = ends["Spot color"].astype(str).to_numpy()
                    else:
                        cols = ends["Track color"].astype(str).to_numpy() if "Track color" in ends.columns else np.array(["red"] * len(ends))
                    m = np.isfinite(xe) & np.isfinite(ye)
                    if m.any():
                        ax.scatter(
                            xe[m],
                            ye[m],
                            marker=marker.get("symbol", "o"),
                            s=markersize,
                            edgecolor=cols[m],
                            facecolor=cols[m] if marker.get("fill", True) else "none",
                            linewidths=lw,
                            zorder=12,
                        )

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
            plt.close(fig)
            buf.seek(0)
            im = Image.open(buf).convert("RGBA")
            image_stack.append(np.asarray(im, dtype=np.uint8))

        if not image_stack:
            return

        np_stack = np.stack(image_stack, axis=0)

        return np_stack


    
    @staticmethod
    def save_image_stack_as_mp4(
        stack: np.ndarray,
        path: str,
        fps: int = 30,
        codec: str = "libx264",
        crf: int | None = 18,
        pix_fmt: str = "yuv420p",
        background: tuple[int, int, int] = (255, 255, 255),
        bitrate: str | None = None,
    ) -> None:
        """
        Save an image stack to MP4.

        Parameters
        ----------
        stack : np.ndarray
            Frames with shape (N, H, W[, C]). C in {1, 3, 4}. uint8 preferred.
            Your create_image_stack returns (N, H, W, 4) RGBA uint8.
        path : str
            Output filename, e.g. 'movie.mp4'.
        fps : int
            Frames per second.
        codec : str
            FFmpeg video codec. 'libx264' is widely compatible.
        crf : int | None
            Constant rate factor for x264. Lower is higher quality. None to skip.
        pix_fmt : str
            Pixel format. 'yuv420p' maximizes compatibility.
        background : (R, G, B)
            Used only if input has alpha. Composites RGBA over this color.
        bitrate : str | None
            e.g. '6M'. If set, FFmpeg will target this bitrate. Usually omit when using CRF.
        """
        if stack.ndim not in (3, 4):
            raise ValueError("stack must have shape (N,H,W) or (N,H,W,C)")
        if stack.ndim == 3:
            stack = stack[..., None]  # (N,H,W,1)

        N, H, W, C = stack.shape
        if C not in (1, 3, 4):
            raise ValueError("last dimension must be 1, 3, or 4 channels")

        # Ensure uint8
        if stack.dtype != np.uint8:
            stack = np.clip(stack, 0, 255).astype(np.uint8)

        # Expand to RGB
        if C == 1:
            rgb = np.repeat(stack, 3, axis=-1)
        elif C == 3:
            rgb = stack
        else:
            # RGBA -> RGB over background
            rgb = _rgba_over_background(stack, background)

        # Build ffmpeg output params
        out_params: list[str] = ["-pix_fmt", pix_fmt]
        if crf is not None:
            out_params += ["-crf", str(crf)]
        if bitrate is not None:
            out_params += ["-b:v", str(bitrate)]

        return rgb, out_params


