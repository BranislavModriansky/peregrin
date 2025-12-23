from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from .._common import Colors, Categorizer, Values
from io import BytesIO
from ..._handlers._reports import Level
from ..._infra._selections import Metrics



class ReconstructTracks:
    
    def __init__(self, Spots_df: pd.DataFrame, Tracks_df: pd.DataFrame, *args,
                 conditions: list, replicates: list,
                 c_mode: str, only_one_color: str,
                 use_stock_palette: None | bool, stock_palette: None | str,
                 lut_scaling_stat: str, auto_lut_scaling: bool,
                 lut_vmin: None | float, lut_vmax: None | float,
                 smoothing_index: int, lw: float,
                 mark_heads: bool, marker: dict, markersize: float,
                 background: str, grid: bool,
                 title: None | str, **kwargs):

        self.Spots_df = Spots_df
        self.Tracks_df = Tracks_df
        self.conditions = conditions
        self.replicates = replicates
        self.c_mode = c_mode

        if lut_scaling_stat == 'Speed instantaneous':
            self.lut_scaling_stat = 'Distance'
        else:
            self.lut_scaling_stat = lut_scaling_stat

        self.only_one_color = only_one_color
        self.stock_palette = stock_palette if use_stock_palette else None
        self.lut_vmin = lut_vmin if not auto_lut_scaling else None
        self.lut_vmax = lut_vmax if not auto_lut_scaling else None
        self.smoothing_index = smoothing_index
        self.lw = lw
        self.mark_heads = mark_heads
        self.marker = marker
        self.markersize = markersize
        self.background = background
        self.grid = grid
        self.title = title
        self.noticequeue = kwargs.get('noticequeue', None)
        self.gridstyle = kwargs.get('gridstyle', 'simple-1')
        self.Spots, self.Tracks = None, None
        self.segments, self.segment_colors = [], []
        self.grid_color, self.face_color, self.grid_alpha, self.grid_ls = None, None, None, None

    KEY_COLS = ['Condition', 'Replicate', 'Track ID']


    def _arrange_data(self):
        Spots =  Categorizer(
            data=self.Spots_df,
            conditions=self.conditions,
            replicates=self.replicates,
            noticequeue=self.noticequeue
        )()
        Tracks = Categorizer(
            data=self.Tracks_df,
            conditions=self.conditions,
            replicates=self.replicates,
            noticequeue=self.noticequeue
        )()

        self.Spots = Spots.sort_values(['Condition', 'Replicate', 'Track ID', 'Time point'])
        self.Tracks = Tracks.sort_values(['Condition', 'Replicate', 'Track ID'])


        if list(Spots.index.names) != self.KEY_COLS:
            Spots = Spots.set_index(self.KEY_COLS)
        if list(Tracks.index.names) != self.KEY_COLS:
            Tracks = Tracks.set_index(self.KEY_COLS)

        self.Spots = Spots
        self.Tracks = Tracks

    def _convert_polar(self):
        self.Spots['X coordinate'] = self.Spots['X coordinate'] - self.Spots.groupby(level=self.KEY_COLS)['X coordinate'].transform('first')
        self.Spots['Y coordinate'] = self.Spots['Y coordinate'] - self.Spots.groupby(level=self.KEY_COLS)['Y coordinate'].transform('first')

        self.Spots['r'] = np.sqrt(self.Spots['X coordinate']**2 + self.Spots['Y coordinate']**2)
        self.Spots['theta'] = np.arctan2(self.Spots['Y coordinate'], self.Spots['X coordinate'])

    def _get_radius(self, alldata: pd.DataFrame):
        """
        Compute global maximum radius from (X, Y) positions.

        Works with both:
        - MultiIndex with levels ['Condition', 'Replicate', 'Track ID']
        - Regular Index with those columns present.
        """
        if alldata.empty:
            self.y_max_global = 100.0
            self.y_max_label_global = "100 μm"
            return

        # Choose grouping mode based on index type
        if isinstance(alldata.index, pd.MultiIndex) and list(alldata.index.names) == self.KEY_COLS:
            group = alldata.groupby(level=self.KEY_COLS)
        else:
            # Fall back to grouping by columns
            missing = [c for c in self.KEY_COLS if c not in alldata.columns]
            if missing:
                self.noticequeue.Report(
                    Level.warning,
                    'Invalid chart radius. Using default "100 μm".',
                    f"Missing grouping columns for radius computation: {missing}.",
                )
                self.y_max_global = 100.0
                self.y_max_label_global = "100 μm"
                return
            group = alldata.groupby(self.KEY_COLS)

        x = alldata['X coordinate']
        y = alldata['Y coordinate']
        x0 = group['X coordinate'].transform('first')
        y0 = group['Y coordinate'].transform('first')
        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

        self.y_max_global = r.max() + 10.0
        self.y_max_label_global = f"{round(r.max()) + 10} μm"

        if not np.isfinite(self.y_max_global):
            self.noticequeue.Report(
                Level.warning,
                'Invalid chart radius. Setting to "100 μm."',
                f"Maximum radius was not finite: {self.y_max_global}.",
            )
            self.y_max_global = 100.0
        if not (self.y_max_global > 0):
            self.noticequeue.Report(
                Level.warning,
                'Negative maximum radius. Setting to default "100 μm".',
                f"Maximum radius was a negative value: {self.y_max_global}.",
            )
            self.y_max_global = 100.0
            self.y_max_label_global = "100 μm"
        
    def _smooth(self):
        if isinstance(self.smoothing_index, (int, float)) and self.smoothing_index >= 1:
            _smoothing_window = self.smoothing_index
            if isinstance(_smoothing_window, float):
                _smoothing_window = round(self.smoothing_index)
                self.noticequeue.Report(Level.warning, f"Smoothing index rounded to nearest integer: {_smoothing_window}.", f"Invalid smoothing index type: float ({self.smoothing_index}).")
            
            self.Spots['X coordinate'] = (
                self.Spots.groupby(level=self.KEY_COLS)['X coordinate'].transform(lambda s: s.rolling(_smoothing_window, min_periods=1).mean())
            )
            self.Spots['Y coordinate'] = (
                self.Spots.groupby(level=self.KEY_COLS)['Y coordinate'].transform(lambda s: s.rolling(_smoothing_window, min_periods=1).mean())
            )
        else:
            self.noticequeue.Report(Level.warning, f"Invalid smoothing index. No smoothing applied.", f"Smoothing index must be an integer type and must be greater than 1. {type(self.smoothing_index)}: {self.smoothing_index}.")

    def _assign_colors(self):
        rng = np.random.default_rng(42)
        track_index = self.Tracks.index.unique()

        if self.c_mode == 'only-one-color':
            self.Tracks['Track color'] = mcolors.to_hex(self.only_one_color)

        elif self.c_mode in ['random colors', 'random greys']:
            if self.c_mode == 'random colors':
                cols = [mcolors.to_hex(rng.random(3)) for _ in range(len(track_index))]
            elif self.c_mode == 'random greys':
                cols = [mcolors.to_hex((float(g), float(g), float(g))) for g in rng.random(len(track_index))]
            self.Tracks['Track color'] = [dict(zip(track_index, cols))[idx] for idx in self.Tracks.index]

        elif self.c_mode in ['differentiate conditions', 'differentiate replicates']:
            if self.c_mode == 'differentiate replicates':
                category = 'replicate'
            if self.c_mode == 'differentiate conditions':
                category = 'condition'

            if self.stock_palette is None:
                mp = Colors.BuildQualPalette(self.Tracks, tag=category.capitalize(), noticequeue=self.noticequeue)
                self.Tracks = self.Tracks.reset_index()
                self.Tracks['Track color'] = self.Tracks[category.capitalize()].map(mp)
                self.Tracks = self.Tracks.set_index(['Condition', 'Replicate', 'Track ID'])
            

            if self.stock_palette is not None:
                categories = self.Tracks.reset_index()[category.capitalize()].unique().tolist()
                palette = Colors.StockQualPalette(categories, self.stock_palette, noticequeue=self.noticequeue)
                compiled = dict(zip(categories, palette))

                self.Tracks = self.Tracks.reset_index()
                self.Tracks['Track color'] = self.Tracks[category.capitalize()].map(compiled)
                self.Tracks = self.Tracks.set_index(['Condition', 'Replicate', 'Track ID'])

        else:
            self.cmap = Colors.GetCmap(self.c_mode)
            norm, vals = Values.LutMapper(self.Tracks if self.lut_scaling_stat in self.Tracks.columns else self.Spots, self.lut_scaling_stat, min=self.lut_vmin, max=self.lut_vmax, noticequeue=self.noticequeue)

            if not (norm is None or vals is None):

                if self.lut_scaling_stat in self.Tracks.columns:
                    self.Tracks['Track color'] = [mcolors.to_hex(self.cmap(norm(v))) for v in vals]

                if self.lut_scaling_stat in self.Spots.columns:
                    self.Spots['Spot color'] = [mcolors.to_hex(self.cmap(norm(v))) for v in vals]

            else:
                self.Tracks['Track color'] = mcolors.to_hex('black')
    
    def _color_tracks(self, polar: bool = False):
        """
        Build self.segments and self.segment_colors from self.Spots/self.Tracks.

        Parameters
        ----------
        polar : bool
            If False (default), use Cartesian coordinates (X/Y).
            If True, use polar coordinates (theta/r), suitable for normalized
            trajectory reconstruction plots.

        Logic:
        - If Tracks has 'Track color' but Spots doesn't, join it down.
        - If Spots has 'Spot color', color per segment (ending spot).
        - Else if Spots has 'Track color', color per whole track.
        - Else, fall back to black.
        """

        # Ensure per-track colors are available on Spots when defined on Tracks
        if 'Track color' in self.Tracks.columns and 'Track color' not in self.Spots.columns:
            self.Spots = self.Spots.join(
                self.Tracks[['Track color']],
                how='left',
                validate='many_to_one',
            )

        # Select coordinate columns based on mode
        if polar:
            coord_cols = ('theta', 'r')
        else:
            coord_cols = ('X coordinate', 'Y coordinate')

        # Convenience alias
        segments = self.segments
        seg_colors = self.segment_colors

        # Per-spot coloring (instantaneous LUTs, e.g. speed)
        if 'Spot color' in self.Spots.columns:
            for _, g in self.Spots.groupby(level=self.KEY_COLS, sort=False):
                coords = g[list(coord_cols)].to_numpy(dtype=float, copy=False)
                cols = g['Spot color'].astype(str).to_numpy()
                n = coords.shape[0]

                if n >= 2:
                    # Always build one segment per consecutive pair
                    for i in range(1, n):
                        segments.append(coords[i - 1:i + 1])
                        # Color by ending spot
                        seg_colors.append(cols[i])

        elif 'Track color' in self.Spots.columns:
            for _, g in self.Spots.groupby(level=self.KEY_COLS, sort=False):
                coords = g[list(coord_cols)].to_numpy(dtype=float, copy=False)
                if coords.shape[0] >= 2:
                    segments.append(coords)
                    seg_colors.append(g['Track color'].iloc[0])

        else:
            default_col = mcolors.to_hex('black')
            for _, g in self.Spots.groupby(level=self.KEY_COLS, sort=False):
                coords = g[list(coord_cols)].to_numpy(dtype=float, copy=False)
                if coords.shape[0] >= 2:
                    segments.append(coords)
                    seg_colors.append(default_col)

    def _background_color(self):
        if self.background == 'white':    self.face_color = 'white'
        elif self.background == 'light':  self.face_color = 'lightgrey'
        elif self.background == 'mid':    self.face_color = 'darkgrey'
        elif self.background == 'dark':   self.face_color = 'dimgrey'
        elif self.background == 'black':  self.face_color = 'black'

    def _grid_color(self, coord_system: str = 'cartesian'):

        if coord_system == 'cartesian':
            if self.background == 'white':    self.grid_color, self.grid_alpha = 'gainsboro', 0.5
            elif self.background == 'light':  self.grid_color, self.grid_alpha = 'silver', 0.5
            elif self.background == 'mid':    self.grid_color, self.grid_alpha = 'silver', 0.5
            elif self.background == 'dark':   self.grid_color, self.grid_alpha = 'grey', 0.5
            elif self.background == 'black':  self.grid_color, self.grid_alpha = 'dimgrey', 0.5

        elif coord_system == 'polar':
            if self.background == 'white':    self.grid_color, self.grid_a_alpha, self.grid_b_alpha = 'lightgrey', 0.7, 0.8
            elif self.background == 'light':  self.grid_color, self.grid_a_alpha, self.grid_b_alpha = 'darkgrey', 0.7, 0.6
            elif self.background == 'mid':    self.grid_color, self.grid_a_alpha, self.grid_b_alpha = 'dimgrey', 0.7, 0.5
            elif self.background == 'dark':   self.grid_color, self.grid_a_alpha, self.grid_b_alpha = 'grey', 0.7, 0.6
            elif self.background == 'black':  self.grid_color, self.grid_a_alpha, self.grid_b_alpha = 'dimgrey', 0.5, 0.4

    def _grid_style(self, ax: plt.Axes, coord_system: str = 'cartesian'):

        if coord_system == 'cartesian':
            self.grid_ls = '-.'
            ax.grid(True, which='both', axis='both', color=self.grid_color, linestyle=self.grid_ls, linewidth=1, alpha=self.grid_alpha)
        
        elif coord_system == 'polar':
            if self.gridstyle in ['simple-1', 'simple-2']:
                ax.xaxis.grid(True, color=self.grid_color, linestyle='-', linewidth=1, alpha=self.grid_a_alpha)
                ax.yaxis.grid(False)

                if self.gridstyle == 'simple-1':
                    for i, line in enumerate(ax.get_xgridlines()):
                        if i % 2 != 0:
                            line.set_color('none')
                if self.gridstyle == 'simple-2':
                    for i, line in enumerate(ax.get_xgridlines()):
                        if i % 2 == 0:
                            line.set_color('none')

            if self.gridstyle in ['dartboard-1', 'dartboard-2']:
                ax.grid(True, lw=0.75, color=self.grid_color, alpha=self.grid_a_alpha)
                if self.gridstyle == 'dartboard-1':
                    for i, line in enumerate(ax.get_xgridlines()):
                        if i % 2 == 0:
                            line.set_linestyle('-.'); line.set_color(self.grid_color); line.set_linewidth(0.75), line.set_alpha(self.grid_a_alpha)
                    for line in ax.get_ygridlines():
                        line.set_linestyle('--'); line.set_color(self.grid_color); line.set_linewidth(0.75), line.set_alpha(self.grid_a_alpha)

                if self.gridstyle == 'dartboard-2':
                    for i, line in enumerate(ax.get_xgridlines()):
                        if i % 2 != 0:
                            line.set_linestyle('-.'); line.set_color(self.grid_color); line.set_linewidth(0.75), line.set_alpha(self.grid_a_alpha)
                    for line in ax.get_ygridlines():
                        line.set_linestyle('--'); line.set_color(self.grid_color); line.set_linewidth(0.75), line.set_alpha(self.grid_a_alpha)
            elif self.gridstyle == 'spindle':
                ax.xaxis.grid(True, color=self.grid_color, linestyle='-', linewidth=1, alpha=self.grid_a_alpha)
                ax.yaxis.grid(False)
            elif self.gridstyle == 'radial':
                ax.xaxis.grid(False)
                ax.yaxis.grid(True, color=self.grid_color, linestyle='-', linewidth=1, alpha=self.grid_a_alpha)


    def _head_markers(self, ax, polar: bool = False):
        """
        Draw markers at track ends.

        polar=False -> use Cartesian coordinates (X/Y).
        polar=True  -> use polar coordinates (theta/r).
        """
        x_coord, y_coord = ('theta', 'r') if polar else ('X coordinate', 'Y coordinate')

        ends = self.Spots.groupby(level=self.KEY_COLS, sort=False).tail(1)
        if len(ends):
            xe = ends[x_coord].to_numpy(dtype=float, copy=False)
            ye = ends[y_coord].to_numpy(dtype=float, copy=False)

            # Prefer per-spot colors if present, else per-track, else fallback to black
            if 'Spot color' in self.Spots.columns:
                cols = ends['Spot color'].astype(str).to_numpy()
            elif 'Track color' in self.Spots.columns:
                cols = ends['Track color'].astype(str).to_numpy()
            else:
                default_col = mcolors.to_hex('black')
                cols = np.array([default_col] * len(ends), dtype=object)

            m = np.isfinite(xe) & np.isfinite(ye)
            if m.any():
                ax.scatter(
                    xe[m],
                    ye[m],
                    marker=self.marker["symbol"],
                    s=self.markersize,
                    edgecolor=cols[m],
                    facecolor=cols[m] if self.marker["fill"] else "none",
                    linewidths=self.lw,
                    zorder=12,
                )

    def _color_segments(self, ax):
        lc = LineCollection(self.segments, colors=self.segment_colors, linewidths=self.lw, zorder=10)
        ax.add_collection(lc)

    def Realistic(self) -> plt.Figure:

        self._arrange_data()    
        if self.smoothing_index is not None and self.smoothing_index > 0:
            self._smooth()
        self._assign_colors()
        self._color_tracks(polar=False)

        fig, ax = plt.subplots(figsize=(13, 10))
        if len(self.Spots):
            x = self.Spots['X coordinate'].to_numpy()
            y = self.Spots['Y coordinate'].to_numpy()
            ax.set_xlim(np.nanmin(x), np.nanmax(x))
            ax.set_ylim(np.nanmin(y), np.nanmax(y))

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X coordinate [microns]')
        ax.set_ylabel('Y coordinate [microns]')
        ax.set_title(self.title, fontsize=12)
        self._background_color(); ax.set_facecolor(self.face_color)

        # Ticks
        ax.xaxis.set_major_locator(MultipleLocator(200))
        ax.yaxis.set_major_locator(MultipleLocator(200))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(MultipleLocator(50))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.tick_params(axis='both', which='major', labelsize=8)

        if self.grid:
            self._grid_color(coord_system='cartesian')
            self._grid_style(coord_system='cartesian', ax=ax)
        else:
            ax.grid(False)
        
        self._color_segments(ax)
        
        if self.mark_heads:
            self._head_markers(ax, polar=False)

        return plt.gcf()
    
    def Normalized(self, all_data: pd.DataFrame) -> plt.Figure:

        self._arrange_data()
        if self.smoothing_index is not None and self.smoothing_index > 0:
            self._smooth()
        self._assign_colors()
        self._convert_polar()
        self._get_radius(all_data)
        self._color_tracks(polar=True)

        fig, ax = plt.subplots(figsize=(12.5, 9.5), subplot_kw={'projection': 'polar'})
        ax.set_title(self.title, fontsize=12)
        ax.set_ylim(0, self.y_max_global)        # <- global, consistent across subsets
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.spines['polar'].set_visible(False)

        self._background_color(); ax.set_facecolor(self.face_color)
        if self.grid:
            self._grid_color(coord_system='polar')
            self._grid_style(coord_system='polar', ax=ax)
        else:
            ax.grid(False)

        self._color_segments(ax)

        if self.mark_heads:
            self._head_markers(ax, polar=True)

        # Scale indicator
        ax.scatter(0, self.y_max_global + 35, color='dimgray', marker='.', s=5, clip_on=False)
        ax.text(0, self.y_max_global + 50, self.y_max_label_global, va='center',
                fontsize=10, color='dimgray', clip_on=False)

        return plt.gcf()

    def GetLutMap(self, units: dict | None = None, _extend: bool = True) -> plt.Figure | None:
        """
        Create a color guide for the current color mode:

        - For continuous LUT modes: a scalar colorbar (same scaling as tracks).
        - For 'differentiate replicates' / 'differentiate conditions':
          a qualitative legend mapping category -> color.

        Parameters
        ----------
        units : dict, optional
            Mapping from metric name (e.g. 'Net distance') to unit string
            (e.g. 'μm'). For 'Speed instantaneous', the key should be the
            original name (e.g. 'Speed instantaneous'), not 'Distance'.
        _extend : bool, optional
            Whether to extend the colorbar at both ends (continuous modes).

        Returns
        -------
        matplotlib.figure.Figure or None
            The created figure, or None if no guide is appropriate.
        """
        units = units or {}

        # Ensure data and colors are ready
        if self.Spots is None or self.Tracks is None:
            self._arrange_data()
        if 'Track color' not in self.Tracks.columns and 'Spot color' not in self.Spots.columns:
            self._assign_colors()

        # Modes where a LUT / legend is not meaningful
        if self.c_mode in ['random colors', 'random greys', 'only-one-color']:
            return None

        # Qualitative legend for categorical modes
        if self.c_mode in ['differentiate replicates', 'differentiate conditions']:
            if self.c_mode == 'differentiate replicates':
                category_col = 'Replicate'
            else:
                category_col = 'Condition'

            df = self.Tracks.reset_index()

            if category_col not in df.columns or 'Track color' not in df.columns:
                return None

            # Preserve first-seen order of categories
            seen: dict = {}
            for _, row in df[[category_col, 'Track color']].dropna().iterrows():
                key = row[category_col]
                if key not in seen:
                    seen[key] = row['Track color']

            if not seen:
                return None

            labels = list(seen.keys())
            colors = list(seen.values())

            fig, ax = plt.subplots(figsize=(2.5, 0.4 * len(labels) + 0.6))
            ax.axis('off')

            handles = [
                plt.Line2D([0], [0], color=c, lw=4)
                for c in colors
            ]
            ax.legend(
                handles,
                labels,
                loc='center left',
                frameon=False,
            )
            fig.tight_layout()
            return fig

        # Continuous LUT colorbar for other colormap modes
        norm, vals = Values.LutMapper(self.Tracks if self.lut_scaling_stat in self.Tracks.columns else self.Spots, self.lut_scaling_stat, min=self.lut_vmin, max=self.lut_vmax, noticequeue=self.noticequeue)
        if norm is None:
            return None

        colormap = Colors.GetCmap(self.c_mode)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])

        fig_lut, ax_lut = plt.subplots(figsize=(2, 6))
        ax_lut.axis('off')
        cbar = fig_lut.colorbar(
            sm,
            ax=ax_lut,
            orientation='vertical',
            extend='both' if _extend else 'neither',
        )

        # Use the user-visible metric name for labeling (e.g. 'Speed instantaneous')
        label_metric = self.lut_scaling_stat
        unit = units.get(label_metric, "")
        if unit:
            cbar.set_label(f"{label_metric} {unit}", fontsize=10)
        else:
            cbar.set_label(f"{label_metric}", fontsize=10)

        return fig_lut

# TODO: integrate Lut Map generator and animated track viz into ReconstructTracks class

# @staticmethod
# def GetLutMap(Tracks_df: pd.DataFrame, Spots_df: pd.DataFrame, c_mode: str, lut_scaling_metric: str, units: dict, *args, _extend: bool = True):

#     if c_mode not in ['random colors', 'random greys', 'only-one-color', 'diferentiate replicates', 'differentiate conditions']:

#         if lut_scaling_metric != 'Speed instantaneous':
#             lut_norm_df = Tracks_df[[lut_scaling_metric]].drop_duplicates()
#             _lut_scaling_metric = lut_scaling_metric
#         if lut_scaling_metric == 'Speed instantaneous':
#             lut_norm_df = Spots_df[['Distance']].drop_duplicates()
#             _lut_scaling_metric = 'Distance'

#         # Normalize the Net distance to a 0-1 range
#         lut_min = lut_norm_df[_lut_scaling_metric].min()
#         lut_max = lut_norm_df[_lut_scaling_metric].max()
#         norm = plt.Normalize(vmin=lut_min, vmax=lut_max)

#         # Get the colormap based on the selected mode
#         colormap = Colors.GetCmap(c_mode)
    
#         # Add a colorbar to show the LUT map
#         sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
#         sm.set_array([])
#         # Create a separate figure for the LUT map (colorbar)
#         fig_lut, ax_lut = plt.subplots(figsize=(2, 6))
#         ax_lut.axis('off')
#         cbar = fig_lut.colorbar(sm, ax=ax_lut, orientation='vertical', extend='both' if _extend else 'neither')
#         cbar.set_label(f"{lut_scaling_metric} {units.get(lut_scaling_metric)}", fontsize=10)

#         return plt.gcf()

#     else:
#         pass



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


    def GetLutMap(
        self,
        units: dict | None = None,
        _extend: bool = True
    ) -> plt.Figure | None:
        pass