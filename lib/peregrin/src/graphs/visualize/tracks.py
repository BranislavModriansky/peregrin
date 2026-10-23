from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal, Optional

import io
import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter, MultipleLocator, FormatStrFormatter
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_hex, to_rgba_array

from ..._pckg_exceptions._pckg_errors import *
from ..._pckg_exceptions._pckg_warnings import *

from ..painter import paint, scale_cmap, Dyes
from ...utils import get_aliases, ensure_polars, is_empty
from .._tooltip_toolkit import GraphFormatter
from .._tooltip_toolkit.color_formats import (
    uniform, 
    random, 
    qualitative, 
    quantitative, 
    matching
)
from ...data_handler.categorizer import categorize
from ...data_compute.data_frames import calc


plt.rcParams['font.family'] = 'monospace'


class AnimateTracks:
    """
    Growing-trajectory animation for a reconstructed set of tracks.

    Kept separate from :class:`ReconstructTracks` so the animation machinery
    stays decoupled from the static plotting pipeline. Instances are bound to
    a builder whose cached arrays and styling kw drive every frame.
    """

    def __init__(self, builder: "ReconstructTracks"):
        self._builder = builder
        self._anim = None
        self._anim_fig = None

    def animate(
        self,
        *,
        loop: bool = True,
        speed: float = 1.0,
        interval: float = 40.0,
        frames: Optional[int] = None,
        show_heads: bool = True,
        repeat_delay: float = 0.0,
        mode: str = 'auto',  # 'auto' | 'jshtml' | 'video'
    ):
        b = self._builder
        fig, ax = plt.subplots(
            figsize=(13, 10),
            subplot_kw={'projection': 'polar'} if b.align_start else {},
        )
        b._frame_axes(ax)

        plot_x, plot_y = b._plot_coords(polar=b.align_start)
        run_lengths = b._track_lengths

        if run_lengths.size == 0:
            plt.close(fig)
            return FuncAnimation(fig, lambda _f: (), frames=1, blit=False)

        max_len = int(run_lengths.max())
        n_frames = frames if frames is not None else max(2, int(max_len / max(speed, 1e-6)))
        reveal_grid = np.linspace(1, max_len, n_frames).astype(np.intp)

        seg_lens = b._within_track_seg_lengths
        seg_local_idx = (
            np.concatenate([np.arange(c) for c in seg_lens])
            if seg_lens.sum() else np.empty(0, np.intp)
        )
        seg_appear = seg_local_idx + 1

        full_colors = b._segment_colors_for_mask(np.ones(seg_appear.size, dtype=bool))
        uniform_color = isinstance(full_colors, str)

        order = np.argsort(seg_appear, kind='stable')
        seg_sorted = b.segments[order]
        colors_sorted = None if uniform_color else np.asarray(full_colors)[order]
        seg_count = np.searchsorted(seg_appear[order], reveal_grid, side='right')

        head_xy = head_colors = None
        if show_heads:
            has_pt = run_lengths > 0
            last_idx = np.minimum(reveal_grid[:, None], (run_lengths - 1)[None, :])
            head_idx = b._track_starts[None, :] + np.maximum(last_idx, 0)
            head_xy = np.stack(
                (plot_x[head_idx][:, has_pt], plot_y[head_idx][:, has_pt]), axis=-1)
            head_colors = b._colors[b._track_ends][has_pt]

        lc = LineCollection(np.empty((0, 2, 2)),
                            linewidths=b.kw.get('lw', 1), zorder=11)
        if uniform_color:
            lc.set_color(full_colors)
        ax.add_collection(lc)

        head_scatter = None
        if show_heads:
            outline = b.kw.get('outline_head', True)
            fill = b.kw.get('fill_head', False)
            head_scatter = ax.scatter(
                [], [], marker=b.kw.get('head_shape', 'o'),
                s=b.kw.get('head_size', 10),
                linewidths=b.kw.get('head_outline_width', 1.0), zorder=13)
            head_fc = head_colors if fill else 'none'
            head_ec = head_colors if outline else 'none'

        def _draw_frame(i):
            k = int(seg_count[i])
            lc.set_segments(seg_sorted[:k])
            if not uniform_color:
                lc.set_color(colors_sorted[:k])
            arts = [lc]
            if head_scatter is not None:
                head_scatter.set_offsets(head_xy[i])
                head_scatter.set_facecolor(head_fc)
                head_scatter.set_edgecolor(head_ec)
                arts.append(head_scatter)
            return arts

        anim = FuncAnimation(
            fig, _draw_frame, frames=n_frames, interval=interval,
            blit=False, repeat=loop, repeat_delay=repeat_delay,
        )
        self._anim = anim
        self._anim_fig = fig
        plt.close(fig)  # avoid a duplicate static figure in Jupyter

        if mode in ('jshtml', 'video'):
            from IPython.display import HTML
            return HTML(anim.to_jshtml() if mode == 'jshtml' else anim.to_html5_video())
        return anim


class ReconstructTracks:
    """
    Track reconstruction plot built with Matplotlib.

    Every artist is tagged with a ``gid`` (``pg:tracks``, ``pg:heads``,
    ``pg:grid``, ``pg:canvas``, ``pg:figure``) so the exported SVG can be
    edited in place by the JS tooltip (see :class:`InteractiveTracks`).
    """

    ALIASES = {
        'color':    ['color', 'colour'],
        'color_by': ['color_by', 'colour_by', 'colorby', 'colourby'],
        'grid_lw':  ['grid_linewidth', 'grid_line_width', 'grid_lw'],
        'smoothing_window': ['smoothing_window', 'smooth_window', 'smoothing_index', 'smooth_index', 'smoothing', 'smooth'],
    }

    CATEGORY_COLS = ['subsubgroup', 'subgroup', 'group', 'subset',  'set']

    def __init__(self):
        self.figure: Optional[plt.Figure] = None
        self.kw: dict[str, Any] = {}
        self._segments = np.empty((0, 2, 2), dtype=float)
        self._animator: Optional[AnimateTracks] = None
        self._single_color: Optional[str] = 'black'
        self._colors: Optional[np.ndarray] = None
        # Per-spot normalized values + colormap when color_by is quantitative,
        # so the tooltip can offer a client-side colormap selector.
        self._lut_values: Optional[np.ndarray] = None
        self._lut_cmap: Optional[str] = None

    def reconstruct(
        self,
        data: pl.DataFrame,
        categories: Optional[dict[str, list[Any]]] = None,
        *,
        align_start: bool = False,
        fmt: Literal['png', 'svg', 'html'] = 'png',
        **kw,
    ):
        """
        Reconstruct tracks from the given spots DataFrame.

        Parameters
        ----------
        df : pl.DataFrame
            Output DataFrame of the `calc.spots` function.
        align_start : bool, default False
            Whether to align tracks at their start.
        categories : dict[str, list[Any]], optional
            A gate specifying for which data categories the tracks are going to be reconstructed, by default None.
        fmt : {'png', 'svg', 'html'}, optional
            Figure output format, by default 'png'. Possible values include 'png', 'svg' and 'html'.
            If 'html' is passed as `fmt`, an interactive version of the plot will be generated.
        
            *additional keyword arguments:*

        smoothing_window : int, optional
            The size of the smoothing window applied to the tracks, by default None. Must be a positive integer larger than 1. 
            The smoothing window is used to average the positions of the tracks over the given number 
            (`smoothing_window`) of frames.
        
        Returns
        -------
        ReconstructTracks or InteractiveTracks results
            The reconstructed tracks object or an interactive version if fmt is 'interactive' or 'html'.
        """
        data = ensure_polars(data).clone() if data is not None else pl.DataFrame()
        self.align_start = align_start
        self.fmt = fmt.lower()
        self.kw = get_aliases(kw, self.ALIASES)

        if is_empty(data):
            return GraphFormatter(self)
        if categories:
            data = categorize(data, categories)
        
        data = data.sort(['track_uid', 'time_point'])
        self._cache_arrays(data)

        smoothing = self.kw.get('smoothing_window', None)
        if smoothing is not None:
            self._smooth(smoothing)

        self._assign_color(data)

        self.figure = self.polar(data) if self.align_start else self.cartesian(data)

        if kw.get('animate', False):
            return self.animate()

        return GraphFormatter(self)

    @property
    def segments(self) -> np.ndarray:
        return self._segments

    @property
    def style_payload(self) -> dict[str, Any]:
        kw = self.kw
        spec = self._color_specs
        return {
            'gid': {
                'tracks': {
                    'type': 'line', 
                    'label': 'Track',
                    'color': {**spec, **self._per_node(spec, self._same_track)},
                    'values': {'width': float(kw.get('lw', 1.0))}},
                'heads': {
                    'type':  'scatter', 
                    'label': 'Track head', 
                    'color': {**matching('tracks'), **self._per_node(spec, self._track_ends)},
                    'values': {'size': float(kw.get('head_size', 1.0))}
                },
                'grid': {
                    'type':  'grid', 
                    'label': 'Grid',
                    'color': uniform(self.kw.get('grid_color', 'gainsboro')),
                    'values': {'width': float(self.kw.get('grid_lw', 0.75))}
                },
                'canvas': {
                    'type': 'background', 'label': 'Background',
                    'color': uniform(self.kw.get('face_color', 'white'))
                }
            }
        }

    _PER_NODE = ('values', 'indices')

    @classmethod
    def _per_node(cls, spec: dict[str, Any], idx: np.ndarray) -> dict[str, Any]:
        """Slice a descriptor's per-spot arrays down to the nodes actually drawn (`idx` is a mask or index array)."""
        return {k: np.asarray(spec[k])[idx].tolist() for k in cls._PER_NODE if k in spec}

    @staticmethod
    def _get_data_categories(df: pl.DataFrame, color_by: str | list[str]) -> pl.Series:
        """Category label per row; multiple columns are joined with '.', as in the painter's category keys."""
        if isinstance(color_by, list):
            return df.select(pl.concat_str(color_by, separator='.')).to_series()
        return df[color_by]

    @staticmethod
    def _categorical_colors(labels: pl.Series, mapping: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Per-label RGBA colors (n, 4) and category indices; unmapped/null labels get black and index len(mapping)."""
        k = len(mapping)
        idx = (
            labels.cast(pl.Utf8).fill_null('')
            .replace_strict({str(key): i for i, key in enumerate(mapping)}, default=k)
            .to_numpy()
        )
        palette = np.vstack([to_rgba_array(list(mapping.values())), [0, 0, 0, 1]])
        return palette[idx], idx

    def animate(self, **kw):
        self._animator = AnimateTracks(self)
        return self._animator.animate(**kw)
    

    def _cache_arrays(self, data: pl.DataFrame):
        """Materialize plotting arrays and track run-boundaries once, after sorting."""
        self._x = data['x_coordinate'].cast(pl.Float64).to_numpy()
        self._y = data['y_coordinate'].cast(pl.Float64).to_numpy()
        
        n = self._x.size
        if n == 0:
            empty = np.empty(0, dtype=np.intp)
            self._same_track = np.empty(0, dtype=bool)
            self._track_starts = self._track_ends = self._track_lengths = empty
            self._within_track_seg_lengths = empty
            self._track_uids = np.empty(0, dtype=object)
            return

        # (Track-lengths) run ids
        run_id = data['track_uid'].rle_id().to_numpy()  # rle_id(): Get a distinct integer ID for each run of identical values.

        # Boolean mask (shape: (n,)) - at start of each track = False
        same_track = np.empty(n, dtype=bool)
        same_track[0] = False
        np.equal(run_id[1:], run_id[:-1], out=same_track[1:])
        self._same_track = same_track

        # Mask inversion (at start of each track = True) -> np.flatnonzero() to get track start indices
        starts = np.flatnonzero(~same_track)
        self._track_starts = starts
        self._track_ends = np.append(starts[1:] - 1, n - 1)
        self._track_lengths = np.diff(starts, append=n)
        self._within_track_seg_lengths = np.maximum(self._track_lengths - 1, 0)

        # gather only the n_tracks uids instead of converting the whole column
        self._track_uids = data['track_uid'].gather(starts).to_numpy()


    def _smooth(self, window: int) -> pl.DataFrame:
        if not isinstance(window, int) or window <= 1:
            try:
                window = int(window)
                if window <= 1:
                    warnings.warn(
                        f"<smoothing_window> must be greater than 1. <smoothing_window={window}> has no effect.", category=UserWarning)
                warnings.warn(
                    f"Converted <smoothing_window> from ({type(window)}: {window}) to ({type(window)}: {window}). "
                    "<smoothing_window> must be a positive integer.", category=UserWarning)
            except Exception:
                warnings.warn(
                    f"Invalid <smoothing_window> type: {type(window)}. <smoothing_window> must be a positive integer. "
                    "No smoothing applied.", category=UserWarning)

        starts_idxs, local_idxs, local_trail_idxs, global_trail_idxs = self._smooth_indexing(window)

        self._x = self._corrected_rolling_mean(self._x, starts_idxs, local_idxs, local_trail_idxs, global_trail_idxs)
        self._y = self._corrected_rolling_mean(self._y, starts_idxs, local_idxs, local_trail_idxs, global_trail_idxs)

    def _smooth_indexing(self, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Trailing rolling-mean smoothing of every track at once, then a linear
        correction per track so its first and last points keep their original values.
        """
        values = self._x
        
        if values.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # starts indexes array - each start index repeated for each point in its track
        starts_idxs = np.repeat(self._track_starts, self._track_lengths)  
        # local position (index) along each track for every point
        local_idxs = np.arange(values.size) - starts_idxs
        # lowest local index for the rolling window (trailing edge of the window). It doesn't matter from which point we start 
        # the window, as long as it doesn't cross the track boundary. The track always starts at index 0 so it's way better to 
        # find the max local index rather than looking the min as tracks may have different lengths.
        local_trail_idxs = np.maximum(0, local_idxs - window + 1)
        # Trailing edge of the rolling window (lowest local track index) applied to coordinate point indexes.
        global_trail_idxs = starts_idxs + local_trail_idxs
        
        return starts_idxs, local_idxs, local_trail_idxs, global_trail_idxs

    def _corrected_rolling_mean(self, values: np.ndarray, starts_idxs: np.ndarray, local_idxs: np.ndarray, local_trail_idxs: np.ndarray, global_trail_idxs: np.ndarray):
        # Windowed mean from one global cumulative sum. Values are centred on 
        # their track's first point so the cumsum stays small and precise.
        origin = values[starts_idxs]
        csum = np.concatenate(([0.0], np.cumsum(values - origin)))
        idx = np.arange(values.size)
        smoothed = (csum[idx + 1] - csum[global_trail_idxs]) / (local_idxs + 1 - local_trail_idxs) + origin

        d0 = np.repeat(values[self._track_starts] - smoothed[self._track_starts], self._track_lengths)
        d1 = np.repeat(values[self._track_ends] - smoothed[self._track_ends], self._track_lengths)
        t = local_idxs / np.repeat(np.maximum(self._track_lengths - 1, 1), self._track_lengths)  # max(...,1) guards 1-point tracks
        return smoothed + d0 * (1.0 - t) + d1 * t


    # ---- colors ----------------------------------------------------------------
    
    def _assign_color(self, data: pl.DataFrame):
        color = self.kw.get('color', 'default')
        color_by = self.kw.get('color_by', None)

        default_cats = calc.DEFAULT_CATEGORIES[1:]
        if ((isinstance(color_by, list) and all(col in default_cats for col in color_by)) or 
            (isinstance(color_by, str) and color_by in default_cats)):
            for i, col in enumerate(default_cats):
                if col in color_by:
                    color_by = default_cats[i:]
                    break

        if isinstance(color_by, list):
            to_track_segments = all(col in data.columns for col in color_by)
            to_whole_tracks = not to_track_segments and all(col in calc.COLUMNS['TRACKS'] for col in color_by)
        else:  
            to_track_segments = color_by is not None and color_by in data.columns
            to_whole_tracks = not to_track_segments and color_by in calc.COLUMNS['TRACKS']

        if to_whole_tracks:
            track_data = calc.tracks(data, color_by)
            c = paint(
                track_data, 
                color_by=color_by, 
                color=color, 
                n=self._track_starts.size,
                get_dtype=True,
                get_params=True
            )
        else:
            c = paint(
                data, 
                color_by=color_by, 
                color=color, 
                n=self._track_starts.size,
                get_dtype=True,
                get_params=True
            )

        if isinstance(c, tuple) and len(c) == 2:
            self._colortype, c = c

        if color in ('random', 'random greys'):
            self._colortype = 'random'
            self._colors = np.repeat(c, self._track_lengths, axis=0)
            self._color_specs = random()
        elif to_track_segments:
            if self._colortype == 'numeric':
                self._colors = c['colors']
                self._color_specs = quantitative(c['cmap'], c['vals'], c['norm'])
            else:
                self._colors, idx = self._categorical_colors(self._get_data_categories(data, color_by), c)
                self._color_specs = qualitative(c.items(), idx)
        elif to_whole_tracks:
            if self._colortype == 'numeric':
                self._colors = np.repeat(c['colors'], self._track_lengths, axis=0)
                self._color_specs = quantitative(
                    c['cmap'], np.repeat(c['vals'], self._track_lengths), c['norm'])
            else:
                colors, idx = self._categorical_colors(self._get_data_categories(track_data, color_by), c)
                self._colors = np.repeat(colors, self._track_lengths, axis=0)
                self._color_specs = qualitative(c.items(), np.repeat(idx, self._track_lengths))
        elif isinstance(c, str):
            self._colortype = 'uniform'
            self._colors = np.repeat(c, data.height, axis=0)
            self._color_specs = uniform(c)
        else:
            raise PlottingError(
                f"Unexpected return value from paint: {c}"
            )
        

    # ---- static figures ----------------------------------------------------------
    def cartesian(self, data) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(13, 10))
        fig.patch.set_gid('pg:figure')
        ax.patch.set_gid('pg:canvas')

        self._build_tracks(ax)

        if self._x.size:
            gap = (self._x.max() - self._x.min()) * 0.025
            ax.set_xlim(self._x.min() - gap, self._x.max() + gap)
            ax.set_ylim(self._y.min() - gap, self._y.max() + gap)

        ax.set_aspect('equal', adjustable='box')
        text_color = self.kw.get('text_color', 'black')
        ax.set_xlabel('x coordinate [µm]', color=text_color, fontsize=12, labelpad=15)
        ax.set_ylabel('y coordinate [µm]', color=text_color, fontsize=12, labelpad=10)
        ax.set_title(self.kw.get('title', ''), fontsize=12, color=text_color)

        ax.xaxis.set_major_locator(MultipleLocator(200))
        ax.yaxis.set_major_locator(MultipleLocator(200))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(MultipleLocator(50))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.tick_params(axis='both', which='major', labelsize=9,
                       colors=self.kw.get('annotation_color', 'black'))
        for spine in ax.spines.values():
            spine.set_color(self.kw.get('frame_color', 'black'))

        if self.fmt in ('html', 'interactive'):
            fig.text(0.835, 0.885, 'right click to interact', ha='right', va='bottom', color="#4d83e3", fontsize=9)

        self._finish_axes(fig, ax, data, polar=False)
        return fig

    def polar(self, data) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12.5, 9.5), subplot_kw={'projection': 'polar'})
        fig.patch.set_gid('pg:figure')
        ax.patch.set_gid('pg:canvas')

        text_color = self.kw.get('text_color', 'black')
        ax.set_title(self.kw.get('title', ''), fontsize=12, color=text_color)
        ax.set_ylim(0, self._max_radius() + 100.0)
        ax.spines['polar'].set_visible(False)

        ax.set_ylabel('radius [µm]', color=text_color, fontsize=12, labelpad=35)
        # ax.set_xlabel('angle [°]', color=text_color, fontsize=12, labelpad=10)

        self._build_tracks(ax, polar=True)

        if self.fmt in ('html', 'interactive'):
            fig.text(0.84, 0.91, 'right click to interact', ha='right', va='top', color="#4d83e3", fontsize=9)
                            
        self._finish_axes(fig, ax, data, polar=True)
        return fig

    def _finish_axes(self, fig, ax, data, *, polar: bool):
        if self.kw.get('grid', True):
            self._style_grid(ax)
        else:
            ax.grid(False)
        if self.kw.get('show_heads', True):
            self._head_markers(ax, polar=polar)
        if self.kw.get('hide_backdrop', False):
            fig.set_facecolor('none')
        if self.kw.get('annotate', None) is not None:
            self._annotate(data, self.kw['annotate'], ax)

    def _frame_axes(self, ax):
        """Minimal axes framing reused by the animator."""
        if self.kw.get('grid', True):
            self._style_grid(ax)

    def _max_radius(self) -> float:
        if self._x.size == 0:
            return 0.0
        x0 = np.repeat(self._x[self._track_starts], self._track_lengths)
        y0 = np.repeat(self._y[self._track_starts], self._track_lengths)
        return float(np.hypot(self._x - x0, self._y - y0).max())

    # ---- geometry / drawing --------------------------------------------------------
    def _plot_coords(self, *, polar: bool = False):
        """Return per-spot plotting coordinates (polar-transformed if requested)."""
        x, y = self._x, self._y

        if not polar:
            return self._x, self._y

        x0 = np.repeat(x[self._track_starts], self._track_lengths)
        y0 = np.repeat(y[self._track_starts], self._track_lengths)
        dx, dy = x - x0, y - y0
        return np.arctan2(dy, dx), np.hypot(dx, dy)


    def _build_tracks(self, ax: plt.Axes, *, polar: bool = False):
        """Plot all track segments as one gid-tagged LineCollection."""
        if self._x.size < 2:
            self._segments = np.empty((0, 2, 2), dtype=float)
            self._seg_mask = np.empty(0, dtype=bool)
            return

        plot_x, plot_y = self._plot_coords(polar=polar)

        # A segment ends at every point that continues a track; its start is the
        # preceding point. Index directly instead of building n segments and
        # discarding the n_tracks that cross track boundaries.
        end = np.flatnonzero(self._same_track)
        start = end - 1

        segs = np.empty((end.size, 2, 2), dtype=float)
        segs[:, 0, 0] = plot_x[start]
        segs[:, 0, 1] = plot_y[start]
        segs[:, 1, 0] = plot_x[end]
        segs[:, 1, 1] = plot_y[end]

        self._seg_mask = self._same_track
        self._segments = segs

        lc = LineCollection(
            segs,
            colors=self._segment_color_arg(self._same_track),
            linewidths=self.kw.get('lw', 1),
        )
        lc.set_gid('pg:tracks')
        ax.add_collection(lc)


    def _segment_color_arg(self, same: np.ndarray):
        seg_colors = self._colors[same]
        if seg_colors.size and self._all_same_color(seg_colors):
            return seg_colors[0]
        return seg_colors

    def _segment_colors_for_mask(self, mask: np.ndarray):
        """Colors for the within-track segments selected by `mask` (animation)."""
        return self._colors[self._seg_mask][mask]

    @staticmethod
    def _all_same_color(colors: np.ndarray) -> bool:
        first = colors[0]
        if colors.dtype != object:
            return bool((colors == first).all())
        return all(np.array_equal(c, first) for c in colors)

    def _head_markers(self, ax: plt.Axes, *, polar: bool = False):
        """Draw one gid-tagged marker at the last spot of each track."""
        ends = self._track_ends
        if ends.size == 0:
            return

        colors = self._colors[ends]
        
        if polar:
            dx = self._x[ends] - self._x[self._track_starts]
            dy = self._y[ends] - self._y[self._track_starts]
            x_pos, y_pos = np.arctan2(dy, dx), np.hypot(dx, dy)
        else:
            x_pos, y_pos = self._x[ends], self._y[ends]

        scatter = ax.scatter(
            x_pos, y_pos,
            marker=self.kw.get('head_shape', 'o'),
            s=self.kw.get('head_size', 10),
            edgecolor=colors,
            facecolor=colors,
            linewidths=self.kw.get('head_outline_width', 1.0),
            zorder=12,
        )
        scatter.set_gid('pg:heads')


    def _annotate(self, data: pl.DataFrame, annotate: str | list[str], ax: plt.Axes):
        annotate_offset = self.kw.get('annotate_offset', 2.5)
        fontsize = self.kw.get('annotate_fontsize', 9)

        if isinstance(annotate, str):
            annotate = [annotate]
        annotate = [ann for ann in calc.DEFAULT_CATEGORIES if ann in annotate]  # Maintains order of DEFAULT_CATEGORIES

        for ann_index, ann in enumerate(annotate): 
            if ann not in calc.DEFAULT_CATEGORIES:
                warnings.warn(f"<annotate={ann}> was not found. For annotation, only select any of [{(', ').join(calc.DEFAULT_CATEGORIES)}].")
                return

            lookup = dict(zip(data['track_uid'].to_list(), data[ann].to_list()))

            for x, y, c, uid in zip(
                self._x[self._track_ends],
                self._y[self._track_ends],
                self._colors[self._track_ends],
                self._track_uids
            ):
                if uid not in lookup:
                    continue

                ax.annotate(
                    text=str(lookup[uid]),
                    xy=(x, y),
                    xytext=(
                        annotate_offset, 
                        annotate_offset + ann_index * (fontsize + 2)
                    ),  # stack annotations columns vertically
                    textcoords='offset points',
                    fontsize=fontsize,
                    color=c,
                    clip_on=True,
                    zorder=14,
                )

    # ---- styling -------------------------------------------------------------
    def _style_grid(self, ax: plt.Axes):
        grid_color = self.kw.get('grid_color', 'gainsboro')
        grid_lw = self.kw.get('grid_lw', 0.75)
        grid_ls = self.kw.get('grid_ls', '-')

        if not self.align_start:
            ax.grid(True, which='both', axis='both',
                    color=grid_color, linestyle=grid_ls, linewidth=grid_lw)
        else:
            ax.grid(True, lw=grid_lw, color=grid_color)

            get = lambda key, fallback: self.kw.get(key) or fallback
            for i, line in enumerate(ax.get_xgridlines()):
                kind = 'cardinal' if i % 2 == 0 else 'diagonal'
                line.set_linestyle(get(f'grid_ls_{kind}', grid_ls))
                line.set_color(get(f'grid_color_{kind}', grid_color))
            for line in ax.get_ygridlines():
                line.set_linestyle(get('grid_ls_radial', grid_ls))
                line.set_color(get('grid_color_radial', grid_color))

        for line in (*ax.get_xgridlines(), *ax.get_ygridlines()):
            line.set_gid('pg:grid')
        for ax_ in (ax.xaxis, ax.yaxis):
            for tick_ in ax_.get_minor_ticks():
                tick_.gridline.set_gid('pg:grid')


reconstruct = ReconstructTracks().reconstruct