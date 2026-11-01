from __future__ import annotations

from typing import Any, Optional

import warnings

import numpy as np
import polars as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.animation import FuncAnimation, HTMLWriter
from matplotlib.colors import to_rgba, to_rgba_array

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


class _Backdrop(Artist):
    """Pre-rendered figure pixels drawn 1:1 (no resampling, unlike ``figimage``)."""

    def __init__(self, rgba: np.ndarray):
        super().__init__()
        self._rgba = np.ascontiguousarray(rgba[::-1])  # renderer expects bottom row first
        self.set_zorder(-1)

    def draw(self, renderer):
        gc = renderer.new_gc()
        renderer.draw_image(gc, 0, 0, self._rgba)
        gc.restore()


class _TrackAnimation(FuncAnimation):
    """
    FuncAnimation that skips its post-frame ``canvas.draw_idle()`` while saving
    (``to_jshtml`` / ``to_html5_video``): the writer's ``savefig`` renders every frame
    anyway, so each frame would otherwise be drawn twice.
    """

    _saving = False

    def save(self, *args, **kwargs):
        self._saving = True
        try:
            return super().save(*args, **kwargs)
        finally:
            self._saving = False

    def _post_draw(self, framedata, blit):
        if not self._saving:
            super()._post_draw(framedata, blit)


class _RevealingLineCollection(LineCollection):
    """
    LineCollection whose Paths are built once; each animation frame only exposes a
    prefix of them. ``set_segments()`` would rebuild one Path per segment per frame.
    Colors are kept in the same order as the paths, so a prefix of paths draws with
    the matching prefix of colors.
    """

    def __init__(self, segments, **kw):
        super().__init__(segments, **kw)
        self._all_paths = self._paths

    def reveal(self, k: int):
        self._paths = self._all_paths[:k]
        self.stale = True



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
        self._colors: Optional[np.ndarray] = None
        self._colortype: Optional[str] = None
        self._labels: Optional[list[str]] = None
        self._segments = np.empty((0, 2, 2), dtype=float)

    def reconstruct(
        self,
        data: pl.DataFrame,
        categories: Optional[dict[str, list[Any]]] = None,
        *,
        align_start: bool = False,
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
        GraphFormatter
            A GraphFormatter object containing the generated graph.
        """
        data = ensure_polars(data).clone() if data is not None else pl.DataFrame()
        self.align_start = align_start
        self.kw = get_aliases(kw, self.ALIASES)
        self.show_heads = self.kw.get('show_heads', True)

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
        self._build_geometry()   # every array used by figure, html and animate() is computed here, once
        self._labels = self._annotation_labels(data)

        fig, ax = self._new_axes()
        self._build_tracks(ax)
        self._finish_axes(ax)
        self.figure = fig
        return GraphFormatter(self)


    # ---------------------------------------------------------------------------
    # 2. Shared axes setup (limits, aspect, ticks, labels, title)
    # ---------------------------------------------------------------------------
    def _setup_axes(self, ax: plt.Axes):
        text_color = self.kw.get('text_color', 'black')
        ax.set_title(self.kw.get('title', ''), fontsize=12, color=text_color)

        if self.align_start:
            ax.set_ylim(0, self._max_radius() + 100.0)
            ax.spines['polar'].set_visible(False)
            ax.set_ylabel('radius [µm]', color=text_color, fontsize=12, labelpad=35)
            return

        if self._x.size:
            gap = (self._x.max() - self._x.min()) * 0.025
            ax.set_xlim(self._x.min() - gap, self._x.max() + gap)
            ax.set_ylim(self._y.min() - gap, self._y.max() + gap)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x coordinate [µm]', color=text_color, fontsize=12, labelpad=15)
        ax.set_ylabel('y coordinate [µm]', color=text_color, fontsize=12, labelpad=10)

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

        return ax


    # ---- animation ---------------------------------------------------------
    def animate(
        self,
        *,
        loop: bool = True,
        speed: float = 1.0,
        interval: float = 40.0,
        frames: Optional[int] = None,
        max_frames: Optional[int] = 200,
        repeat_delay: float = 0.0,
        dpi: Optional[float] = None,
        **_,
    ) -> FuncAnimation:
        """
        Build a FuncAnimation revealing the tracks over time.

        All geometry comes from the arrays cached in ``reconstruct()``; per frame
        only a path-list slice and the head offsets change.
        """
        if self._x.size == 0:
            raise RuntimeError("Call reconstruct() with non-empty data before animate().")

        starts, last = self._track_starts, self._track_lengths - 1
        max_len = int(self._track_lengths.max())

        n_frames = frames if frames is not None else max(2, int(max_len / max(speed, 1e-6)))
        if max_frames is not None:
            n_frames = min(n_frames, max_frames)
        reveal = np.linspace(0, max(max_len - 1, 1), n_frames).round().astype(np.intp)

        # Segments ordered by the step at which they appear -> each frame shows a prefix.
        order = np.argsort(self._seg_step, kind='stable')
        seg_count = np.searchsorted(self._seg_step[order], reveal, side='right')
        segments = self._segments[order]

        # Fresh figure; its static background (axes, ticks, grid, labels) is rendered once
        # and replayed as a single image, so frames only draw the image + tracks + heads.
        fig, ax = self._new_axes(dpi=dpi)
        fig.canvas.draw()
        background = np.asarray(fig.canvas.buffer_rgba()).copy()
        fig.patch.set_visible(False)
        for artist in ax.get_children():
            artist.set_visible(False)   # ax.patch stays the (invisible) clip path for the tracks
        fig.add_artist(_Backdrop(background))

        segments, transform = self._projected(ax, segments)
        lc = _RevealingLineCollection(
            segments,
            colors=self._seg_colors[:1] if self._uniform else self._seg_colors[order],
            linewidths=self.kw.get('lw', 1),
            transform=transform,
            zorder=11,
        )
        lc.reveal(0)
        ax.add_collection(lc, autolim=False)
        artists = [lc]

        # Head positions of every spot, projected once; a frame just indexes into them.
        spot_xy, point_transform = self._projected(ax, np.column_stack((self._px, self._py)))
        head_idx = starts.copy()

        heads = None
        if self.show_heads:
            heads = self._head_markers(
                ax, spot_xy[head_idx], point_transform, zorder=13,
                fill=self.kw.get('fill_head', True), outline=self.kw.get('outline_head', True),
            )
            artists.append(heads)

        labels = self._annotate(ax, spot_xy[head_idx], point_transform)
        artists.extend(labels)

        def _draw_frame(i: int):
            nonlocal head_idx
            lc.reveal(int(seg_count[i]))
            idx = starts + np.minimum(reveal[i], last)
            xy = spot_xy[idx]
            if heads is not None:
                heads.set_offsets(xy)
            if labels:
                # only labels whose head moved since the last drawn frame (finished tracks stay put)
                for j in np.flatnonzero(idx != head_idx):
                    labels[j].xy = xy[j]
            head_idx = idx
            return artists

        anim = _TrackAnimation(
            fig, _draw_frame, frames=n_frames, init_func=lambda: artists,
            interval=interval, repeat=loop, repeat_delay=repeat_delay,
            blit=False, cache_frame_data=False,
        )
        plt.close(fig)  # avoid a duplicate static figure in Jupyter
        return anim


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

    @classmethod
    def _per_node(cls, spec: dict[str, Any], idx: np.ndarray) -> dict[str, Any]:
        """Slice a descriptor's per-spot arrays down to the nodes actually drawn (`idx` is a mask or index array)."""
        return {k: np.asarray(spec[k])[idx].tolist() for k in ('values', 'indices') if k in spec}

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
            self._local = empty
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

        # local position (index) along its track for every point; reused by smoothing and animation
        self._local = np.arange(n) - np.repeat(starts, self._track_lengths)


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
        # local position (index) along each track for every point (cached in _cache_arrays)
        local_idxs = self._local
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


    def _build_geometry(self):
        """Plot coordinates, segments and their RGBA colors, computed once per reconstruction."""
        self._px, self._py = self._plot_coords(polar=self.align_start)

        # A segment ends at every point that continues a track; its start is the preceding point.
        end = np.flatnonzero(self._same_track)
        segs = np.empty((end.size, 2, 2), dtype=float)
        segs[:, 0, 0], segs[:, 0, 1] = self._px[end - 1], self._py[end - 1]
        segs[:, 1, 0], segs[:, 1, 1] = self._px[end], self._py[end]
        self._segments = segs

        self._seg_colors = self._colors[self._same_track]
        self._seg_step = self._local[self._same_track]   # animation step at which a segment appears
        self._uniform = self._colortype == 'uniform' or (
            self._seg_colors.size > 0 and bool((self._seg_colors == self._seg_colors[0]).all())
        )
        self._head_colors = self._colors[self._track_ends]

    # ---- colors ----------------------------------------------------------------
    
    def _assign_color(self, data: pl.DataFrame):
        color = self.kw.get('color', 'default')
        color_by = self.kw.get('color_by', None)

        if color_by is not None and (color_by not in data.columns and color_by not in calc.COLUMNS['TRACKS']):
            raise ValueError(f"<color_by={color_by}> not found. <color_by> must be one of {', '.join(data.columns) + ', ' + ', '.join(calc.COLUMNS['TRACKS'])}")

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

        # All branches produce float RGBA (n_spots, 4); strings are parsed once (per track where possible).
        if color in ('random', 'random greys'):
            self._colortype = 'random'
            self._colors = np.repeat(to_rgba_array(c), self._track_lengths, axis=0)
            self._color_specs = random()
        elif to_track_segments:
            if self._colortype == 'numeric':
                self._colors = to_rgba_array(c['colors'])
                self._color_specs = quantitative(c['cmap'], c['vals'], c['norm'])
            else:
                self._colors, idx = self._categorical_colors(self._get_data_categories(data, color_by), c)
                self._color_specs = qualitative(c.items(), idx)
        elif to_whole_tracks:
            if self._colortype == 'numeric':
                self._colors = np.repeat(to_rgba_array(c['colors']), self._track_lengths, axis=0)
                self._color_specs = quantitative(c['cmap'], np.repeat(c['vals'], self._track_lengths), c['norm'])
            else:
                colors, idx = self._categorical_colors(self._get_data_categories(track_data, color_by), c)
                self._colors = np.repeat(colors, self._track_lengths, axis=0)
                self._color_specs = qualitative(c.items(), np.repeat(idx, self._track_lengths))
        elif isinstance(c, str):
            self._colortype = 'uniform'
            self._colors = np.broadcast_to(to_rgba(c), (data.height, 4))
            self._color_specs = uniform(c)
        else:
            raise PlottingError(f"Unexpected return value from paint: {c}")


    def _new_axes(self, dpi=None):
        fig, ax = plt.subplots(
            figsize=(12, 10), dpi=dpi or self.kw.get('dpi', 100),
            subplot_kw={'projection': 'polar'} if self.align_start else {},
        )
        fig.patch.set_gid('pg:figure')
        ax.patch.set_gid('pg:canvas')
        self._setup_axes(ax)          # return value ignored -> fixes polar crash
        self._style_axes(fig, ax)
        return fig, ax
        

    def _style_axes(self, fig, ax):
        if self.kw.get('grid', True):
            self._style_grid(ax)
        else:
            ax.grid(False)
        if self.kw.get('hide_backdrop', False):
            fig.set_facecolor('none')
        return fig, ax

    def _finish_axes(self, ax):
        ends = self._track_ends
        if not ends.size:
            return
        xy, transform = self._projected(ax, np.column_stack((self._px[ends], self._py[ends])))
        if self.show_heads:
            self._head_markers(ax, xy, transform, zorder=12).set_gid('pg:heads')
        self._annotate(ax, xy, transform)


    def _max_radius(self) -> float:
        return float(self._py.max()) if self._py.size else 0.0   # polar py == radius


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


    # ---- static drawing uses the cached geometry ---------------------------
    def _build_tracks(self, ax):
        """Plot all track segments as one gid-tagged LineCollection."""
        if not self._segments.size:
            return
        segments, transform = self._projected(ax, self._segments)
        lc = LineCollection(
            segments,
            colors=self._seg_colors[:1] if self._uniform else self._seg_colors,
            linewidths=self.kw.get('lw', 1),
            transform=transform,
        )
        lc.set_gid('pg:tracks')
        ax.add_collection(lc, autolim=False)

    def _projected(self, ax: plt.Axes, points: np.ndarray):
        """
        Coordinates (any shape ``(..., 2)``) + transform for artists on `ax`. On polar axes
        the non-affine projection is applied to all points at once; otherwise Matplotlib
        re-projects every path / annotation individually, in Python, on each draw.
        """
        if not self.align_start:
            return points, ax.transData
        projected = (ax.transScale + ax.transShift + ax.transProjection).transform(
            points.reshape(-1, 2)).reshape(points.shape)
        return projected, ax.transProjectionAffine + ax.transWedge + ax.transAxes

    def _head_markers(self, ax: plt.Axes, xy: np.ndarray, transform, *, zorder: float, fill: bool = True, outline: bool = True):
        """One marker per track at `xy` (already in `transform` coordinates)."""
        return ax.scatter(
            xy[:, 0], xy[:, 1],
            marker=self.kw.get('head_shape', 'o'),
            s=self.kw.get('head_size', 10),
            facecolors=self._head_colors if fill else 'none',
            edgecolors=self._head_colors if outline else 'none',
            linewidths=self.kw.get('head_outline_width', 1.0),
            transform=transform,
            zorder=zorder,
        )

    def _annotation_labels(self, data: pl.DataFrame) -> Optional[list[str]]:
        """One label per track (selected columns stacked as lines), computed once per reconstruction."""
        annotate = self.kw.get('annotate', None)
        if annotate is None or not self._track_starts.size:
            return None
        if isinstance(annotate, str):
            annotate = [annotate]
        unknown = [ann for ann in annotate if ann not in calc.DEFAULT_CATEGORIES]
        if unknown:
            warnings.warn(f"<annotate={unknown}> was not found. For annotation, only select any of [{', '.join(calc.DEFAULT_CATEGORIES)}].")
        # DEFAULT_CATEGORIES order, first column on the bottom line (nearest to the head)
        cols = [ann for ann in calc.DEFAULT_CATEGORIES if ann in annotate and ann in data.columns][::-1]
        if not cols:
            return None
        # data is sorted by track, so the row at each track start carries that track's labels
        return (
            data.select(cols).gather(self._track_starts)
            .select(pl.concat_str([pl.col(c).cast(pl.Utf8).fill_null('') for c in cols], separator='\n'))
            .to_series().to_list()
        )

    def _annotate(self, ax: plt.Axes, xy: np.ndarray, transform) -> list:
        """Place the cached per-track labels next to `xy` (head positions in `transform` coordinates)."""
        if not self._labels:
            return []
        offset = self.kw.get('annotate_offset', 2.5)
        fontsize = self.kw.get('annotate_fontsize', 9)
        return [
            ax.annotate(
                text=label,
                xy=(x, y),
                xycoords=transform,
                xytext=(offset, offset),
                textcoords='offset points',
                va='bottom',
                fontsize=fontsize,
                color=c,
                clip_on=True,
                zorder=14,
            )
            for (x, y), c, label in zip(xy, self._head_colors, self._labels)
        ]

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

        return ax


reconstruct = ReconstructTracks().reconstruct