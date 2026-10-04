from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from typing import Optional, Literal, Any

from warnings import warn
from ..._pckg_exceptions._pckg_errors import *
from ..._pckg_exceptions._pckg_warnings import *

from ...utils import is_empty, get_aliases
from ..painter import paint
from ...data_compute.data_frames import calc
from ...data_handler.categorizer import categorize


plt.rcParams['font.family'] = 'monospace'


class MSD:

    # Constants for color adjustments in linear fits
    SATURATION_SCALE = 0.7
    SATURATION_MIN = 0.02
    SATURATION_MAX = 1.0
    BRIGHTNESS_SCALE = 0.8
    BRIGHTNESS_MIN = 0.06

    ALIASES = {
        'grouping_level': ['grouping', 'groupby', 'group_by', 'grouping_level'],
        'fig_size': ['figsize', 'figure_size', 'fig_size'],
        'color_by': ['color_by', 'colour_by', 'colorby', 'colourby', 'cby', 'c_by'],
        'color': ['color', 'colour', 'c'],
    }

    # Columns produced by calc.time_intervals for MSD.
    MSD_COL = 'MSD'
    MSD_SD_COL = 'MSD_sd'
    MSD_SEM_COL = 'MSD_sem'

    def __init__(self):
        ...


    # Public API
    # --------------

    def plot(
        self,
        data: pl.DataFrame,
        band: Optional[Literal['sd', 'sem', 'min-max', 'ci']] = None,
        categories: Optional[dict[str, list]] = None,
        *,
        grouping_level: Literal['highest', 'lowest'] | str | int = 'highest',
        log: bool = False,
        linear_fit: bool = False,
        **kw,
    ) -> plt.Figure:

        self.data = data.clone() if data is not None else pl.DataFrame()
        self.band = band
        self.categories = categories
        self.grouping_level = grouping_level
        self.log = log
        self.linear_fit = linear_fit

        self.kwargs = get_aliases(kw, self.ALIASES)

        self._arrange_data()

        # ---- compute MSD on call ------------------------------------- #
        self._compute_msd()

        # If nothing could be computed, return an empty figure.
        fig, ax = plt.subplots(figsize=self.kwargs.get('fig_size', (10, 7)))
        if is_empty(self.data):
            return fig

        if self.log:
            ax.set_xscale('log')
            ax.set_yscale('log')

        self.group_keys = [c for c in calc.DEFAULT_CATEGORIES if c in self.data.columns][::-1]  # Reverse the order to prioritize the highest-level grouping keys first.
        color_map = self._build_color_map()

        self._set_axis_labels(ax)

        groups = [ (gdata[self.group_keys[-1]][0], gdata)
                   for gdata in self.data.partition_by(self.group_keys, maintain_order=True) ]
        n_groups = len(groups)

        for idx, (name, gdata) in enumerate(groups):

            group_stamp, group_label = self._get_group_names(gdata, name)

            gdata = gdata.sort('time_lag')

            x_data = gdata['time_lag'].to_numpy().astype(float)
            y_data = gdata['MSD'].to_numpy().astype(float)

            color = self._resolve_color(color_map.get(group_stamp), idx)

            # ---- error band ------------------------------------------ #
            band_bottom, band_top = self._band_bounds(gdata, y_data)
            if band_bottom is not None:
                mask = np.isfinite(band_bottom) & np.isfinite(band_top)
                if np.any(mask):
                    ax.fill_between(
                        x_data[mask], band_bottom[mask], band_top[mask],
                        color=color, alpha=0.10, linewidth=0, zorder=2,
                    )

            # ---- main line ------------------------------------------- #
            if self.kwargs.get('line', True):
                ax.plot(
                    x_data, y_data, marker='none', label=group_label,
                    linestyle='-', color=color, alpha=1.0, zorder=6,
                )

            # ---- scatter markers ------------------------------------- #
            if self.kwargs.get('scatter', False):
                ax.plot(
                    x_data, y_data, marker='o', markersize=6, label=group_label if self.kwargs.get('line', False) else None,
                    linestyle='none', color=color, zorder=5,
                )

            # ---- linear fit ------------------------------------------ #
            if self.linear_fit:
                if not self.log:
                    warn("Linear fit is recommended to be used with log scale.")
                self._add_linear_fit(ax, x_data, y_data, color, idx, n_groups)

        self._set_ylim(ax, self.data['MSD'].to_numpy().astype(float))
        self._style_axes(ax, fig)

        return fig


    def _arrange_data(self) -> None:
        """Ensure the input data is in a suitable format for MSD computation."""
        if is_empty(self.data):
            raise ValueError("Input data is empty.")

        # Categorize the data if categories are provided.
        if self.categories:
            self.data = categorize(self.data, self.categories)

    
    # Computation
    # ----------------

    def _compute_msd(self) -> None:
        """Compute MSD (+ requested dispersion) from spot data via calc."""
        self.data = calc.timelags(
            self.data,
            subset=self._msd_subset(),
            grouping_level=self.grouping_level,
        )

    def _msd_subset(self) -> list[str]:
        """Metric columns to request from calc.time_intervals for MSD."""
        subset = ['MSD']

        self._ci_confidence = calc.ci_confidence
        if self._ci_confidence < 1:
            self._ci_confidence = self._ci_confidence * 100

        match self.band:
            case 'sd':
                subset += ['MSD_sd']
            case 'sem':
                subset += ['MSD_sem']
            case 'ci':
                subset += [f'MSD_ci{self._ci_confidence}_low', f'MSD_ci{self._ci_confidence}_high']
            case 'min-max':
                subset += ['MSD_min', 'MSD_max']
            case _:
                pass
        return subset

    # ------------------------------------------------------------------ #
    # Colors
    # ------------------------------------------------------------------ #

    def _build_color_map(self) -> dict[Any, Any]:
        """One color per group, via the painter (or a supplied color_by)."""
        colors = paint(self.data, color_by=self.group_keys, color=self.kwargs.get('color', 'default'))
        return colors

    def _paint_kwargs(self) -> dict:
        allowed = ('palette', 'cmap', 'lut_vmin', 'lut_vmax')
        return {k: v for k, v in self.kwargs.items() if k in allowed}


    def _band_bounds(self, gdata: pl.DataFrame, y_data: np.ndarray):
        """Return (bottom, top) arrays for the error band, or (None, None)."""
        match self.band:
            case None:
                return None, None
            case 'sd':
                err = gdata['MSD_sd'].to_numpy().astype(float) / 2.0
                return np.maximum(y_data - err, 0.0), y_data + err
            case 'sem':
                err = gdata['MSD_sem'].to_numpy().astype(float)
                return np.maximum(y_data - err, 0.0), y_data + err
            case 'min-max':
                min = gdata['MSD_min'].to_numpy().astype(float)
                max = gdata['MSD_max'].to_numpy().astype(float)
                return np.maximum(min, 0.0), max
            case 'ci':
                low  = gdata[f'MSD_ci{self._ci_confidence}_low'].to_numpy().astype(float)
                high = gdata[f'MSD_ci{self._ci_confidence}_high'].to_numpy().astype(float)
                return np.maximum(low, 0.0), high
            case _:
                raise ValueError(f"<band> parameter '{self.band}' was not recognized -> ignoring error band. <band> must be one of 'sd', 'sem', 'min-max', 'ci', or None.")
    
            
    # ------------------------------------------------------------------ #
    # Styling
    # ------------------------------------------------------------------ #
    def _get_group_names(self, group_data: pl.DataFrame, name: str) -> tuple[str, str]:
        """ Return group names and legend labels for a given group. """

        group_stamp = '.'.join(f'{group_data[i][0]}' for i in self.group_keys)

        match self.kwargs.get('legend_names', 'full'):
            case 'full' | 'complete' | 'detailed':
                return group_stamp, group_stamp
            case 'last' | 'short':
                return group_stamp, name
            case _:
                warn(f"Unrecognized legend_names option '{self.kwargs.get('legend_names')}', defaulting to 'full'.")
                return group_stamp, group_stamp


    def _set_axis_labels(self, ax: plt.Axes) -> None:
        ax.set_xlabel(f"Time lag [{calc.metadata['timeunits']}]", fontsize=11, labelpad=15)
        ax.set_ylabel(f'MSD [{calc.metadata["spatialunits"]}²]', fontsize=11, labelpad=15)

    def _set_ylim(self, ax: plt.Axes, y_vals: np.ndarray) -> None:
        if self.log:
            return
        finite = y_vals[np.isfinite(y_vals)]
        if finite.size == 0:
            return
        miny, maxy = float(np.min(finite)), float(np.max(finite))
        lower = miny - 0.5 * abs(miny)
        upper = maxy + 0.05 * abs(maxy) if maxy != 0 else 1.0
        ax.set_ylim(lower, upper)

    def _style_axes(self, ax: plt.Axes, fig: plt.Figure) -> None:
        if self.kwargs.get('title'):
            ax.set_title(
                self.kwargs['title'],
                color=self.kwargs.get('text_color', 'black'),
                fontsize=self.kwargs.get('title_fontsize', 14),
                fontweight=self.kwargs.get('title_fontweight', 'bold'),
            )

        if self.kwargs.get('grid', False):
            ax.grid(True, color='whitesmoke', zorder=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False)

        fig.set_facecolor(self.kwargs.get('fig_background', 'white'))

    # ------------------------------------------------------------------ #
    # Colors helpers
    # ------------------------------------------------------------------ #
    def _resolve_color(self, color: Optional[Any], idx: int = 0) -> Any:
        if color is not None and mcolors.is_color_like(color):
            return color
        return f"C{idx % 10}"

    def _compute_fit_color(self, base_color: Any) -> str:
        safe_color = self._resolve_color(base_color, 0)
        base_rgb = mcolors.to_rgb(safe_color)
        hsv = mcolors.rgb_to_hsv(np.array(base_rgb))
        hsv[1] = np.clip(hsv[1] * self.SATURATION_SCALE, self.SATURATION_MIN, self.SATURATION_MAX)
        hsv[2] = np.clip(hsv[2] * self.BRIGHTNESS_SCALE, self.BRIGHTNESS_MIN, hsv[2])
        return mcolors.to_hex(mcolors.hsv_to_rgb(hsv))

    # ------------------------------------------------------------------ #
    # Linear fit
    # ------------------------------------------------------------------ #
    def _add_linear_fit(self, ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray,
                        color: Any, idx: int, n_tags: int) -> None:

        xv = x_data[np.isfinite(x_data) & (x_data > 0)]
        yv = y_data[np.isfinite(y_data) & (y_data > 0)]
        if xv.size < 2:
            return

        if self.log:
            lxv, lyv = np.log10(xv), np.log10(yv)
            a, b = np.polyfit(lxv, lyv, 1)
            x_fit = np.logspace(lxv.min(), lxv.max(), 200)
            y_fit = (10.0 ** b) * (x_fit ** a)
        else:
            a, b = np.polyfit(xv, yv, 1)
            x_fit = np.linspace(xv.min(), xv.max(), 200)
            y_fit = a * x_fit + b

        # print(f"a: {a}, b: {b}")

        fit_color = self._compute_fit_color(color)
        ax.plot(
            x_fit, y_fit, linestyle='-.', color=fit_color,
            linewidth=2, zorder=7, alpha=0.8,
        )

        try:
            if self.log:
                lxv, lyv = np.log10(xv), np.log10(yv)
                lxrange = lxv.max() - lxv.min()
                lyrange = lyv.max() - lyv.min()
                lxrange = lxrange if np.isfinite(lxrange) and lxrange > 0 else 1.0
                lyrange = lyrange if np.isfinite(lyrange) and lyrange > 0 else 1.0
                x_text = 10.0 ** (lxv.max() - 0.03 * lxrange)
                y_base_log = b + a * lxv.max()
                v_offset = (idx - (n_tags - 1) / 2.0) * 0.03 * lyrange
                y_text = 10.0 ** (y_base_log + v_offset)
            else:
                x_text = xv.max() - 0.03 * (xv.max() - xv.min())
                y_base = a * xv.max() + b
                v_offset = (idx - (n_tags - 1) / 2.0) * 0.03 * (yv.max() - yv.min())
                y_text = y_base + v_offset

            slope_text = f"D = {a:.2f} [µm²·{calc.t_unit}⁻¹]"
            ax.text(
                x_text, y_text, slope_text, color=color,
                fontsize=7, fontweight='bold',
                verticalalignment='center', horizontalalignment='left',
                bbox=dict(facecolor='none', alpha=1, edgecolor='none'),
                zorder=7,
            )
        except Exception:
            pass


def turn_angles(
    data: pl.DataFrame,
    *,
    grouping_level: Literal['highest', 'lowest'] | str | int = 'highest',
    angle_range: int = 15,
    tlag_range: int = 1,
    cmap: str = "plasma",
    **kwargs,
) -> Optional[plt.Figure]:
    """Plot mean directional change (turning angle) over time lags as a colormesh.

    Directional-change statistics are computed on call via :class:`calc`.
    """
    text_color = kwargs.get('text_color', 'black')
    title = kwargs.get('title', '')

    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (6, 6)))

    # engine = calc(cat_descr=True, cat_descr_err=True, cat_infer_err=False)
    data = calc.timelags(
        data,
        subset=['directional_change_mean'],
        grouping_level=grouping_level,
    )

    if is_empty(data) or 'directional_change_mean' not in data.columns:
        return None

    lags = np.sort(data['time_lag'].unique().to_numpy())
    if lags.size < 2:
        return None

    tlag_range = lags[1] - lags[0]

    # One "sample" per group per lag.
    hierarchy = calc.DEFAULT_CATEGORIES
    group_key = next((c for c in reversed(hierarchy) if c in data.columns), None)
    n = data[group_key].n_unique() if group_key else 1

    xvals = data['directional_change_mean'].to_numpy().astype(float)
    yvals = data['time_lag'].to_numpy().astype(float)

    x_bins = np.arange(0, 181, angle_range)
    y_bins = np.arange(0, lags.max() + tlag_range, tlag_range)

    H, xe, ye = np.histogram2d(xvals, yvals, bins=[x_bins, y_bins])

    pcm = ax.pcolormesh(
        xe, ye, H.T / max(n, 1),
        cmap=cmap, shading='auto',
        norm=mcolors.Normalize(vmin=0, vmax=np.nanmax(H / max(n, 1)) or 1.0),
    )

    ax.set_xlabel("Mean directional change [°]", color=text_color)
    ax.set_ylabel(f"Time lag [{calc.t_unit}]", color=text_color)
    ax.tick_params(colors=text_color, width=0.5)
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(text_color)
        spine.set_linewidth(0.5)

    ax.set_title(title, color=text_color)

    if kwargs.get('strip_background', True):
        fig.set_facecolor('none')

    cbar = plt.colorbar(pcm, ax=ax, aspect=25, pad=0.04)
    cbar.set_label('Fraction of groups', color=text_color)
    cbar.ax.tick_params(colors=text_color, width=0.5)
    for spine in cbar.ax.spines.values():
        spine.set_color(text_color)
        spine.set_linewidth(0.5)

    return fig



msd = MSD().plot