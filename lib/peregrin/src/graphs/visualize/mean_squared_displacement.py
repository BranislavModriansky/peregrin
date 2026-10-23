from __future__ import annotations

from matplotlib.colors import Normalize, to_hex
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from typing import Optional, Literal, Any

from warnings import warn

from ..._pckg_exceptions._pckg_errors import *
from ..._pckg_exceptions._pckg_warnings import *

from .._tooltip_toolkit import GraphFormatter
from ...utils import ensure_polars, is_empty, get_aliases, round_sigfigs
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
        'grouping_level': ['grouping', 'groupby', 'group_by', 'group_level', 'groupby_level'],
        'fig_size': ['figsize', 'figure_size'],
        'color_by': ['colour_by', 'colorby', 'colourby', 'cby', 'c_by'],
        'color': ['colour', 'c'],
        'alpha': ['a', 'scatter_alpha', 'line_alpha', 'scatteralpha', 'linealpha'],
        'scatter_size': ['marker_size', 'scattersize', 'markersize'],
        'scatter_fill': ['marker_fill', 'fill', 'scatterfill', 'markerfill'],
        'line_style': ['linestyle', 'ls'],
        'line_width': ['linewidth', 'lw'],
        'fit_line_style': ['fit_linestyle', 'fitlinestyle', 'fit_ls', 'fitls'],
        'fit_line_width': ['fit_linewidth', 'fitlinewidth'],
        'fit_alpha': ['fit_a', 'fitalpha'],
        'legend_names': ['legendnames', 'legend_labels', 'legendlabels', 'group_labels', 'grouplabels'],
        'weights': ['weight', 'w', 'fit_weights', 'fitweights']
    }

    def __init__(self):
        self._figure: Optional[plt.Figure] = None
        self._result: Optional[dict] = None

    def __call__(
        self,
        data: pl.DataFrame,
        categories: Optional[dict[str, list]] = None,
        band: Optional[Literal['sd', 'sem', 'min-max', 'ci']] = None,
        *,
        log: bool = False,
        linear_fit: bool = False,
        fmt: Literal['png', 'svg', 'html'] = 'png',
        **kw,
    ) -> plt.Figure | "GraphFormatter":
        """Plot MSD versus time lag, optionally with a diffusion-coefficient fit.

        Parameters
        ----------
        data : pl.DataFrame
            Spots DataFrame (output of `calc.spots`).
        categories : dict[str, list], optional
            Category gate applied before plotting.
        band : {'sd', 'sem', 'min-max', 'ci'}, optional
            Dispersion band to draw around each MSD curve.
        log : bool, default False
            Use log-log axes. Required for the anomalous (`linear_fit`) model.
        linear_fit : bool, default False
            Fit a linear model using `np.polyfit` on log-log transformed MSD data. Requires `log=True`.
        fmt : {'png', 'svg', 'html'}, default 'png'
            Output format of the figure.

            *additional keyword arguments:*

        diffusion_coefficient : bool, default True
            Controls whether the D̃ and α are annotated. Requires `linear_fit=True`.
        weights : str, default 'tracks_contributing'
            Column name in `data` containing per-track weights for the fit.

        Returns
        -------
        plt.Figure | "GraphFormatter"
        """
        data = ensure_polars(data).clone() if data is not None else pl.DataFrame()
        self.fmt = fmt.lower()

        self.kw = get_aliases(kw, self.ALIASES)

        self._result = None
        self._figure = None

        if is_empty(data):
            raise ValueError("Input data is empty.")
        
        if categories:
            data = categorize(data, categories)

        self._result = {
            'spatial_dimensions': self.spatial_dimensions, 
            'results': {}
        }

        # ---- compute MSD on call ----------
        if not all(col in data.columns for col in self._required_cols(band)):
            data = self._compute_msd(data, band)

        # If nothing could be computed, return an empty figure.
        self._figure, ax = plt.subplots(figsize=self.kw.get('fig_size', (10, 7)))
        self._figure.patch.set_gid('pg:figure')
        ax.patch.set_gid('pg:canvas')

        if linear_fit and not log:
            warn("MSD linear model requires a log scale; <log> set to <True>.")
            log = True

        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')

        group_keys = [c for c in calc.DEFAULT_CATEGORIES if c in data.columns][::-1]  # Reverse the order to prioritize the highest-level grouping keys first.
        color_map = self._build_color_map(data, group_keys)

        self._set_axis_labels(ax)

        groups = [ (gdata[group_keys[-1]][0], gdata)
                   for gdata in data.partition_by(group_keys, maintain_order=True) ]

        # (label, color) per group, in draw order, for the tooltip's per-category
        # color controls. Each group's line is tagged pg:msd:c{idx} accordingly.
        self._categories: list[tuple[str, Any]] = []

        for idx, (name, gdata) in enumerate(groups):

            group_stamp, group_label = self._get_group_names(gdata, name, group_keys)

            gdata = gdata.sort('time_lag')

            x_data = gdata['time_lag'].to_numpy().astype(float)
            y_data = gdata['MSD'].to_numpy().astype(float)

            color = color_map.get(group_stamp)

            # ---- error band ------------------------------------------
            band_bottom, band_top = self._band_bounds(gdata, y_data, band)
            if band_bottom is not None:
                mask = np.isfinite(band_bottom) & np.isfinite(band_top)
                if np.any(mask):
                    ax.fill_between(
                        x_data[mask], 
                        band_bottom[mask], 
                        band_top[mask],
                        color=color, 
                        alpha=self.kw.get('band_alpha', 0.15), 
                        zorder=2,
                        linewidth=0,
                    )

            # ---- plot MSD ------------------------------------------
            line = self.kw.get('line', '-')
            scatter = self.kw.get('scatter', None)
            if self.kw.get('scatter_fill', True) in (True, ' full', 'on'): 
                scatter_fill = 'full'
            else:
                scatter_fill = 'none'
            
            msd_lines = ax.plot(
                x_data, y_data, 
                marker     = 'none' if scatter in (False, None, 'none', 'off') else scatter, 
                linestyle  = 'none' if line in (False, None, 'none', 'off') else line,
                markersize = self.kw.get('scatter_size', 6), 
                linewidth  = self.kw.get('line_width', 1.5),
                fillstyle  = scatter_fill,
                alpha      = self.kw.get('alpha', 1.0),
                label      = group_label, color=color, zorder=5,
            )
            for msd_line in msd_lines:
                msd_line.set_gid(category_gid('msd', idx))
            self._categories.append((group_label, color))

            # ---- linear fit ------------------------------------------ 
            if linear_fit:
                weights = self.kw.get('weights', 'tracks_contributing')
                if weights not in (False, None, 'none', 'off'):
                    weights = gdata[weights].to_numpy().astype(float)
                else:
                    weights = None

                self._linear_fit(ax, x_data, y_data, weights, color, idx, group_label)        

        self._set_ylim(ax, data['MSD'].to_numpy().astype(float), log)
        self._style_axes(ax)

        return GraphFormatter(self)


    # Computation
    # ------------

    def _compute_msd(self, data: pl.DataFrame, band: str | None) -> pl.DataFrame:
        """ Compute MSD (+ requested error statistics) from spot data via calc. """
        return calc.timelags(
            data,
            subset = ['MSD', 'tracks_contributing'],
            grouping_level = self.kw.get('grouping_level', 'highest'),
            inferative_error = (band == 'sem'),
            bootstrap_ci = (band == 'ci'),
            ci_confidence = self.kw.get('ci_confidence', None),
            bootstrap_resamples = self.kw.get('bootstrap_resamples', 500),
            bootstrap_ci_method = self.kw.get('bootstrap_ci_method', None),
            ci_statistic = self.kw.get('ci_statistic', None),
        )

    def _required_cols(self, band: str | None) -> list[str]:
        """Columns the plot needs; recompute via calc if any are missing."""
        required = ['MSD', 'tracks_contributing']

        self._ci_confidence = calc.ci_confidence
        if self._ci_confidence <= 1:
            self._ci_confidence = self._ci_confidence * 100
        self._ci_confidence = int(round(self._ci_confidence))

        match band:
            case 'sd':
                required += ['MSD_sd']
            case 'sem':
                required += ['MSD_sem']
            case 'ci':
                required += [f'MSD_ci{self._ci_confidence}_low', f'MSD_ci{self._ci_confidence}_high']
            case 'min-max':
                required += ['MSD_min', 'MSD_max']
            case _:
                pass
        
        return required


    # Colors
    # -------

    def _build_color_map(self, data: pl.DataFrame, group_keys: list[str]) -> dict[Any, Any]:
        """One color per group, via the painter (or a supplied color_by)."""
        colors = paint(data, color_by = group_keys, color = self.kw.get('color', 'default'))
        return colors


    def _band_bounds(self, gdata: pl.DataFrame, y_data: np.ndarray, band: str | None):
        """ Return (bottom, top) arrays for the error band, or (None, None). """
        match band:
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
                raise ValueError(f"<band> parameter '{band}' was not recognized -> ignoring error band. <band> must be one of 'sd', 'sem', 'min-max', 'ci', or None.")
    
            
    # Styling
    # --------

    def _get_group_names(self, group_data: pl.DataFrame, name: str, group_keys: list[str]) -> tuple[str, str]:
        """ Return group names and legend labels for a given group. """
        group_stamp = '.'.join(f'{group_data[i][0]}' for i in group_keys)

        match self.kw.get('legend_names', 'full'):
            case 'full' | 'complete' | 'detailed':
                return group_stamp, group_stamp
            case 'last' | 'short':
                return group_stamp, name
            case _:
                warn(f"Unrecognized legend_names option '{self.kw.get('legend_names')}', defaulting to 'full'.")
                return group_stamp, group_stamp


    def _set_axis_labels(self, ax: plt.Axes) -> None:
        ax.set_xlabel(f"Time lag [{calc.metadata['timeunits']}]", fontsize=11, labelpad=15)
        ax.set_ylabel(f"MSD [{calc.metadata['spatialunits']}²]", fontsize=11, labelpad=15)


    def _set_ylim(self, ax: plt.Axes, y_vals: np.ndarray, log: bool) -> None:
        if log:
            return
        finite = y_vals[np.isfinite(y_vals)]
        if finite.size == 0:
            return
        miny, maxy = float(np.min(finite)), float(np.max(finite))
        lower = miny - 0.5 * abs(miny)
        upper = maxy + 0.05 * abs(maxy) if maxy != 0 else 1.0
        ax.set_ylim(lower, upper)


    def _style_axes(self, ax: plt.Axes) -> None:
        if self.kw.get('title'):
            ax.set_title(
                self.kw['title'],
                color=self.kw.get('text_color', 'black'),
                fontsize=self.kw.get('title_fontsize', 14),
                fontweight=self.kw.get('title_fontweight', 'bold'),
            )

        if self.kw.get('grid', True):
            ax.grid(True, which='both', color='whitesmoke', zorder=0)
            for grid_line in (*ax.get_xgridlines(), *ax.get_ygridlines()):
                grid_line.set_gid('pg:grid')

        borders = self.kw.get('borders', False)
        ax.spines['top'].set_visible(borders)
        ax.spines['right'].set_visible(borders)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, loc='best')

        self._figure.set_facecolor(self.kw.get('fig_background', 'white'))


    @property
    def style_payload(self) -> dict[str, Any]:
        kw = self.kw
        categories = getattr(self, '_categories', None)
        msd_color = (GraphFormatter.qualitative(categories) if categories
                     else GraphFormatter.uniform('#000000'))
        return {
            'gids': [
                {'id': 'msd', 'type': 'line', 'label': 'Curves',
                 'color': msd_color,
                 'values': {'width': float(kw.get('line_width', 1.5))}},
                {'id': 'grid', 'type': 'grid', 'label': 'Grid',
                 'color': GraphFormatter.uniform(to_hex(kw.get('grid_color', 'whitesmoke'))),
                 'values': {'width': float(kw.get('grid_lw', 0.75))}},
                {'id': 'canvas', 'type': 'background', 'label': 'Background',
                 'color': GraphFormatter.uniform(to_hex(kw.get('fig_background', 'white')))},
            ],
        }


    # Linear fit
    # -----------

    def _linear_fit(self, ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray, weights: Optional[np.ndarray],
                        color: Any, idx: int, label: str) -> None:
        
        weights = weights / np.max(weights) if weights is not None else np.ones(y_data.size)

        slope, intercept, lxv, _ = self._linear_model(x_data, y_data, weights)

        if lxv.size < 2:
            warn(f"Not enough valid short-lag points to fit the anomalous MSD model for group '{label}'.")
            return

        D = self.diffusion_coefficient(slope, self.spatial_dimensions)

        x_fit = np.logspace(lxv.min(), lxv.max(), 200)
        y_fit = (10.0 ** intercept) * (x_fit ** slope)

        ax.plot(
            x_fit, y_fit, 
            linestyle = self.kw.get('fit_line_style', '-.'), 
            linewidth = self.kw.get('fit_line_width', 1.5), 
            alpha     = self.kw.get('fit_alpha', 1),
            color     = color,
            zorder    = 7, 
        )

        sig_figs = self.kw.get('round_result', 4)
        self._result['results'][label] = {
            'slope': round_sigfigs(slope, sig_figs),
            'intercept': round_sigfigs(intercept, sig_figs),
            'diffusion_coefficient': round_sigfigs(D, sig_figs)
        }

        if self.kw.get('diffusion_coefficient', True):
            self._annotate_fit(ax, idx, color, self._format_D_label(D, slope))


    @staticmethod
    def diffusion_coefficient(slope: float, spatial_dimensions: int) -> float:
        """Diffusion coefficient D̃ = 10^slope / (2·spatial_dimensions) from the log-log slope."""
        return (10.0 ** slope) / (2.0 * spatial_dimensions)

    @property
    def result(self) -> dict:
        return self._result

    @property
    def figure(self) -> plt.Figure:
        return self._figure


    def _linear_model(self, x_data: np.ndarray, y_data: np.ndarray, weights: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
        """ Return the slope (a) and intercept (b) of the log-log linear fit, along with the log-transformed x and y data.

        Non-finite and non-positive points are dropped jointly across both
        arrays so that each retained log-log point comes from a matching
        ``(t, MSD)`` pair.

        Returns:
            a (float): Slope of the log-log linear fit.
            b (float): Intercept of the log-log linear fit.
            lxv (np.ndarray): Log-transformed x data.
            lyv (np.ndarray): Log-transformed y data.
        """
        mask = (
            np.isfinite(x_data) & np.isfinite(y_data)
            & (x_data > 0) & (y_data > 0)
        )
        xv, yv = x_data[mask], y_data[mask]

        if xv.size < 2:
            return 0.0, 0.0, np.array([]), np.array([])

        lxv, lyv = np.log10(xv), np.log10(yv)
        slope, intercept = np.polyfit(lxv, lyv, 1, w=weights)
        return slope, intercept, lxv, lyv


    # Fit helpers
    # ------------

    @property
    def spatial_dimensions(self) -> int:
        """ Spatial dimensionality used in the the diffusion coefficient calculation. """
        return 2
        

    def _format_D_label(self, D: float, alpha: float) -> str:
        return rf"$\tilde{{D}}$ = {D:.3g}"


    def _annotate_fit(self, ax: plt.Axes, idx: int, color: Any, text: str) -> None:
        """Place a per-group fit annotation, stacked in the bottom-right corner.

        MSD curves increase with lag, so the lower-right region is empty and
        keeps the annotations clear of the (upper-left) legend.
        """
        y = 0.03 + idx * 0.05
        ax.text(
            0.97, y, text, transform=ax.transAxes, color=color,
            fontsize=8, fontweight='bold',
            verticalalignment='bottom', horizontalalignment='right',
            zorder=8, alpha=1,
        )


def turn_angles(
    data: pl.DataFrame,
    *,
    grouping_level: Literal['highest', 'lowest'] | str | int = 'highest',
    angle_range: int = 15,
    tlag_range: int = 1,
    cmap: str = "plasma",
    **kw,
) -> Optional[plt.Figure]:
    """Plot mean directional change (turning angle) over time lags as a colormesh.

    Directional-change statistics are computed on call via :class:`calc`.
    """
    text_color = kw.get('text_color', 'black')
    title = kw.get('title', '')

    fig, ax = plt.subplots(figsize=kw.get('figsize', (6, 6)))

    # engine = calc(cat_descr=True, cat_descr_err=True, cat_infer_err=False)
    data = calc.timelags(
        data,
        subset=['directional_change'],
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
        norm=Normalize(vmin=0, vmax=np.nanmax(H / max(n, 1)) or 1.0),
    )

    ax.set_xlabel(f"Mean directional change [°]", color=text_color)
    ax.set_ylabel(f"Time lag [{calc.metadata['timeunits']}]", color=text_color)
    ax.tick_params(colors=text_color, width=0.5)
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(text_color)
        spine.set_linewidth(0.5)

    ax.set_title(title, color=text_color)

    if kw.get('strip_background', True):
        fig.set_facecolor('none')

    cbar = plt.colorbar(pcm, ax=ax, aspect=25, pad=0.04)
    cbar.set_label('Fraction of groups', color=text_color)
    cbar.ax.tick_params(colors=text_color, width=0.5)
    for spine in cbar.ax.spines.values():
        spine.set_color(text_color)
        spine.set_linewidth(0.5)

    return fig



msd = MSD()