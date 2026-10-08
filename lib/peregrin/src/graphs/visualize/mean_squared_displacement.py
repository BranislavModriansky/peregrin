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

    # Diffusion-coefficient fitting defaults.
    # The MSD is computed from the (x, y) projection, so the dimensionality of
    # the fitted model is 2 unless the caller overrides it.
    DIMENSIONS = 2
    # Fraction of the (sorted, unique) time lags used for the fit. Restricting
    # to short lags keeps the estimate in the regime where the power law holds.
    FIT_FRACTION = 0.25
    # Minimum number of lag points required to attempt a fit.
    MIN_FIT_POINTS = 3

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
        log: bool = False,
        linear_fit: bool = False,
        return_data: bool = False,
        **kw,
    ) -> plt.Figure | tuple[plt.Figure, pl.DataFrame]:
        """Plot MSD versus time lag, optionally with a diffusion-coefficient fit.

        Parameters
        ----------
        data : pl.DataFrame
            Spots DataFrame (output of ``calc.spots``).
        band : {'sd', 'sem', 'min-max', 'ci'}, optional
            Dispersion band to draw around each MSD curve.
        categories : dict[str, list], optional
            Category gate applied before plotting.
        log : bool, default False
            Use log-log axes. Required for the anomalous (``linear_fit``) model.
        linear_fit : bool, default False
            Fit the anomalous model ``MSD(t) = 2·d·D̃·t^α`` on short lags via a
            log-log linear regression and annotate D̃ (generalized transport
            coefficient) together with α. Requires ``log=True``.

        Other Parameters
        ----------------
        diffusion_coefficient : bool
            When ``linear_fit`` is used, controls whether the D̃/α annotation is
            drawn (default True). When ``linear_fit`` is False, setting this to
            True instead fits a classical coefficient ``MSD(t) = 2·d·D·t + offset``
            on the early lags and annotates D (default False). For subdiffusive
            cells the classical D is only an effective short-time number.
        dimensions : int, default 2
            Spatial dimensionality ``d`` used in the fit. The MSD is computed
            from the (x, y) projection, so 2 is the matching default.
        fit_fraction : float, default 0.25
            Fraction of the sorted unique time lags (the short-lag regime) used
            for the fit. Must be in (0, 1].

        Returns
        -------
        plt.Figure
            The MSD figure. Per-group fit results are also stored on
            ``self.fit_results``.
        """

        self.data = data.clone() if data is not None else pl.DataFrame()
        self.band = band
        self.categories = categories
        self.log = log
        self.linear_fit = linear_fit

        self.kwargs = get_aliases(kw, self.ALIASES)

        self._arrange_data()

        # ---- compute MSD on call ------------------------------------- #
        required_cols = self._msd_columns_required()
        if not all(col in self.data.columns for col in required_cols):
            self.data = self._compute_msd()

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

        # Collects per-group fit results (anomalous D̃/α or classical D).
        self.fit_results: list[dict[str, Any]] = []

        for idx, (name, gdata) in enumerate(groups):

            group_stamp, group_label = self._get_group_names(gdata, name)

            gdata = gdata.sort('time_lag')

            x_data = gdata['time_lag'].to_numpy().astype(float)
            y_data = gdata['MSD'].to_numpy().astype(float)

            color = self._resolve_color(color_map.get(group_stamp), idx)

            # ---- error band ------------------------------------------

            band_bottom, band_top = self._band_bounds(gdata, y_data)
            if band_bottom is not None:
                mask = np.isfinite(band_bottom) & np.isfinite(band_top)
                if np.any(mask):
                    ax.fill_between(
                        x_data[mask], band_bottom[mask], band_top[mask],
                        color=color, alpha=0.10, linewidth=0, zorder=2,
                    )

            # ---- plot MSD ------------------------------------------

            line = self.kwargs.get('line', '-')
            scatter = self.kwargs.get('scatter', None)
            
            ax.plot(
                x_data, y_data, 
                marker     = 'none' if scatter is None else scatter, 
                markersize = self.kwargs.get('scattersize', 6), 
                linestyle  = 'none' if line is None else line,
                linewidth  = self.kwargs.get('linewidth', 1),
                label = group_label, color=color, zorder=5,
            )

            # ---- linear fit ------------------------------------------ 

            if self.linear_fit:
                if self.log:
                    self._add_linear_fit(ax, x_data, y_data, color, idx, n_groups, group_label)
                else:
                    warn("Anomalous (log-log) MSD model requires a log scale; skipping fit.")
                    self._add_linear_fit(ax, x_data, y_data, color, idx, n_groups, group_label)
            # elif self.kwargs.get('diffusion_coefficient', False):
            #     self._add_diffusion_coefficient(ax, x_data, y_data, color, idx, group_label)


            

        self._set_ylim(ax, self.data['MSD'].to_numpy().astype(float))
        self._style_axes(ax, fig)

        if return_data:
            return fig, self.data
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

    def _compute_msd(self, subset: Optional[list[str]] = None) -> pl.DataFrame:
        """Compute MSD (+ requested error statistics) from spot data via calc."""
        return calc.timelags(
            self.data,
            subset=subset if subset is not None else ['MSD'],
            grouping_level=self.kwargs.get('grouping_level', 'highest'),
            inferative_error=(self.band == 'sem'),
            bootstrap_ci=(self.band == 'ci'),
        )

    def _msd_columns_required(self) -> list[str]:
        """Columns the plot needs; recompute via calc if any are missing."""
        required = ['MSD']

        self._ci_confidence = calc.ci_confidence
        if self._ci_confidence <= 1:
            self._ci_confidence = self._ci_confidence * 100
        self._ci_confidence = int(round(self._ci_confidence))

        match self.band:
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
            loc = 'upper left' if getattr(self, 'fit_results', None) else 'best'
            ax.legend(frameon=False, loc=loc)

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
    # Diffusion-coefficient fitting
    # ------------------------------------------------------------------ #
    def _add_linear_fit(self, ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray,
                        color: Any, idx: int, n_tags: int, label: str) -> None:
        """Fit the anomalous MSD model on short lags and annotate D̃ and α.

        The model is ``MSD(t) = 2·d·D̃·t^α``. Taking a log-log linear fit on the
        short-lag regime gives ``log10(MSD) = a·log10(t) + b`` so that
        ``α = a`` and ``D̃ = 10^b / (2·d)``.
        """
        short = self._short_lag_mask(x_data)
        a, b, lxv, _ = self._log_linear_model(x_data[short], y_data[short])

        if lxv.size < 2:
            warn(f"Not enough valid short-lag points to fit the anomalous MSD model for group '{label}'.")
            return

        d = self._get_dimensions()
        alpha = a
        d_tilde = self._diffusion_coefficient(b, d)

        x_fit = np.logspace(lxv.min(), lxv.max(), 200)
        y_fit = (10.0 ** b) * (x_fit ** a)

        fit_color = self._compute_fit_color(color)
        ax.plot(
            x_fit, y_fit, linestyle='-.', color=fit_color,
            linewidth=2, zorder=7, alpha=0.8,
        )

        self.fit_results.append({
            'group': label,
            'model': 'anomalous',
            'alpha': alpha,
            'D_tilde': d_tilde,
            'intercept': b,
            'dimensions': d,
            'n_points': int(lxv.size),
        })
        self.diffusion_coefficient = d_tilde

        if self.kwargs.get('diffusion_coefficient', True):
            self._annotate_fit(ax, idx, color, self._format_anomalous_label(d_tilde, alpha))


    def _add_diffusion_coefficient(self, ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray,
                                   color: Any, idx: int, label: str) -> None:
        """Fit a classical diffusion coefficient on the early lags.

        Uses the linear model ``MSD(t) = 2·d·D·t + offset`` restricted to the
        short-lag regime. For subdiffusive cells this is only an effective
        short-time number, not a true diffusion coefficient.
        """
        d = self._get_dimensions()
        short = self._short_lag_mask(x_data)
        D, offset, xv = self._linear_diffusion_model(x_data[short], y_data[short], d)

        if D is None:
            warn(f"Not enough valid short-lag points to fit a diffusion coefficient for group '{label}'.")
            return

        x_fit = np.linspace(xv.min(), xv.max(), 200)
        y_fit = 2.0 * d * D * x_fit + offset

        fit_color = self._compute_fit_color(color)
        ax.plot(
            x_fit, y_fit, linestyle='-.', color=fit_color,
            linewidth=2, zorder=7, alpha=0.8,
        )

        self.fit_results.append({
            'group': label,
            'model': 'linear',
            'D': D,
            'offset': offset,
            'dimensions': d,
            'n_points': int(xv.size),
        })
        self.diffusion_coefficient = D

        self._annotate_fit(ax, idx, color, self._format_linear_label(D))


    def _diffusion_coefficient(self, b: float, d: int) -> float:
        """Generalized transport coefficient D̃ = 10^b / (2·d) from the log-log intercept."""
        return (10.0 ** b) / (2.0 * d)


    def _linear_diffusion_model(self, x_data: np.ndarray, y_data: np.ndarray, d: int
                                ) -> tuple[Optional[float], Optional[float], np.ndarray]:
        """Classical diffusion fit ``MSD = 2·d·D·t + offset`` on the given lags.

        Returns ``(D, offset, x_used)`` or ``(None, None, empty)`` when there
        are too few finite points.
        """
        mask = np.isfinite(x_data) & np.isfinite(y_data)
        xv, yv = x_data[mask], y_data[mask]

        if xv.size < 2:
            return None, None, np.array([])

        slope, offset = np.polyfit(xv, yv, 1)
        return slope / (2.0 * d), offset, xv


    def _log_linear_model(self, x_data: np.ndarray, y_data: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
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
        a, b = np.polyfit(lxv, lyv, 1)
        return a, b, lxv, lyv


    # ------------------------------------------------------------------ #
    # Fit helpers
    # ------------------------------------------------------------------ #
    def _get_dimensions(self) -> int:
        """Spatial dimensionality used in the MSD model (defaults to 2)."""
        d = self.kwargs.get('dimensions', self.DIMENSIONS)
        try:
            d = int(d)
        except (TypeError, ValueError):
            warn(f"Invalid <dimensions> value '{d}'; falling back to {self.DIMENSIONS}.")
            return self.DIMENSIONS
        if d < 1:
            warn(f"<dimensions> must be >= 1; falling back to {self.DIMENSIONS}.")
            return self.DIMENSIONS
        return d

    def _fit_fraction(self) -> float:
        """Fraction of the sorted unique lags used for the fit (defaults to 0.25)."""
        frac = self.kwargs.get('fit_fraction', self.FIT_FRACTION)
        try:
            frac = float(frac)
        except (TypeError, ValueError):
            warn(f"Invalid <fit_fraction> value '{frac}'; falling back to {self.FIT_FRACTION}.")
            return self.FIT_FRACTION
        if not (0.0 < frac <= 1.0):
            warn(f"<fit_fraction> must be in (0, 1]; falling back to {self.FIT_FRACTION}.")
            return self.FIT_FRACTION
        return frac

    def _short_lag_mask(self, x_data: np.ndarray) -> np.ndarray:
        """Boolean mask selecting the first ``fit_fraction`` of the sorted unique lags."""
        finite = np.isfinite(x_data)
        if not finite.any():
            return finite

        lags = np.unique(x_data[finite])
        frac = self._fit_fraction()
        n_keep = int(np.ceil(lags.size * frac))
        n_keep = max(self.MIN_FIT_POINTS, n_keep)
        n_keep = min(n_keep, lags.size)
        cutoff = lags[n_keep - 1]
        return finite & (x_data <= cutoff)

    def _units(self) -> tuple[str, str]:
        """Return ``(spatial_units, time_units)`` from calc metadata, or empty strings."""
        meta = getattr(calc, 'metadata', None)
        space, time = '', ''
        if isinstance(meta, dict):
            space = meta.get('spatialunits', '') or ''
            time = meta.get('timeunits', '') or ''
        return space, time

    def _format_anomalous_label(self, d_tilde: float, alpha: float) -> str:
        space, time = self._units()
        unit = f" [{space}²·{time}" + r"$^{-\alpha}$]" if (space or time) else ''
        return rf"$\tilde{{D}}$ = {d_tilde:.3g}{unit}   $\alpha$ = {alpha:.2f}"

    def _format_linear_label(self, D: float) -> str:
        space, time = self._units()
        unit = f" [{space}²·{time}" + r"$^{-1}$]" if (space or time) else ''
        return rf"$D$ = {D:.3g}{unit}"

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
            zorder=8,
        )


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
        norm=mcolors.Normalize(vmin=0, vmax=np.nanmax(H / max(n, 1)) or 1.0),
    )

    ax.set_xlabel("Mean directional change [°]", color=text_color)
    ax.set_ylabel(f"Time lag [{calc.units('time_lag')}]", color=text_color)
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