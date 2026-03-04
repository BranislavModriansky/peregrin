import traceback

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from typing import *


from ..._handlers._reports import Level, Reporter
from .._common import Categorizer, Painter
from ..._general import is_empty


class MSD:
    """
    #### *Mean Squared Displacement analysis and visualization class.*
    
    Provides methods for computing and plotting MSD statistics across time lags,
    with support for multiple conditions, linear fitting, and various scaling options.
    """

    # Constants for color adjustments in linear fits
    SATURATION_SCALE = 0.7
    SATURATION_MIN = 0.02
    SATURATION_MAX = 1.0
    BRIGHTNESS_SCALE = 0.8
    BRIGHTNESS_MIN = 0.06


    def __init__(
            self, data: pd.DataFrame, conditions: list, replicates: list, *, 
            c_mode: str = None, separate_replicates: bool = False, **kwargs):
        
        self.data = data
        self.conditions = conditions
        self.replicates = replicates
        
        self.aggregate = not separate_replicates
        self.disaggregate = False
        self.c_mode = c_mode

        self.color = kwargs.get('color', None)
        self.palette = kwargs.get('palette', None)
        self.noticequeue = kwargs.get('noticequeue', None)

        self.instance_kwargs = kwargs

        self.painter = Painter(noticequeue=self.noticequeue)

        self._check_errors()


    def plot(self,
             statistic: str = 'mean',
             linear_fit: bool = False,
             errorband: Optional[Literal['sd', 'sem', 'min-max', 'ci', False]] = False,
             *,
             line: bool = True,
             scatter: bool = False,
             **kwargs) -> plt.Figure:

        from ..._compute._stats import Stats

        fig, ax = plt.subplots(figsize=(kwargs.get('fig_height', 5), kwargs.get('fig_width', 3.5)))
        
        if self.aggregate:
            prefix = '{per condition}'
            if not any('{per condition}' in col for col in self.data.columns.to_list()):
                Reporter(Level.warning, "Expected aggregated data per condition but no columns with '{per condition}' found. Going to use '{per replicate}' data instead.", noticequeue=self.noticequeue)
                prefix = '{per replicate}'
        else:
            prefix = '{per replicate}'
        

        self.agg_dict, cols = self._build_agg_dict(prefix, Stats.CI_STATISTIC, Stats.CONFIDENCE_LEVEL)
        self._arrange_data()

        mi = cols['mean'] if statistic == 'mean' else cols['median']

        self._set_axis_labels(ax)
            
        color_map = self._get_colors()
        
        # Plot each condition
        for idx, condition in enumerate(self.conditions):
            cond_data = self.data[self.data['Condition'] == condition]

            if self.aggregate:
                # grouped replicates
                groups = [(condition, None, cond_data)]
            else:
                # separate replicates
                groups = []
                for rep, rep_df in cond_data.groupby('Replicate'):
                    groups.append((condition, rep, rep_df))

            for g_idx, (cond_name, rep_name, gdata) in enumerate(groups):

                x_data = gdata['Frame lag'].values
                y_data = gdata[f'{prefix} MSD {statistic}'].values
              
                match errorband:

                    case 'sd':

                        if statistic != 'mean':
                            Reporter(Level.warning, "Error band with SD is only meaningful when plotting the mean. Ignoring error band.", noticequeue=self.noticequeue)
                            errorband = False
                        else:
                            err_data = gdata[f'{prefix} MSD sd'].values / 2

                    case 'sem':

                        if f'{prefix} MSD sem' not in gdata.columns:
                            Reporter(Level.error, "SEM data not available for error band. Make sure to compute SEM in the statistics step. Ignoring error band.", noticequeue=self.noticequeue)
                            errorband = False

                        else:
                            if statistic != 'mean':
                                Reporter(Level.warning, "Error band with SEM is only meaningful when plotting the mean. Ignoring error band.", noticequeue=self.noticequeue)
                                errorband = False
                            else:
                                err_data = gdata[f'{prefix} MSD sem'].values

                    case 'min-max':
                        err_data = (gdata[f'{prefix} MSD max'].values, gdata[f'{prefix} MSD min'].values)

                    case 'ci':
                        
                        if f'{prefix} MSD {Stats.CI_STATISTIC} ci{Stats.CONFIDENCE_LEVEL} low' not in gdata.columns or f'{prefix} MSD {Stats.CI_STATISTIC} ci{Stats.CONFIDENCE_LEVEL} high' not in gdata.columns:
                            Reporter(Level.error, "Confidence interval data not available for error band. Make sure to compute confidence intervals in the statistics step. Ignoring error band.", noticequeue=self.noticequeue)
                            errorband = False

                        else:
                            if statistic != Stats.CI_STATISTIC:
                                Reporter(Level.warning, f"Confidence interval was coputed for {Stats.CI_STATISTIC}, not {statistic}. Ignoring error band.", noticequeue=self.noticequeue)
                                errorband = False
                            else:
                                err_data = (gdata[f'{prefix} MSD {Stats.CI_STATISTIC} ci{Stats.CONFIDENCE_LEVEL} high'].values, gdata[f'{prefix} MSD {Stats.CI_STATISTIC} ci{Stats.CONFIDENCE_LEVEL} low'].values)

                    case _:
                        Reporter(Level.error, f"Invalid errorband type '{errorband}'.", noticequeue=self.noticequeue)
                        errorband = False

                if errorband:
                    if isinstance(err_data, tuple):
                        band_top_y, band_bottom_y = err_data
                    else:
                        band_bottom_y = np.maximum(y_data - err_data, 0.0)
                        band_top_y = y_data + err_data
                
                color = color_map.get(cond_name) if self.c_mode == 'differentiate conditions' else color_map.get(rep_name)
                color = self._resolve_color(color, idx if self.aggregate else g_idx)

                # label: show replicate name only once in legend when not grouped
                if self.aggregate:
                    label = cond_name
                else:
                    label = f"{cond_name} | {rep_name}"

                # Plot error band
                if errorband:
                    mask = ~np.isnan(band_bottom_y) & ~np.isnan(band_top_y)
                    if np.any(mask):
                        ax.fill_between(
                            x_data[mask],
                            band_bottom_y[mask],
                            band_top_y[mask],
                            color=color,
                            alpha=0.08 if self.aggregate else 0.05,
                            linewidth=0,
                            zorder=2
                        )
                
                # Plot main line
                if line:
                    ax.plot(
                        x_data,
                        y_data,
                        marker='none',
                        label=label,
                        linestyle='-',
                        color=color,
                        alpha=1 if self.aggregate else 0.8,
                        zorder=6
                    )
                
                # Plot scatter markers
                if scatter:
                    ax.plot(
                        x_data,
                        y_data,
                        marker='|',
                        markersize=5,
                        label=None,
                        linestyle='none',
                        color='lightgrey',
                        zorder=5
                    )
                
                # Add linear fit (per replicate when not grouped)
                if linear_fit:
                    # pass a unique condition key for annotation placement
                    fit_key = f"{condition}" if self.aggregate else f"{condition}-{rep_name}"
                    self._add_linear_fit(
                        ax,
                        x_data,
                        y_data,
                        fit_key,
                        color,
                        idx if self.aggregate else g_idx,
                        len(self.conditions) if self.aggregate else len(groups)
                    )
        
        # Set y-limits
        self._set_ylim(ax, self.data[mi].values)

        if kwargs.get('title', None):
            ax.set_title(kwargs.get('title'), color=kwargs.get('text_color', 'black'))
        
        if kwargs.get('grid', False):
            ax.grid(True, color='whitesmoke', zorder=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(frameon=False)

        fig.set_facecolor(kwargs.get('fig_background', 'white'))
        
        return plt.gcf()
    
    
    def _arrange_data(self) -> pd.DataFrame:
        self.data = Categorizer(
            data=self.data,
            conditions=self.conditions,
            replicates=self.replicates,
            aggby=['Condition', 'Frame lag'] if self.aggregate else ['Condition', 'Replicate', 'Frame lag'],
            aggdict=self.agg_dict,
            noticequeue=self.noticequeue
        )()

    
    def _get_colors(self) -> dict:

        tag = 'Condition' if self.c_mode == 'differentiate conditions' else 'Replicate'
        tags = self.conditions if self.c_mode == 'differentiate conditions' else self.replicates

        if self.c_mode in ['differentiate conditions', 'differentiate replicates']:
            mp = {}
            if self.palette:
                try:
                    mp = self.painter.StockQualPalette(data=self.data, tag=tag, palette=self.palette)
                except Exception as e:
                    Reporter(Level.error, f"Failed to apply palette '{self.palette}'. Falling back to default colors.", details=traceback.format_exc(), noticequeue=self.noticequeue)
                    pass

            else:
                mp = self.painter.BuildQualPalette(
                    data=self.data,
                    tag=tag,
                    which=tags
                )

            return mp

        elif self.c_mode == 'single color':
            safe_single = self._resolve_color(self.color, 0)
            keys = list(self.conditions) + list(self.replicates) + [None]
            return {k: safe_single for k in keys}

        # fallback for undefined c_mode
        return {k: self._resolve_color(None, i) for i, k in enumerate(list(self.conditions) + list(self.replicates) + [None])}

    def _resolve_color(self, color: Optional[str], idx: int = 0) -> str:
        if color is not None and mcolors.is_color_like(color):
            return color
        return f"C{idx % 10}"

    def _compute_fit_color(self, base_color: str) -> str:
        safe_color = self._resolve_color(base_color, 0)
        base_rgb = mcolors.to_rgb(safe_color)
        hsv = mcolors.rgb_to_hsv(np.array(base_rgb))

        hsv[1] = np.clip(hsv[1] * self.SATURATION_SCALE,
                        self.SATURATION_MIN, self.SATURATION_MAX)
        hsv[2] = np.clip(hsv[2] * self.BRIGHTNESS_SCALE,
                        self.BRIGHTNESS_MIN, hsv[2])

        return mcolors.to_hex(mcolors.hsv_to_rgb(hsv))
    
    def _set_axis_labels(self, ax: plt.Axes):
        ax.set_xlabel('Time lag [frame]')
        ax.set_ylabel('MSD [μm²]')
    
    def _set_ylim(self, ax: plt.Axes, y_vals: np.ndarray):
        miny, maxy = np.nanmin(y_vals), np.nanmax(y_vals)
        lower = miny - 0.5 * abs(miny)
        upper = maxy + 0.05 * abs(maxy) if maxy != 0 else 1.0
        ax.set_ylim(lower, upper)
    
    
    
    def _add_linear_fit(self, ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray,
                       tag: str, color: str, idx: int, n_tags: int):
        
        valid = ~np.isnan(x_data) & ~np.isnan(y_data)
        xv = x_data[valid]
        yv = y_data[valid]
        
        if xv.size < 2:
            return
        
        coeffs = np.polyfit(xv, yv, 1)
        slope, intercept = coeffs[0], coeffs[1]
        
        x_fit = np.linspace(xv.min(), xv.max(), 100)
        y_fit = intercept + slope * x_fit
        
        fit_color = self._compute_fit_color(color)
        
        ax.plot(
            x_fit, y_fit,
            linestyle='dotted',
            color=fit_color,
            linewidth=1.25,
            zorder=7,
            alpha=0.6
        )
        
        try:
            xrange = xv.max() - xv.min()
            x_text = xv.max() + (0.03 * xrange if np.isfinite(xrange) and xrange > 0 else 0.5)
            
            y_text = intercept + slope * xv.max()
            yrange = np.nanmax(yv) - np.nanmin(yv) if np.nanmax(yv) != np.nanmin(yv) else 1.0
            v_offset = ((idx - (n_tags - 1) / 2.0) * 0.02 * yrange)
            y_text += v_offset
            
            slope_text = f"D = {round(slope, 2)} [μm²·s⁻¹]"
            ax.text(
                x_text, y_text, slope_text,
                color=color,
                fontsize=7,
                fontweight='bold',
                verticalalignment='center',
                horizontalalignment='left',
                bbox=dict(facecolor='none', alpha=1, edgecolor='none'),
                zorder=7
            )
        except Exception:
            pass


    def _build_agg_dict(self, prefix: str, ci_stat: str, ci_lvl: int) -> dict:
        

        min     = f"{prefix} MSD min"
        max     = f"{prefix} MSD max"
        mean    = f"{prefix} MSD mean"
        median  = f"{prefix} MSD median"
        sd      = f"{prefix} MSD sd"
        sem     = f"{prefix} MSD sem"
        ci_low  = f"{prefix} MSD {ci_stat} ci{ci_lvl} low"
        ci_high = f"{prefix} MSD {ci_stat} ci{ci_lvl} high"

        agg_dict = {
            min:     'min',
            max:     'max',
            mean:    'mean',
            sd:      'mean',
            median:  'median',
        }

        if sem in self.data.columns:
            agg_dict.update({
                sem: 'mean',
            })

        if ci_low in self.data.columns and ci_high in self.data.columns:
            agg_dict.update({
                ci_low:  'mean',
                ci_high: 'mean'
            })

        cols = {v: k for k, v in agg_dict.items()}

        # if self.aggregate:
        #     agg_dict.pop('Replicate', None)

        return agg_dict, cols


    def _check_errors(self) -> None:
        if not self.aggregate:
            if 'Replicate' not in self.data.columns or not any('{per replicate}' in col for col in self.data.columns.to_list()):
                Reporter(f"Cannot separate replicates <- missing 'Replicate' data.", noticequeue=self.noticequeue)
                self.aggregate = True
                self.disaggregate = False
        if not self.aggregate and self.c_mode == 'differentiate conditions':
            self.c_mode = 'differentiate replicates'
            Reporter("Cannot differentiate conditions when replicate grouping is disabled. Switching to differentiating replicates instead.", noticequeue=self.noticequeue)
        
        if self.aggregate:
            if 'Condition' not in self.data.columns or not any('{per condition}' in col for col in self.data.columns.to_list()):
                Reporter(f"Cannot aggregate per condition <- missing 'Condition' data.", noticequeue=self.noticequeue)
                self.aggregate = False
                self.disaggregate = True
        if self.aggregate and self.c_mode == 'differentiate replicates':
            self.c_mode = 'differentiate conditions'
            Reporter("Cannot differentiate replicates when replicate grouping is enabled. Switching to differentiating conditions instead.", noticequeue=self.noticequeue)


        # if self.aggregate and self.c_mode == 'differentiate replicates':
        #     self.aggregate = False
        #     self.disaggregate = True



def TurnAnglesHeatmap(
        data: pd.DataFrame,
        conditions: list[str],
        replicates: list[str],
        *,
        angle_range: int = 15,
        cmap="plasma",
        **kwargs
    ) -> plt.Figure:
    """
    #### *Plots a heatmap of mean turn angles across frame lags.*

    This function creates a colormesh showing the distribution of mean turn angles (0-180°) across different frame lags in the data. 
    The x-axis represents binned mean turn angles, while the y-axis represents frame lags. 
    The color intensity indicates the fraction of data points within each angle-lag bin.
    """

    # Keyword arguments
    noticequeue = kwargs.get('noticequeue', None)
    text_color = kwargs.get('text_color', 'black')
    title = kwargs.get('title', None)
    strip_background = kwargs.get('strip_background', True)
    
    # Define angle bins based on the specified range
    angle_bin_edges = np.arange(0, 181, angle_range)
    
    # Get data of selected categories
    data = Categorizer(
        data=data,
        conditions=conditions,
        replicates=replicates,
        noticequeue=noticequeue
    )()

    # Check if data is empty after categorization
    if is_empty(data):
        noticequeue.Report(Level.info, "No data. Cannot generate heatmap.")
        return None
    
    cmap = Painter(noticequeue=noticequeue).GetCmap(cmap)
    
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(4, 3.8))

    # X (mean turn angles - clipped to [0, 180]) and Y (frame lags) values for the heatmap
    xvals = np.clip(data['{per condition} Turn mean'].to_numpy(float), 0, 180)
    yvals = data['Frame lag'].to_numpy(float)

    # Build contiguous lag edges -> each lag becomes a row
    lags = np.unique(yvals)
    # Create a bin for each lag value. Lag value <- bin's center value <- boundaries halfway to the next lag value.
    if lags.size > 1:
        mids = (lags[1:] + lags[:-1]) / 2
        y_edges = np.r_[lags[0]-(mids[0]-lags[0]), mids, lags[-1]+(lags[-1]-mids[-1])]
    else:
        y_edges = np.array([lags[0]-0.5, lags[0]+0.5])

    # Compute 2D histogram of mean turn angles vs frame lags
    H, xe, ye = np.histogram2d(xvals, yvals, bins=[angle_bin_edges, y_edges])

    # Create a colormesh plot using the histogram data
    pcm = ax.pcolormesh(
        xe, ye, H.T, 
        cmap=cmap, shading="auto",
        norm=mcolors.Normalize(vmin=0, vmax=np.nanmax(H)),
    )

    # Set axis labels, limits, ticks, and title
    ax.set_xlabel("Mean turn angle (°)", color=text_color)
    ax.set_ylabel("Frame lag", color=text_color)
    ax.set_xlim(angle_bin_edges[0], angle_bin_edges[-1])
    ax.set_xticks(np.arange(0, 181, 30))
    ax.tick_params(colors=text_color, width=0.5)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(text_color)
        spine.set_linewidth(0.5)

    ax.set_title(title, color=text_color)

    if strip_background:
        fig.set_facecolor('none')

    # Add colorbar with custom ticks and labels
    cbar = plt.colorbar(pcm, ax=ax, aspect=25, pad=0.04)
    cbar.set_label("Data fraction per lag", color=text_color)
    cbar.set_ticks(np.linspace(0.0, 1.0, 5))
    cbar.formatter = mticker.FormatStrFormatter("%.2f")
    cbar.update_ticks()
    cbar.ax.tick_params(colors=text_color, width=0.5)
    
    for spine in cbar.ax.spines.values():
        spine.set_color(text_color)
        spine.set_linewidth(0.5)

    return plt.gcf()