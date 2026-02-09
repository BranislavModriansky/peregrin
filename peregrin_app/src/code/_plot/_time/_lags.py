import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from typing import *


from ..._handlers._reports import Level
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

    AGG_DICT = {
        'MSD min': 'min',
        'MSD max': 'max',
        'MSD mean': 'mean',
        'MSD sem': 'mean',
        'MSD sd': 'mean',
        'MSD median': 'median',
        'MSD CI95 low': 'mean',
        'MSD CI95 high': 'mean'
    }
    
    def __init__(
            self, data: pd.DataFrame, conditions: list, replicates: list, *args, 
            group_replicates: bool = True, c_mode: str = None, **kwargs):
        
        self.data = data
        self.conditions = conditions
        self.replicates = replicates
        
        self.aggregate = group_replicates
        self.disaggregate = False
        self.c_mode = c_mode

        self.color = kwargs.get('color', None) if 'color' in kwargs else None

        self.palette = kwargs.get('palette', None) if 'palette' in kwargs else None
        # TODO: create a possibility to use predefined qualitative color maps ["Set1","Set2","Set3","tab10","Accent","Dark2","Pastel1","Pastel2"] when selecting differentiate conditions / replicates

        self.noticequeue = kwargs.get('noticequeue', None) if 'noticequeue' in kwargs else None

        self.painter = Painter(noticequeue=self.noticequeue)

        self._check_errors()

    def _check_errors(self) -> None:
        if self.aggregate and self.c_mode == 'differentiate replicates':
            self.aggregate = False
            self.disaggregate = True
        if self.disaggregate:
            self.noticequeue.Report(Level.warning, "Cannot differentiate replicates when replicate grouping is enabled. Disabling replicate grouping.")

    def _arrange_data(self) -> pd.DataFrame:
        return Categorizer(
            data=self.data,
            conditions=self.conditions,
            replicates=self.replicates,
            aggby=['Condition', 'Frame lag'] if self.aggregate else ['Condition', 'Replicate', 'Frame lag'],
            aggdict=self.AGG_DICT,
            noticequeue=self.noticequeue
        )()
    
    def _get_colors(self) -> dict:

        tag = 'Condition' if self.c_mode == 'differentiate conditions' else 'Replicate'
        tags = self.conditions if self.c_mode == 'differentiate conditions' else self.replicates
        
        if self.c_mode in ['differentiate conditions', 'differentiate replicates']:
            if self.palette:
                try:
                    colors_list = self.painter.StockQualPalette(tags, self.palette)
                    if colors_list:
                        return dict(zip(tags, colors_list))
                except Exception:
                    pass

            else:
                mp = self.painter.BuildQualPalette(
                    data=self.data,
                    tag=tag,
                    which=tags
                )

            return mp

        elif self.c_mode == 'single color':
            keys = list(self.conditions) + list(self.replicates) + [None]
            return {k: self.color for k in keys}
            

    def _compute_fit_color(self, base_color: str) -> str:
        
        base_rgb = mcolors.to_rgb(base_color)
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
    
    def plot(self,
             statistic: str = 'mean',
             line: bool = True,
             scatter: bool = False,
             linear_fit: bool = False,
             title: Optional[str] = None,
             errorband: Optional[Literal['sd', 'sem', 'min-max', 'CI95', False]] = False,
             grid: bool = True,
             figsize: tuple = (5, 3.5),
             ax: Optional[plt.Axes] = None) -> plt.Figure:

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        if title:
            ax.set_title(title)

        data = self._arrange_data()
        
        self._set_axis_labels(ax)
        
        conditions = data['Condition'].unique()

        # Determine which tags we need colors for
        if self.c_mode == 'differentiate conditions':
            tags = self.conditions
        else:
            tags = self.replicates
            
        color_map = self._get_colors()
        
        # Plot each condition
        for idx, condition in enumerate(conditions):
            cond_data = data[data['Condition'] == condition]

            if self.aggregate:
                # one line per condition (replicates already aggregated)
                groups = [(condition, None, cond_data)]
            else:
                # separate line per replicate
                groups = []
                for rep, rep_df in cond_data.groupby('Replicate'):
                    groups.append((condition, rep, rep_df))

            for g_idx, (cond_name, rep_name, gdata) in enumerate(groups):

                x_data = gdata['Frame lag'].values
                y_data = gdata[f'MSD {statistic}'].values
                err_anchor = gdata[f'MSD mean'].values
              
                match errorband:
                    case 'sd':
                        err_data = gdata['MSD sd'].values / 2
                    case 'sem':
                        err_data = gdata['MSD sem'].values
                    case 'min-max':
                        err_data = (gdata['MSD max'].values, gdata['MSD min'].values)
                    case 'CI95':
                        err_data = (gdata['MSD CI95 high'].values, gdata['MSD CI95 low'].values)
                    case _:
                        raise ValueError("Invalid errorband type specified.")

                if isinstance(err_data, tuple):
                    band_top_y, band_bottom_y = err_data
                else:
                    band_bottom_y = np.maximum(err_anchor - err_data, 0.0)
                    band_top_y = err_anchor + err_data
                
                color = color_map.get(cond_name) if self.c_mode == 'differentiate conditions' else color_map.get(rep_name)

                # label: show replicate name only once in legend when not grouped
                if self.aggregate:
                    label = cond_name
                else:
                    # e.g. "naive-ctr | BC39"
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
                            zorder=0
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
                        len(conditions) if self.aggregate else len(groups)
                    )
        
        # Set y-limits
        self._set_ylim(ax, data['MSD mean'].values)
        
        # Grid and legend
        if grid:
            ax.grid(grid, color='whitesmoke', zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False)
        
        return plt.gcf()
    
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

    def __call__(self):
        return self.plot()



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
    xvals = np.clip(data['Turn mean'].to_numpy(float), 0, 180)
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