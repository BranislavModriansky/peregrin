import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Any, Optional, Tuple, Literal, List


from ..._handlers._reports import Level
from .._common import Categorizer, Colors


class MSD:
    """
    Mean Squared Displacement analysis and visualization class.
    
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
        'MSD mean': 'mean',
        'MSD sem': 'sem',
        'MSD sd': 'mean',
        'MSD median': 'median'
    }
    
    def __init__(
            self, data: pd.DataFrame, conditions: list, replicates: list, *args, 
            group_replicates: bool = True, c_mode: str = None, **kwargs):
        
        self.data = data
        self.conditions = conditions
        self.replicates = replicates
        self.aggregate = group_replicates
        self.c_mode = c_mode

        self.noticequeue = kwargs.get('noticequeue', None) if 'noticequeue' in kwargs else None

    def _arrange_data(self) -> pd.DataFrame:
        return Categorizer(
            data=self.data,
            conditions=self.conditions,
            replicates=self.replicates,
            aggby=['Condition', 'Frame lag'] if self.aggregate else ['Condition', 'Replicate', 'Frame lag'],
            aggdict=self.AGG_DICT,
            noticequeue=self.noticequeue
        )()

    def _compute_fit_color(self, base_color: str) -> str:
        base_rgb = mcolors.to_rgb(base_color)
        hsv = mcolors.rgb_to_hsv(np.array(base_rgb))
        
        hsv[1] = np.clip(hsv[1] * self.SATURATION_SCALE, 
                        self.SATURATION_MIN, self.SATURATION_MAX)
        hsv[2] = np.clip(hsv[2] * self.BRIGHTNESS_SCALE, 
                        self.BRIGHTNESS_MIN, hsv[2])
        
        return mcolors.to_hex(mcolors.hsv_to_rgb(hsv))
    
    def _set_axis_labels(self, ax: plt.Axes, scale: str):
        if scale == 'log':
            ax.set_xlabel('log Time lag [frame]')
            ax.set_ylabel('log MSD [μm²]')
        elif scale == 'sqrt':
            ax.set_xlabel('√ Time lag [frame]')
            ax.set_ylabel('√ MSD [μm²]')
        else:
            ax.set_xlabel('Time lag [frame]')
            ax.set_ylabel('MSD [μm²]')
    
    def _set_ylim(self, ax: plt.Axes, y_vals: np.ndarray, scale: str):
        if scale == 'log':
            y_pos = y_vals[y_vals > 0]
            if y_pos.size:
                y_plot_all = np.log10(y_pos)
                margin = max(1e-3, 0.05 * (y_plot_all.max() - y_plot_all.min()))
                ax.set_ylim(y_plot_all.min() - margin, y_plot_all.max() + margin * 2)
        elif scale == 'sqrt':
            y_plot_all = np.sqrt(y_vals)
            margin = 0.1 * (y_plot_all.max() - y_plot_all.min())
            ax.set_ylim(max(0.0, y_plot_all.min() - margin), y_plot_all.max() + margin)
        else:
            miny, maxy = np.nanmin(y_vals), np.nanmax(y_vals)
            lower = miny - 0.5 * abs(miny)
            upper = maxy + 0.05 * abs(maxy) if maxy != 0 else 1.0
            ax.set_ylim(lower, upper)
    
    def plot(self, 
             x_scale: Literal['linear', 'log', 'sqrt'] = 'linear',
             line: bool = True,
             scatter: bool = True,
             linear_fit: bool = False,
             title: Optional[str] = None,
             errorband: Optional[Literal['sem', 'sd', False]] = False,
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
        
        self._set_axis_labels(ax, x_scale)
        
        conditions = data['Condition'].unique()
        colors = {cond: self.c_mode.get(cond, '#000000') for cond in conditions}
        
        # Plot each condition
        for idx, condition in enumerate(conditions):
            cond_data = data[data['Condition'] == condition]

            # --- NEW: decide grouping level based on self.aggregate ---
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
                y_data = gdata['MSD mean'].values

                if errorband == 'sem':
                    err_data = gdata['MSD sem'].values
                elif errorband == 'sd':
                    err_data = gdata['MSD sd'].values / 2
                else:
                    err_data = np.zeros_like(y_data)

                band_bottom_y = np.maximum(y_data - err_data, 0.0)
                band_top_y = y_data + err_data
                
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
                            color=colors[condition],
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
                        color=colors[condition],
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
                        colors[condition],
                        idx if self.aggregate else g_idx,
                        len(conditions) if self.aggregate else len(groups)
                    )
        
        # Set y-limits
        self._set_ylim(ax, data['MSD mean'].values, x_scale)
        
        # Grid and legend
        if grid:
            ax.grid(grid, color='whitesmoke', zorder=0)
        ax.legend(frameon=False)
        
        return plt.gcf()
    
    def _add_linear_fit(self, ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray,
                       condition: str, color: str, idx: int, n_conditions: int):
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
            v_offset = ((idx - (n_conditions - 1) / 2.0) * 0.02 * yrange)
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



# # example usage

# MSD(
#     data=LAGS_STATS,
#     conditions=['naive-ctr', 'naive-cxcl12'],
#     replicates=LAGS_STATS['Replicate'].unique().tolist(),
#     group_replicates=False,  # will now plot one line per replicate per condition
#     c_mode={
#         'naive-ctr': "#093AFF",
#         'naive-cxcl12': "#FF0000",
#     }
# )()
