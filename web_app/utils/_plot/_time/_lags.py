import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Dict, Literal

from utils import Colors


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
    
    def __init__(self, data: pd.DataFrame, *args, c_mode: str, **kwargs):
        """
        Initialize MSD analyzer.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with MSD statistics. Must contain columns:
            'Condition', 'Frame lag', 'MSD mean', 'MSD sem', 'MSD sd', 'MSD median'
        color_map : dict, optional
            Custom color mapping for conditions. Uses DEFAULT_COLOR_MAP if None.
        """
        self.data = data
        self.c_mode = c_mode

        self._validate_data()
    
    def _validate_data(self):
        """Validate that required columns exist in the data."""
        required_cols = ['Condition', 'Frame lag', 'MSD mean']
        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def filter_conditions(self, conditions: list) -> 'MSD':
        """
        Filter data to specific conditions.
        
        Parameters
        ----------
        conditions : list
            List of condition names to include
            
        Returns
        -------
        MSD
            New MSD instance with filtered data
        """
        filtered = self.data[self.data['Condition'].isin(conditions)]
        return MSD(filtered, self.color_map)
    
    def aggregate_replicates(self) -> 'MSD':
        """
        Aggregate data across replicates by condition and frame lag.
        
        Returns
        -------
        MSD
            New MSD instance with aggregated data
        """
        agg_dict = {
            'MSD mean': 'mean',
            'MSD sem': 'sem',
            'MSD sd': 'mean',
            'MSD median': 'median'
        }
        aggregated = self.data.groupby(['Condition', 'Frame lag']).agg(agg_dict).reset_index()
        return MSD(aggregated, self.color_map)
    
    def _transform_scale(self, x: np.ndarray, y: np.ndarray, err: np.ndarray, 
                        scale: Literal['linear', 'log', 'sqrt']) -> tuple:
        """
        Transform data according to specified scale.
        
        Parameters
        ----------
        x : np.ndarray
            X-axis data (frame lags)
        y : np.ndarray
            Y-axis data (MSD values)
        err : np.ndarray
            Error values
        scale : str
            Scale type: 'linear', 'log', or 'sqrt'
            
        Returns
        -------
        tuple
            (x_plot, y_plot, y_low_plot, y_high_plot)
        """
        if scale == 'log':
            x_plot = np.log10(x)
            y_low = y - err
            y_high = y + err
            valid_mask = (y_low > 0) & (y_high > 0) & (y > 0)
            y_plot = np.where(y > 0, np.log10(y), np.nan)
            y_low_plot = np.where(valid_mask, np.log10(y_low), np.nan)
            y_high_plot = np.where(valid_mask, np.log10(y_high), np.nan)
        elif scale == 'sqrt':
            x_plot = np.sqrt(x)
            y_low = np.maximum(y - err, 0.0)
            y_high = y + err
            y_plot = np.sqrt(np.maximum(y, 0.0))
            y_low_plot = np.sqrt(y_low)
            y_high_plot = np.sqrt(y_high)
        else:  # linear
            x_plot = x
            y_plot = y
            y_low_plot = np.maximum(y - err, 0.0)
            y_high_plot = y + err
        
        return x_plot, y_plot, y_low_plot, y_high_plot
    
    def _compute_fit_color(self, base_color: str) -> str:
        """
        Create desaturated variant of base color for fit lines.
        
        Parameters
        ----------
        base_color : str
            Base color in hex format
            
        Returns
        -------
        str
            Adjusted color in hex format
        """
        base_rgb = mcolors.to_rgb(base_color)
        hsv = mcolors.rgb_to_hsv(np.array(base_rgb))
        
        hsv[1] = np.clip(hsv[1] * self.SATURATION_SCALE, 
                        self.SATURATION_MIN, self.SATURATION_MAX)
        hsv[2] = np.clip(hsv[2] * self.BRIGHTNESS_SCALE, 
                        self.BRIGHTNESS_MIN, hsv[2])
        
        return mcolors.to_hex(mcolors.hsv_to_rgb(hsv))
    
    def _set_axis_labels(self, ax: plt.Axes, scale: str):
        """Set appropriate axis labels based on scale."""
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
        """Set appropriate y-axis limits with margin."""
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
        """
        Plot MSD per time lag with optional linear fits.
        
        Parameters
        ----------
        x_scale : str
            X-axis scale: 'linear', 'log', or 'sqrt'
        line : bool
            Whether to plot the main line
        scatter : bool
            Whether to overlay scatter markers
        linear_fit : bool
            Whether to add linear regression fit lines
        title : str, optional
            Plot title
        errorband : str or False
            Error band type: 'sem', 'sd', or False for no band
        grid : bool
            Whether to show grid
        figsize : tuple
            Figure size (width, height)
        ax : plt.Axes, optional
            Existing axes to plot on
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        if title:
            ax.set_title(title)
        
        self._set_axis_labels(ax, x_scale)
        
        conditions = self.data['Condition'].unique()
        colors = Colors.BuildQualPalette(self.data, tag='Condition')
        colors = [colors[c] for c in conditions]
        
        # Plot each condition
        for idx, condition in enumerate(conditions):
            cond_data = self.data[self.data['Condition'] == condition]
            
            x_data = cond_data['Frame lag'].values
            y_data = cond_data['MSD mean'].values
            
            # Determine error values
            if errorband == 'sem':
                err_data = cond_data['MSD sem'].values
            elif errorband == 'sd':
                err_data = cond_data['MSD sd'].values / 2
            else:
                err_data = np.zeros_like(y_data)
            
            # Transform according to scale
            x_plot, y_plot, y_low_plot, y_high_plot = self._transform_scale(
                x_data, y_data, err_data, x_scale
            )
            
            # Plot error band
            if errorband:
                mask = ~np.isnan(y_low_plot) & ~np.isnan(y_high_plot)
                if np.any(mask):
                    ax.fill_between(x_plot[mask], y_low_plot[mask], y_high_plot[mask],
                                   color=colors[condition], alpha=0.1, linewidth=0, zorder=0)
            
            # Plot main line
            if line:
                ax.plot(x_plot, y_plot, marker='none', label=condition,
                       linestyle='-', color=colors[condition], alpha=1, zorder=6)
            
            # Plot scatter markers
            if scatter:
                ax.plot(x_plot, y_plot, marker='|', markersize=5, label=None,
                       linestyle='none', color='lightgrey', zorder=5)
            
            # Add linear fit
            if linear_fit:
                self._add_linear_fit(ax, x_plot, y_plot, condition, colors[condition], idx, len(conditions))
        
        # Set y-limits
        self._set_ylim(ax, self.data['MSD mean'].values, x_scale)
        
        # Grid and legend
        if grid:
            ax.grid(grid, color='whitesmoke', zorder=0)
        ax.legend(frameon=False)
        
        return plt.gcf()
    
    def _add_linear_fit(self, ax: plt.Axes, x_plot: np.ndarray, y_plot: np.ndarray,
                       condition: str, color: str, idx: int, n_conditions: int):
        """Add linear regression fit line and annotation."""
        valid = ~np.isnan(x_plot) & ~np.isnan(y_plot)
        xv = x_plot[valid]
        yv = y_plot[valid]
        
        if xv.size < 2:
            return
        
        # Fit line
        coeffs = np.polyfit(xv, yv, 1)
        slope, intercept = coeffs[0], coeffs[1]
        
        x_fit = np.linspace(xv.min(), xv.max(), 100)
        y_fit = intercept + slope * x_fit
        
        fit_color = self._compute_fit_color(color)
        
        ax.plot(x_fit, y_fit, linestyle='dotted', color=fit_color,
               linewidth=1.25, zorder=7, alpha=0.6)
        
        # Add annotation
        try:
            xrange = xv.max() - xv.min()
            x_text = xv.max() + (0.03 * xrange if np.isfinite(xrange) and xrange > 0 else 0.5)
            
            y_text = intercept + slope * xv.max()
            yrange = np.nanmax(yv) - np.nanmin(yv) if np.nanmax(yv) != np.nanmin(yv) else 1.0
            v_offset = ((idx - (n_conditions - 1) / 2.0) * 0.02 * yrange)
            y_text += v_offset
            
            slope_text = f"D = {round(slope, 2)} [μm²·s⁻¹]"
            ax.text(x_text, y_text, slope_text, color=color,
                   fontsize=7, fontweight='bold',
                   verticalalignment='center', horizontalalignment='left',
                   bbox=dict(facecolor='none', alpha=1, edgecolor='none'),
                   zorder=7)
        except Exception:
            pass


# Usage example:
msd_analyzer = MSD(LAGS_STATS)

# Filter and aggregate
msd_filtered = msd_analyzer.filter_conditions(['naive-ctr', 'naive-cxcl12'])
msd_agg = msd_filtered.aggregate_replicates()

# Create plots
fig1 = msd_agg.plot(x_scale='linear', linear_fit=True, line=True, scatter=False, grid=False)
fig2 = msd_agg.plot(x_scale='linear', line=True, scatter=False, grid=True)
fig3 = msd_agg.plot(x_scale='linear', line=True, scatter=False, grid=False, errorband='sd')
fig4 = msd_agg.plot(x_scale='linear', line=True, scatter=False, grid=False, errorband='sem')