import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Any
from scipy.stats import gaussian_kde

from .._common import Colors, Categorizer, Values
from ..._handlers._reports import Level



class PolarDataDistribute:

    def __init__(self, data: pd.DataFrame, conditions: list, replicates: list, 
                 *args, normalization: str = 'globally', weight: str = None, 
                 cmap: str = 'plasma', face: str = 'none', 
                 text_color: str = 'black', title: str = None, 
                 label_x: bool = True, label_y: bool = True, 
                 label_y_color: str = 'lightgrey',
                 **kwargs):
        
        # - Shared arguments for all polar plots
        self.noticequeue = kwargs.get('noticequeue', None)

        self.data = data
        self.conditions = conditions
        self.replicates = replicates
        
        self.normalization = normalization
        self.weight = weight

        self.cmap = Colors.GetCmap(cmap, noticequeue=self.noticequeue)
        self.face = face
        self.text_color = text_color
        self.title = title
        self.label_x = label_x
        self.label_y = label_y
        self.label_y_color = label_y_color

        self.background = kwargs.get('background', 'white')
        self.min_density = kwargs.get('min_density', None)
        self.max_density = kwargs.get('max_density', None)

        # - Keyword arguments for Rose Chart
        self.bins = kwargs.get('bins', 16)
        self.ntiles = kwargs.get('ntiles', 10)
        self.alignment = kwargs.get('alignment', 'edge')
        self.gap = kwargs.get('gap', 0.1)
        self.outline = kwargs.get('outline', False)
        self.outline_color = kwargs.get('outline_color', 'blend in')
        if self.outline_color == 'blend in':
            self.outline_color = self.background
        self.outline_width = kwargs.get('outline_width', 0)
        self.radial_step = kwargs.get('radial_step', None)
        self.single_color = kwargs.get('single_color', 'coral')

        # - Keyword arguments for KDE Line plot
        self.bw = kwargs.get('bandwidth', 0.02)
        self.num_points = kwargs.get('num_points', 1000)
        self.normalize = kwargs.get('normalize', True)
        self.kde_fill = kwargs.get('kde_fill', True)
        self.kde_fill_color = kwargs.get('kde_fill_color', 'blue')
        self.kde_fill_alpha = kwargs.get('kde_fill_alpha', 0.4)
        # outline, its color and its width already mentioned
        self.show_abs_average = kwargs.get('absolute_average', True)
        self.mean_angle_color = kwargs.get('mean_angle_color', 'black')
        self.mean_angle_width = kwargs.get('mean_angle_width', 3)

        # - Keyword arguments for Rose chart & KDE overlay
        self.fraction_size = kwargs.get('fraction_size', 0.5)

        # - Keyword arguments and constants for Gaussian KDE Colormesh
        # bins and bandwidth already mentioned
        self.OUTER_RADIUS = 1.0
        self.INNER_RADIUS = 0.7
        self.WIDTH = self.OUTER_RADIUS - self.INNER_RADIUS
        self.D_THETA = 2 * np.pi / self.bins
        self.auto_lut_scale = kwargs.get('auto_lut_scale', False)

        self._check_input()


    def _check_input(self):

        if 'Direction mean' not in self.data.columns:
            self.noticequeue.Report(Level.error, "Missing Direction data in the input DataFrame.")

        if self.bins < 2:
            self.bins = 2
            self.noticequeue.Report(Level.error, "Number of bins must be at least 2.")
        
        if self.ntiles < 1:
            self.ntiles = 1
            self.noticequeue.Report(Level.error, "Number of tiles must be at least 1.")

        if self.gap < 0 or self.gap >= 1:
            self.gap = Values.Clamp01(self.gap)
            self.noticequeue.Report(Level.error, "Gap must be in range (0, 1). Clamped to: " + str(self.gap))


    def _arrange_data(self):
        data = Categorizer(
            data=self.data,
            conditions=self.conditions,
            replicates=self.replicates,
            noticequeue=self.noticequeue
        )()

        angles = np.asarray(data['Direction mean'], dtype=float)
        self.angles = angles % (2 * np.pi)
        self.weights = np.asarray(data[self.weight], dtype=float) if self.weight else None

    def _theta_density(self, a: np.ndarray, weights: np.ndarray, wrap: bool = False):
        
        if weights is not None:
            check = np.isfinite(a) & np.isfinite(weights)
            angles, weights = a[check] % (2 * np.pi), weights[check]
        else:
            check = np.isfinite(a)
            angles = a[check] % (2 * np.pi)

        if angles.size < 2:
            theta, density = np.linspace(0, 2 * np.pi, self.bins, endpoint=False), np.zeros(self.bins, dtype=float)
        else:
            kde = gaussian_kde(angles, bw_method=self.bw, weights=weights if weights is not None else None)
            theta = np.linspace(0, 2 * np.pi, self.bins, endpoint=False)
        
            density = sum(kde(theta + (2 * np.pi * k)) for k in range(-2, 3)) if wrap else kde(theta)

        return theta, density
    
    def _density_norm(self):
        if self.auto_lut_scale:
            if self.normalization == 'locally':
                _min_density = np.min(self.density)
                _max_density = np.max(self.density)
            elif self.normalization == 'globally':
                angles_total = np.asarray(self.data['Direction mean'], dtype=float)
                angles_total = angles_total % (2 * np.pi)
                weights_total = np.asarray(self.data[self.weight], dtype=float) if self.weight else None
                _, all_density = self._theta_density(angles_total, weights_total)
                _min_density = np.min(all_density)
                _max_density = np.max(all_density)
        else:
            _min_density = self.min_density
            _max_density = self.max_density
            if self.min_density is None:
                self.noticequeue.Report(Level.warning, "Minimum density not provided -> setting 0.")
                _min_density = 0.0
            if self.max_density is None:
                self.noticequeue.Report(Level.warning, "Maximum density not provided -> setting 1.")
                _max_density = 1.0
            if self.min_density >= self.max_density:
                self.noticequeue.Report(Level.error, "Minimum density must be less than maximum density -> resetting to 0 and 1.")
                _min_density = 0.0
                _max_density = 1.0
        
        self._min_density = _min_density
        self._max_density = _max_density
            
        self.norm = Normalize(self._min_density, self._max_density)
        self.normalized_density = self.norm(self.density)

    def _mean_direction(self, ax):
        mean_angle = np.arctan2(np.sum(np.sin(self.angles)), np.sum(np.cos(self.angles)))
        mean_angle_wrapped = mean_angle % (2 * np.pi)
        density_at_mean = np.interp(mean_angle_wrapped, self.theta, self.density)

        ax.vlines(mean_angle_wrapped, 0, density_at_mean, color=self.mean_angle_color, linewidth=self.mean_angle_width, zorder=11)
    
    def _annotate_x_axis(self, ax, create: bool = False):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        if self.label_x:
            if create:
                for angle in range(0, 360, 45):
                    ax.text(np.deg2rad(angle), self.OUTER_RADIUS + 0.1, f"{angle}°", 
                            ha='center', va='center', fontsize=10, color=self.text_color, fontweight="medium", zorder=100)
        elif not self.label_x:
            ax.set_xticklabels([])
        
    def _annotate_y_axis(self, ax):
        if self.label_y:
            for label in ax.get_yticklabels():
                label = int(label.get_text())
                ax.text(np.deg2rad(75), label, str(label),
                    ha="center", va="center", fontsize=10, zorder=100,
                    color=self.label_y_color, fontweight="medium")
        ax.set_yticklabels([])

    def _polar_hist(self, ax, data: np.ndarray, heights: np.ndarray, widths: np.ndarray, 
                    bottom: float = 0.0, color: Any = None, **kwargs):
        ax.bar(
            data,
            heights,
            width=widths,
            bottom=bottom,
            color=color,
            align=self.alignment,
            edgecolor=self.outline_color if self.outline else 'none',
            linewidth=self.outline_width if self.outline else 0,
            **kwargs
        )

    def GaussianKDEColormesh(self):
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

        self._arrange_data()
        self.theta, self.density = self._theta_density(self.angles, self.weights, wrap=True)
        self._density_norm()

        self._polar_hist(
            ax, 
            data=self.theta, 
            heights=np.full_like(self.theta, self.WIDTH),
            widths=self.D_THETA * 1.1,
            bottom=self.INNER_RADIUS,
            color=self.cmap(self.normalized_density),
            antialiased=False
        )

        theta_line = np.linspace(0, 2*np.pi, 500)
        for circuit in (self.INNER_RADIUS, self.OUTER_RADIUS):
            ax.plot(theta_line, np.full_like(theta_line, circuit), color='black', linewidth=0.9)

        self._annotate_x_axis(ax, create=True)
        if self.title:
            ax.set_title(self.title, color=self.text_color, fontsize=14, pad=20)
        ax.set_axis_off()

        sm = plt.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.115, fraction=0.045)
        cbar.set_ticks([])
        for cap in ('min', 'max'):
            cbar.ax.text(0.035 if cap == 'min' else 0.965, -0.5, cap, va='center', ha='center', color=self.text_color, transform=cbar.ax.transAxes, fontsize=9, fontstyle='italic')
        cbar.set_label("Density" + (f" weighted by {self.weight}" if self.weight else ""), labelpad=10, color=self.text_color)
        
        fig.set_facecolor(self.face)

        return plt.gcf()


    def KDELinePlot(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

        self._arrange_data()
        self.theta, self.density = self._theta_density(self.angles, self.weights)
        self._density_norm()

        if self.outline:
            ax.plot(
                self.theta, 
                self.normalized_density, 
                color=self.outline_color,
                linewidth=self.outline_width
                 
            )
        
        if self.kde_fill:
            ax.fill_between(
                self.theta, 
                0, 
                self.normalized_density, 
                color=self.kde_fill_color, 
                alpha=self.kde_fill_alpha,
                zorder=5,
                linewidth=0
            )

        if self.show_abs_average:
            self._mean_direction(ax)

        self._annotate_x_axis(ax)
        self._annotate_y_axis(ax)
        if self.title:
            ax.set_title(self.title, color=self.text_color, fontsize=14, pad=20)
        ax.set_facecolor(self.background)
        fig.set_facecolor(self.face)

        return plt.gcf()



    def RoseChart(self):
        pass


    def Overlay(self):
        pass


    def get_density_caps(self):
        
        self._arrange_data()
        self.theta, self.density = self._theta_density(self.angles, self.weights)
        self._density_norm()

        return self._min_density, self._max_density