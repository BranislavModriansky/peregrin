import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Any
from scipy.stats import gaussian_kde, vonmises

from .._common import Colors, Categorizer, Values
from ..._handlers._reports import Level



class PolarDataDistribute:

    def __init__(self, data: pd.DataFrame, conditions: list, replicates: list, 
                 *args, normalization: str = 'globally', weight: str = None, 
                 cmap: str = 'plasma LUT', face: str = 'none', 
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
        self.kappa = kwargs.get('kappa', 100)
        self.num_points = kwargs.get('num_points', 1000)
        self.normalize = kwargs.get('normalize', True)
        self.kde_fill = kwargs.get('kde_fill', True)
        self.kde_fill_color = kwargs.get('kde_fill_color', 'blue')
        self.kde_fill_alpha = kwargs.get('kde_fill_alpha', 0.4)
        # outline, its color and its width already mentioned
        self.show_abs_average = kwargs.get('show_abs_average', True)
        self.mean_angle_color = kwargs.get('mean_angle_color', 'black')
        self.mean_angle_width = kwargs.get('mean_angle_width', 3)
        self.r_locate = kwargs.get('r_locate', 75)

        # - Keyword arguments for Rose chart & KDE overlay
        self.bw = kwargs.get('bandwidth', 0.02)
        self.fraction_size = kwargs.get('fraction_size', 0.5)

        # - Keyword arguments and constants for Gaussian KDE Colormesh
        # bins and bandwidth already mentioned
        self.OUTER_RADIUS = 1.0
        self.INNER_RADIUS = 0.7
        self.WIDTH = self.OUTER_RADIUS - self.INNER_RADIUS
        self.D_THETA = 2 * np.pi / self.bins
        self.auto_lut_scale = kwargs.get('auto_lut_scale', True)

        self._check_input()

    def GaussianKDEColormesh(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

        self._arrange_data()
        self.theta, self.density = self._theta_density(self.angles, self.weights)
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

        self._annotate_theta_axis(ax, create=True)
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

    def KDELinePlot(self) -> plt.Figure:
        num_points=3600

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

        self._arrange_data()
        self.kappa = self._bandwidth_to_kappa()
        self.theta, self.density = self._theta_density(self.angles, self.weights, num_points=num_points)
        self._density_norm(num_points=num_points)

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

        self._annotate_theta_axis(ax)
        self._annotate_r_axis(ax)
        if self.title:
            ax.set_title(self.title, color=self.text_color, fontsize=14, pad=20)
        ax.set_facecolor(self.background)
        fig.set_facecolor(self.face)

        return plt.gcf()

    def RoseChart(self) -> plt.Figure:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

        self._arrange_data()
        self.theta, self.density = self._theta_density(self.angles, self.weights)
        self._density_norm()

        self._bins()











    def Overlay(self) -> plt.Figure:
        ...

    def get_density_caps(self) -> tuple[float, float]:
        
        self._arrange_data()
        self.theta, self.density = self._theta_density(self.angles, self.weights)
        self._density_norm()

        return self._min_density, self._max_density
    

    def _check_input(self) -> None:

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

        if not (0 <= self.r_locate <= 360):
            self.r_locate = 75
            self.noticequeue.Report(Level.warning, "R axis labels position must be in range (0, 360). Resetting to 75.")

        if not (0.0 < self.gap <= 1.0):
            self.gap = 0.0
            self.noticequeue.Report(Level.warning, "Gap must be in range (0.0, 1.0). Resetting to 0.0.")
        else:
            self.gap = float(self.gap)


    def _arrange_data(self) -> None:
        data = Categorizer(
            data=self.data,
            conditions=self.conditions,
            replicates=self.replicates,
            noticequeue=self.noticequeue
        )()

        angles = np.asarray(data['Direction mean'], dtype=float)
        self.angles = angles % (2 * np.pi)
        self.weights = np.asarray(data[self.weight], dtype=float) if self.weight else None

    def _theta_density(self, a: np.ndarray, weights: np.ndarray, wrap: bool = True, num_points: int = None) -> tuple[np.ndarray, np.ndarray]:
        
        if weights is not None:
            check = np.isfinite(a) & np.isfinite(weights)
            angles, weights = a[check] % (2 * np.pi), weights[check]
        else:
            check = np.isfinite(a)
            angles = a[check] % (2 * np.pi)

        if angles.size < 2:
            theta, density = np.linspace(0, 2 * np.pi, self.bins if num_points is None else num_points, endpoint=False), np.zeros(self.bins if num_points is None else num_points, dtype=float)
        else:
            theta = np.linspace(0, 2 * np.pi, self.bins if num_points is None else num_points, endpoint=False)
        
            density = np.mean(
                (weights[:, None] if weights is not None else 1) *
                vonmises.pdf(theta[:, None], self.kappa, loc=angles),
                axis=1
            )

        return theta, density
    
    def _density_norm(self, wrap: bool = True, num_points: int = None) -> None:
        if self.auto_lut_scale:
            if self.normalization == 'locally':
                _min_density = np.min(self.density)
                _max_density = np.max(self.density)
            elif self.normalization == 'globally':
                angles_total = np.asarray(self.data['Direction mean'], dtype=float)
                angles_total = angles_total % (2 * np.pi)
                weights_total = np.asarray(self.data[self.weight], dtype=float) if self.weight else None
                _, all_density = self._theta_density(angles_total, weights_total, wrap=wrap, num_points=num_points)
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

    def _mean_direction(self, ax) -> None:
        mean_angle = np.arctan2(np.sum(np.sin(self.angles)), np.sum(np.cos(self.angles)))
        mean_angle_wrapped = mean_angle % (2 * np.pi)
        density_at_mean = np.interp(mean_angle_wrapped, self.theta, self.normalized_density)

        ax.vlines(mean_angle_wrapped, 0, density_at_mean, color=self.mean_angle_color, linewidth=self.mean_angle_width, zorder=11)
    
    def _annotate_theta_axis(self, ax, create: bool = False) -> None:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        if self.label_x:
            if create:
                for angle in range(0, 360, 45):
                    ax.text(np.deg2rad(angle), self.OUTER_RADIUS + 0.1, f"{angle}°", 
                            ha='center', va='center', fontsize=10, color=self.text_color, fontweight="medium", zorder=500)
            else:
                ax.tick_params(axis="x", labelcolor=self.text_color)
        elif not self.label_x:
            ax.set_xticklabels([])
        
    def _annotate_r_axis(self, ax, axtext: bool = False) -> None:
        if self.label_y:
            if axtext:
                for label in ax.get_yticklabels():
                    label = int(label.get_text())
                    ax.text(np.deg2rad(self.r_locate), label, str(label),
                        ha="center", va="center", fontsize=10, zorder=100,
                        color=self.label_y_color, fontweight="medium")
                ax.set_yticklabels([])
            else:
                ax.tick_params(axis="y", labelcolor=self.label_y_color)
                ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_rlabel_position(self.r_locate)
        else:
            ax.set_yticklabels([])

    def _polar_hist(self, ax, data: np.ndarray, heights: np.ndarray, widths: np.ndarray, 
                    bottom: float = 0.0, color: Any = None, **kwargs) -> None:
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

    def _space_bins(self) -> None:
        # Bin edges
        bin_edges = np.linspace(0.0, 2 * np.pi, int(self.bins) + 1)
        bin_widths = np.diff(bin_edges)

        # Gaps between bars
        self.bin_locs = bin_edges[:-1] + (bin_widths * self.gap / 2.0)
        self.bin_widths = bin_widths * (1.0 - self.gap)

    def _color_bins(self) -> None:
        # TODO: upgrade the tile coloring options. 
        #       A. single color => color the bins with single color
        #       B. colormap => divide bins into tiles
        #           - no mapping 
        #               -> color bin tiles solely based on their order
        #                  and distance from the center
        #               -> allow the user to set the number of bin tiles created
        #           - mapping to conditions / replicates [differentiate conditions / differentiate replicates]
        #               -> each condition / replicate inherrits its own color from:
        #                  a) a qualitative colormap
        #                  b) user-defined colors'
        #               -> create a cbar legend indicating which color corresponds to which condition / replicate
        #           - mapping to quantiles of weight values
        #               -> divide the bins and  the weight values into the corresponding number of quantiles (based on user selection)
        #               -> map bin tile proportions and colors to the values in quantiles of weight values 
        #               -> create a cbar legend indicating which color corresponds to which weight value range

        ...

    def _bandwidth_to_kappa(self, kappa_min=0, kappa_max=1e4):
        return np.clip(1.0 / (self.bw ** 2), kappa_min, kappa_max)