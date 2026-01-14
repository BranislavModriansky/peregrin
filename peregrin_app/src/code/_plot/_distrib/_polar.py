import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Any
from scipy.stats import vonmises

from .._common import Colors, Categorizer, Values
from ..._handlers._reports import Level



class PolarDataDistribute:

    def __init__(self, data: pd.DataFrame, conditions: list, replicates: list, 
                 *args, normalization: str = 'globally', weight: str = None, 
                 cmap: str = 'plasma LUT', face: str = 'none', 
                 text_color: str = 'black', title: str = None, 
                 label_theta: bool = True, label_r: bool = True, 
                 label_r_color: str = 'lightgrey',
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
        self.label_theta = label_theta
        self.label_r = label_r
        self.label_r_color = label_r_color

        self.background = kwargs.get('background', 'white')
        self.min_density = kwargs.get('min_density', None)
        self.max_density = kwargs.get('max_density', None)

        # Keyword arguments for KDE Colormesh
        self.bw = kwargs.get('bandwidth', 0.02)

        # Keyword arguments for Rose Chart
        self.bins = kwargs.get('bins', 16)
        self.alignment = kwargs.get('alignment', 'edge')
        self.gap = float(kwargs.get('gap', 0.1))
        self.levels = kwargs.get('levels', 5)
        self.ntiles = kwargs.get('ntiles', 10)
        self.discretize = kwargs.get('discretize', None)
        self.c_mode = kwargs.get('c_mode', 'single color')
        self.default_colors = kwargs.get('default_colors', True)
        self.single_color = kwargs.get('single_color', 'coral')
        self.outline = kwargs.get('outline', False)
        self.outline_color = kwargs.get('outline_color', 'blend in')
        if self.outline_color == 'blend in':
            self.outline_color = self.background
        self.outline_width = kwargs.get('outline_width', 0)
        

        # Keyword arguments for KDE Line plot
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
        self.r_loc = kwargs.get('r_loc', 75)

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

        self.kappa = self._bandwidth_to_kappa()

        self._arrange_data()
        self.theta, self.density = self._theta_density(self.angles, self.weights)
        self._density_norm()

        self._plot_tiles(
            ax, 
            data=self.theta, 
            height=np.full_like(self.theta, self.WIDTH),
            widths=self.D_THETA * 1.1,
            bottom=self.INNER_RADIUS,
            color=self.cmap(self.norm(self.density)),
            antialiased=False
        )

        theta_line = np.linspace(0, 2 * np.pi, 500)
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
        num_points=1440

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

        self.kappa = self._bandwidth_to_kappa()

        self._arrange_data()

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

        self._mean_dir_dial(ax)
        self._annotate_theta_axis(ax)
        self._annotate_dens_r_axis(ax)
        ax.set_facecolor(self.background)
        fig.set_facecolor(self.face)

        if self.title:
            ax.set_title(self.title, color=self.text_color, fontsize=14, pad=20)

        return plt.gcf()

    def RoseChart(self) -> plt.Figure:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

        self._arrange_data(wrap=True)

        self._define_bins()
        self._plot_histogram(ax, fig=fig)
        self._annotate_theta_axis(ax)
        self._annotate_hist_r_axis(ax)
        ax.set_facecolor(self.background)
        fig.set_facecolor(self.face)
        if self.title:
            ax.set_title(self.title, color=self.text_color, fontsize=14, pad=20)

        return plt.gcf()


    def get_density_range(self) -> tuple[float, float]:

        self.kappa = self._bandwidth_to_kappa()
        
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

        if not 0.0 <= self.gap <= 1.0:
            self.gap = Values.Clamp01(self.gap)
            self.noticequeue.Report(Level.error, "Gap must be in range (0, 1). Clamped to: " + str(self.gap))

        if not (0 <= self.r_loc <= 360):
            self.r_loc = 75
            self.noticequeue.Report(Level.warning, "R axis labels position must be in range (0, 360). Resetting to 75.")


    def _arrange_data(self, wrap: bool = False) -> None:
        self.data = Categorizer(
            data=self.data,
            conditions=self.conditions,
            replicates=self.replicates,
            noticequeue=self.noticequeue
        )()

        # self.data = data
        self.angles = self.data['Direction mean'] % (2 * np.pi) if wrap else self.data['Direction mean']
        self.weights = self.data[self.weight] if self.weight else None

    def _theta_density(self, a: np.ndarray, weights: np.ndarray, wrap: bool = True, 
                       num_points: int = None) -> tuple[np.ndarray, np.ndarray]:
        
        if weights is not None:
            check = np.isfinite(a) & np.isfinite(weights)
            angles, weights = a[check], weights[check] 
        else:
            check = np.isfinite(a)
            angles = a[check]

        if angles.size < 2:
            theta, density = np.linspace(0, 2 * np.pi, self.bins if num_points is None else num_points, endpoint=False), np.zeros(self.bins if num_points is None else num_points, dtype=float)
        else:
            theta = np.linspace(0, 2 * np.pi, self.bins if num_points is None else num_points, endpoint=False)
        
            density = np.mean(
                (weights if weights is not None else 1) *
                vonmises.pdf(theta[:, None], self.kappa, loc=angles),
                axis=1
            )

        return theta, density
    
    def _density_norm(self, wrap: bool = True, num_points: int = None) -> None:

        if self.normalization == 'locally':
            _min_density = np.min(self.density)
            _max_density = np.max(self.density)
        elif self.normalization == 'globally':
            angles_total = np.asarray(self.data['Direction mean'], dtype=float)
            angles_total = angles_total
            weights_total = np.asarray(self.data[self.weight], dtype=float) if self.weight else None
            _, all_density = self._theta_density(angles_total, weights_total, wrap=wrap, num_points=num_points)
            _min_density = np.min(all_density)
            _max_density = np.max(all_density)
        else:
            _min_density = 0.0
            _max_density = 1.0

        if not self.auto_lut_scale:
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

        print(f"Density range: {_min_density} to {_max_density}")
            
        self.norm = Normalize(self._min_density, self._max_density)
        self.normalized_density = self.density / _max_density
        
    def _polar_hist(self, ax, *, fig = None) -> None:

        match self.c_mode:

            case 'single color':
                colors = [[self.single_color] for _ in range(self.bins)]
                heights = None

                for i in range(self.bins):
                    abs_height = float(self.bin_counts[i])
                    if abs_height <= 0:
                        continue

                    self._plot_tiles(
                        ax, 
                        data=self.bin_locs[i],
                        height=abs_height,
                        widths=self.bin_widths[i],
                        color=colors,
                        zorder=10
                    )


            case 'level-based':
                colors = self._slice_cmap(self.levels)
                max_height = max(self.bin_counts)
                level_height = max_height / self.levels

                for i in range(self.bins):

                    abs_height = float(self.bin_counts[i])
                    if abs_height <= 0:
                        continue

                    for lvl in range(self.levels):
                        bottom = lvl * level_height

                        height = min((lvl + 1) * level_height, abs_height) - bottom
                        if height <= 0:
                            continue

                        self._plot_tiles(
                            ax,
                            data = self.bin_locs[i],
                            height = height,
                            widths = self.bin_widths[i],
                            color = colors[lvl],
                            bottom = bottom,
                            zorder = 10
                        )
                    

            case 'n-tiles':
                colors = self._slice_cmap(self.ntiles)
                max_height = max(self.bin_counts)

                discretize = self.data[self.discretize]

                bin_idx = np.digitize(self.angles, self.bin_edges, right=False) - 1
                bin_idx = np.clip(bin_idx, 0, self.bins - 1)
 
                qs = np.linspace(0.0, 1.0, self.ntiles + 1)
                global_edges = np.quantile(discretize, qs)

                # If all weights are (effectively) identical, put everything in first tile
                if np.allclose(global_edges, global_edges[0]):
                    weights_tile_id = np.zeros(discretize.size, dtype=int)
                else:
                    # digitize against inner edges -> ids 0..ntiles-1
                    weights_tile_id = np.digitize(discretize, global_edges[1:-1], right=False)

                for i in range(self.bins):

                    m = bin_idx == i
                    if not np.any(m):
                        continue

                    # Use the globally-derived tile ids, then count composition within this direction bin
                    tile_id = weights_tile_id[m]
                    if tile_id.size == 0:
                        continue

                    tile_counts = np.bincount(tile_id, minlength=self.ntiles).astype(float)

                    # Draw stacked bars; drop empty levels
                    bottom = 0.0

                    for t in range(self.ntiles):

                        height = tile_counts[t]
                        
                        if height <= 0:
                            continue

                        self._plot_tiles(
                            ax,
                            data = self.bin_locs[i],
                            height = height,
                            widths = self.bin_widths[i],
                            color = colors[t],
                            bottom = bottom,
                            zorder = 10
                        )

                        bottom += tile_counts[t]

                self._plot_cbar(fig, ax, colors, global_edges)


            case 'differentiate conditions':
                unique_cond, cond_idx = np.unique(self.data['Condition'], return_inverse=True)
                
                if self.default_colors:
                    colors = self._slice_cmap(len(unique_cond))
                else:
                    colors = [self.data[self.data['Condition'] == cond].iloc[0]['Condition color'] for cond in unique_cond]
                
                bin_idx = np.digitize(self.angles, self.bin_edges, right=False) - 1
                bin_idx = np.clip(bin_idx, 0, self.bins - 1)
                
                for i in range(self.bins):
                    m = bin_idx == i
                    if not np.any(m):
                        continue

                    counts_per_cond = np.bincount(cond_idx[m], minlength=unique_cond.size).astype(float)

                    bottom = 0.0

                    for c in range(len(unique_cond)):

                        height = counts_per_cond[c]
                        
                        if height <= 0:
                            continue

                        self._plot_tiles(
                            ax,
                            data = self.bin_locs[i],
                            height = height,
                            widths = self.bin_widths[i],
                            color = colors[c],
                            bottom = bottom,
                            zorder = 10
                        )

                        bottom += counts_per_cond[c]

                    handles = [mpl.patches.Patch(color=colors[i], label=unique_cond[i]) for i in range(len(unique_cond))]
                self._write_legend(ax, handles)

                
            case 'differentiate replicates':
                unique_repl, repl_idx = np.unique(self.data['Replicate'], return_inverse=True)
                
                if self.default_colors:
                    colors = self._slice_cmap(len(unique_repl))
                else:
                    colors = [self.data[self.data['Replicate'] == repl].iloc[0]['Replicate color'] for repl in unique_repl]
                
                bin_idx = np.digitize(self.angles, self.bin_edges, right=False) - 1
                bin_idx = np.clip(bin_idx, 0, self.bins - 1)
                
                for i in range(self.bins):
                    m = bin_idx == i
                    if not np.any(m):
                        continue

                    counts_per_repl = np.bincount(repl_idx[m], minlength=unique_repl.size).astype(float)

                    bottom = 0.0

                    for r in range(len(unique_repl)):

                        height = counts_per_repl[r]
                        
                        if height <= 0:
                            continue

                        self._plot_tiles(
                            ax,
                            data = self.bin_locs[i],
                            height = height,
                            widths = self.bin_widths[i],
                            color = colors[r],
                            bottom = bottom,
                            zorder = 10
                        )

                        bottom += counts_per_repl[r]

                    handles = [mpl.patches.Patch(color=colors[i], label=unique_repl[i]) for i in range(len(unique_repl))]
                self._write_legend(ax, handles)

    def _plot_histogram(self, ax, *, fig=None) -> None:

        match self.c_mode:

            # Computations are light, so performance should be acceptable
            # even with this amount of for cycles and clarity can be prioritized

            case 'single color':

                levels = 1
                colors = [self.single_color]
                heights = [[float(self.bin_counts[bin])] for bin in range(self.bins)]
                bottoms = [[0.0] for _ in range(self.bins)]

                bin_idx = 'pass'
                

            case 'level-based':

                levels = self.levels
                colors = self._slice_cmap(self.levels)
                
                level_height = max(self.bin_counts) / self.levels
                heights = [
                    [min((lvl + 1) * level_height, float(self.bin_counts[bin])) - lvl * level_height 
                     for lvl in range(self.levels)] 
                    for bin in range(self.bins)
                ]
                bottoms = [
                    [lvl * level_height 
                     for lvl in range(self.levels)] 
                    for _ in range(self.bins)
                ]

                bin_idx = 'pass'
                

            case 'n-tiles':

                colors = self._slice_cmap(self.ntiles)

                levels = self.ntiles

                discretize = self.data[self.discretize]

                bin_idx = np.digitize(self.angles, self.bin_edges, right=False) - 1
                bin_idx = np.clip(bin_idx, 0, self.bins - 1)
 
                qs = np.linspace(0.0, 1.0, self.ntiles + 1)
                global_edges = np.quantile(discretize, qs)

                # Check if discretization values are (effectively) identical
                if np.allclose(global_edges, global_edges[0]): 
                    tile_spans = np.zeros(discretize.size, dtype=int)
                    self.noticequeue.Report(Level.warning, "All values in the discretization column are identical. All data points will be assigned to the first tile.")
                else:
                    tile_spans = np.digitize(discretize, global_edges[1:-1], right=False)

                tile_counts = [
                    np.bincount(tile_spans[bin_idx == bin], minlength=self.ntiles).astype(float)
                    for bin in range(self.bins)
                ]

                heights = [
                    [tile_counts[bin][n] 
                     for n in range(self.ntiles)]
                    for bin in range(self.bins)
                ]
                bottoms = [
                    [sum(tile_counts[bin][0:n]) 
                     for n in range(self.ntiles)]
                    for bin in range(self.bins)
                ]

                self._plot_cbar(fig, ax, colors, global_edges)


            case 'differentiate conditions':

                unique_cond, cond_idx = np.unique(self.data['Condition'], return_inverse=True)

                levels = len(unique_cond)
                
                if self.default_colors:
                    colors = self._slice_cmap(len(unique_cond))
                else:
                    try:
                        colors = [self.data[self.data['Condition'] == cond].iloc[0]['Condition color'] for cond in unique_cond]
                    except KeyError:
                        self.noticequeue.Report(Level.error, "Condition color information is missing in the data. Reverting to default colors.")
                        colors = self._slice_cmap(len(unique_cond))
                    except Exception as e:
                        self.noticequeue.Report(Level.error, f"An error occurred while retrieving condition colors: {e}. Reverting to default colors.")
                        colors = self._slice_cmap(len(unique_cond))
                
                bin_idx = np.digitize(self.angles, self.bin_edges, right=False) - 1
                bin_idx = np.clip(bin_idx, 0, self.bins - 1)

                cond_score = [
                    np.bincount(cond_idx[bin_idx == bin], minlength=unique_cond.size).astype(float)
                    for bin in range(self.bins)
                ]

                heights = [
                    [cond_score[bin][c]
                     for c in range(len(unique_cond))]
                    for bin in range(self.bins)
                ]
                bottoms = [
                    [sum(cond_score[bin][0:c])
                     for c in range(len(unique_cond))]
                    for bin in range(self.bins)
                ]

                handles = [mpl.patches.Patch(color=colors[i], label=unique_cond[i]) for i in range(len(unique_cond))]
                
                self._plot_legend(ax, handles)

                
            case 'differentiate replicates':

                unique_repl, repl_idx = np.unique(self.data['Replicate'], return_inverse=True)

                levels = len(unique_repl)
                
                if self.default_colors:
                    colors = self._slice_cmap(len(unique_repl))
                else:
                    try:
                        colors = [self.data[self.data['Replicate'] == repl].iloc[0]['Replicate color'] for repl in unique_repl]
                    except KeyError:
                        self.noticequeue.Report(Level.error, "Replicate color information is missing in the data. Reverting to default colors.")
                        colors = self._slice_cmap(len(unique_repl))
                    except Exception as e:
                        self.noticequeue.Report(Level.error, f"An error occurred while retrieving replicate colors: {e}. Reverting to default colors.")
                        colors = self._slice_cmap(len(unique_repl))
                
                bin_idx = np.digitize(self.angles, self.bin_edges, right=False) - 1
                bin_idx = np.clip(bin_idx, 0, self.bins - 1)

                repl_score = [
                    np.bincount(repl_idx[bin_idx == bin], minlength=unique_repl.size).astype(float)
                    for bin in range(self.bins)
                ]

                heights = [
                    [repl_score[bin][r]
                     for r in range(len(unique_repl))]
                    for bin in range(self.bins)
                ]
                bottoms = [
                    [sum(repl_score[bin][0:r])
                     for r in range(len(unique_repl))]
                    for bin in range(self.bins)
                ]

                handles = [mpl.patches.Patch(color=colors[i], label=unique_repl[i]) for i in range(len(unique_repl))]
                
                self._plot_legend(ax, handles)



        for bin in range(self.bins):

            try:
                if bin_idx == 'pass':
                    pass
            except Exception:
                if not np.any(bin_idx == bin):
                    continue

            for lvl in range(levels):

                if heights[bin][lvl] <= 0:
                    continue

                self._plot_tiles(
                    ax,
                    data = self.bin_locs[bin],
                    height = heights[bin][lvl],
                    widths = self.bin_widths[bin],
                    color = colors[lvl],
                    bottom = bottoms[bin][lvl],
                    zorder = 10
                )    

    def _plot_tiles(self, ax, data: np.ndarray, height: np.ndarray, widths: np.ndarray, 
                    bottom: float = 0.0, color: Any = None, **kwargs) -> None:
        ax.bar(
            data,
            height,
            width=widths,
            bottom=bottom,
            color=color,
            align=self.alignment,
            edgecolor=self.outline_color if self.outline else 'none',
            linewidth=self.outline_width if self.outline else 0,
            **kwargs
    )

    def _plot_legend(self, ax, handles: list) -> None:

        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=len(handles),
            frameon=False,
            fontsize=9,
            labelcolor=self.text_color,
            handlelength=1.2,
            handletextpad=0.5,
            columnspacing=1.0
        )

    def _plot_cbar(self, fig, ax, colors, global_edges) -> None:
        """
        Discrete horizontal colorbar for n-tiles.

        IMPORTANT: Uses fig.colorbar(..., ax=ax) instead of fig.add_axes(...)
        to avoid Shiny coordmap crashing on Axes without SubplotSpec.
        """
        if fig is None:
            fig = ax.figure

        n = int(self.ntiles)
        if n < 1:
            return

        # Discrete mapping: bins [0..n] with n colors
        cmap = mpl.colors.ListedColormap(list(colors))
        boundaries = np.arange(n + 1)
        norm = mpl.colors.BoundaryNorm(boundaries, cmap.N)

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # older Matplotlib expects an array sometimes

        # This creates a gridspec-managed cbar axis (has SubplotSpec)
        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation="horizontal",
            pad=0.10,
            fraction=0.045,
            ticks=boundaries,          # label bin edges
            spacing="uniform",
            drawedges=bool(self.outline),
        )

        # Style
        cbar.ax.tick_params(axis="x", length=0, pad=4, colors=self.text_color)
        cbar.set_label(self.discretize, color=self.text_color, fontsize=10, labelpad=10)

        # Edge labels from quantiles / global_edges
        def fmt_val(v):
            try:
                span = float(np.max(global_edges) - np.min(global_edges))
            except Exception:
                span = 0.0
            return f"{v:.2g}" if span < 10 else str(int(round(v)))

        cbar.ax.set_xticklabels([fmt_val(v) for v in global_edges], fontsize=9, color=self.text_color)

        # Optional outline control
        if not self.outline:
            cbar.outline.set_visible(False)
        else:
            cbar.outline.set_edgecolor(self.outline_color)
            cbar.outline.set_linewidth(self.outline_width)

    def _mean_dir_dial(self, ax) -> None:
        if self.show_abs_average:
            mean_angle = np.arctan2(np.sum(np.sin(self.angles)), np.sum(np.cos(self.angles)))
            mean_angle_wrapped = mean_angle % (2 * np.pi)
            density_at_mean = np.interp(mean_angle_wrapped, self.theta, self.normalized_density)

        ax.vlines(mean_angle_wrapped, 0, density_at_mean, color=self.mean_angle_color, linewidth=self.mean_angle_width, zorder=11)
    
    def _define_bins(self) -> None:
        # Bin edges
        self.bin_edges = np.linspace(0.0, 2 * np.pi, self.bins + 1)
        bin_width = np.diff(self.bin_edges)

        # Gaps between bars
        self.bin_locs = self.bin_edges[:-1] + (bin_width * self.gap / 2.0)
        self.bin_widths = bin_width * (1.0 - self.gap)

        self.bin_counts, _ = np.histogram(self.angles, bins=self.bin_edges, weights=self.weights)

    def _annotate_theta_axis(self, ax, create: bool = False) -> None:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        if self.label_theta:
            if create:
                for angle in range(0, 360, 45):
                    ax.text(np.deg2rad(angle), self.OUTER_RADIUS + 0.1, f"{angle}°", 
                            ha='center', va='center', fontsize=10, color=self.text_color, fontweight="medium", zorder=500)
            else:
                ax.tick_params(axis="x", labelcolor=self.text_color)
        elif not self.label_theta:
            ax.set_xticklabels([])
        
    def _annotate_dens_r_axis(self, ax) -> None:
        if self.label_r:
            ax.tick_params(axis="y", labelcolor=self.label_r_color)
            if self.normalize in ['globally', 'locally']:
                ax.set_ylim(0, self.normalized_density.max() * 1.05)
                ax.set_yticks(np.linspace(self.normalized_density, self.normalized_density.max(), 6))
                
                max = self.normalized_density.max()
                start = max / 5
                rlabels = [round(r) for r in range(start, max, 5)]
                rlabels.insert(0, None)
                ax.set_yticklabels(rlabels)
            else:
                ax.set_ylim(0, 1.05)
                ax.set_yticks(np.linspace(0, 1, 6))
                ax.set_yticklabels([None, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_rlabel_position(self.r_loc)
            ax.yaxis.label.set_color(self.label_r_color)
            ax.yaxis.label.set_fontweight("medium")
        else:
            ax.set_yticklabels([])

    def _annotate_hist_r_axis(self, ax) -> None:
        ax.set_ylim(0, self.bin_counts.max() * 1.05)
        ax.set_yticks(np.linspace(0, self.bin_counts.max(), 6))
        ax.set_yticklabels([])
        if self.label_r:
            max = round(self.bin_counts.max())
            step = round(max / 5)
            rlabels = [r for r in range(step, max, step)]
            for r in rlabels:
                ax.text(
                    np.deg2rad(self.r_loc),
                    r,
                    str(r),
                    ha="center",
                    va="center",
                    fontsize=10,
                    zorder=100,
                    color=self.label_r_color,
                    fontweight="medium",
                )

    def _slice_cmap(self, slices: int) -> list:
        return self.cmap(np.linspace(0.05, 0.95, slices))

    def _bandwidth_to_kappa(self, kappa_min=0, kappa_max=1e4) -> float:
        return np.clip(1.0 / (self.bw ** 2), kappa_min, kappa_max)