import __future__
from typing import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .._common import Categorizer, Painter
from ..._handlers._reports import Reporter, Level
from ..._general import is_empty
from ..._compute._stats import Stats



class MotionFlowPlot:
    """
    Creates a quiver/streamplot from cell trajectory spot data.

    The spatial extent is divided into an (n_arrows_x × n_arrows_y) grid.
    Each occupied cell computes:
    * DIRECTION  — circular-mean / min / max of all 'Direction' values in that cell
    * SCALE      — aggregated scale_by metric (density or any numeric column)

    Two render modes:
    * mode='quiver'   — straight arrows, centered on cell, length ∝ scale
    * mode='stream'   — curved streamlines (matplotlib streamplot), linewidth/color ∝ scale
    """

    SCALE_METHODS = ('density', 'min', 'max', 'mean', 'median', 'sum', 'sd', 'add', 'subtract', 'multiply', 'divide')
    RENDER_MODES  = ('quiver', 'stream')

    def __init__(
        self,
        data: pd.DataFrame,
        conditions: list = None,
        replicates: list = None,
        *,
        ignore_categories: bool = True,
        n_arrows_x: int = 10,
        n_arrows_y: int = 7,
        cmap: str = 'viridis',
        scale_by: str | Tuple[str, str] = 'density',
        scale_method: str = 'mean',
        mode: str = 'stream',
        **kwargs
    ):
        """
        Parameters
        ----------
        data : pd.DataFrame
        conditions, replicates : list, optional
        method : str
            Direction aggregation: 'mean' (circular), 'min', 'max'.
        n_arrows_x, n_arrows_y : int
            Grid dimensions (number of arrows / streamline seed columns × rows).
        cmap : str
            Matplotlib colormap name.
        scale_by : str
            'density' → point count per cell, or any numeric column name.
        scale_method : str
            Aggregation for scale_by: 'density' | 'min' | 'max' | 'mean' | 'median' | 'sum' | 'sd' | 'add' | 'subtract' | 'multiply' | 'divide'.
        mode : str
            'quiver' → straight centered arrows.
            'stream' → curved streamlines via streamplot (matches reference image).

        kwargs (styling)
        ----------------
        fig_width, fig_height   : float   — figure size in inches
        max_arrow_size          : float   — maximum arrow/line length or linewidth
                                            quiver: max arrow length in data units (default: auto = 90% of cell)
                                            stream: max linewidth (default: 3.0)
        min_arrow_frac          : float   — min size as fraction of max (default: 0.0 → invisible at zero)
        arrow_width             : float   — quiver shaft width in axes fraction (default: 0.002)
        head_width              : float   — quiver headwidth (default: 4)
        head_length             : float   — quiver headlength (default: 4)
        head_axislength         : float   — quiver headaxislength (default: 3)
        stream_density          : float   — streamplot density (default: 1.0)
        stream_arrowsize        : float   — streamplot arrowsize (default: 1.2)
        show_grid               : bool    — draw faint cell-boundary lines
        title                   : str     — override auto-title
        """

        assert scale_method in self.SCALE_METHODS, \
            f"scale_method must be one of {self.SCALE_METHODS}, got '{scale_method}'"
        assert mode in self.RENDER_MODES, \
            f"mode must be one of {self.RENDER_MODES}, got '{mode}'"

        self.data              = data
        self.ignore_categories = ignore_categories
        self.conditions        = conditions  if conditions  is not None else []
        self.replicates        = replicates  if replicates  is not None else []
        self.n_arrows_x        = n_arrows_x
        self.n_arrows_y        = n_arrows_y
        self.cmap              = cmap
        self.scale_by          = scale_by
        self.scale_method      = scale_method
        self.mode              = mode
        self.noticequeue       = kwargs.get('noticequeue', None)
        self.kwargs            = kwargs



    def plot(self) -> plt.Figure:
        
        if not self.ignore_categories:
            self._arrange_data()

        if is_empty(self.data):
            Reporter(Level.error, "Input data is empty -> returning None.")
            return None

        self.data = self.data.dropna(subset=['Direction'])

        if is_empty(self.data):
            Reporter(Level.error, "No data left after dropping rows with missing 'Direction' values -> returning None.")
            return None

        grid_x, grid_y, U, V, S = self._build_grid()
        sv_norm, sv_min, sv_max = self._normalise_scale(S)

        x_min, x_max = grid_x[0] - (grid_x[1]-grid_x[0])*0.5, grid_x[-1] + (grid_x[1]-grid_x[0])*0.5
        y_min, y_max = grid_y[0] - (grid_y[1]-grid_y[0])*0.5, grid_y[-1] + (grid_y[1]-grid_y[0])*0.5
        x_range      = x_max - x_min
        y_range      = y_max - y_min
        cell_w       = x_range / self.n_arrows_x
        cell_h       = y_range / self.n_arrows_y

        fig, ax = plt.subplots(figsize=(self.kwargs.get('fig_width', 12), self.kwargs.get('fig_height', 10)))

        colormap = Painter().GetCmap(self.cmap)
        norm     = mcolors.Normalize(vmin=sv_min,vmax=sv_max)

        if self.mode == 'quiver':
            self._render_quiver(ax, grid_x, grid_y, U, V, S, sv_norm,
                                cell_w, cell_h, colormap, norm)
        else:
            self._render_stream(ax, grid_x, grid_y, U, V, S, sv_norm,
                                colormap, norm)
        if self.kwargs.get('show_colorbar', True):
            self._make_cbar(fig, ax, colormap, norm)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('X coordinate [µm]')
        ax.set_ylabel('Y coordinate [µm]')
        ax.set_title(self.kwargs.get('title', None),
                     fontsize=self.kwargs.get('title_fontsize', 12),
                     color=self.kwargs.get('text_color', 'black'))
        
        fig.set_facecolor(self.kwargs.get('background_color', 'whitesmoke'))

        if self.kwargs.get('show_grid', False):
            self._draw_grid(ax, grid_x, grid_y, cell_w, cell_h)

        plt.tight_layout()
        return plt.gcf()


    def _render_quiver(self, ax, grid_x, grid_y, U, V, S, sv_norm,
                       cell_w, cell_h, colormap, norm):
        """
        Straight arrows rooted at cell centre, tip pointing in direction of motion.

        kwargs
        ------
        min_arrow_frac : float  smallest arrow as a fraction of cell size  (default 0.0)
        max_arrow_frac : float  longest  arrow as a fraction of cell size  (default 0.9)
                                set > 1.0 to let arrows overflow their cell
        arrow_width    : float  shaft width in axes-fraction units          (default 0.002)
        head_width     : float  quiver headwidth                            (default 4)
        head_length    : float  quiver headlength                           (default 4)
        head_axislength: float  quiver headaxislength                       (default 3)
        """

        cell_size = min(cell_w, cell_h)
        min_frac  = self.kwargs.get('min_arrow_frac', 0.0)
        max_frac  = self.kwargs.get('max_arrow_frac', 1.0)

        min_len = cell_size * min_frac
        max_len = cell_size * max_frac

        for row in range(self.n_arrows_y):
            for col in range(self.n_arrows_x):

                if not np.isfinite(S[row, col]):
                    continue # skip empty cells

                sn     = float(sv_norm[row, col])
                length = min_len + (max_len - min_len) * sn

                # Fix the tail at the "cell" centre
                cx, cy = grid_x[col], grid_y[row]
                u,  v  = U[row, col], V[row, col]

                color = colormap(norm(S[row, col]))

                ax.quiver(
                    cx, cy, 
                    u * length,
                    v * length,
                    color=color,
                    angles='xy', scale_units='xy', scale=1,
                    width=self.kwargs.get('arrow_width', 0.002),
                    headwidth=self.kwargs.get('head_width', 4),
                    headlength=self.kwargs.get('head_length', 4),
                    headaxislength=self.kwargs.get('head_axislength', 3),
                )


    def _render_stream(self, ax, grid_x, grid_y, U, V, S, sv_norm,
                       colormap, norm):
        """
        Curved streamlines via ax.streamplot.

        streamplot needs a *regular* grid and no NaNs in U/V (it will
        terminate lines that hit masked regions).  We replace NaN cells
        with zero-magnitude vectors so no line is seeded there.
        """
        min_frac   = self.kwargs.get('min_arrow_frac', 0.0)
        max_lw     = self.kwargs.get('max_arrow_size', 3.0)   # linewidth units
        arrowsize  = self.kwargs.get('stream_arrowsize', 1.2)
        density    = self.kwargs.get('stream_density', 1.0)

        # Build per-cell linewidth array
        lw_grid = np.where(
            np.isfinite(sv_norm),
            max_lw * (min_frac + (1.0 - min_frac) * np.where(np.isfinite(sv_norm), sv_norm, np.nan)),
            np.nan
        )

        # Replace NaN direction cells with zero-magnitude vectors (no line will be seeded there)
        U_plot = np.where(np.isfinite(U), U, np.nan)
        V_plot = np.where(np.isfinite(V), V, np.nan)

        # Color grid: map actual scale values to colors, but fill NaN cells with the color of the minimum finite value (or leave as NaN to be transparent)
        fin        = S[np.isfinite(S)]
        fill_val   = fin.min() if len(fin) else np.nan
        color_grid = np.where(np.isfinite(S), S, fill_val)

        ax.streamplot(
            grid_x, grid_y,
            U_plot, V_plot,
            color=color_grid,
            cmap=colormap,
            norm=norm,
            linewidth=lw_grid,
            arrowsize=arrowsize,
            density=density,
            start_points=np.column_stack([
                np.tile(grid_x, self.n_arrows_y),
                np.repeat(grid_y, self.n_arrows_x),
            ]),
        )


    def _arrange_data(self) -> pd.DataFrame:
        self.data = Categorizer(
            data=self.data,
            conditions=self.conditions,
            replicates=self.replicates,
        )()


    def _agg_scale(self, vals: np.ndarray) -> float:

        v = vals[np.isfinite(vals)]
        if len(v) == 0:
            return np.nan
        
        a_func, b_func = self.kwargs.get('metric_a_func', 'mean'), self.kwargs.get('metric_b_func', 'mean')
        
        return {
            # 'density':  lambda v: len(v),
            'min':      np.min,      
            'max':      np.max,
            'mean':     np.mean,     
            'median':   np.median, 
            'sd':       np.std,      
            'sum':      np.sum,
            'add':      lambda v: getattr(np, a_func)(v[0]) + getattr(np, b_func)(v[-1]),
            'subtract': lambda v: getattr(np, a_func)(v[0]) - getattr(np, b_func)(v[-1]),
            'multiply': lambda v: getattr(np, a_func)(v[0]) * getattr(np, b_func)(v[-1]),
            'divide':   lambda v: getattr(np, a_func)(v[0]) / getattr(np, b_func)(v[-1]) if getattr(np, b_func)(v[-1]) != 0 else np.nan,
        }[self.scale_method](v)


    def _build_grid(self):
        """
        Returns
        -------
        grid_x  : (n_arrows_x,)          cell-centre x coordinates
        grid_y  : (n_arrows_y,)          cell-centre y coordinates
        U, V    : (n_arrows_y, n_arrows_x) unit direction components
        S       : (n_arrows_y, n_arrows_x) aggregated scale values (NaN = empty cell)
        """
        x          = self.data['X coordinate'].values
        y          = self.data['Y coordinate'].values
        directions = self.data['Direction'].values

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        dx = (x_max - x_min) / self.n_arrows_x
        dy = (y_max - y_min) / self.n_arrows_y

        col_idx = np.clip(((x - x_min) / dx).astype(int), 0, self.n_arrows_x - 1)
        row_idx = np.clip(((y - y_min) / dy).astype(int), 0, self.n_arrows_y - 1)

        scale_data = self._scale_getitems()

        dir_acc   = {}
        scale_acc = {}

        for i in range(len(directions)):
            key = (col_idx[i], row_idx[i])
            dir_acc.setdefault(key, []).append(directions[i])
            if scale_data is not None:
                scale_acc.setdefault(key, []).append(scale_data[i] if isinstance(scale_data, np.ndarray) else (scale_data[0][i], scale_data[-1][i]))


        # build 2D grids  (row = y axis, col = x axis)
        U = np.full((self.n_arrows_y, self.n_arrows_x), np.nan)
        V = np.full((self.n_arrows_y, self.n_arrows_x), np.nan)
        S = np.full((self.n_arrows_y, self.n_arrows_x), np.nan)

        grid_x = x_min + (np.arange(self.n_arrows_x) + 0.5) * dx
        grid_y = y_min + (np.arange(self.n_arrows_y) + 0.5) * dy

        for row in range(self.n_arrows_y):
            for col in range(self.n_arrows_x):
                key  = (col, row)
                dirs = dir_acc.get(key)
                if dirs:
                    da = np.array(dirs)
                    angle = np.arctan2(np.mean(np.sin(da)), np.mean(np.cos(da)))

                    U[row, col] = np.cos(angle)
                    V[row, col] = np.sin(angle)
                    S[row, col] = (float(len(dirs)) if scale_data is None
                                   else self._agg_scale(
                                       np.array(scale_acc.get(key, [np.nan]))))

        return grid_x, grid_y, U, V, S
    

    def _scale_getitems(self) -> np.ndarray | None:

        if self.scale_by == 'density' or self.scale_method == 'density':
            return None

        if self.scale_by == 'density': 
            return None
        
        elif isinstance(self.scale_by, str):
            if self.scale_by not in self.data.columns:
                Reporter(Level.warning, f"Column '{self.scale_by}' not found in data for scale_by -> Falling back to 'density'.")
                return None    
            return self.data[self.scale_by].values

        elif isinstance(self.scale_by, (tuple, list)) and len(self.scale_by) == 2:
            for col in self.scale_by:
                if col not in self.data.columns:
                    Reporter(Level.warning, f"Column '{col}' not found in data for scale_by -> Falling back to 'density'.")
                    return None
                
            return self.data[self.scale_by[0]].values, self.data[self.scale_by[1]].values
            
        else:
            if len(self.scale_by) > 2:
                Reporter(Level.warning, f"<scale_by> must be either a single str or a tuple/list of two str from the column names -> Falling back to 'density'.")
            else:
                Reporter(Level.warning, f"Invalid scale_by '{self.scale_by}' -> Falling back to 'density'.")
            return None



    def _normalise_scale(self, S: np.ndarray):
        """Return (sv_norm, sv_min, sv_max) — NaN cells stay NaN."""

        fin = S[np.isfinite(S)]
        if len(fin) == 0:
            return np.zeros_like(S), 0.0, 1.0
        
        sv_min, sv_max = fin.min(), fin.max()

        if sv_min == sv_max:
            sv_max = sv_min + 1.0
        sv_norm = (S - sv_min) / (sv_max - sv_min)   # NaN preserved

        return sv_norm, sv_min, sv_max
    

    def _draw_grid(self, ax, grid_x, grid_y, cell_w, cell_h):

        kw = dict(color=self.kwargs.get('grid_color', 'grey'), 
                  lw=self.kwargs.get('grid_linewidth', 0.3), 
                  alpha=self.kwargs.get('grid_alpha', 0.35), 
                  zorder=-1)
        
        match self.kwargs.get('grid_type', 'lines'):
            case 'lines':
                for c in range(self.n_arrows_x):
                    ax.axvline(grid_x[c], **kw)
                for r in range(self.n_arrows_y):
                    ax.axhline(grid_y[r], **kw)

            case 'horizontal lines':
                for r in range(self.n_arrows_y):
                    ax.axhline(grid_y[r], **kw)
            
            case 'vertical lines':
                for c in range(self.n_arrows_x):
                    ax.axvline(grid_x[c], **kw)
            
            case 'circles':
                for r in range(self.n_arrows_y):
                    for c in range(self.n_arrows_x):
                        ax.add_patch(plt.Circle((grid_x[c], grid_y[r]), min(cell_w, cell_h)*0.15, fill=False, **kw))

            case 'crosshairs':
                for r in range(self.n_arrows_y):
                    for c in range(self.n_arrows_x):
                        ax.plot([grid_x[c]-cell_w*0.15, grid_x[c]+cell_w*0.15], [grid_y[r], grid_y[r]], **kw)
                        ax.plot([grid_x[c], grid_x[c]], [grid_y[r]-cell_h*0.15, grid_y[r]+cell_h*0.15], **kw)

            case 'radial lines':
                match self.kwargs.get('radial_grid_centre_position', 'center'):
                    case 'center':
                        cx, cy = grid_x.mean(), grid_y.mean()
                    case 'top':
                        cx, cy = grid_x.mean(), grid_y[-1]
                    case 'bottom':
                        cx, cy = grid_x.mean(), grid_y[0]
                    case 'left':
                        cx, cy = grid_x[0], grid_y.mean()
                    case 'right':
                        cx, cy = grid_x[-1], grid_y.mean()
                    case 'bottom left':
                        cx, cy = grid_x[0], grid_y[0]
                    case 'bottom right':
                        cx, cy = grid_x[-1], grid_y[0]
                    case 'top left':
                        cx, cy = grid_x[0], grid_y[-1]
                    case 'top right':
                        cx, cy = grid_x[-1], grid_y[-1]
                for r in range(round((self.n_arrows_y + self.n_arrows_x) / 2)):
                    radius = np.hypot(grid_x[-1] - cx, grid_y[-1] - cy) * (r + 1) / (max(self.n_arrows_x, self.n_arrows_y) / 2)
                    ax.add_patch(plt.Circle((cx, cy), radius, fill=False, **kw))
                    


    
    def _make_cbar(self, fig, ax, colormap, norm, label=None):

        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])

        if label is None:
            label = self._write_cbar_label(self.scale_by, self.scale_method)

        fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02, aspect=30, label=label)


    def _write_cbar_label(self, scale_by, scale_method):
        match scale_method:
            case 'density':
                return 'Track point count'
            
            case 'min' | 'max' | 'mean' | 'median' | 'sum' | 'sd':
                units = Stats().get_units(scale_by)
                if units is None:
                    units = ''
                else:
                    units = f' [{units}]'
                return f'{scale_by} {scale_method} {units}'
            
            case 'add' | 'subtract' | 'multiply' | 'divide':
                units = Stats().get_units(scale_by[0]), Stats().get_units(scale_by[1])
                for unit in units:
                    if unit is None:
                        unit = ''
                    else:
                        unit = f' [{unit}]'
                    units[units.index(unit)] = unit

                return f'{scale_by[0]}{units[0]} {self._operator(scale_method)} {scale_by[1]}{units[1]}'
            
            case _:
                return f'Scale ({scale_method})'
            
    
    def _operator(self, scale_method):
        match scale_method:
            case 'add':       return '+'
            case 'subtract':  return '-'
            case 'multiply':  return '⋅'
            case 'divide':    return '÷'
            case _:           return scale_method.capitalize()
    

    def _any_cherries(self, basket):
        if isinstance(basket, np.ndarray):
            return [basket]
        else:
            basket = [basket[0], basket[1]]
        
        return not None in basket

        
        