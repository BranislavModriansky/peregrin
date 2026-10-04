from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional, Tuple, Literal

import seaborn as sns
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl

from warnings import warn
from .._pckg_exceptions._pckg_errors import *
from .._pckg_exceptions._pckg_warnings import *

from ..utils import ensure_polars


@dataclass
class Dyes:
    """
    Class holding color options.
    """

    # peregrin_cmap_qn = np.asarray(['#002288ff', '#2555bdff', '#3e73ceff', '#608dceff', '#a0a376ff', '#bdb74bff', '#cfc532ff', '#eadd24ff', '#ffee00ff', "#fcf4ecff"])
    # peregrin_cmap_ql = np.asarray(['#093190ff', "#a9290fff", '#197fdfff', '#edb717ff', '#799adcff', '#ffee00ff'])

    peregrin_cmap_qn = ['#002288ff', '#2555bdff', '#3e73ceff', '#608dceff', '#a0a376ff', '#bdb74bff', '#cfc532ff', '#eadd24ff', '#ffee00ff', "#fcf4ecff"]
    peregrin_cmap_ql = ['#093190ff', "#a9290fff", '#197fdfff', '#edb717ff', '#799adcff', '#ffee00ff']

    _base_quantitative_cmaps = [
        'gist_grey', 'gist_yarg', 'viridis', 'cividis', 'plasma', 'inferno',
        'magma', 'gist_heat', 'hot', 'afmhot', 'copper', 'Wistia', 'pink',
        'bone', 'spring', 'summer', 'autumn', 'winter', 'cool', 'ocean',
        'gist_earth', 'terrain', 'cubehelix', 'CMRmap', 'gnuplot2', 'gnuplot',
        'gist_stern', 'nipy_spectral', 'gist_ncar', 'brg', 'jet', 'turbo',
        'rainbow', 'gist_rainbow', 'twilight', 'twilight_shifted', 'hsv',
        'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd',
        'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn',
        'BuGn', 'YlGn', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
        'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
        'berlin', 'managua', 'vanimo',
    ]

    quantitative_cmaps = ["peregrin_qn"]
    for _cmap in _base_quantitative_cmaps:
        quantitative_cmaps.append(_cmap)
        if f"{_cmap}_r" in mpl.colormaps:
            quantitative_cmaps.append(f"{_cmap}_r")
    del _base_quantitative_cmaps

    qualitative_cmaps_matplotlib = [
        'Pastel1', 'Pastel2', 'Paired', 'Accent', 'okabe_ito', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'
    ]
    qualitative_cmaps_seaborn = [
        "deep", "muted", "bright", "pastel", "dark", "colorblind", "husl", "hsl"
    ]
    qualitative_cmaps = ["peregrin_ql"] + qualitative_cmaps_matplotlib + qualitative_cmaps_seaborn
    cmaps = quantitative_cmaps + qualitative_cmaps


class ColorGenerator:
    _HEX = np.array([f"{i:02x}" for i in range(256)], dtype="<U2")

    def __init__(self): ...

    def random_color(self, n: Optional[int] = 1, *, 
                     code: str = "hex", a: float = 1.0, **kwargs) -> np.ndarray:
        if not isinstance(n, int) or n < 1:
            raise ColorGeneratorError("n must be a positive integer.")
        rng = np.random.default_rng(kwargs.get("seed", 42))
        rgb = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
        return self._color_value(rgb, code=code, a=a)

    def random_grey(self, n: Optional[int] = 1, *, code: str = "hex",
                    a: float = 1.0, **kwargs) -> np.ndarray:
        if not isinstance(n, int) or n < 1:
            raise ColorGeneratorError("n must be a positive integer.")
        rng = np.random.default_rng(kwargs.get("seed", 42))
        grey = rng.integers(0, 240, size=(n, 1), dtype=np.uint8)
        rgb = np.repeat(grey, 3, axis=1)
        return self._color_value(rgb, code=code, a=a)

    def _color_value(self, rgb: np.ndarray, *, code: str = "hex",
                     a: float = 1.0) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.ndim == 1:
            rgb = rgb.reshape(1, -1)
        alpha = float(np.clip(a, 0.0, 1.0))
        match code:
            case "hex":
                alpha_hex = np.full((rgb.shape[0], 1), round(alpha * 255), dtype=np.uint8)
                rgba = np.hstack((rgb, alpha_hex))
                parts = self._HEX[rgba]
                out = np.char.add("#", parts[:, 0])
                out = np.char.add(out, parts[:, 1])
                out = np.char.add(out, parts[:, 2])
                out = np.char.add(out, parts[:, 3])
                return out
            case "rgb":
                return np.array([f"rgb({r}, {g}, {b})" for r, g, b in rgb], dtype=object)
            case "rgba":
                return np.array([f"rgba({r}, {g}, {b}, {alpha})" for r, g, b in rgb], dtype=object)
            case _:
                raise ValueError("Unsupported color code. Use one of: 'hex', 'rgb', 'rgba'.")

    @staticmethod
    def is_color_code(value, raise_on_out_of_range=True):
        if not isinstance(value, str):
            return False
        s = value.strip()
        if mcolors.is_color_like(s):
            return True
        rgb_pattern = re.compile(
            r"^rgb\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)%?)\s*,"
            r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)%?)\s*,"
            r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)%?)\s*\)$",
            re.IGNORECASE)
        rgba_pattern = re.compile(
            r"^rgba\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)%?)\s*,"
            r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)%?)\s*,"
            r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)%?)\s*,"
            r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)%?)\s*\)$",
            re.IGNORECASE)

        def _channel_in_range(channel: str) -> bool:
            if channel.endswith("%"):
                return 0.0 <= float(channel[:-1]) <= 100.0
            return 0.0 <= float(channel) <= 255.0

        def _alpha_in_range(alpha: str) -> bool:
            if alpha.endswith("%"):
                return 0.0 <= float(alpha[:-1]) <= 100.0
            return 0.0 <= float(alpha) <= 1.0

        match = rgb_pattern.match(s)
        if match:
            if all(_channel_in_range(c) for c in match.groups()):
                return True
            if raise_on_out_of_range:
                raise InvalidColorRangeError(
                    f"'{s}' has the shape of an rgb() color but values are out of range "
                    f"(expected 0-255 or 0%-100% per channel).")
            return False

        match = rgba_pattern.match(s)
        if match:
            *channels, alpha = match.groups()
            if all(_channel_in_range(c) for c in channels) and _alpha_in_range(alpha):
                return True
            if raise_on_out_of_range:
                raise InvalidColorRangeError(
                    f"'{s}' has the shape of an rgba() color but values are out of range "
                    f"(channels 0-255 or 0%-100%, alpha 0-1 or 0%-100%).")
            return False
        return False


class Cmaps:

    def __init__(self): ...

    @staticmethod
    def retrieve_palette(palette: Optional[str | list] = "peregrin_ql", categories: list = None) -> dict:
        
        if palette in ('peregrin_ql', 'default'):
            palette = Dyes.peregrin_cmap_ql
        elif palette == 'peregrin_qn':
            palette = Dyes.peregrin_cmap_qn

        if isinstance(palette, list):
            if categories is None:
                return palette
            if len(palette) < len(categories):
                warn(f"More categories ({len(categories)}) than colors ({len(palette)}). "
                     "Randomly generating additional colors.")
                print(type(palette))
                palette.extend(random_color(len(categories) - len(palette)))
                print(palette)
                print(type(palette))
            return dict(zip(categories, palette))
        else:
            try:
                palette = mpl.colormaps[palette]
            except ValueError:
                palette = sns.color_palette(palette)
            except Exception as e:
                warn(
                    message=f"An error occurred while retrieving the palette '{palette}': {str(e)}. "
                            "<- Defaulting to 'tab10' colormap.",
                    category=PaletteBuilderWarning, stacklevel=2)
                palette = mpl.colormaps['tab10']

            if categories is None:
                return palette
            
            cat_count = len(categories)
            return {cat: mcolors.to_hex(palette(i / cat_count)) for i, cat in enumerate(categories)}

    @staticmethod
    def retrieve_cmap(cmap: str | mcolors.Colormap) -> mcolors.Colormap:

        if isinstance(cmap, mcolors.Colormap):
            return cmap

        if cmap in Dyes.qualitative_cmaps + ['peregrin_qn']:
            return mcolors.LinearSegmentedColormap.from_list(f'_{cmap}', Cmaps.retrieve_palette(cmap))

        try:
            return mpl.colormaps[cmap]
        except Exception as e:
            warn(
                message=f"An error occurred while retrieving the colormap for '{cmap}': {str(e)}. "
                        f"Available colormaps are: {', '.join(Dyes.quantitative_cmaps)}. "
                        "Defaulting to 'jet' colormap.",
                category=PainterWarning, stacklevel=2)
            return mpl.colormaps['jet']


    @staticmethod
    def scale_cmap(data: pl.Series, *, min: float = None, max: float = None) -> Tuple[Any, Any]:
        """Build a Normalize + float64 value array from a polars Series."""
        try:
            vals = data.cast(pl.Float64, strict=False).to_numpy()

            if not isinstance(min, (int, float)):
                min = float(np.nanmin(vals)) if vals.size else 0.0
            if not isinstance(max, (int, float)):
                max = float(np.nanmax(vals)) if vals.size else 100.0

            if not (np.isfinite(max) or np.isfinite(min)):
                warn(
                    message="Invalid LUT range. Max and min values are not finite. "
                            "Using default range (0.0, 100.0).",
                    category=LUTWarning, stacklevel=2)
                if not np.isfinite(min):
                    min = 0.0
                if not np.isfinite(max):
                    max = 100.0

            if max <= min:
                warn(
                    message="Invalid LUT range. Max value must be greater than min value. Swapping values.",
                    category=LUTWarning, stacklevel=2)
                min, max = max, min

            return plt.Normalize(min, max), vals

        except Exception as e:
            raise LUTError(f"Error while computing LUT for {data.name or 'unknown data'}: {str(e)}")

    @staticmethod
    def showcase_colormaps(*, which: str = "quantitative", **kwargs) -> plt.Figure:
        text_color = kwargs.get('text_color', 'black')
        strip_background = kwargs.get('strip_background', False)

        if kwargs.get('cmaps') is None:
            match which:
                case "quantitative":
                    cmaps = Dyes.quantitative_cmaps
                case "qualitative":
                    cmaps = Dyes.qualitative_cmaps
                case _:
                    raise ValueError(
                        f"Unknown colormap type '{which}'. Supported types are "
                        "'quantitative' and 'qualitative'.")
        else:
            cmaps = kwargs.get('cmaps')

        n = len(cmaps)
        if n == 0:
            raise ValueError("No colormaps provided for showcasing.")

        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        height = 0.35 + 0.15 + (n + (n - 1) * 0.1) * 0.22
        fig, axs = plt.subplots(nrows=n + 1, figsize=(6.4, height))
        fig.subplots_adjust(top=1 - 0.35 / height, bottom=0.15 / height, left=0.2, right=0.99)

        for ax, name in zip(axs, cmaps):
            ax.imshow(gradient, aspect='auto', cmap=Cmaps.retrieve_cmap(name))
            ax.text(-0.02, 0.5, name[:-4] if name.endswith(' LUT') else name,
                    va='center', ha='right', fontsize=10, color=text_color,
                    fontfamily='monospace', transform=ax.transAxes)

        for ax in axs:
            ax.set_axis_off()
        if strip_background:
            fig.set_facecolor('none')
        return plt.gcf()


class Painter:

    def __init__(self):
        pass

    def paint(
        self,
        data: pl.DataFrame,
        *,
        color_by: Optional[str | tuple[str, Literal['categorical', 'numeric']] | list[str]] = None,
        color: Any = 'black',
        **kwargs
    ) -> None:
        """
        Assign colors to the data based on the provided parameters and return format.

        Parameters
        ----------
        data : pl.DataFrame
            The data to be colored.
        color : Any, optional
            The default color to use if 'color_by' is not specified. Default is 'black'.
        color_by : Optional[str | tuple[str, Literal['categorical', 'numeric']]], optional
            The column or specification to color by. Default is None.
        **kwargs
            Additional keyword arguments for color assignment.

        Returns
        -------
        dict or list
        """

        self.data = ensure_polars(data)
        self.color = color if color != 'default' else 'peregrin_ql'
        self.color_by = color_by
        self.kwargs = kwargs

        self.colors = None

        print(self.color_by, self.color, self.kwargs)

        if self.color_by is not None:
            self._color_by()
        # elif (self.color in list(mcolors.CSS4_COLORS.keys()) or 
        #       self.color in list(mcolors.BASE_COLORS.keys()) or
        #       self.color in list(mcolors.TABLEAU_COLORS.keys()) or
        #       self.color in list(mcolors.XKCD_COLORS.keys()) or
        #       is_color_code(self.color)):
        #     pass
        # elif self.color is not None:
        #     self._color()

        if self.colors is not None:
            return self.colors
        else:
            return self.color


    def _color_by(self) -> None:
        datatype = None
        if isinstance(self.color_by, tuple) and len(self.color_by) == 2:
            self.color_by, datatype = self.color_by
            if datatype not in ('categorical', 'numeric'):
                raise ValueError(
                    f"Invalid datatype parameter '{datatype}' for color_by. "
                    "Must be one of ['categorical', 'numeric'].")

        if isinstance(self.color_by, list):
            datatype = 'categorical'
            if any(col not in self.data.columns for col in self.color_by):
                raise InvalidParameterValueError(
                    f"<color_by> input columns: [{', '.join(f'{c}' for c in self.color_by)}] not found.")
        else:
            if self.color_by not in self.data.columns:
                raise InvalidParameterValueError(
                    f"<color_by> column: '{self.color_by}' not found in DataFrame.")

            dtype = self.data.schema[self.color_by]

        if datatype == 'categorical' or dtype in (pl.Categorical, pl.Enum, pl.Utf8, pl.Boolean):
            self.colors = self._categorical_colors()
        elif datatype == 'numeric' or dtype.is_numeric():
            self.colors = self._numeric_colors()
        else:
            raise InvalidParameterValueError(
                f"Invalid color_by value: '{self.color_by}'. Must be a column name in "
                "spot_data with categorical or numeric data or a tuple where the "
                "data type is specified (column_name, 'categorical'|'numeric').")

        self.color = None

    def _color(self) -> None:
        # No index in polars: 'n' defaults to the row count (callers that want
        # one-color-per-track pass n explicitly).
        n = self.kwargs.get('n', max(self.data.height, 1))
        if self.color == 'random':
            self.colors = random_color(n)
            self.color = None
        elif self.color == 'random greys':
            self.colors = random_grey(n)
            self.color = None
        else:
            raise InvalidParameterValueError(
                f"Invalid color parameter: {self.color}. Must be either a valid color name, hex "
                "code, or one of ['random', 'random greys'].")

    def _categorical_colors(self) -> np.ndarray:
        palette = self.color
        if isinstance(self.color_by, list):
            categories = pl.concat([self.data[[c for c in self.color_by]]], how='horizontal')
            categories = list(
                pl.Series(
                    [ '.'.join(row) for row in categories.to_numpy() ]
                ).drop_nulls().unique(maintain_order=True)
            )
        else:
            categories = self.data[self.color_by].drop_nulls().unique(maintain_order=True).to_list()
        print(categories)

        if isinstance(palette, str):
            if palette not in Dyes.cmaps:
                warn(
                    f"Colormap '{palette}' was not recognized. Defaulting to the 'peregrin_ql' colormap. "
                    "To paint categorical variables, choose from the selected colormaps, including qualitative: "
                    f"{', '.join(Dyes.qualitative_cmaps)}, or quantitative: {', '.join(Dyes.quantitative_cmaps)}",
                    category=PainterWarning, stacklevel=1)
                palette = 'peregrin_ql'
            mapped_colors = retrieve_palette(palette, categories)
        elif isinstance(palette, list):
            mapped_colors = retrieve_palette(palette, categories)
        elif isinstance(palette, dict):
            mapped_colors = palette
        else:
            raise PlottingError(
                f"Invalid palette type: {type(palette)}. Must be str, list, or dict.")


        # Vectorized value -> color mapping; unmapped/null values fall back to black.
        match self.kwargs.get('return_', 'dict'):
            case 'dict':
                return mapped_colors
            case 'array':
                categories = pl.Series(categories)
                return categories.cast(pl.Utf8).replace_strict(
                    {str(k): v for k, v in mapped_colors.items()},
                    default="#000000FF",
                ).to_numpy()
            case _:
                raise PlottingError(
                    f"Invalid <return_> argument: {self.kwargs.get('return_', 'dict')}. Must be 'dict' or 'array'.")

    def _numeric_colors(self) -> np.ndarray:
        cmap_name = self.kwargs.get('cmap', 'peregrin_gn')
        if cmap_name not in Dyes.quantitative_cmaps:
            warn(
                f"Colormap '{cmap_name}' is not a recognized quantitative colormap. "
                "Defaulting to 'peregrin_gn'. Supported colormaps include: "
                f"{', '.join(Dyes.quantitative_cmaps)}.",
                category=PainterWarning, stacklevel=2)
            cmap = retrieve_cmap('peregrin_gn')
        else:
            cmap = retrieve_cmap(cmap_name)

        norm, vals = Cmaps.scale_cmap(
            self.data[self.color_by],
            min=self.kwargs.get('lut_vmin'),
            max=self.kwargs.get('lut_vmax'),
        )
        try:
            # RGBA (N,4) float array — far cheaper downstream than an object column.
            return cmap(norm(vals))
        except Exception as e:
            raise PlottingError(
                f"Error applying quantitative colormap: '{cmap}' to data: {e}")


dyes = Dyes()

retrieve_palette = Cmaps.retrieve_palette
retrieve_cmap = Cmaps.retrieve_cmap
scale_cmap = Cmaps.scale_cmap

color_generator = ColorGenerator()
random_color = color_generator.random_color
random_grey = color_generator.random_grey
is_color_code = color_generator.is_color_code

paint = Painter().paint