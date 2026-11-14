from __future__ import annotations

import io, json
from pathlib import Path
from functools import lru_cache
from typing import Any, Iterable
from IPython.core.display import HTML
from matplotlib.animation import HTMLWriter
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


import warnings

from ...utils import get_aliases
from ..painter import Dyes, retrieve_cmap


_TOOLKIT_DIR = Path(__file__).resolve().parent
CMAP_SAMPLES = 32  # Number of stops sampled per colormap.

@lru_cache(maxsize=1)  # caches the results of the function after the first call
def load_assets():
    return {
        'css': (_TOOLKIT_DIR / '_tooltip_styling.css').read_text(encoding='utf-8'),
        'js': (_TOOLKIT_DIR / '_tooltip.js').read_text(encoding='utf-8'),
    }

tooltip_assets = load_assets()


@lru_cache(maxsize=1)  # caches the results of the function after the first call
def cmap_registry() -> dict[str, Any]:
    """ Selectable quantitative colormaps + sampled hex stops for the JS side. """
    samples: dict[str, list[str]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for name in Dyes.quantitative_cmaps:
            samples[name] = [
                mcolors.to_hex(
                    retrieve_cmap(name)(i / (CMAP_SAMPLES - 1))
                ) for i in range(CMAP_SAMPLES)
            ]
    return {
        'quantitative': list(samples.keys()), 
        'samples': samples
    }


class GraphFormatter:
    """ Figure wrapper for formatting. """


    def __init__(self, builder):
        self.builder = builder
        self._svg: str | None = None
        self._html: str | None = None
        self._anim_cache: dict[tuple, Any] = {}
        if builder.figure is not None:
            plt.close(builder.figure)


    def save(
        self,
        how: str | Path = 'figure.png',
        **kw
    ) -> None:
        kw = get_aliases(kw, {'fmt': ['fileformat', 'format', 'filetype', 'type', 'extension', 'ext']})

        if not isinstance(how, str):
            how = str(how)
    
        path = Path(f"{how}").parent
        if path == Path('.'):
            path = self._caller_path()
        path = path / how

        if '.' in how: ext = how.split('.')[-1].lower()
        else: ext = kw.get('fmt', 'png').lower()

        match ext:
            case 'svg':
                print(f"Saving SVG: {path.with_suffix('.' + ext)}")
                self.builder.figure.savefig(path, format='svg', bbox_inches='tight')
            case 'png':
                print(f"Saving PNG: {path.with_suffix('.' + ext)}")
                self.builder.figure.savefig(path, format='png', bbox_inches='tight')
            case 'html':
                print(f"Saving HTML: {path.with_suffix('.' + ext)}")
                (path.with_suffix("." + ext)).write_text(self._repr_html_, encoding='utf-8')
            case _:
                try:
                    print(f"Trying save {ext.upper()}: {path.with_suffix('.' + ext)}")
                    self.builder.figure.savefig(path, format=ext, bbox_inches='tight')
                except Exception as e:
                    raise ValueError(f"Unsupported format: <ext={ext}>. Supported formats are 'svg', 'png', 'html'. Error: {e}")

    @property
    def figure(self) -> plt.Figure:
        return self.builder.figure

    @property
    def cbar(self):
        return getattr(self.builder, 'cbar', None)

    @property
    def svg(self) -> str:
        if self._svg is None:  # rendered once
            buf = io.StringIO()
            self.builder.figure.savefig(buf, format='svg', bbox_inches='tight')
            svg = buf.getvalue()
            self._svg = svg[svg.find('<svg'):]  # strip XML prolog for inlining
        return self._svg

    @property
    def html(self) -> HTML:
        return HTML(self._repr_html_)

    def animate(self, mode: str = 'jshtml', **kw) -> Any:
        """
        **`Not fully developed!`**
        
        Animate the graph.

        Parameters
        ----------
        mode : {'jshtml', 'video', 'animation'}
            'jshtml' -> ``FuncAnimation.to_jshtml()``, 'video' -> ``FuncAnimation.to_html5_video()``
            (requires ffmpeg), anything else returns the raw ``FuncAnimation``.

            *additional keyword arguments:*

        embed_limit : float, default 100.0
            Max size (MB) of the embedded animation; Matplotlib's default of 20 MB silently drops frames.
        frame_format : {'jpeg', 'png', 'tiff', 'svg'}, default 'png'
            Format of the individual frames in the animation.
        fps : int, default 30
            Frames per second for the animation.
        """
        if not hasattr(self.builder, 'animate'):
            raise NotImplementedError("Animation unavailable for this graph.")

        kw = get_aliases(kw, {'frame_format': ['frame_fmt', 'embed_format']})
        supported_formats = ('jpeg', 'png', 'tiff', 'svg')

        embed_limit = kw.get('embed_limit', 100.0)
        frame_format = kw.get('frame_format', 'png').lower()

        if frame_format not in supported_formats:
            raise ValueError(f"Unsupported embed format: {frame_format}. Supported formats are {', '.join(supported_formats)}.")

        key = (mode, tuple(sorted((k, repr(v)) for k, v in kw.items())))
        if key not in self._anim_cache:
            animation = self.builder.animate(**kw)
            match mode:
                case 'jshtml':
                    with mpl.rc_context({
                        'animation.frame_format': frame_format,
                        'animation.embed_limit': embed_limit
                    }): out = HTML(animation.to_jshtml(fps=kw.get('fps', 30)))
                case 'video':
                    if frame_format == 'svg':
                        warnings.warn("SVG format is not supported for an image sequence video format. Falling back to PNG.")
                        frame_format = 'png'
                    with mpl.rc_context({
                        'animation.frame_format': frame_format,
                        'animation.embed_limit': embed_limit
                    }): out = HTML(animation.to_html5_video())
                case _:
                    out = animation
            self._anim_cache[key] = out
        return self._anim_cache[key]


    @property
    def result(self) -> Any:
        if hasattr(self.builder, 'result'):
            return self.builder.result
        return 'No result available.'


    @property
    def style_payload(self) -> dict[str, Any]:
        if hasattr(self.builder, 'style_payload'):
            return self.builder.style_payload
        return {}


    def _caller_path(self, debug: bool = False) -> Path | None:

        # Source - https://stackoverflow.com/a/51488346
        # Posted by martineau, modified by community. See post 'Timeline' for change history
        # Retrieved 2026-10-20, License - CC BY-SA 4.0

        import sys
        namespace = sys._getframe(2).f_globals  # caller's globals

        if debug:
            from pprint import pprint
            pprint(namespace)
        if namespace.get('__file__', None) is not None:
            if debug:
                print("caller's path:", namespace['__file__'])
            return Path(rf"{namespace['__file__']}").parent
        if namespace.get('__vsc_ipynb_file__', None) is not None:
            if debug:
                print("caller's VS Code IPython Notebook file:", namespace['__vsc_ipynb_file__'])
            return Path(rf"{namespace['__vsc_ipynb_file__']}").parent
        if namespace.get('__ipynb_file__', None) is not None:
            if debug:
                print("caller's IPython Notebook file:", namespace['__ipynb_file__'])
            return Path(rf"{namespace['__ipynb_file__']}").parent
        if namespace.get('_dh', None) is not None:
            if debug:
                print("caller's current working directory stack (_dh):", namespace['_dh'])
            return Path(rf"{namespace['_dh'][0]}").parent if isinstance(namespace['_dh'], list) else Path(rf"{namespace['_dh']}").parent
        return None


    @property
    def _repr_html_(self) -> str:
        if self._html is not None:
            return self._html
        js = tooltip_assets['js'].replace(
            '"__PEREGRIN_TOOLTIP_CSS__"', json.dumps(tooltip_assets['css']))
        payload = json.dumps({
            'style': self.builder.style_payload,  # gids
            'cmaps': cmap_registry(),             # colormaps with <CMAP_SAMPLES> number of samples
        })
        self._html = (
            f'<div class="peregrin-graph" id="{self.builder.__class__.__name__}">{self.svg}</div>\n'
            f'<script>window.GRAPH_STATE = {payload};</script>\n'
            f'<script>{js}</script>'
        )
        return self._html
