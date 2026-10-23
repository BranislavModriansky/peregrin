from __future__ import annotations

import io, json
from pathlib import Path
from functools import lru_cache
from typing import Any, Iterable
from IPython.core.display import HTML
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
        plt.close(self.builder.figure)

    def save(
        self,
        how: str | Path = 'figure.png',
        **kw
    ) -> None:
        kw = get_aliases(kw, {'fmt': ['fileformat', 'format', 'filetype', 'type', 'extension', 'ext']})

        if not isinstance(how, str):
            how = str(how)

        print(f"Saving figure as: {how}")
    
        path = Path(f"{how}").parent
        if path == Path('.'):
            path = self._caller_path()
        path = path / how

        print(f"Resolved path: {path}")

        if '.' in how:
            ext = how.split('.')[-1].lower()
        else:
            ext = kw.get('fmt', 'png').lower()

        print(f"File ext: {ext}")

        match ext:
            case 'svg':
                self.builder.figure.savefig(path, format='svg', bbox_inches='tight')
            case 'png':
                self.builder.figure.savefig(path, format='png', bbox_inches='tight')
            case 'html':
                print(f"Saving HTML as: {path.with_suffix('.' + ext)}")
                (path.with_suffix("." + ext)).write_text(self.__repr_html__, encoding='utf-8')
            case _:
                try:
                    self.builder.figure.savefig(path, format=ext, bbox_inches='tight')
                except Exception as e:
                    raise ValueError(f"Unsupported format: <ext={ext}>. Supported formats are 'svg', 'png', 'html'. Error: {e}")

    @property
    def figure(self) -> plt.Figure:
        return self.builder.figure

    @property
    def svg(self) -> str:
        buf = io.StringIO()
        self.builder.figure.savefig(buf, format='svg', bbox_inches='tight')
        svg = buf.getvalue()
        return svg[svg.find('<svg'):]  # strip XML prolog for inlining

    @property
    def html(self) -> HTML:
        return HTML(self.__repr_html__)

    def animate(self) -> Any:
        if hasattr(self.builder, 'animate'):
            return self.builder.animate()
        raise NotImplementedError("Animation not available for this graph.")

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
    def __repr_html__(self) -> str:
        js = tooltip_assets['js'].replace(
            '"__PEREGRIN_TOOLTIP_CSS__"', json.dumps(tooltip_assets['css']))
        payload = json.dumps({
            'style': self.builder.style_payload,  # gids
            'cmaps': cmap_registry(),             # colormaps with <CMAP_SAMPLES> number of samples
        })
        return (
            f'<div class="peregrin-graph" id="{self.builder.__class__.__name__}">{self.svg}</div>\n'
            f'<script>window.GRAPH_STATE = {payload};</script>\n'
            f'<script>{js}</script>'
        )
