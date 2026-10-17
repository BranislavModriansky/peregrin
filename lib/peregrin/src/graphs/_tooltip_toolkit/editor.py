from functools import lru_cache
from pathlib import Path

_TOOLKIT_DIR = Path(__file__).resolve().parent

@lru_cache(maxsize=1)
def load_assets():
    return {
        'css': (_TOOLKIT_DIR / '_tooltip_styling.css').read_text(encoding='utf-8'),
        'js': (_TOOLKIT_DIR / '_tooltip.js').read_text(encoding='utf-8'),
    }

tooltip_assets = load_assets()


import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import io
from typing import Any
from pathlib import Path
import json


class Interactive:
    """
    Wraps the actual Matplotlib figure, exported as SVG markup with tagged
    (gid) element groups. The JS tooltip attaches to those groups and edits
    SVG attributes in place — nothing is re-rendered client-side.
    """

    def __init__(self, builder):
        self._builder = builder

    @property
    def fig(self) -> plt.Figure:
        return self._builder.figure

    @staticmethod
    def _html_color(color: Any) -> str:
        try:
            return to_hex(color, keep_alpha=True)
        except (TypeError, ValueError):
            return '#000000ff'

    def animate(self, **kwargs):
        return self._builder.animate(**kwargs)

    def _figure_svg(self) -> str:
        buf = io.StringIO()
        self._builder.figure.savefig(buf, format='svg', bbox_inches='tight')
        svg = buf.getvalue()
        return svg[svg.find('<svg'):]  # strip XML prolog for inlining

    def _style_payload(self) -> dict[str, Any]:
        k = self._builder.kwargs
        hex_ = self._html_color
        return {
            'lw': float(k.get('lw', 1.0)),
            'headSize': 1.0,
            'trackColor': hex_(k.get('color', 'black')),
            'faceColor': hex_(k.get('face_color', 'white')),
            'gridColor': hex_(k.get('grid_color', 'gainsboro')),
            'gridLw': float(k.get('grid_lw', 0.75)),
        }

    def to_html(self) -> str:
        js = tooltip_assets['js'].replace(
            '"__PEREGRIN_TOOLTIP_CSS__"', json.dumps(tooltip_assets['css']))
        payload = json.dumps({'style': self._style_payload()})
        return (
            f'<div class="peregrin-graph">{self._figure_svg()}</div>\n'
            f'<script>window.GRAPH_STATE = {payload};</script>\n'
            f'<script>{js}</script>'
        )

    def _repr_html_(self):
        return self.to_html()

    def save(self, path, **kwargs):
        path = Path(path)
        path.write_text(self.to_html(), encoding='utf-8')
        return path