
from typing import Any, Iterable, Optional

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def uniform(color: Any) -> dict[str, Any]:
    """ Uniform color descriptor. """
    return {
        'colortype': 'uniform', 
        'value': mcolors.to_hex(color)
    }


def random() -> dict[str, Any]:
    """ Random color descriptor. """
    return {
        'colortype': 'random'
    }


def qualitative(items: Iterable[tuple[Any, Any]], indices: Optional[Iterable[int]] = None) -> dict[str, Any]:
    """ Qualitative colormap descriptor acquiring `(label, color)` pairs and optional per-node category indices. """
    desc: dict[str, Any] = {
        'colortype': 'qualitative',
        'categories': [{
                'token': f'c{i}', 
                'label': str(label), 
                'value': mcolors.to_hex(color)
            } for i, (label, color) in enumerate(items)],
    }
    if indices is not None:
        desc['indices'] = np.asarray(indices, dtype=int).tolist()
    return desc


def quantitative(cmap: str, values: Iterable[float], norm: plt.Normalize = None) -> dict[str, Any]:
    """ Quantitative colormap descriptor acquiring continuous color values. """
    # vals = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    values = np.asarray(values, dtype=float)
    if norm is not None: values = norm(values)
    else: values = np.clip(values, 0.0, 1.0)
    return {
        'colortype': 'quantitative',
        'cmap': cmap,
        'values': np.round(values, 4).tolist(),
    }


def matching(target: str, values: Optional[Iterable[float]] = None, norm: plt.Normalize = None) -> dict[str, Any]:
    """
    Color descriptor that mirrors another gid's color instead of exposing
    its own color control in the tooltip (e.g. track heads matching the
    color of their track).

    `target` is the suffix of the gid to mirror. If the target uses a
    quantitative colormap, `values` supplies this gid's own per-node
    normalized positions on that colormap (e.g. each track head's value at
    its track's last point), so every node is colored to match its own
    spot rather than sharing one color. `values` is ignored when the
    target isn't quantitative.
    """
    desc: dict[str, Any] = {'colortype': 'matching', 'target': target}
    try:
        if values is not None:
            vals = np.asarray(values, dtype=float)
            if norm is not None: vals = norm(vals)
            else: vals = np.clip(vals, 0.0, 1.0)
            desc['values'] = np.round(vals, 4).tolist()
    except Exception:
        pass
    return desc

def category_gid(layer_id: str, index: int) -> str:
    """ gid for the `index`-th category's artist, matched by the JS tooltip. """
    return f'pg:{layer_id}:c{index}'



