import matplotlib
import polars as pl

matplotlib.use('Agg')

from src.data_compute.data_frames import Calc
from src.graphs.visualize.mean_squared_displacement import msd


def test_msd_html_exposes_category_colors_and_editable_labels(monkeypatch):
    monkeypatch.setattr(
        Calc,
        'metadata',
        {'timeunits': 's', 'spatialunits': 'um'},
        raising=False,
    )
    data = pl.DataFrame({
        'subset': ['A', 'A', 'B', 'B'],
        'time_lag': [1.0, 2.0, 1.0, 2.0],
        'MSD': [1.0, 4.0, 2.0, 8.0],
        'MSD_sd': [0.2, 0.4, 0.3, 0.5],
        'tracks_contributing': [5, 5, 6, 6],
    })

    graph = msd(
        data,
        band='sd',
        log=True,
        linear_fit=True,
        scatter='o',
        scatter_fill=None,
        fmt='html',
        color={'A': '#123456', 'B': '#abcdef'},
        title='Editable MSD title',
    )

    payload = graph.style_payload['gid']
    svg = graph.svg
    for gid in ('title', 'xlabel', 'ylabel'):
        assert f'id="pg:{gid}"' in svg

    for idx in range(2):
        prefix = f'pg:msd-category-{idx}-'
        for artist in ('series-curve', 'series-legend', 'band', 'fit', 'diffusion'):
            assert f'id="{prefix}{artist}"' in svg

    for idx, (label, color) in enumerate((('A', '#123456'), ('B', '#abcdef'))):
        series = payload[f'msd-category-{idx}-series-']
        assert series['type'] == 'msd_category'
        assert series['values'] == {}
        curve = series
        assert curve['color']['colortype'] == 'qualitative'
        assert curve['color']['categories'] == [
            {'token': 'c0', 'label': label, 'color': color}
        ]
        for artist in ('band', 'fit', 'diffusion'):
            assert payload[f'msd-category-{idx}-{artist}']['color'] == {
                'colortype': 'matching',
                'target': f'msd-category-{idx}-series-',
                'indices': [0],
            }
        assert payload[f'msd-category-{idx}-fit']['type'] == 'msd_fit'
        assert payload[f'msd-category-{idx}-fit']['values'] == {}

    shared = graph.style_payload['shared_controls']
    assert [control['key'] for control in shared] == ['width', 'fit_width', 'size']
    assert [control['apply'] for control in shared] == [
        'line_width',
        'line_width',
        'scatter_size',
    ]
    assert shared[0]['targets'] == [
        'msd-category-0-series-curve',
        'msd-category-0-series-legend',
        'msd-category-1-series-curve',
        'msd-category-1-series-legend',
    ]
    assert shared[1]['targets'] == [
        'msd-category-0-fit',
        'msd-category-1-fit',
    ]
    assert shared[2]['targets'] == [
        'msd-category-0-series-curve',
        'msd-category-1-series-curve',
    ]

def test_msd_line_only_legend_shares_the_series_width_control(monkeypatch):
    monkeypatch.setattr(
        Calc,
        'metadata',
        {'timeunits': 's', 'spatialunits': 'um'},
        raising=False,
    )
    data = pl.DataFrame({
        'subset': ['A', 'A'],
        'time_lag': [1.0, 2.0],
        'MSD': [1.0, 4.0],
        'tracks_contributing': [5, 5],
    })

    graph = msd(data, log=True, line='--', scatter=None, color={'A': '#123456'})

    series = graph.style_payload['gid']['msd-category-0-series-']
    assert series['type'] == 'msd_category'
    assert series['values'] == {}
    assert [control['key'] for control in graph.style_payload['shared_controls']] == ['width']
    assert 'id="pg:msd-category-0-series-curve"' in graph.svg
    assert 'id="pg:msd-category-0-series-legend"' in graph.svg
