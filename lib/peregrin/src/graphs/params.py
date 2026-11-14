from dataclasses import dataclass


@dataclass
class GraphParams:

    theme: str = 'default'
    GRAPH_THEMES = {
        'default': {
            'text_color': 'black',
            'frame_color': '#252525',
            'grid_color': '#e0e0e0',
            'tick_color': 'none',
            'tick_length': 1,
            'grid_linestyle': '-',
        }
    }

    def __init__(self):
        pass

    @property
    def get(self):
        return self.GRAPH_THEMES.get(self.theme, {})
    
GraphParams = GraphParams()