from shiny import reactive
import pandas as pd

def build_state():
    return type("State", (object,), {
        "NOTICES": reactive.Value([]),
        "PLAY": reactive.Value(False),
        "BG_SWITCH": reactive.Value(False),
        "BG_ENABLED": reactive.Value(True),
        "READYTORUN": reactive.Value(False),
        "INPUTS": reactive.Value(1),
        "THRESHOLDS": reactive.Value(None),
        "THRESHOLDS_ID": reactive.Value(1),
        "RAWDATA": reactive.Value(pd.DataFrame()),
        "UNFILTERED_SPOTSTATS": reactive.Value(pd.DataFrame()),
        "UNFILTERED_TRACKSTATS": reactive.Value(pd.DataFrame()),
        "UNFILTERED_FRAMESTATS": reactive.Value(pd.DataFrame()),
        "UNFILTERED_TINTERVALSTATS": reactive.Value(pd.DataFrame()),
        "SPOTSTATS": reactive.Value(pd.DataFrame()),
        "TRACKSTATS": reactive.Value(pd.DataFrame()),
        "FRAMESTATS": reactive.Value(pd.DataFrame()),
        "UNITS": reactive.Value(),
        "REPLAY_ANIMATION": reactive.Value(None),
    })

