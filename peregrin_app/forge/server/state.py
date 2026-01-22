from shiny import reactive
import pandas as pd

def build_state():
    return type("State", (object,), {
        "PLAY": reactive.Value(False),
        "BG_SWITCH": reactive.Value(False),
        "BG_ENABLED": reactive.Value(True),
        "IMPORT_MODE": reactive.Value("raw"),
        "READYTORUN": reactive.Value(False),
        "INPUTS": reactive.Value(1),
        "THRESHOLDS_ID": reactive.Value(1),
        "THRESHOLDS": reactive.Value({
            0: {
                "property": None,
                "filter": None,
                "selection": None,
                "mask": None,
                "series": None,
                "ambit": None,
            }
        }),
        "RAWDATA": reactive.Value(pd.DataFrame()),
        "UNITS": reactive.Value(),
        "UNFILTERED_SPOTSTATS": reactive.Value(pd.DataFrame()),
        "UNFILTERED_TRACKSTATS": reactive.Value(pd.DataFrame()),
        "UNFILTERED_FRAMESTATS": reactive.Value(pd.DataFrame()),
        "UNFILTERED_TINTERVALSTATS": reactive.Value(pd.DataFrame()),
        "SPOTSTATS": reactive.Value(pd.DataFrame()),
        "TRACKSTATS": reactive.Value(pd.DataFrame()),
        "FRAMESTATS": reactive.Value(pd.DataFrame()),
        "TINTERVALSTATS": reactive.Value(pd.DataFrame()),
        "SPOTSUMMARY": reactive.Value(),
        "TRACKSUMMARY": reactive.Value(),
        "FRAMESUMMARY": reactive.Value(),
        "TINTERVALSUMMARY": reactive.Value(),
        "SPOTSTATS_COLUMNS": reactive.Value([]),
        "TRACKSTATS_COLUMNS": reactive.Value([]),
        "FRAMESTATS_COLUMNS": reactive.Value([]),
        "TINTERVALSTATS_COLUMNS": reactive.Value([]),
        "REPLAY_ANIMATION": reactive.Value(),
        "MIN_DENSITY": reactive.Value(),
        "MAX_DENSITY": reactive.Value(),
        "_init_thresh": reactive.Value(True),
    })

