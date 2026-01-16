import time
from shiny import reactive, ui, render


def mount_statcols(input, output, session, S, noticequeue):
    
    @reactive.effect
    def update_statcols():
        try:
            spot_cols = S.SPOTSTATS.get().columns.tolist()
            track_cols = S.TRACKSTATS.get().columns.tolist()
            frame_cols = S.FRAMESTATS.get().columns.tolist()
            tinterval_cols = S.TINTERVALSTATS.get().columns.tolist()
            
            S.SPOTSTATS_COLUMNS.set(spot_cols)
            S.TRACKSTATS_COLUMNS.set(track_cols)
            S.FRAMESTATS_COLUMNS.set(frame_cols)
            S.TINTERVALSTATS_COLUMNS.set(tinterval_cols)
            
        except Exception:
            pass