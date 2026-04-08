import traceback
from shiny import reactive, ui, render, req

from src.code import Level, Reporter, get_logger

_log = get_logger(__name__)


def mount_statcols(input, output, session, S, noticequeue):
    
    @reactive.Effect()
    @reactive.event(S.UNFILTERED_SPOTSTATS, S.UNFILTERED_TRACKSTATS, S.UNFILTERED_FRAMESTATS, S.UNFILTERED_TINTERVALSTATS, ignore_init=True, ignore_none=True)
    def update_statcols():
        
        try:
            spot_cols = S.UNFILTERED_SPOTSTATS.get().columns.tolist()
            track_cols = S.UNFILTERED_TRACKSTATS.get().columns.tolist()

            ignore = ["Track ID", "Track UID", "Condition", "Replicate", "Condition color", "Replicate color"]

            try:
                for col in ignore:
                    if col in spot_cols:
                        spot_cols.remove(col)
                    if col in track_cols:
                        track_cols.remove(col)
                

            except Exception as e:
                Reporter(Level.error, f"Could not drop some default columns from the stats column lists: {e}", noticequeue=noticequeue, trace=traceback.format_exc())
                pass

            try:
                spot_cols = dict(zip(spot_cols, spot_cols))
                track_cols = dict(zip(track_cols, track_cols))

            except Exception as e:
                Reporter(Level.error, f"Could not convert stats column lists to dictionaries: {e}", noticequeue=noticequeue, trace=traceback.format_exc())
                pass
            
            S.SPOTSTATS_COLUMNS.set(spot_cols)
            S.TRACKSTATS_COLUMNS.set(track_cols)

            _log.info(f"[INFO] Updated stats column lists: \nspots columns: {spot_cols}\ntracks columns: {track_cols}")
            
        except Exception as e:
            Reporter(Level.error, f"Error updating statcols: {e}", noticequeue=noticequeue, trace=traceback.format_exc())
            pass