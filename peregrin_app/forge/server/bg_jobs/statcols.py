import traceback
from shiny import reactive, ui, render

from src.code import Level


def mount_statcols(input, output, session, S, noticequeue):
    
    @reactive.effect
    def update_statcols():
        try:
            spot_cols = S.SPOTSTATS.get().columns.tolist()
            track_cols = S.TRACKSTATS.get().columns.tolist()
            frame_cols = S.FRAMESTATS.get().columns.tolist()
            tinterval_cols = S.TINTERVALSTATS.get().columns.tolist()

            try:
                spot_cols.remove(['Track UID', 'Condition', 'Replicate'])
                track_cols.remove(['Track UID', 'Condition', 'Replicate'])
                frame_cols.remove(['Condition', 'Replicate'])
                tinterval_cols.remove(['Frame lag', 'Time lag', 'Condition', 'Replicate'])
            except Exception as e:

                # Debugging lines commented out

                # error_trace = traceback.format_exc()
                # noticequeue.Report(Level.error, f"Error clearing columns: {e}", error_trace)

                pass
            
            S.SPOTSTATS_COLUMNS.set(spot_cols)
            S.TRACKSTATS_COLUMNS.set(track_cols)
            S.FRAMESTATS_COLUMNS.set(frame_cols)
            S.TINTERVALSTATS_COLUMNS.set(tinterval_cols)
            
        except Exception as e:

            # Debugging lines commented out

            # error_trace = traceback.format_exc()
            # noticequeue.Report(Level.error, f"Error updating statcols: {e}", error_trace)

            pass