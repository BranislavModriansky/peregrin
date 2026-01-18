import traceback
from shiny import reactive, ui, render, req

from src.code import Level


def mount_statcols(input, output, session, S, noticequeue):
    
    @reactive.Effect
    def update_statcols():

        try:
            spot_cols = S.SPOTSTATS.get().columns
            track_cols = S.TRACKSTATS.get().columns
            frame_cols = S.FRAMESTATS.get().columns
            tinterval_cols = S.TINTERVALSTATS.get().columns

            try:
                spot_cols = spot_cols.drop(['Track ID', 'Track UID', 'Condition', 'Replicate'])
                track_cols = track_cols.drop(['Condition', 'Replicate'])
                frame_cols = frame_cols.drop(['Condition', 'Replicate'])
                tinterval_cols = tinterval_cols.drop(['Frame lag', 'Time lag', 'Condition', 'Replicate'])

            except Exception as e:

                # Debugging lines commented out

                # error_trace = traceback.format_exc()
                # print(error_trace)
                # noticequeue.Report(Level.error, f"Error clearing columns: {e}", error_trace)

                pass
            
            S.SPOTSTATS_COLUMNS.set(spot_cols.to_list())
            S.TRACKSTATS_COLUMNS.set(track_cols.to_list())
            S.FRAMESTATS_COLUMNS.set(frame_cols.to_list())
            S.TINTERVALSTATS_COLUMNS.set(tinterval_cols.to_list())
            
        except Exception as e:

            # Debugging lines commented out

            # error_trace = traceback.format_exc()
            # noticequeue.Report(Level.error, f"Error updating statcols: {e}", error_trace)

            pass