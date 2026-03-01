import traceback
from shiny import reactive, ui, render, req

from src.code import Level, Reporter


def mount_statcols(input, output, session, S, noticequeue):
    
    @reactive.Effect()
    @reactive.event(S.UNFILTERED_SPOTSTATS, S.UNFILTERED_TRACKSTATS, S.UNFILTERED_FRAMESTATS, S.UNFILTERED_TINTERVALSTATS, ignore_init=True, ignore_none=True)
    def update_statcols():

        try:
            spot_cols = S.UNFILTERED_SPOTSTATS.get().columns
            track_cols = S.UNFILTERED_TRACKSTATS.get().columns
            frame_cols = S.UNFILTERED_FRAMESTATS.get().columns
            tinterval_cols = S.UNFILTERED_TINTERVALSTATS.get().columns

            try:
                spot_cols = spot_cols.drop(['Track ID', 'Track UID', 'Condition', 'Replicate'])
                spot_cols = spot_cols.drop([col for col in spot_cols if ('{per replicate}' in col) or ('{per condition}' in col)], errors='ignore')
                track_cols = track_cols.drop(['Condition', 'Replicate'])
                track_cols = track_cols.drop([col for col in track_cols if ('{per replicate}' in col) or ('{per condition}' in col)], errors='ignore')
                

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
            # S.FRAMESTATS_COLUMNS.set(frame_cols)
            # S.TINTERVALSTATS_COLUMNS.set(tinterval_cols)
            
        except Exception as e:

            # Debugging lines commented out

            # error_trace = traceback.format_exc()
            # noticequeue.Report(Level.error, f"Error updating statcols: {e}", error_trace)

            Reporter(Level.error, f"Error updating statcols: {e}", noticequeue=noticequeue, trace=traceback.format_exc())

            pass