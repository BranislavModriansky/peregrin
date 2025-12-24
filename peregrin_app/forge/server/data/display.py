import io
from datetime import date
from shiny import render


def mount_data_display(input, output, session, S):

    # _ _ _ _ RENDERING DATA FRAMES _ _ _ _
    
    @render.data_frame
    def render_spot_stats():
        spot_stats = S.SPOTSTATS.get()
        if spot_stats is not None and not spot_stats.empty:
            return spot_stats
        else:
            pass

    @render.data_frame
    def render_track_stats():
        track_stats = S.TRACKSTATS.get()
        if track_stats is not None and not track_stats.empty:
            return track_stats
        else:
            pass

    @render.data_frame
    def render_frame_stats():
        frame_stats = S.FRAMESTATS.get()
        if frame_stats is not None and not frame_stats.empty:
            return frame_stats
        else:
            pass

    @render.data_frame
    def render_time_interval_stats():
        time_interval_stats = S.TINTERVALSTATS.get()
        if time_interval_stats is not None and not time_interval_stats.empty:
            return time_interval_stats
        else:
            pass
    

    # _ _ _ _ DATAFRAME CSV DOWNLOADS _ _ _ _

    @render.download(filename=f"Spot stats {date.today()}.csv")
    def download_spot_stats():
        spot_stats = S.SPOTSTATS.get()
        if spot_stats is not None and not spot_stats.empty:
            with io.BytesIO() as buffer:
                spot_stats.to_csv(buffer, index=False)
                yield buffer.getvalue()
        else:
            pass

    @render.download(filename=f"Track stats {date.today()}.csv")
    def download_track_stats():
        track_stats = S.TRACKSTATS.get()
        if track_stats is not None and not track_stats.empty:
            with io.BytesIO() as buffer:
                track_stats.to_csv(buffer, index=False)
                yield buffer.getvalue()
        else:
            pass
    
    @render.download(filename=f"Frame stats {date.today()}.csv")
    def download_frame_stats():
        frame_stats = S.FRAMESTATS.get()
        if frame_stats is not None and not frame_stats.empty:
            with io.BytesIO() as buffer:
                frame_stats.to_csv(buffer, index=False)
                yield buffer.getvalue()
        else:
            pass

    @render.download(filename=f"Time interval stats {date.today()}.csv")
    def download_time_interval_stats():
        time_interval_stats = S.TINTERVALSTATS.get()
        if time_interval_stats is not None and not time_interval_stats.empty:
            with io.BytesIO() as buffer:
                time_interval_stats.to_csv(buffer, index=False)
                yield buffer.getvalue()
        else:
            pass
