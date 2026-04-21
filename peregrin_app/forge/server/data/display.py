import io
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from shiny import render, reactive, ui, req
from src.code import Summarize, Values, DebounceCalc, is_empty, Stats


def mount_data_display(input, output, session, S):
    HIST_EXCLUDE = {
        "Condition", "Replicate", "Condition color", "Replicate color",
        "Frame", "Time point", "Frame lag", "Time lag", "Track ID", "Track UID"
    }

    def _col_to_stat_id(col: str) -> str:
        return (
            col.strip()
            .lower()
            .replace(" ", "_")
            .replace(".", "_")
            .replace("{", "_l_br_")
            .replace("}", "_r_br_")
        )

    def _should_render_hist(col: str, stats: dict) -> bool:
        return stats.get("type") == "type_one" and col not in HIST_EXCLUDE

    # @reactive.extended_task
    def hist(data: pd.Series, app_theme: str = "light") -> plt.Figure:
            fig, ax = plt.subplots(figsize=(3, 3), dpi=72)
            plt.hist(
                data.dropna(), 
                bins=15,
                align='left',
                color='#337ab7', 
                edgecolor='#fafafa' if app_theme == "light" else '#2b2b2b',
            )
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            fig.set_facecolor('none')
            ax.set_facecolor('none')
            ax.margins(0)
            ax.autoscale(enable=True, tight=True)
            fig.tight_layout(pad=0.05)

            return plt.gcf()

    # @reactive.extended_task
    def rend_summary(summary: dict) -> ui.TagList:
        return ui.div(
            ui.tags.h5("Summary"),
            ui.br(),
            ui.markdown(
                f"""
                **Rows:** {summary['rows']} <br/>
                **Columns:** {summary['columns']} <br/>
                **Missing cells:** {summary['missing_cells']} <br/>
                **Memory:** {summary['memory_mb']} MB <br/>
                """
            )
        )

    
    def rend_summaries(column_stats: dict, tag: str) -> ui.TagList:
        cards = []
        for col, stats in column_stats.items():
            if col not in ["Track ID", "Track UID"]:
                if _should_render_hist(col, stats):
                    stat = _col_to_stat_id(col)
                    body = ui.div(
                        ui.div(ui.output_plot(f"hist_{tag}_{stat}", height="125px"), style="width: 125px; height: 127px;"),
                        ui.div(
                            ui.div(str(Values.RoundSigFigs(stats['min'], 5)), style="align-self: flex-start;"),
                            ui.div(str(Values.RoundSigFigs(stats['max'], 5)), style="align-self: flex-end;"),
                            style="display: flex; justify-content: space-between; width: 100%;"
                        ),
                        ui.div(ui.tags.b("missing: "), str(stats['missing'])),
                        ui.div(ui.tags.b("distinct: "), str(stats['distinct'])),
                        ui.div(ui.tags.b("mean: "), str(Values.RoundSigFigs(stats['mean'], 5))),
                        ui.div(ui.tags.b("median: "), str(Values.RoundSigFigs(stats['median'], 5))),
                        ui.div(ui.tags.b("sd: "), str(Values.RoundSigFigs(stats['sd'], 5))),
                        ui.div(ui.tags.b("variance: "), str(Values.RoundSigFigs(stats['variance'], 5))),
                        ui.div(ui.tags.b("mode: "), str(Values.RoundSigFigs(stats['mode'], 5))),
                        class_="column-body"
                    )
                else:
                    body = ui.div(
                        ui.div(ui.tags.b("missing: "), str(stats['missing'])),
                        ui.div(ui.tags.b("distinct: "), str(stats['distinct'])),
                        ui.div(ui.tags.b("top values:")) if stats.get("top") else None,
                        *[
                            ui.div(ui.tags.b(f"{val} "), f" {pct}%")
                            for val, pct in stats.get("top", [])
                        ],
                        class_="column-body"
                    )

                cards.append(
                    ui.div(
                        ui.div(col, class_="column-title"),
                        body,
                        class_="column-card"
                    )
                )

        return ui.TagList(*cards)


    # _ _ _ _ RENDERING DATA FRAMES _ _ _ _

    @reactive.extended_task
    async def summarize(spots, tracks, frames, tintervals):
        return [
            {
                "general": Summarize.dataframe_summary(spots),
                "columns": {col: Summarize.column_summary(spots[col]) for col in spots.columns}
            },
            {
                "general": Summarize.dataframe_summary(tracks),
                "columns": {col: Summarize.column_summary(tracks[col]) for col in tracks.columns}
            },
            {
                "general": Summarize.dataframe_summary(frames),
                "columns": {col: Summarize.column_summary(frames[col]) for col in frames.columns}
            },
            {
                "general": Summarize.dataframe_summary(tintervals),
                "columns": {col: Summarize.column_summary(tintervals[col]) for col in tintervals.columns}
            }
        ]

    @reactive.Effect
    @reactive.event(S.SPOTSTATS, S.TRACKSTATS, S.FRAMESTATS, S.TINTERVALSTATS)
    def _():
        spots = S.SPOTSTATS.get()
        tracks = S.TRACKSTATS.get()
        frames = S.FRAMESTATS.get()
        tintervals = S.TINTERVALSTATS.get()
        req(
            not is_empty(spots),
            not is_empty(tracks),
            not is_empty(frames),
            not is_empty(tintervals)
        )
        summarize(spots, tracks, frames, tintervals)

    @reactive.Effect
    def _():
        result = summarize.result()
        if result is not None and len(result) == 4:
            S.SPOTSUMMARY.set(result[0])
            S.TRACKSUMMARY.set(result[1])
            S.FRAMESUMMARY.set(result[2])
            S.TINTERVALSUMMARY.set(result[3])


    @output
    @render.ui
    def spots_summary():
        req(S.SPOTSUMMARY.get() is not None and "general" in S.SPOTSUMMARY.get())
        return rend_summary(S.SPOTSUMMARY.get()["general"])
    
    @output
    @render.ui
    def tracks_summary():
        req(S.TRACKSUMMARY.get() is not None and "general" in S.TRACKSUMMARY.get())
        return rend_summary(S.TRACKSUMMARY.get()["general"])
    
    @output
    @render.ui
    def frames_summary():
        req(S.FRAMESUMMARY.get() is not None and "general" in S.FRAMESUMMARY.get())
        return rend_summary(S.FRAMESUMMARY.get()["general"])
    
    @output
    @render.ui
    def tintervals_summary():
        req(S.TINTERVALSUMMARY.get() is not None and "general" in S.TINTERVALSUMMARY.get())
        return rend_summary(S.TINTERVALSUMMARY.get()["general"])


    @DebounceCalc(3)
    @reactive.calc
    def _rend_spot_summaries():
        return rend_summaries(S.SPOTSUMMARY.get()["columns"], "spots")

    @DebounceCalc(3)
    @reactive.calc
    def _rend_track_summaries():
        return rend_summaries(S.TRACKSUMMARY.get()["columns"], "tracks")
    
    @DebounceCalc(3)
    @reactive.calc
    def _rend_frame_summaries():
        return rend_summaries(S.FRAMESUMMARY.get()["columns"], "frames")
    
    @DebounceCalc(3)
    @reactive.calc
    def _rend_tinterval_summaries():
        return rend_summaries(S.TINTERVALSUMMARY.get()["columns"], "tintervals")
    

    @DebounceCalc(3)
    @reactive.calc
    def _rend_spots_tbl():
        tbl = Stats().format_digits(S.SPOTSTATS.get(), sig_figs=input.significant_figures(), decimals=input.decimal_places())
        return tbl
    
    @DebounceCalc(3)
    @reactive.calc
    def _rend_tracks_tbl():
        tbl = Stats().format_digits(S.TRACKSTATS.get(), sig_figs=input.significant_figures(), decimals=input.decimal_places())
        return tbl
    
    @DebounceCalc(3)
    @reactive.calc
    def _rend_frames_tbl():
        tbl = Stats().format_digits(S.FRAMESTATS.get(), sig_figs=input.significant_figures(), decimals=input.decimal_places())
        return tbl
    
    @DebounceCalc(3)
    @reactive.calc
    def _rend_tintervals_tbl():
        tbl = Stats().format_digits(S.TINTERVALSTATS.get(), sig_figs=input.significant_figures(), decimals=input.decimal_places())
        return tbl
    

    @output
    @render.ui
    def spots_summaries():
        req(S.SPOTSUMMARY.get() is not None and "columns" in S.SPOTSUMMARY.get())
        if not input.show_spot_stats_sums():
            return
        return _rend_spot_summaries()

    @output
    @render.ui
    def tracks_summaries():
        req(S.TRACKSUMMARY.get() is not None and "columns" in S.TRACKSUMMARY.get())
        if not input.show_track_stats_sums():
            return
        return _rend_track_summaries()
    
    @output
    @render.ui
    def frame_summaries():
        req(S.FRAMESUMMARY.get() is not None and "columns" in S.FRAMESUMMARY.get())
        if not input.show_frame_stats_sums():
            return
        return _rend_frame_summaries()

    @output
    @render.ui
    def tinterval_summaries():
        req(S.TINTERVALSUMMARY.get() is not None and "columns" in S.TINTERVALSUMMARY.get())
        if not input.show_tinterval_stats_sums():
            return
        return _rend_tinterval_summaries()



    @output
    @render.data_frame
    def spot_stats():
        req(not is_empty(S.SPOTSTATS.get()))
        if not input.show_spot_stats_tbl():
            return
        return _rend_spots_tbl()

    @output
    @render.data_frame
    def track_stats():
        req(not is_empty(S.TRACKSTATS.get()))
        if not input.show_track_stats_tbl():
            return
        return _rend_tracks_tbl()
    
    @output
    @render.data_frame
    def frame_stats():
        req(not is_empty(S.FRAMESTATS.get()))
        if not input.show_frame_stats_tbl():
            return
        return _rend_frames_tbl()
    
    @output
    @render.data_frame
    def tinterval_stats():
        req(not is_empty(S.TINTERVALSTATS.get()))
        if not input.show_tinterval_stats_tbl():
            return
        return _rend_tintervals_tbl()
    

    
    @reactive.effect
    def _():
        for col, stats in S.SPOTSUMMARY.get()["columns"].items():
            if stats["type"] == "type_one" and col not in ["Condition", "Replicate"]:
                stat_id = _col_to_stat_id(col)

                @output(id=f"hist_spots_{stat_id}")
                @render.plot
                def histogram(c=col): 
                    return hist(data=S.SPOTSTATS.get()[c], app_theme=input.app_theme())

    @reactive.effect
    def _():
        for col, stats in S.TRACKSUMMARY.get()["columns"].items():
            if stats["type"] == "type_one" and col not in ["Condition", "Replicate"]:
                stat_id = _col_to_stat_id(col)

                @output(id=f"hist_tracks_{stat_id}")
                @render.plot
                def histogram(c=col): 
                    return hist(data=S.TRACKSTATS.get()[c], app_theme=input.app_theme())

    @reactive.effect
    def _():
        for col, stats in S.FRAMESUMMARY.get()["columns"].items():
            if stats["type"] == "type_one" and col not in ["Condition", "Replicate"]:
                stat_id = _col_to_stat_id(col)

                @output(id=f"hist_frames_{stat_id}")
                @render.plot
                def histogram(c=col): 
                    return hist(data=S.FRAMESTATS.get()[c], app_theme=input.app_theme())

    @reactive.effect
    def _():
        for col, stats in S.TINTERVALSUMMARY.get()["columns"].items():
            if stats["type"] == "type_one" and col not in ["Condition", "Replicate"]:
                stat_id = _col_to_stat_id(col)

                @output(id=f"hist_tintervals_{stat_id}")
                @render.plot
                def histogram(c=col):
                    return hist(data=S.TINTERVALSTATS.get()[c], app_theme=input.app_theme())

    

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
    def download_tinterval_stats():
        time_interval_stats = S.TINTERVALSTATS.get()
        if time_interval_stats is not None and not time_interval_stats.empty:
            with io.BytesIO() as buffer:
                time_interval_stats.to_csv(buffer, index=False)
                yield buffer.getvalue()
        else:
            pass
