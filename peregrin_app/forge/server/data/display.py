import io
from datetime import date
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib as mpl
from shiny import render, reactive, ui, req
from shinywidgets import output_widget, render_widget
from src.code import Summarize, Values


def mount_data_display(input, output, session, S):

    def create_figure(data: pd.Series, app_theme: str = "Shiny") -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=data.dropna(), nbinsx=15,
                marker_color='#337ab7' if app_theme == "Shiny" else '#a15c5c',
                marker_line_width=0,
                hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
            )
        )

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=240),
            xaxis=dict(visible=False, fixedrange=True),
            yaxis=dict(visible=False, fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            dragmode=False,
            bargap=0.1,
            hoverlabel=dict(
                bgcolor="white" if app_theme == "Shiny" else "#1d1d1d",
                font_size=10,
                bordercolor="#ddd" if app_theme == "Shiny" else "#2b2b2b",
                font_color="#333" if app_theme == "Shiny" else "#dcdcdc"
            ),
            modebar=dict(remove=["zoom", "pan", "select", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d", "hoverClosestCartesian", "hoverCompareCartesian"])
        )

        return fig
    
    def hist(data: pd.Series, app_theme: str = "Shiny") -> go.Figure:
            fig, ax = plt.subplots(figsize=(3, 3), dpi=72)
            plt.hist(
                data.dropna(), 
                bins=15,
                align='left',
                color='#337ab7' if app_theme == "Shiny" else '#a15c5c', 
                edgecolor='#fafafa' if app_theme == "Shiny" else '#2b2b2b',
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
                
                if (stats["type"] == "type_one" 
                    and col not in ["Condition", "Replicate", "Condition color", "Replicate color", 
                                    "Frame", "Time point", "Frame lag", "Time lag"]):
                    stat = col.strip().lower().replace(" ", "_")
                    body = ui.div(
                        ui.div(ui.output_plot(f"hist_{tag}_{stat}", height="125px"), style="width: 125px; height: 127px;"),
                        ui.div(ui.tags.b("missing: "), str(stats['missing'])),
                        ui.div(ui.tags.b("distinct: "), str(stats['distinct'])),
                        ui.div(ui.tags.b("min: "), str(Values.RoundSigFigs(stats['min'], 5))),
                        ui.div(ui.tags.b("mean: "), str(Values.RoundSigFigs(stats['mean'], 5))),
                        ui.div(ui.tags.b("median: "), str(Values.RoundSigFigs(stats['median'], 5))),
                        ui.div(ui.tags.b("max: "), str(Values.RoundSigFigs(stats['max'], 5))),
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

    @reactive.Effect
    @reactive.event(S.SPOTSTATS, S.TRACKSTATS, S.FRAMESTATS, S.TINTERVALSTATS)
    def summarize():
        req(S.SPOTSTATS.get() is not None and not S.SPOTSTATS.get().empty
            and S.TRACKSTATS.get() is not None and not S.TRACKSTATS.get().empty
            and S.FRAMESTATS.get() is not None and not S.FRAMESTATS.get().empty
            and S.TINTERVALSTATS.get() is not None and not S.TINTERVALSTATS.get().empty)
        
        S.SPOTSUMMARY.set({
            "general": Summarize.dataframe_summary(S.SPOTSTATS.get()),
            "columns": {
                col: Summarize.column_summary(S.SPOTSTATS.get()[col])
                for col in S.SPOTSTATS.get().columns
            }
        })
        S.TRACKSUMMARY.set({
            "general": Summarize.dataframe_summary(S.TRACKSTATS.get()),
            "columns": {
                col: Summarize.column_summary(S.TRACKSTATS.get()[col])
                for col in S.TRACKSTATS.get().columns
            }
        })
        S.FRAMESUMMARY.set({
            "general": Summarize.dataframe_summary(S.FRAMESTATS.get()),
            "columns": {
                col: Summarize.column_summary(S.FRAMESTATS.get()[col])
                for col in S.FRAMESTATS.get().columns
            }
        })
        S.TINTERVALSUMMARY.set({
            "general": Summarize.dataframe_summary(S.TINTERVALSTATS.get()),
            "columns": {
                col: Summarize.column_summary(S.TINTERVALSTATS.get()[col])
                for col in S.TINTERVALSTATS.get().columns
            }
        })

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

    @output
    @render.ui
    def spots_summaries():
        req(S.SPOTSUMMARY.get() is not None and "columns" in S.SPOTSUMMARY.get())
        return rend_summaries(S.SPOTSUMMARY.get()["columns"], "spots")
    
    @output
    @render.ui
    def tracks_summaries():
        req(S.TRACKSUMMARY.get() is not None and "columns" in S.TRACKSUMMARY.get())
        return rend_summaries(S.TRACKSUMMARY.get()["columns"], "tracks")
    
    @output
    @render.ui
    def frame_summaries():
        req(S.FRAMESUMMARY.get() is not None and "columns" in S.FRAMESUMMARY.get())
        return rend_summaries(S.FRAMESUMMARY.get()["columns"], "frames")
    
    @output
    @render.ui
    def tinterval_summaries():
        req(S.TINTERVALSUMMARY.get() is not None and "columns" in S.TINTERVALSUMMARY.get())
        return rend_summaries(S.TINTERVALSUMMARY.get()["columns"], "tintervals")

    @output
    @render.data_frame
    def spot_stats():
        req(S.SPOTSTATS.get() is not None and not S.SPOTSTATS.get().empty)
        return S.SPOTSTATS.get()

    @output
    @render.data_frame
    def track_stats():
        req(S.TRACKSTATS.get() is not None and not S.TRACKSTATS.get().empty)
        return S.TRACKSTATS.get()
    
    @output
    @render.data_frame
    def frame_stats():
        req(S.FRAMESTATS.get() is not None and not S.FRAMESTATS.get().empty)
        return S.FRAMESTATS.get()

    @output
    @render.data_frame
    def tinterval_stats():
        req(S.TINTERVALSTATS.get() is not None and not S.TINTERVALSTATS.get().empty)
        return S.TINTERVALSTATS.get()
    

    
    
    @reactive.effect
    @reactive.event(S.SPOTSUMMARY)
    def _():
        for col, stats in S.SPOTSUMMARY.get()["columns"].items():
            if stats["type"] == "type_one" and col not in ["Condition", "Replicate"]:
                stat_id = col.strip().lower().replace(" ", "_")
                
                @output(id=f"hist_spots_{stat_id}")
                @render.plot
                def histogram(c=col): 
                    return hist(data=S.SPOTSTATS.get()[c], app_theme=input.app_theme())
                
                
                    
    @reactive.effect
    @reactive.event(S.TRACKSUMMARY)
    def _():
        for col, stats in S.TRACKSUMMARY.get()["columns"].items():
            if stats["type"] == "type_one" and col not in ["Condition", "Replicate"]:
                stat_id = col.strip().lower().replace(" ", "_")

                @output(id=f"hist_tracks_{stat_id}")
                @render.plot
                def histogram(c=col): 
                    fig = hist(data=S.TRACKSTATS.get()[c], app_theme=input.app_theme())

                    return fig
                
    @reactive.effect
    @reactive.event(S.FRAMESUMMARY)
    def _():
        for col, stats in S.FRAMESUMMARY.get()["columns"].items():
            if stats["type"] == "type_one" and col not in ["Condition", "Replicate"]:
                stat_id = col.strip().lower().replace(" ", "_")

                @output(id=f"hist_frames_{stat_id}")
                @render.plot
                def histogram(c=col): 
                    fig = hist(data=S.FRAMESTATS.get()[c], app_theme=input.app_theme())

                    return fig
                
    @reactive.effect
    @reactive.event(S.TINTERVALSUMMARY)
    def _():
        for col, stats in S.TINTERVALSUMMARY.get()["columns"].items():
            if stats["type"] == "type_one" and col not in ["Condition", "Replicate"]:
                stat_id = col.strip().lower().replace(" ", "_")

                @output(id=f"hist_tintervals_{stat_id}")
                @render.plot
                def histogram(c=col):
                    fig = hist(data=S.TINTERVALSTATS.get()[c], app_theme=input.app_theme())

                    return fig

    

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
