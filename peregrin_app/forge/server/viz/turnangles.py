import io
import asyncio
import warnings

from datetime import date
import pandas as pd

from shiny import render, reactive, ui, req
from src.code import Level, TurnAnglesHeatmap, Dyes, is_empty




def mount_plot_turnangles(input, output, session, S, noticequeue):


    @reactive.Effect
    def update_choices_turnangles():

        @reactive.Effect
        @reactive.event(S.TINTERVALSTATS)
        def _():
            if S.TINTERVALSTATS.get() is None or S.TINTERVALSTATS.get().empty:
                return [
                    ui.update_selectize(id="conditions_ta", choices=[]),
                    ui.update_selectize(id="replicates_ta", choices=[])
                ]
            return [
                ui.update_selectize(id="conditions_ta", choices=S.TINTERVALSTATS.get()["Condition"].unique().tolist(), selected=S.TINTERVALSTATS.get()["Condition"].unique().tolist()),
                ui.update_selectize(id="replicates_ta", choices=S.TINTERVALSTATS.get()["Replicate"].unique().tolist(), selected=S.TINTERVALSTATS.get()["Replicate"].unique().tolist())
            ]
        
        @reactive.Effect
        @reactive.event(input.conditions_reset_ta)
        def _():

            req(S.TINTERVALSTATS.get() is not None and not S.TINTERVALSTATS.get().empty)

            ui.update_selectize(
                id="conditions_ta",
                selected=[]
            ) if input.conditions_ta() else \
            ui.update_selectize(
                id="conditions_ta",
                selected=S.TINTERVALSTATS.get()["Condition"].unique().tolist()
            )

        @reactive.Effect
        @reactive.event(input.replicates_reset_ta)
        def _():

            req(S.TINTERVALSTATS.get() is not None and not S.TINTERVALSTATS.get().empty)

            ui.update_selectize(
                id="replicates_ta",
                selected=[]
            ) if input.replicates_ta() else \
            ui.update_selectize(
                id="replicates_ta",
                selected=S.TINTERVALSTATS.get()["Replicate"].unique().tolist()
            )

    
    @ui.bind_task_button(button_id="generate_ta")
    @reactive.extended_task
    async def output_plot_turnangles(
        data,
        conditions,
        replicates,
        angle_range,
        cmap,
        title,
        text_color,
        strip_background
    ):

        def _build():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )

                return TurnAnglesHeatmap(
                    data=data,
                    conditions=conditions,
                    replicates=replicates,
                    angle_range=angle_range,
                    cmap=cmap,
                    title=title,
                    text_color=text_color,
                    strip_background=strip_background,
                    noticequeue=noticequeue
                )

        return await asyncio.get_running_loop().run_in_executor(None, _build)
    

    @reactive.Effect
    @reactive.event(input.generate_ta)
    def _():
        output_plot_turnangles.cancel()

        req(not is_empty(S.TINTERVALSTATS.get()))

        output_plot_turnangles(
            data=S.TINTERVALSTATS.get(),
            conditions=input.conditions_ta(),
            replicates=input.replicates_ta(),
            angle_range=input.angle_range_ta(),
            cmap=input.cmap_ta(),
            title=input.title_ta(),
            text_color='white' if input.app_theme() == 'dark' else 'black',
            strip_background=True
        )

    @render.plot
    def plot_ta():
        return output_plot_turnangles.result()
    
    @render.download(filename=f"Turn Angles Colormesh {date.today()}.svg")
    def download_plot_ta():
        
        req(not is_empty(S.TINTERVALSTATS.get()))

        fig = TurnAnglesHeatmap(
            data=S.TINTERVALSTATS.get(),
            conditions=input.conditions_ta(),
            replicates=input.replicates_ta(),
            angle_range=input.angle_range_ta(),
            cmap=input.cmap_ta(),
            title=input.title_ta(),
            text_color='black',
            strip_background=False,
            noticequeue=noticequeue
        )

        if fig is not None:
            buf = io.BytesIO()
            fig.savefig(buf, format="svg", bbox_inches='tight')
            buf.seek(0)
            return buf.read()
        else:
            noticequeue.Report(Level.error, "Failed to generate the plot for download.", "Please try adjusting the plot settings or check the input data.")
            return None