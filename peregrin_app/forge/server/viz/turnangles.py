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

            if is_empty(S.TINTERVALSTATS.get()):
                conditions = []
                replicates = []

            elif not is_empty(S.TINTERVALSTATS.get()) :
                conditions = S.TINTERVALSTATS.get()['Condition'].unique().tolist()
                
                if 'Replicate' not in S.TINTERVALSTATS.get().columns:
                    replicates = []
                else:
                    replicates = S.TINTERVALSTATS.get()['Replicate'].unique().tolist()
            
            return [
                ui.update_selectize(id='condition_ta', choices=conditions, selected=conditions[0] if conditions else None),
                ui.update_selectize(id='replicates_ta', choices=replicates, selected=replicates)
            ]
        
        @reactive.Effect
        @reactive.event(input.conditions_reset_ta)
        def _():

            req(not is_empty(S.TINTERVALSTATS.get()))

            ui.update_selectize(
                id="condition_ta",
                selected=[]
            ) if input.condition_ta() else \
            ui.update_selectize(
                id="condition_ta",
                selected=S.TINTERVALSTATS.get()["Condition"].unique().tolist()
            )

        @reactive.Effect
        @reactive.event(input.replicates_reset_ta)
        def _():

            req(not is_empty(S.TINTERVALSTATS.get()) and "Replicate" in S.TINTERVALSTATS.get().columns)

            ui.update_selectize(
                id="replicates_ta",
                selected=[]
            ) if input.replicates_ta() else \
            ui.update_selectize(
                id="replicates_ta",
                selected=S.TINTERVALSTATS.get()["Replicate"].unique().tolist()
            )


    def _ta_kwargs():
        return {
            "data": S.TINTERVALSTATS.get(),
            "condition": input.condition_ta(),
            "replicates": input.replicates_ta(),
            "angle_range": input.angle_range_ta(),
            "tlag_range": input.tlag_range_ta(),
            "cmap": input.cmap_ta(),
            "figsize": (input.ta_fig_width(), input.ta_fig_height()),
            "title": input.title_ta(),
            "text_color": 'white' if input.app_theme() == 'dark' else 'black',
            "strip_background": True,
        }

    
    @ui.bind_task_button(button_id="generate_ta")
    @reactive.extended_task
    async def output_plot_turnangles(kwargs):

        def _build(_kwargs=kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )

                return TurnAnglesHeatmap(**_kwargs)

        return await asyncio.get_running_loop().run_in_executor(None, _build)
    

    @reactive.Effect
    @reactive.event(input.generate_ta)
    def _():
        output_plot_turnangles.cancel()

        req(not is_empty(S.TINTERVALSTATS.get()))

        output_plot_turnangles(_ta_kwargs())


    @render.plot
    def plot_ta():
        return output_plot_turnangles.result()
    

    @render.download(filename=f"Directional change colormesh {date.today()}.svg", media_type="svg")
    def ta_download_svg():
        fig = TurnAnglesHeatmap(**_ta_kwargs())

        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="svg", bbox_inches='tight')
                yield buffer.getvalue()

    @render.download(filename=f"Directional change colormesh {date.today()}.png", media_type="png")
    def ta_download_png():
        fig = TurnAnglesHeatmap(**_ta_kwargs())

        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="png", bbox_inches='tight')
                yield buffer.getvalue()