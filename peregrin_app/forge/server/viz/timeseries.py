import io
import asyncio
import warnings

from datetime import date
import pandas as pd

from shiny import render, reactive, ui, req
from src.code import TSeries, is_empty




def mount_plot_ts(input, output, session, S, noticequeue):

    @reactive.Effect
    def update_choices():

        @reactive.Effect
        @reactive.event(S.FRAMESTATS)
        def _():
            if is_empty(S.FRAMESTATS.get()):
                conditions = []
                replicates = []

            elif not is_empty(S.FRAMESTATS.get()) :
                conditions = S.FRAMESTATS.get()['Condition'].unique().tolist()
                
                if 'Replicate' not in S.FRAMESTATS.get().columns:
                    replicates = []
                else:
                    replicates = S.FRAMESTATS.get()['Replicate'].unique().tolist()
            
            ui.update_selectize(id="conditions_ts", choices=conditions, selected=conditions[0] if conditions else None)
            ui.update_selectize(id="replicates_ts", choices=replicates, selected=replicates if replicates else None)
            
        
        @reactive.Effect
        @reactive.event(input.conditions_reset_ts)
        def _():

            req(not is_empty(S.FRAMESTATS.get()))

            ui.update_selectize(
                id="conditions_ts",
                selected=[]
            ) if input.conditions_ts() else \
            ui.update_selectize(
                id="conditions_ts",
                selected=S.FRAMESTATS.get()["Condition"].unique().tolist()
            )
        
        @reactive.Effect
        @reactive.event(input.replicates_reset_ts)
        def _():

            req(not is_empty(S.FRAMESTATS.get()))

            ui.update_selectize(
                id="replicates_ts",
                selected=[]
            ) if input.replicates_ts() else \
            ui.update_selectize(
                id="replicates_ts",
                selected=S.FRAMESTATS.get()["Replicate"].unique().tolist()
            )

    def _ts_kwargs() -> dict:
        return dict(
            data = S.FRAMESTATS.get(),
            conditions = input.conditions_ts(),
            replicates = input.replicates_ts(),
            metric = input.metric_ts(),
            stat = input.statistic_ts(),
            disper = input.dispertion_ts(),
            level = input.level_ts(),
            palette = input.stock_palette_ts(),
            stock_palette = input.palette_ts(),
            figsize = (input.fig_width_ts(), input.fig_height_ts()),
            xscale = input.xscale_ts(),
            title = input.title_ts(),
            noticequeue = noticequeue
        )

    ui.bind_task_button(button_id="sp_generate")
    @reactive.extended_task
    async def output_time_series(kwargs: dict):
        def _build(_kwargs=kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )
                return TSeries(**_kwargs).plot()

        return await asyncio.get_running_loop().run_in_executor(None, _build)
    

    @reactive.Effect
    @reactive.event(input.generate_ts, ignore_none=False)
    def _trigger():
        output_time_series.cancel()

        req(not is_empty(S.FRAMESTATS.get()))

        kwargs = _ts_kwargs()

        output_time_series(kwargs)

    
    @render.plot
    def plot_ts():
        return output_time_series.result()


    @render.download(filename=f"Time series {date.today()}.svg", media_type="svg")
    def ts_download_svg():
        req(not is_empty(S.FRAMESTATS.get()))

        fig = TSeries(**_ts_kwargs()).plot()

        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="svg", bbox_inches='tight')
                yield buffer.getvalue()

    @render.download(filename=f"Time series {date.today()}.png", media_type="png")
    def ts_download_png():
        req(not is_empty(S.FRAMESTATS.get()))

        fig = TSeries(**_ts_kwargs()).plot()

        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="png", bbox_inches='tight')
                yield buffer.getvalue()