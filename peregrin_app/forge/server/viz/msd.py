import io
import asyncio
import warnings

from datetime import date
import pandas as pd

from shiny import render, reactive, ui, req
from src.code import MSD, Dyes, is_empty




def mount_plot_msd(input, output, session, S, noticequeue):

    @reactive.Effect
    def update_choices():

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
            
            ui.update_selectize(id="conditions_msd", choices=conditions, selected=conditions[0] if conditions else None)
            ui.update_selectize(id="replicates_msd", choices=replicates, selected=replicates if replicates else None)
            
        
        @reactive.Effect
        @reactive.event(input.conditions_reset_msd)
        def _():

            req(not is_empty(S.TINTERVALSTATS.get()))

            ui.update_selectize(
                id="conditions_msd",
                selected=[]
            ) if input.conditions_msd() else \
            ui.update_selectize(
                id="conditions_msd",
                selected=S.TINTERVALSTATS.get()["Condition"].unique().tolist()
            )
        
        @reactive.Effect
        @reactive.event(input.replicates_reset_msd)
        def _():

            req(not is_empty(S.TINTERVALSTATS.get()) and "Replicate" in S.TINTERVALSTATS.get().columns)

            ui.update_selectize(
                id="replicates_msd",
                selected=[]
            ) if input.replicates_msd() else \
            ui.update_selectize(
                id="replicates_msd",
                selected=S.TINTERVALSTATS.get()["Replicate"].unique().tolist()
            )


    def _get_class_kwargs():
        return dict(
            data=S.TINTERVALSTATS(),
            conditions=input.conditions_msd(),
            replicates=input.replicates_msd(),
            level=input.aggregation_msd(),
            stock_palette=input.palette_stock_msd(),
            palette=input.palette_stock_type_msd(),
            noticequeue=noticequeue
        )

    def _get_plot_kwargs():
        return dict(
            statistic=input.statistic_msd(),
            line=input.line_show_msd(),
            scatter=input.scatter_show_msd(),
            linear_fit=input.fit_show_msd(),
            errorband=input.error_band_type_msd(),
            grid=input.grid_show_msd(),
            title=input.title_msd()
        )


    @ui.bind_task_button(button_id="generate_msd")
    @reactive.extended_task
    async def output_plot_msd(class_kwargs, plot_kwargs):
        def _build(_class_kwargs=class_kwargs, _plot_kwargs=plot_kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )
                return MSD(**_class_kwargs).plot(**_plot_kwargs)
            
        return await asyncio.get_running_loop().run_in_executor(None, _build)


    @reactive.Effect
    @reactive.event(input.generate_msd, ignore_init=True, ignore_none=False)
    def _():

        req(not is_empty(S.TINTERVALSTATS.get())
            and "Condition" in S.TINTERVALSTATS.get().columns 
            and "Replicate" in S.TINTERVALSTATS.get().columns)
        
        output_plot_msd.cancel()
        output_plot_msd(_get_class_kwargs(), _get_plot_kwargs())

    
    @render.plot
    def plot_msd():
        return output_plot_msd.result()
    

    
    @render.download(filename=f"MSD {date.today()}.svg")
    def download_plot_msd():

        req(S.TINTERVALSTATS() is not None and not S.TINTERVALSTATS().empty)

        fig = MSD(_get_class_kwargs()).plot(_get_plot_kwargs())


        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="svg", bbox_inches="tight")
                yield buffer.getvalue()