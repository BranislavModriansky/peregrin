import io
import asyncio
import warnings

from datetime import date
import pandas as pd

from shiny import render, reactive, ui, req
from peregrin_app.src.code._plot._tracks._reconstruct import VisualizeTracksRealistics
from src.code import MSD




def mount_plot_msd(input, output, session, S, noticequeue):

    @reactive.Effect
    @reactive.event(S.UNFILTERED_TINTERVALSTATS)
    def update_choices_msd():
        if S.UNFILTERED_TINTERVALSTATS.get() is None or S.UNFILTERED_TINTERVALSTATS.get().empty:
            return [
                ui.update_selectize(id="conditions_msd", choices=[]),
                ui.update_selectize(id="replicates_msd", choices=[])
            ]
        return [
            ui.update_selectize(id="conditions_msd", choices=S.UNFILTERED_TINTERVALSTATS.get()["Condition"].unique().tolist(), selected=S.UNFILTERED_TINTERVALSTATS.get()["Condition"].unique().tolist()),
            ui.update_selectize(id="replicates_msd", choices=S.UNFILTERED_TINTERVALSTATS.get()["Replicate"].unique().tolist(), selected=S.UNFILTERED_TINTERVALSTATS.get()["Replicate"].unique().tolist())
        ]
        
    @ui.bind_task_button(button_id="generate_msd")
    @reactive.extended_task
    async def output_plot_msd(
        data,
        conditions,
        replicates,
        group_replicates,
        c_mode,
        color,
        line,
        scatter,
        linear_fit,
        errorband,
        grid,
        title
    ):
        def _build():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )

                return MSD(
                    data=data,
                    conditions=conditions,
                    replicates=replicates,
                    group_replicates=group_replicates,
                    c_mode=c_mode,
                    color=color,
                    noticequeue=noticequeue
                ).plot(
                    line=line,
                    scatter=scatter,
                    errorband=errorband,
                    linear_fit=linear_fit,
                    grid=grid,
                    title=title
                )
            
        # Run in executor, then store the figure in the reactive value
        return await asyncio.get_running_loop().run_in_executor(None, _build)
        
    @reactive.Effect
    @reactive.event(input.generate_msd, ignore_none=False)
    def _():

        output_plot_msd.cancel()

        req(S.UNFILTERED_TINTERVALSTATS() is not None and not S.UNFILTERED_TINTERVALSTATS().empty)

        if input.error_band_show_msd():
            errorband = input.error_band_type_msd()
        else:
            errorband = None

        if input.c_mode_msd() == "only-one-color":
            color = input.only_one_color_msd()
        else:
            color = None

        output_plot_msd(
            data=S.UNFILTERED_TINTERVALSTATS(), #TODO: implement filtering for timelags stats
            conditions=input.conditions_msd(),
            replicates=input.replicates_msd(),
            group_replicates=input.replicates_group_msd(),
            c_mode=input.c_mode_msd(),
            color=color,
            line=input.line_show_msd(),
            scatter=input.scatter_show_msd(),
            linear_fit=input.fit_show_msd(),
            errorband=errorband,
            grid=input.grid_show_msd(),
            title=input.title_msd()
        )

    
    @render.plot
    def plot_msd():
        return output_plot_msd.result()
    
    