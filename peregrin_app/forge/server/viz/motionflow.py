import io
import warnings
import asyncio

from shiny import req, ui, render, reactive
from src.code import is_empty, Dyes, Level, MotionFlowPlot



# TODO: arrow_scale_by_mf update the selectize for spotstats columns




def mount_motionflow(input, output, session, S, noticequeue):

    @reactive.Effect
    def update_choices():

        @reactive.Effect
        @reactive.event(S.TRACKSTATS)
        def _():
            if is_empty(S.TRACKSTATS.get()):
                conditions = []
                replicates = []

            elif not is_empty(S.TRACKSTATS.get()) :
                conditions = S.TRACKSTATS.get()['Condition'].unique().tolist()
                
                if 'Replicate' not in S.TRACKSTATS.get().columns:
                    replicates = []
                else:
                    replicates = S.TRACKSTATS.get()['Replicate'].unique().tolist()
            
            ui.update_selectize(id="conditions_mf", choices=conditions, selected=conditions[0] if conditions else None)
            ui.update_selectize(id="replicates_mf", choices=replicates, selected=replicates if replicates else None)


        @reactive.Effect
        @reactive.event(input.conditions_reset_mf)
        def _():
            req(not is_empty(S.TRACKSTATS.get()))

            ui.update_selectize(
                id="conditions_mf",
                selected=[]
            ) if input.conditions_mf() else \
            ui.update_selectize(
                id="conditions_mf",
                selected=S.TRACKSTATS.get()["Condition"].unique().tolist()
            )
        
        @reactive.Effect
        @reactive.event(input.replicates_reset_mf)
        def _():
            req(not is_empty(S.TRACKSTATS.get()) and "Replicate" in S.TRACKSTATS.get().columns)

            ui.update_selectize(
                id="replicates_mf",
                selected=[]
            ) if input.replicates_mf() else \
            ui.update_selectize(
                id="replicates_mf",
                selected=S.TRACKSTATS.get()["Replicate"].unique().tolist()
            )


        @reactive.Effect
        @reactive.event(S.SPOTSTATS)
        def _():
            if not is_empty(S.SPOTSTATS.get()):
                columns = S.SPOTSTATS.get().columns.tolist()
            else:
                columns = []
                
            ui.update_selectize(id="arrow_scale_by_mf", choices=columns, selected=columns[0] if columns else None)
            ui.update_selectize(id="arrow_scale_by_a_mf", choices=columns, selected=columns[0] if columns else None)
            ui.update_selectize(id="arrow_scale_by_b_mf", choices=columns, selected=columns[0] if columns else None)


    def motionflow_kwargs():
        if input.arrow_scaling_method_mf() in ['min', 'max', 'mean', 'median', 'sum', 'sd']:
            _scale_by = input.arrow_scale_by_mf()
        else:
            _scale_by = (input.arrow_scale_by_a_mf(), input.arrow_scale_by_b_mf())

        return dict(
            data=S.SPOTSTATS.get(),
            ignore_categories=False,
            conditions=input.conditions_mf(),
            replicates=input.replicates_mf(),
            mode=input.chart_type_mf(),
            n_arrows_y=input.n_arrows_y_mf(),
            n_arrows_x=input.n_arrows_x_mf(),
            min_arrow_frac=input.min_arrow_size_mf(),
            max_arrow_frac=input.max_arrow_size_mf(),
            stream_density=input.stream_density_mf(),
            scale_method=input.arrow_scaling_method_mf(),
            scale_by=_scale_by,
            cmap=input.color_cmap_mf(),
            title=input.title_mf(),
            noticequeue=noticequeue,
        )
    

    @ui.bind_task_button(button_id="generate_mf")
    @reactive.extended_task
    async def output_motionflow_plot(kwargs: dict):
        def _build(_kwargs=kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )
                return MotionFlowPlot(**_kwargs).plot()

        return await asyncio.get_running_loop().run_in_executor(None, _build)
    

    @reactive.Effect
    @reactive.event(input.generate_mf)
    def _():
        output_motionflow_plot.cancel()

        req(not is_empty(S.SPOTSTATS.get())
            and "Condition" in S.SPOTSTATS.get().columns 
            and "Replicate" in S.SPOTSTATS.get().columns)
        
        kwargs = motionflow_kwargs()

        output_motionflow_plot(kwargs)




    @render.plot
    def motionflow_plot():
        return output_motionflow_plot.result()
    
