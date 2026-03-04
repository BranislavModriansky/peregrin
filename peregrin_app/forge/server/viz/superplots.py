import io
import asyncio
import warnings
import pandas as pd
from datetime import date
from shiny import ui, reactive, req, render
from src.code import SuperPlots, DebounceCalc, Dyes, is_empty


def mount_superplots(input, output, session, S, noticequeue):
    

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
            
            ui.update_selectize(id="conditions_sp", choices=conditions, selected=conditions if conditions else None)
            ui.update_selectize(id="replicates_sp", choices=replicates, selected=replicates if replicates else None)


        @reactive.Effect
        @reactive.event(input.conditions_reset_sp)
        def _():
            req(not is_empty(S.TRACKSTATS.get()))

            ui.update_selectize(
                id="conditions_sp",
                selected=[]
            ) if input.conditions_sp() else \
            ui.update_selectize(
                id="conditions_sp",
                selected=S.TRACKSTATS.get()["Condition"].unique().tolist()
            )
        
        @reactive.Effect
        @reactive.event(input.replicates_reset_sp)
        def _():
            req(not is_empty(S.TRACKSTATS.get()) and "Replicate" in S.TRACKSTATS.get().columns)

            ui.update_selectize(
                id="replicates_sp",
                selected=[]
            ) if input.replicates_sp() else \
            ui.update_selectize(
                id="replicates_sp",
                selected=S.TRACKSTATS.get()["Replicate"].unique().tolist()
            )


    def _superplot_common_kwargs() -> dict:

        return dict(
            data=S.TRACKSTATS.get(),
            statistic=input.metric_sp(),
            conditions=input.conditions_sp(),
            replicates=input.replicates_sp(),
            ignore_categories=False,
            noticequeue=noticequeue,
            ci_statistic=input.sp_error_statistic(),
            confidence_level=input.sp_error_ci_level(),
            n_resamples=input.sp_error_bootstraps(),
            monochrome=input.sp_monochrome(),
        )

    def _superplot_hybrid_kwargs() -> dict:

        return dict(
            orient=input.sp_orientation(),
            density_norm=input.sp_density_norm(),

            cond_mean=input.sp_cond_mean(),
            mean_ls=input.sp_cond_mean_ls(),
            cond_median=input.sp_cond_median(),
            median_ls=input.sp_cond_median_ls(),

            error=input.sp_error(),
            error_type=input.sp_error_type(),

            use_stock_palette=input.stock_palette_sp(),
            palette=input.palette_sp(),

            rep_means=input.sp_rep_means(),
            rep_mean_size=input.sp_rep_mean_marker_size(),
            rep_mean_outline_width=input.sp_rep_mean_marker_outline_lw(),
            rep_mean_alpha=input.sp_rep_mean_marker_alpha(),

            connect_rep_mean=input.sp_rep_mean_marker_join(),
            connect_rep_mean_ls=input.sp_rep_mean_join_ls(),

            rep_medians=input.sp_rep_medians(),
            rep_median_size=input.sp_rep_median_marker_size(),
            rep_median_outline_width=input.sp_rep_median_marker_outline_lw(),
            rep_median_alpha=input.sp_rep_median_marker_alpha(),

            connect_rep_median=input.sp_rep_median_marker_join(),
            connect_rep_median_ls=input.sp_rep_median_join_ls(),

            violins=input.sp_show_violins_2(),
            violin_fill_color=input.sp_violin_fill_color_2(),
            violin_outline_width=input.sp_violin_outline_lw_2() if input.sp_show_violins_2() else 0,
            violin_outline_color=input.sp_violin_outline_color_2() if input.sp_show_violins_2() else None,
            violin_alpha=input.sp_violin_fill_alpha_2(),

            scatter=input.sp_show_scatter(),
            scatter_style=input.sp_scatter_type(),

            scatter_size=input.sp_scatter_marker_size(),
            scatter_alpha=input.sp_scatter_marker_alpha(),
            scatter_outline_width=input.sp_scatter_marker_outline_lw(),

            kde=input.sp_show_kde(),
            kde_outline=input.sp_kde_outline(),
            kde_outline_lw=input.sp_kde_outline_lw(),
            kde_fill=input.sp_kde_fill(),
            kde_alpha=input.sp_kde_fill_alpha(),

            title=input.sp_title(),

            fig_width=input.sp_fig_width(), 
            fig_height=input.sp_fig_height(),
        )

    def _superplot_superviolins_kwargs() -> dict:

        return dict(
            orient=input.sp_orientation(),
            density_norm=input.sp_density_norm(),

            cond_mean=input.sp_cond_mean(),
            mean_ls=input.sp_cond_mean_ls(),

            cond_median=input.sp_cond_median(),
            median_ls=input.sp_cond_median_ls(),

            error=input.sp_error(),
            error_type=input.sp_error_type(),

            rep_markers=input.sp_show_rep_markers(),
            middle_vals=input.sp_rep_center(),

            outline_lw=input.sp_violin_outline_lw_1() if input.sp_violin_outline_1() else 0,
            sep_lw=input.sp_subviolins_outline_lw_1() if input.sp_subviolins_outline_1() else 0,

            use_stock_palette=input.stock_palette_sp(),
            palette=input.palette_sp(),

            title=input.sp_title(),

            fig_width=input.sp_fig_width(), 
            fig_height=input.sp_fig_height(),
        )

    @reactive.Effect
    @reactive.event(input.sp_generate, ignore_none=False)
    def _card_layout():
        
        @output(id="sp_plot_card")
        @render.ui
        def sp_plot_card():

            with reactive.isolate():
                req(input.sp_fig_height() is not None and input.sp_fig_width() is not None)
                fig_height, fig_width = input.sp_fig_height() * 96, input.sp_fig_width() * 96

            return ui.card(
                ui.div(
                    # Make the *plot image* larger than the panel so scrolling kicks in
                    ui.output_plot("hybrid_superplot", width=f"{fig_width}px", height=f"{fig_height}px"),
                    style=f"height: {fig_height}px; width: {fig_width}px; margin: auto",
                    
                    # style=f"overflow: auto;",
                    class_="scroll-panel",
                ),
                full_screen=True, fill=False
            )
        
    @reactive.Effect
    @reactive.event(input.vp_generate, ignore_none=False)
    def _card_layout():

        @output(id="vp_plot_card")
        @render.ui
        def vp_plot_card():

            with reactive.isolate():
                req(input.sp_fig_height() is not None and input.sp_fig_width() is not None)
                fig_height, fig_width = input.sp_fig_height() * 96, input.sp_fig_width() * 96

            return ui.card(
                ui.div(
                    # Make the *plot image* larger than the panel so scrolling kicks in
                    ui.output_plot("superviolins_superplot", width=f"{fig_width}px", height=f"{fig_height}px"),
                    style=f"height: {fig_height}px; width: {fig_width}px; margin: auto",

                    # style=f"overflow: auto;",
                    class_="scroll-panel",
                ),
                full_screen=True, fill=False
            )
        
    

    ui.bind_task_button(button_id="sp_generate")
    @reactive.extended_task
    async def output_hybrid_superplot(class_kwargs: dict, func_kwargs: dict):
        def _build(_class_kwargs=class_kwargs, _func_kwargs=func_kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )
                return SuperPlots(**_class_kwargs).hybrid(**_func_kwargs)

        return await asyncio.get_running_loop().run_in_executor(None, _build)
    

    ui.bind_task_button(button_id="vp_generate")
    @reactive.extended_task
    async def output_superviolins_superplot(class_kwargs: dict, func_kwargs: dict):
        def _build(_class_kwargs=class_kwargs, _func_kwargs=func_kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )
                return SuperPlots(**_class_kwargs).superviolins(**_func_kwargs)

        return await asyncio.get_running_loop().run_in_executor(None, _build)
        


    @reactive.Effect
    @reactive.event(input.sp_generate, ignore_none=False)
    def _trigger():
        output_hybrid_superplot.cancel()

        req(S.TRACKSTATS.get() is not None and not S.TRACKSTATS.get().empty
            and "Condition" in S.TRACKSTATS.get().columns and "Replicate" in S.TRACKSTATS.get().columns)

        class_kwargs = _superplot_common_kwargs()
        func_kwargs = _superplot_hybrid_kwargs()

        output_hybrid_superplot(class_kwargs, func_kwargs)


    @reactive.Effect
    @reactive.event(input.vp_generate, ignore_none=False)
    def _trigger():
        output_superviolins_superplot.cancel()

        req(S.TRACKSTATS.get() is not None and not S.TRACKSTATS.get().empty
            and "Condition" in S.TRACKSTATS.get().columns and "Replicate" in S.TRACKSTATS.get().columns)

        class_kwargs = _superplot_common_kwargs()
        func_kwargs = _superplot_superviolins_kwargs()

        output_superviolins_superplot(class_kwargs, func_kwargs)

    
    @render.plot
    def hybrid_superplot():
        return output_hybrid_superplot.result()

    @render.plot
    def superviolins_superplot():
        return output_superviolins_superplot.result()


