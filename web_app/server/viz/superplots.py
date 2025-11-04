import io
import asyncio
import warnings
import pandas as pd
from datetime import date
from shiny import ui, reactive, req, render
from utils import SwarmsAndBeyond, Superviolins, Debounce


def mount_superplots(input, output, session, S, queue):

    # Don't create a new instance - use the imported singleton
    # buffer = _message_buffer.MessageBuffer()  # REMOVE THIS LINE

    # _ UPDATES ON SWARMPLOT PRESETS SELECTION _
    @reactive.Effect
    def get_preset():
        # Get the selected preset and update the UI accordingly
        preset = input.sp_preset()
        if preset == "Swarms":
            ( # Swarms
                ui.update_checkbox(id=f"sp_show_swarms", value=True),
                ui.update_numeric(id=f"sp_swarm_marker_size", value=2.5),
                ui.update_numeric(id=f"sp_swarm_marker_alpha", value=1),
            )
            ( # Violins
                ui.update_checkbox(id=f"sp_show_violins", value=False),
            )
            ( # KDEs
                ui.update_checkbox(id=f"sp_show_kde", value=False),
            )
            ( # Bullets
                ui.update_checkbox(id=f"sp_show_rep_medians", value=True),
                ui.update_numeric(id=f"sp_median_bullet_size", value=90),
            )
            ( # Cond Stats
                ui.update_checkbox(id=f"sp_show_cond_mean", value=True),
                ui.update_checkbox(id=f"sp_show_cond_median", value=False),
                ui.update_checkbox(id=f"sp_show_errbars", value=True),
            )
            ( # Palette
                ui.update_selectize(id=f"sp_palette", selected="Set2"),
            )

        if preset == "Swarms & Violins":
            ( # Swarms
                ui.update_checkbox(id=f"sp_show_swarms", value=True),
                ui.update_numeric(id=f"sp_swarm_marker_size", value=2.5),
            )
            ( # Violins
                ui.update_checkbox(id=f"sp_show_violins", value=True),
                ui.update_numeric(id=f"sp_violin_alpha", value=0.65),
                ui.update_selectize(id=f"sp_violin_fill", selected="whitesmoke"),
                ui.update_selectize(id=f"sp_violin_outline", selected="lightgrey"),
                ui.update_numeric(id=f"sp_violin_outline_width", value=1),
            )
            ( # KDEs
                ui.update_checkbox(id=f"sp_show_kde", value=False),
            )
            ( # Bullets
                ui.update_checkbox(id=f"sp_show_rep_medians", value=True),
                ui.update_numeric(id=f"sp_median_bullet_size", value=70),
            )
            ( # Cond Stats
                ui.update_checkbox(id=f"sp_show_cond_mean", value=True),
                ui.update_checkbox(id=f"sp_show_cond_median", value=False),
                ui.update_checkbox(id=f"sp_show_errbars", value=True),
            )
            ( # Palette
                ui.update_selectize(id=f"sp_palette", selected="Set2"),
            )

        if preset == "Violins & KDEs":
            ( # Swarms
                ui.update_checkbox(id=f"sp_show_swarms", value=False),
            )
            ( # Violins
                ui.update_checkbox(id=f"sp_show_violins", value=True),
                ui.update_numeric(id=f"sp_violin_alpha", value=1),
                ui.update_selectize(id=f"sp_violin_fill", selected="lightgrey"),
                ui.update_selectize(id=f"sp_violin_outline", selected="dimgrey"),
                ui.update_numeric(id=f"sp_violin_outline_width", value=1),
            )
            ( # KDEs
                ui.update_checkbox(id=f"sp_show_kde", value=True),
                ui.update_numeric(id=f"sp_kde_alpha", value=0.5),
                ui.update_numeric(id=f"sp_kde_line_width", value=0),
                ui.update_checkbox(id=f"sp_kde_fill", value=True),
            )
            ( # Bullets
                ui.update_checkbox(id=f"sp_show_rep_medians", value=False),
            )
            ( # Cond Stats
                ui.update_checkbox(id=f"sp_show_cond_mean", value=True),
                ui.update_checkbox(id=f"sp_show_cond_median", value=True),
                ui.update_checkbox(id=f"sp_show_errbars", value=True),
            )
            ( # Palette
                ui.update_selectize(id=f"sp_palette", selected="pastel"),
            )

        if preset == "Swarms & Violins & KDEs":
            ( # Swarms
                ui.update_checkbox(id=f"sp_show_swarms", value=True),
                ui.update_numeric(id=f"sp_swarm_marker_size", value=1.5),
                ui.update_numeric(id=f"sp_swarm_marker_alpha", value=0.75),
            )
            ( # Violins
                ui.update_checkbox(id=f"sp_show_violins", value=True),
                ui.update_numeric(id=f"sp_violin_alpha", value=0.5),
                ui.update_selectize(id=f"sp_violin_fill", selected="whitesmoke"),
                ui.update_selectize(id=f"sp_violin_outline", selected="lightgrey"),
                ui.update_numeric(id=f"sp_violin_outline_width", value=1),
            )
            ( # KDEs
                ui.update_checkbox(id=f"sp_show_kde", value=True),
                ui.update_numeric(id=f"sp_kde_alpha", value=0.75),
                ui.update_numeric(id=f"sp_kde_line_width", value=1),
                ui.update_checkbox(id=f"sp_kde_fill", value=False),
            )
            ( # Bullets
                ui.update_checkbox(id=f"sp_show_rep_medians", value=True),
                ui.update_numeric(id=f"sp_median_bullet_size", value=70),
                ui.update_numeric(id=f"sp_mean_bullet_size", value=50),
            )
            ( # Cond Stats
                ui.update_checkbox(id=f"sp_show_cond_mean", value=False),
                ui.update_checkbox(id=f"sp_show_cond_median", value=False),
                ui.update_checkbox(id=f"sp_show_errbars", value=False),
            )
            ( # Palette
                ui.update_selectize(id=f"sp_palette", selected="muted"),
            )


    # _ _ _ _ SWARMPPLOT _ _ _ _
    
    @ui.bind_task_button(button_id="sp_generate")
    @reactive.extended_task
    async def output_swarmplot(
        df,
        metric,
        title,
        palette,
        use_stock_palette,
        show_swarm,
        swarm_size,
        swarm_outline_color,
        swarm_alpha,
        show_violin,
        violin_fill_color,
        violin_edge_color,
        violin_alpha,
        violin_outline_width,
        show_mean,
        mean_span,
        mean_color,
        show_median,
        median_span,
        median_color,
        line_width,
        show_error_bars,
        errorbar_capsize,
        errorbar_color,
        errorbar_lw,
        errorbar_alpha,
        show_mean_balls,
        mean_ball_size,
        mean_ball_outline_color,
        mean_ball_outline_width,
        mean_ball_alpha,
        show_median_balls,
        median_ball_size,
        median_ball_outline_color,
        median_ball_outline_width,
        median_ball_alpha,
        show_kde,
        kde_inset_width,
        kde_outline,
        kde_alpha,
        kde_fill,
        show_legend,
        show_grid,
        open_spine,
        plot_width,
        plot_height
    ):  
        def build():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )

                return SwarmsAndBeyond(
                    df=df,
                    metric=metric,
                    title=title,
                    palette=palette,
                    use_stock_palette=use_stock_palette,
                    show_swarm=show_swarm,
                    swarm_size=swarm_size,
                    swarm_outline_color=swarm_outline_color,
                    swarm_alpha=swarm_alpha,
                    show_violin=show_violin,
                    violin_fill_color=violin_fill_color,
                    violin_edge_color=violin_edge_color,
                    violin_alpha=violin_alpha,
                    violin_outline_width=violin_outline_width,
                    show_mean=show_mean,
                    mean_span=mean_span,
                    mean_color=mean_color,
                    show_median=show_median,
                    median_span=median_span,
                    median_color=median_color,
                    line_width=line_width,
                    show_error_bars=show_error_bars,
                    errorbar_capsize=errorbar_capsize,
                    errorbar_color=errorbar_color,
                    errorbar_lw=errorbar_lw,
                    errorbar_alpha=errorbar_alpha,
                    show_mean_balls=show_mean_balls,
                    mean_ball_size=mean_ball_size,
                    mean_ball_outline_color=mean_ball_outline_color,
                    mean_ball_outline_width=mean_ball_outline_width,
                    mean_ball_alpha=mean_ball_alpha,
                    show_median_balls=show_median_balls,
                    median_ball_size=median_ball_size,
                    median_ball_outline_color=median_ball_outline_color,
                    median_ball_outline_width=median_ball_outline_width,
                    median_ball_alpha=median_ball_alpha,
                    show_kde=show_kde,
                    kde_inset_width=kde_inset_width,
                    kde_outline=kde_outline,
                    kde_alpha=kde_alpha,
                    kde_fill=kde_fill,
                    show_legend=show_legend,
                    show_grid=show_grid,
                    open_spine=open_spine,
                    plot_width=plot_width,
                    plot_height=plot_height,
                    queue=queue,
                )

        result = await asyncio.get_running_loop().run_in_executor(None, build)
        
        return result
    
    # @reactive.Effect
    # @reactive.event(input.sp_generate, ignore_none=False)
    # def trigger_swarmplot():

    @reactive.Effect
    @reactive.event(input.sp_generate, ignore_none=False)
    def _():
        @output(id="sp_plot_card")
        @render.ui
        def plot_card():

            with reactive.isolate():
                req(input.sp_fig_height() is not None and input.sp_fig_width() is not None)
                fig_height, fig_width = input.sp_fig_height() * 96, input.sp_fig_width() * 96
        
            return ui.card(
                ui.div(
                    # Make the *plot image* larger than the panel so scrolling kicks in
                    ui.output_plot("swarmplot", width=f"{fig_width}px", height=f"{fig_height}px"),
                    # ui.output_plot(id="swarmplot"),
                    style=f"height: {fig_height}px; width: {fig_width}px; margin: auto",
                    # style=f"overflow: auto;",
                    class_="scroll-panel",
                ),
                full_screen=True, fill=False
            ), 



    @reactive.Effect
    @reactive.event(input.sp_generate, ignore_none=False)
    def make_swarmplot():
            
        output_swarmplot.cancel()
        
        output_swarmplot(
            df=S.TRACKSTATS.get() if S.TRACKSTATS.get() is not None else pd.DataFrame(),
            metric=input.sp_metric(),
            title=input.sp_title(), 
            palette=input.sp_palette(),
            use_stock_palette=input.sp_use_stock_palette(),

            show_swarm=input.sp_show_swarms(),
            swarm_size=input.sp_swarm_marker_size(),
            swarm_outline_color=input.sp_swarm_marker_outline(),
            swarm_alpha=input.sp_swarm_marker_alpha() if 0.0 <= input.sp_swarm_marker_alpha() <= 1.0 else 1.0,

            show_violin=input.sp_show_violins(),
            violin_fill_color=input.sp_violin_fill(),
            violin_edge_color=input.sp_violin_outline(),
            violin_alpha=input.sp_violin_alpha() if 0.0 <= input.sp_violin_alpha() <= 1.0 else 1.0,
            violin_outline_width=input.sp_violin_outline_width(),

            show_mean=input.sp_show_cond_mean(),
            mean_span=input.sp_mean_line_span(),
            mean_color=input.sp_mean_line_color(),
            show_median=input.sp_show_cond_median(),
            median_span=input.sp_median_line_span(),
            median_color=input.sp_median_line_color(),
            line_width=input.sp_lines_lw(),
            show_error_bars=input.sp_show_errbars(),
            errorbar_capsize=input.sp_errorbar_capsize(),
            errorbar_color=input.sp_errorbar_color(),
            errorbar_lw=input.sp_errorbar_lw(),
            errorbar_alpha=input.sp_errorbar_alpha() if 0.0 <= input.sp_errorbar_alpha() <= 1.0 else 1.0,

            show_mean_balls=input.sp_show_rep_means(),
            mean_ball_size=input.sp_mean_bullet_size(),
            mean_ball_outline_color=input.sp_mean_bullet_outline(),
            mean_ball_outline_width=input.sp_mean_bullet_outline_width(),
            mean_ball_alpha=input.sp_mean_bullet_alpha() if 0.0 <= input.sp_mean_bullet_alpha() <= 1.0 else 1.0,
            show_median_balls=input.sp_show_rep_medians(),
            median_ball_size=input.sp_median_bullet_size(),
            median_ball_outline_color=input.sp_median_bullet_outline(),
            median_ball_outline_width=input.sp_median_bullet_outline_width(),
            median_ball_alpha=input.sp_median_bullet_alpha() if 0.0 <= input.sp_median_bullet_alpha() <= 1.0 else 1.0,

            show_kde=input.sp_show_kde(),
            kde_inset_width=input.sp_kde_bandwidth(),
            kde_outline=input.sp_kde_line_width(),
            kde_alpha=input.sp_kde_fill_alpha() if 0.0 <= input.sp_kde_fill_alpha() <= 1.0 else 1.0,
            kde_fill=input.sp_kde_fill(),

            show_legend=input.sp_show_legend(),
            show_grid=input.sp_grid(),
            open_spine=input.sp_spine(),
            
            plot_width=input.sp_fig_width(),
            plot_height=input.sp_fig_height(),
        )

                

    @render.plot
    def swarmplot():
        # Only update when output_swarmplot task completes (not reactively)
        req(output_swarmplot.result())
        return output_swarmplot.result()

    @render.download(filename=f"Beyond Swarmplot {date.today()}.svg")
    def sp_download_svg():
        if S.TRACKSTATS.get() is None or S.TRACKSTATS.get().empty: return
        fig = SwarmsAndBeyond(
            df=S.TRACKSTATS.get(),
            metric=input.sp_metric(),
            title=input.sp_title(),
            palette=input.sp_palette(),
            use_stock_palette=input.sp_use_stock_palette(),

            show_swarm=input.sp_show_swarms(),
            swarm_size=input.sp_swarm_marker_size(),
            swarm_outline_color=input.sp_swarm_marker_outline(),
            swarm_alpha=input.sp_swarm_marker_alpha(),

            show_violin=input.sp_show_violins(),
            violin_fill_color=input.sp_violin_fill(),
            violin_edge_color=input.sp_violin_outline(),
            violin_alpha=input.sp_violin_alpha(),
            violin_outline_width=input.sp_violin_outline_width(),

            show_mean=input.sp_show_cond_mean(),
            mean_span=input.sp_mean_line_span(),
            mean_color=input.sp_mean_line_color(),
            show_median=input.sp_show_cond_median(),
            median_span=input.sp_median_line_span(),
            median_color=input.sp_median_line_color(),
            line_width=input.sp_lines_lw(),
            show_error_bars=input.sp_show_errbars(),
            errorbar_capsize=input.sp_errorbar_capsize(),
            errorbar_color=input.sp_errorbar_color(),
            errorbar_lw=input.sp_errorbar_lw(),
            errorbar_alpha=input.sp_errorbar_alpha(),

            show_mean_balls=input.sp_show_rep_means(),
            mean_ball_size=input.sp_mean_bullet_size(),
            mean_ball_outline_color=input.sp_mean_bullet_outline(),
            mean_ball_outline_width=input.sp_mean_bullet_outline_width(),
            mean_ball_alpha=input.sp_mean_bullet_alpha(),
            show_median_balls=input.sp_show_rep_medians(),
            median_ball_size=input.sp_median_bullet_size(),
            median_ball_outline_color=input.sp_median_bullet_outline(),
            median_ball_outline_width=input.sp_median_bullet_outline_width(),
            median_ball_alpha=input.sp_median_bullet_alpha(),

            show_kde=input.sp_show_kde(),
            kde_inset_width=input.sp_kde_bandwidth(),
            kde_outline=input.sp_kde_line_width(),
            kde_alpha=input.sp_kde_fill_alpha(),
            kde_fill=input.sp_kde_fill(),

            show_legend=input.sp_show_legend(),
            show_grid=input.sp_grid(),
            open_spine=input.sp_spine(),
            
            plot_width=input.sp_fig_width(),
            plot_height=input.sp_fig_height(),
        )
        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="svg", bbox_inches="tight")
                yield buffer.getvalue()


    
    # _ _ _ _ SUPERVIOLINPLOT _ _ _ _ 

    @ui.bind_task_button(button_id="vp_generate")
    @reactive.extended_task
    async def output_superviolinplot(
        df,
        metric,
        title,
        palette,
        use_stock_palette,
        units,
        centre_val,
        middle_vals,
        error_bars,
        total_width,
        outline_lw,
        sep_lw,
        bullet_lw,
        errorbar_lw,
        bullet_size,
        plot_width,
        plot_height,
    ):
        # run sync plotting off the event loop
        def build():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )

                if df is None or df.empty: return 

                return Superviolins(
                    df=df,
                    metric=metric,
                    title=title,
                    units=units,
                    palette=palette,
                    use_stock_palette=use_stock_palette,
                    centre_val=centre_val,
                    middle_vals=middle_vals,
                    error_bars=error_bars,
                    total_width=total_width,
                    outline_lw=outline_lw,
                    sep_lw=sep_lw,
                    bullet_lw=bullet_lw,
                    errorbar_lw=errorbar_lw,
                    bullet_size=bullet_size,
                    plot_width=plot_width,
                    plot_height=plot_height,
                )

        # Either form is fine; pick one:
        return await asyncio.get_running_loop().run_in_executor(None, build)
        # return await asyncio.to_thread(build)
    

    @reactive.Effect
    @reactive.event(input.vp_generate, ignore_none=False)
    def trigger_superviolinplot():

        @reactive.Effect
        @reactive.event(input.vp_generate, ignore_none=False)
        def _():
            @output(id="vp_plot_card")
            @render.ui
            def plot_card():

                with reactive.isolate():
                    req(input.vp_fig_height() is not None and input.vp_fig_width() is not None)
                    fig_height, fig_width = input.vp_fig_height() * 96, input.vp_fig_width() * 96

                return ui.card(
                    ui.div(
                        # Make the *plot image* larger than the panel so scrolling kicks in
                        ui.output_plot("superviolinplot", width=f"{fig_width}px", height=f"{fig_height}px"),
                        # ui.output_plot(id="superviolinplot"),
                        style=f"height: {fig_height}px; width: {fig_width}px; margin: auto",
                        # style=f"overflow: auto;",
                        class_="scroll-panel",
                    ),
                    full_screen=True, fill=False
                ), 


        @reactive.Effect
        @reactive.event(input.vp_generate, ignore_none=False)
        def make_superviolinplot():
            output_superviolinplot.cancel()

            output_superviolinplot(
                df=S.TRACKSTATS.get(),
                metric=input.vp_metric(),
                title=input.vp_title(),
                palette=input.vp_palette(),
                use_stock_palette=input.vp_use_stock_palette(),

                units=S.UNITS.get().get(input.vp_metric()),
                centre_val=input.vp_skeleton_centre(),
                middle_vals=input.vp_replicate_bullets(),
                error_bars=input.vp_errorbars_method(),
                errorbar_lw=input.vp_errorbar_linewidth(),

                total_width=input.vp_violin_bandwidth(),
                outline_lw=input.vp_violin_outline_width(),
                sep_lw=input.vp_subviolin_outline_width(),

                bullet_lw=input.vp_bullet_outline_linewidth(),
                bullet_size=input.vp_bullet_size(),

                plot_width=input.vp_fig_width(),
                plot_height=input.vp_fig_height(),
            )

    @render.plot
    def superviolinplot():
        # Only update when output_superviolinplot task completes (not reactively)
        return output_superviolinplot.result()

    @render.download(filename=f"Superviolinplot {date.today()}.svg")
    def vp_download_svg():
        if S.TRACKSTATS.get() is None or S.TRACKSTATS.get().empty: return
        fig = Superviolins(
            df=S.TRACKSTATS.get(),
            metric=input.vp_metric(),
            title=input.vp_title(),
            palette=input.vp_palette(),
            use_stock_palette=input.vp_use_stock_palette(),
            units=S.UNITS.get(),
            centre_val=input.vp_skeleton_centre(),
            middle_vals=input.vp_replicate_bullets(),
            error_bars=input.vp_errorbars_method(),
            errorbar_lw=input.vp_errorbar_linewidth(),
            total_width=input.vp_violin_bandwidth(),
            outline_lw=input.vp_violin_outline_width(),
            sep_lw=input.vp_subviolin_outline_width(),
            bullet_lw=input.vp_bullet_outline_linewidth(),
            bullet_size=input.vp_bullet_size(),
            plot_width=input.vp_fig_width(),
            plot_height=input.vp_fig_height(),
        )
        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="svg", bbox_inches="tight")
                yield buffer.getvalue()

