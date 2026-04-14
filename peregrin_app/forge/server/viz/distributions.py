import io
import asyncio
import warnings

from datetime import date
import pandas as pd

from shiny import render, reactive, ui, req
from src.code import PolarDataDistribute, is_empty

import numpy as np
from io import BytesIO
from PIL import Image
import imageio.v3 as iio
import base64



def MountDistributions(input, output, session, S, noticequeue):

    # _ _ _ UPDATE CATEGORY SELECTION _ _ _

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
            
            ui.update_selectize(id="conditions_dd", choices=conditions, selected=conditions[0] if conditions else None)
            ui.update_selectize(id="replicates_dd", choices=replicates, selected=replicates if replicates else None)


        @reactive.Effect
        @reactive.event(input.conditions_reset_dd)
        def _():
            req(not is_empty(S.TRACKSTATS.get()))

            ui.update_selectize(
                id="conditions_dd",
                selected=[]
            ) if input.conditions_dd() else \
            ui.update_selectize(
                id="conditions_dd",
                selected=S.TRACKSTATS.get()["Condition"].unique().tolist()
            )
        
        @reactive.Effect
        @reactive.event(input.replicates_reset_dd)
        def _():
            req(not is_empty(S.TRACKSTATS.get()) and "Replicate" in S.TRACKSTATS.get().columns)

            ui.update_selectize(
                id="replicates_dd",
                selected=[]
            ) if input.replicates_dd() else \
            ui.update_selectize(
                id="replicates_dd",
                selected=S.TRACKSTATS.get()["Replicate"].unique().tolist()
            )

        @reactive.Effect
        def _():
            ui.update_selectize(
                id="dd_rosechart_discretize",
                choices=S.TRACKSTATS_COLUMNS.get(),
            )
            


    # _ _ _ FUNCTIONS FOR DRAWING KWARGS _ _ _

    def _common_kwargs(remove: list = []) -> dict:
        _kwargs =  dict(
            data=S.TRACKSTATS.get(),
            conditions=input.conditions_dd(),
            replicates=input.replicates_dd(),
            normalization=input.dd_normalization(),
            noticequeue=noticequeue,
        )

        for key in remove:
            _kwargs.pop(key, None)

        return _kwargs

     
    def _distribution_kde_colormesh_kwargs() -> dict:
        if input.dd_kde_colormesh_auto_scale_lut():
            min_density=None
            max_density=None
        else:
            min_density=input.dd_kde_colormesh_lutmap_scale_min()
            max_density=input.dd_kde_colormesh_lutmap_scale_max()

        return dict(
            **_common_kwargs(),
            cmap=input.dd_kde_colormesh_lut_map(),
            text_color='black' if input.app_theme() == "light" else 'white',
            label_theta=input.dd_kde_colormesh_theta_labels(),
            bins=input.dd_kde_colormesh_bins(),
            bandwidth=input.dd_kde_colormesh_bandwidth(),
            auto_lut_scale=input.dd_kde_colormesh_auto_scale_lut(),
            min_density=min_density,
            max_density=max_density,
            title=input.dd_kde_colormesh_title(),
        )
        
    
    def _distribution_kde_line_kwargs() -> dict:
        return dict(
            **_common_kwargs(),
            text_color='black' if input.app_theme() == "light" else 'white',
            bandwidth=input.dd_kde_line_bandwidth(),
            label_theta=input.dd_kde_line_theta_labels(),
            label_r=input.dd_kde_line_r_labels(),
            label_r_color=input.dd_kde_line_r_label_color(),
            r_loc=input.dd_kde_line_r_axis_position(),
            outline=input.dd_kde_line_outline(),
            outline_color=input.dd_kde_line_outline_color(),
            outline_width=input.dd_kde_line_outline_width(),
            kde_fill=input.dd_kde_line_fill(),
            kde_fill_color=input.dd_kde_line_fill_color(),
            kde_fill_alpha=input.dd_kde_line_fill_alpha(),
            show_abs_average=input.dd_kde_line_mean(),
            mean_angle_color=input.dd_kde_line_mean_color(),
            mean_angle_width=input.dd_kde_line_mean_width(),
            peak_direction_trend=input.dd_kde_line_peak_direction_trend(),
            peak_direction_trend_color=input.dd_kde_line_peak_direction_trend_color(),
            peak_direction_trend_width=input.dd_kde_line_peak_direction_trend_width(),
            background=input.dd_kde_line_bg_color(),
            title=input.dd_kde_line_title(),
        )
    
    def _distribution_rose_chart_kwargs() -> dict:
        return dict(
            **_common_kwargs(),
            bins=input.dd_rosechart_bins(),
            text_color='black' if input.app_theme() == "light" else 'white',
            alignment=input.dd_rosechart_alignment(),
            gap=input.dd_rosechart_gap(),
            levels=input.dd_rosechart_levels(),
            ntiles=input.dd_rosechart_ntiles(),
            discretize=input.dd_rosechart_discretize(),
            c_mode=input.dd_rosechart_cmode(),
            cmap=input.dd_rosechart_lut_map(),
            single_color=input.dd_rosechart_single_color(),
            default_colors=not input.dd_rosechart_custom_colors(),
            outline=input.dd_rosechart_outline(),
            outline_color=input.dd_rosechart_outline_color(),
            outline_width=input.dd_rosechart_outline_width(),
            background=input.dd_rosechart_bg_color(),
            title=input.dd_rosechart_title(),
        )
    

    def _dd_build(which="rosechart"):
        if is_empty(S.TRACKSTATS.get()):
            return None
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning,
                message="Starting a Matplotlib GUI outside of the main thread will likely fail")
            
            match which:
                case "rosechart":
                    kwargs = _distribution_rose_chart_kwargs()
                    kwargs['text_color'] = 'black'  # Override text color on download
                    return PolarDataDistribute(**kwargs).RoseChart()
                
                case "kde_line":
                    kwargs = _distribution_kde_line_kwargs()
                    kwargs['text_color'] = 'black'  # Override text color on download
                    return PolarDataDistribute(**kwargs).KDELinePlot()
                
                case "kde_colormesh":
                    kwargs = _distribution_kde_colormesh_kwargs()
                    kwargs['text_color'] = 'black'  # Override text color on download
                    return PolarDataDistribute(**kwargs).GaussianKDEColormesh()


    # _ _ _ ROSE CHART _ _ _

    ui.bind_task_button(button_id="generate_dd_rosechart")
    @reactive.extended_task
    async def output_data_distribution_rose_chart(kwargs: dict):
        def _build(_kwargs=kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )
                return PolarDataDistribute(**_kwargs).RoseChart()
            
        return await asyncio.get_running_loop().run_in_executor(None, _build)
    

    @reactive.Effect
    @reactive.event(input.generate_dd_rosechart, ignore_none=False)
    def _():
        output_data_distribution_rose_chart.cancel()
        
        req(not is_empty(S.TRACKSTATS.get()))
        output_data_distribution_rose_chart(_distribution_rose_chart_kwargs())


    @render.plot
    def dd_plot_rosechart():
        return output_data_distribution_rose_chart.result()
    

    @render.download(filename=f"Rose Chart {date.today()}.svg", media_type="svg")
    def dd_rosechart_download_svg():
        fig = _dd_build(which="rosechart")

        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="svg", bbox_inches="tight")
                yield buffer.getvalue()

    @render.download(filename=f"Rose Chart {date.today()}.png", media_type="png")
    def dd_rosechart_download_png():
        fig = _dd_build(which="rosechart")

        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="png", bbox_inches="tight")
                yield buffer.getvalue()


    # _ _ _ KDE LINE PLOT _ _ _

    ui.bind_task_button(button_id="generate_dd_kde_line")
    @reactive.extended_task
    async def output_data_distribution_kde_line(kwargs: dict):
        def _build(_kwargs=kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )
                return PolarDataDistribute(**_kwargs).KDELinePlot()
            
        return await asyncio.get_running_loop().run_in_executor(None, _build)


    @reactive.Effect
    @reactive.event(input.generate_dd_kde_line, ignore_none=False)
    def _():
        output_data_distribution_kde_line.cancel()

        req(not is_empty(S.TRACKSTATS.get()))

        output_data_distribution_kde_line(_distribution_kde_line_kwargs())


    @render.plot
    def dd_plot_kde_line():
        return output_data_distribution_kde_line.result()
    

    @render.download(filename=f"KDE Line Plot {date.today()}.svg", media_type="svg")
    def dd_kde_line_download_svg():
        fig = _dd_build(which="kde_line")

        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="svg", bbox_inches="tight")
                yield buffer.getvalue()

    @render.download(filename=f"KDE Line Plot {date.today()}.png", media_type="png")
    def dd_kde_line_download_png():
        fig = _dd_build(which="kde_line")

        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="png", bbox_inches="tight")
                yield buffer.getvalue()
    

    # _ _ _ KDE COLORMESH _ _ _
    
    @render.text
    def dd_kde_colormesh_density_range():

        if (not is_empty(S.TRACKSTATS.get())):

            kwargs = _distribution_kde_colormesh_kwargs()

            min, max = PolarDataDistribute(**kwargs).get_density_range()
            S.MIN_DENSITY.set(f'{min:.4f}'); S.MAX_DENSITY.set(f'{max:.4f}')

        if (is_empty(S.TRACKSTATS.get())
            or S.MIN_DENSITY.get() is None
            or S.MAX_DENSITY.get() is None
        ): return "No data."

        return f"min: {S.MIN_DENSITY.get()}; max: {S.MAX_DENSITY.get()}"
    

    ui.bind_task_button(button_id="generate_dd_kde_colormesh")
    @reactive.extended_task
    async def output_data_distribution_colormesh(kwargs: dict):
        def _build(_kwargs=kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )
                return PolarDataDistribute(**_kwargs).GaussianKDEColormesh()
            
        return await asyncio.get_running_loop().run_in_executor(None, _build)
    

    @reactive.Effect
    @reactive.event(input.generate_dd_kde_colormesh, ignore_none=False)
    def _():
        output_data_distribution_colormesh.cancel()

        req(not is_empty(S.TRACKSTATS.get()))
        output_data_distribution_colormesh(_distribution_kde_colormesh_kwargs())


    @render.plot
    def dd_plot_kde_colormesh():
        return output_data_distribution_colormesh.result()
    
    
    @render.download(filename=f"KDE Colormesh {date.today()}.svg", media_type="svg")
    def dd_kde_colormesh_download_svg():
        fig = _dd_build(which="kde_colormesh")

        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="svg", bbox_inches="tight")
                yield buffer.getvalue()

    @render.download(filename=f"KDE Colormesh {date.today()}.png", media_type="png")
    def dd_kde_colormesh_download_png():
        fig = _dd_build(which="kde_colormesh")

        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="png", bbox_inches="tight")
                yield buffer.getvalue()
    