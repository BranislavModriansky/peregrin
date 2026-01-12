import io
import asyncio
import warnings

from datetime import date
import pandas as pd

from shiny import render, reactive, ui, req
from src.code import PolarDataDistribute, Metrics

import numpy as np
from io import BytesIO
from PIL import Image
import imageio.v3 as iio
import base64



def MountDistributions(input, output, session, S, noticequeue):

    @reactive.Effect
    def update_choices():

        @reactive.Effect
        @reactive.event(S.TRACKSTATS)
        def _():
            if S.TRACKSTATS.get() is None or S.TRACKSTATS.get().empty:
                return
            ui.update_selectize(id="conditions_dd", choices=S.TRACKSTATS.get()["Condition"].unique().tolist(), selected=S.TRACKSTATS.get()["Condition"].unique().tolist()[0] if S.TRACKSTATS.get()["Condition"].unique().tolist() else None)
            ui.update_selectize(id="replicates_dd", choices=S.TRACKSTATS.get()["Replicate"].unique().tolist(), selected=S.TRACKSTATS.get()["Replicate"].unique().tolist() if S.TRACKSTATS.get()["Replicate"].unique().tolist() else None)


        @reactive.Effect
        @reactive.event(input.conditions_reset_dd)
        def _():
            req(S.TRACKSTATS.get() is not None and not S.TRACKSTATS.get().empty)

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
            req(S.TRACKSTATS.get() is not None and not S.TRACKSTATS.get().empty)

            ui.update_selectize(
                id="replicates_dd",
                selected=[]
            ) if input.replicates_dd() else \
            ui.update_selectize(
                id="replicates_dd",
                selected=S.TRACKSTATS.get()["Replicate"].unique().tolist()
            )


    def _common_kwargs(remove: list = []) -> dict:
        if input.dd_add_weights():
            weight=input.dd_weight()
        else: 
            weight=None

        _kwargs =  dict(
            data=S.TRACKSTATS.get(),
            conditions=input.conditions_dd(),
            replicates=input.replicates_dd(),
            normalization=input.dd_normalization(),
            weight=weight,
            noticequeue=noticequeue,
        )

        for key in remove:
            _kwargs.pop(key, None)

        return _kwargs

    def _density_caps_kwargs() -> dict:
        if input.dd_kde_colormesh_auto_scale_lut():
            min_density=input.dd_kde_colormesh_lutmap_scale_min()
            max_density=input.dd_kde_colormesh_lutmap_scale_max()
        else:
            min_density=0.0
            max_density=1.0

        return dict(
            **_common_kwargs(),
            auto_lut_scale=input.dd_kde_colormesh_auto_scale_lut(),
            cmap=input.dd_kde_colormesh_lut_map(),
            min_density=min_density,
            max_density=max_density,
        )

    def _distribution_kde_colormesh_kwargs() -> dict:
        if input.dd_kde_colormesh_auto_scale_lut():
            min_density=0.0
            max_density=1.0
        else:
            min_density=input.dd_kde_colormesh_lutmap_scale_min()
            max_density=input.dd_kde_colormesh_lutmap_scale_max()

        return dict(
            **_common_kwargs(),
            cmap=input.dd_kde_colormesh_lut_map(),
            text_color='black' if input.app_theme() == "Shiny" else 'white',
            label_theta=input.dd_kde_colormesh_theta_labels(),
            bins=input.dd_kde_colormesh_bins(),
            bandwidth=input.dd_kde_colormesh_bandwidth(),
            auto_lut_scale=input.dd_kde_colormesh_auto_scale_lut(),
            min_density=min_density,
            max_density=max_density,
        )
    
    def _distribution_kde_line_kwargs() -> dict:
        return dict(
            **_common_kwargs(),
            text_color='black' if input.app_theme() == "Shiny" else 'white',
            bandwidth=input.dd_kde_line_bandwidth(),
            label_theta=input.dd_kde_line_theta_labels(),
            label_r=input.dd_kde_line_r_labels(),
            label_r_color=input.dd_kde_line_r_label_color(),
            r_loc=input.dd_kde_line_r_axis_position(),
            outline=True,
            outline_color=input.dd_kde_line_outline_color(),
            outline_width=input.dd_kde_line_outline_width(),
            kde_fill=input.dd_kde_line_fill(),
            kde_fill_color=input.dd_kde_line_fill_color(),
            kde_fill_alpha=input.dd_kde_line_fill_alpha(),
            show_abs_average=input.dd_kde_line_dial(),
            mean_angle_color=input.dd_kde_line_dial_color(),
            mean_angle_width=input.dd_kde_line_dial_width(),
            background=input.dd_kde_line_bg_color(),
        )
    
    def _distribution_rose_chart_kwargs() -> dict:
        return dict(
            **_common_kwargs(),
            bins=input.dd_rosechart_bins(),
            text_color='black' if input.app_theme() == "Shiny" else 'white',
            c_mode=input.dd_rosechart_cmode(),
            ntiles=input.dd_rosechart_ntiles() if input.dd_rosechart_cmode() == "n-tiles" else None,
            discretize=input.dd_rosechart_partition_selector() if input.dd_rosechart_cmode() == "n-tiles" else None,
            # outline=True,
            # outline_color=input.dd_rosechart_outline_color(),
            # outline_width=input.dd_rosechart_outline_width(),
            # c_mode=input.dd_rosechart_color_mode(),
            # single_color=input.dd_rosechart_single_color(),
            # ntiles=input.dd_rosechart_ntiles(),
            # background=input.dd_rosechart_bg_color(),
        )
    
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
                fig = PolarDataDistribute(**_kwargs).GaussianKDEColormesh()

                return fig
            
        return await asyncio.get_running_loop().run_in_executor(None, _build)
    
    @reactive.Effect
    @reactive.event(input.generate_dd_kde_colormesh, ignore_none=False)
    def _():
        output_data_distribution_colormesh.cancel()

        req(
            S.TRACKSTATS.get() is not None and not S.TRACKSTATS.get().empty
            and "Condition" in S.TRACKSTATS.get().columns and "Replicate" in S.TRACKSTATS.get().columns
        )

        kwargs = _distribution_kde_colormesh_kwargs()

        output_data_distribution_colormesh(kwargs)

    @render.plot
    def dd_plot_kde_colormesh():
        return output_data_distribution_colormesh.result()
    
    @reactive.Effect
    def _get_density_range():
        if (S.TRACKSTATS.get() is not None and not S.TRACKSTATS.get().empty
            and noticequeue is not None):

            kwargs = _density_caps_kwargs()
            min, max = PolarDataDistribute(**kwargs).get_density_caps()
            S.MIN_DENSITY.set(f'{min:.4f}'); S.MAX_DENSITY.set(f'{max:.4f}')
    
    @render.text
    def dd_kde_colormesh_density_range():
        if (
            S.TRACKSTATS.get() is None or S.TRACKSTATS.get().empty
            or S.MIN_DENSITY.get() is None
            or S.MAX_DENSITY.get() is None
        ): return "No data."

        return f"min: {S.MIN_DENSITY.get()}; max: {S.MAX_DENSITY.get()}"
    

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
                fig = PolarDataDistribute(**_kwargs).KDELinePlot()

                return fig
            
        return await asyncio.get_running_loop().run_in_executor(None, _build)
    
    @reactive.Effect
    @reactive.event(input.generate_dd_kde_line, ignore_none=False)
    def _():
        output_data_distribution_kde_line.cancel()

        req(
            S.TRACKSTATS.get() is not None and not S.TRACKSTATS.get().empty
            and "Condition" in S.TRACKSTATS.get().columns and "Replicate" in S.TRACKSTATS.get().columns
        )

        kwargs = _distribution_kde_line_kwargs()

        output_data_distribution_kde_line(kwargs)

    @render.plot
    def dd_plot_kde_line():
        return output_data_distribution_kde_line.result()
    
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
                fig = PolarDataDistribute(**_kwargs).RoseChart()

                return fig
            
        return await asyncio.get_running_loop().run_in_executor(None, _build)
    
    @reactive.Effect
    @reactive.event(input.generate_dd_rosechart, ignore_none=False)
    def _():
        output_data_distribution_rose_chart.cancel()

        req(
            S.TRACKSTATS.get() is not None and not S.TRACKSTATS.get().empty
            and "Condition" in S.TRACKSTATS.get().columns and "Replicate" in S.TRACKSTATS.get().columns
        )

        kwargs = _distribution_rose_chart_kwargs()

        output_data_distribution_rose_chart(kwargs)

    @render.plot
    def dd_plot_rosechart():
        return output_data_distribution_rose_chart.result()
    