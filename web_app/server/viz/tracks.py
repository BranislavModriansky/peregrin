import io
import asyncio
import warnings

from datetime import date
import pandas as pd

from shiny import render, reactive, ui, req
from utils import VisualizeTracksRealistics, VisualizeTracksNormalized, GetLutMap, Markers



def mount_tracks(input, output, session, S):

    

    @ui.bind_task_button(button_id="trr_generate")
    @reactive.extended_task
    async def output_track_reconstruction_realistic(
        Spots_df,
        Tracks_df,
        condition,
        replicate,
        c_mode,
        only_one_color,
        lut_scaling_metric,
        background,
        smoothing_index,
        lw,
        grid,
        mark_heads,
        marker,
        markersize,
        title
    ):
        
        # run sync plotting off the event loop
        def _build():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )

                return VisualizeTracksRealistics(
                    Spots_df=Spots_df,
                    Tracks_df=Tracks_df,
                    condition=condition,
                    replicate=replicate,
                    c_mode=c_mode,
                    only_one_color=only_one_color,
                    lut_scaling_metric=lut_scaling_metric,
                    background=background,
                    smoothing_index=smoothing_index,
                    lw=lw,
                    grid=grid,
                    mark_heads=mark_heads,
                    marker=marker,
                    markersize=markersize,
                    title=title
                )

        # Either form is fine; pick one:
        return await asyncio.get_running_loop().run_in_executor(None, _build)
        # return await asyncio.to_thread(_build)
    
    @reactive.Effect
    @reactive.event(input.trr_generate, ignore_none=False)
    def _():
            
        output_track_reconstruction_realistic.cancel()

        req(
            S.SPOTSTATS.get() is not None and not S.SPOTSTATS.get().empty 
            and S.TRACKSTATS.get() is not None and not S.TRACKSTATS.get().empty
            and "Condition" in S.SPOTSTATS.get().columns and "Replicate" in S.SPOTSTATS.get().columns
        )

        output_track_reconstruction_realistic(
            Spots_df=S.SPOTSTATS.get(),
            Tracks_df=S.TRACKSTATS.get(),
            condition=input.tracks_conditions(),
            replicate=input.tracks_replicates(),
            c_mode=input.tracks_color_mode(),
            only_one_color=input.tracks_only_one_color(),
            lut_scaling_metric=input.tracks_lut_scaling_metric(),
            background=input.tracks_background(),
            smoothing_index=input.tracks_smoothing_index(),
            lw=input.tracks_line_width(),
            grid=input.tracks_show_grid(),
            mark_heads=input.tracks_mark_heads(),
            marker=Markers.TrackHeads.get(input.tracks_marker_type()),
            markersize=input.tracks_marks_size()*10,
            title=input.tracks_title()
        )

    @render.plot
    def track_reconstruction_realistic():
        return output_track_reconstruction_realistic.result()

    @render.download(filename=f"Realistic Track Reconstruction {date.today()}.svg")
    def trr_download():
        req(
            S.SPOTSTATS.get() is not None and not S.SPOTSTATS.get().empty
            and S.TRACKSTATS.get() is not None and not S.TRACKSTATS.get().empty
            and "Condition" in S.SPOTSTATS.get().columns and "Replicate" in S.SPOTSTATS.get().columns
        )

        fig = VisualizeTracksRealistics(
            Spots_df=S.SPOTSTATS.get(),
            Tracks_df=S.TRACKSTATS.get(),
            condition=input.tracks_conditions(),
            replicate=input.tracks_replicates(),
            c_mode=input.tracks_color_mode(),
            only_one_color=input.tracks_only_one_color(),
            lut_scaling_metric=input.tracks_lut_scaling_metric(),
            background=input.tracks_background(),
            smoothing_index=input.tracks_smoothing_index(),
            lw=input.tracks_line_width(),
            grid=input.tracks_show_grid(),
            mark_heads=input.tracks_mark_heads(),
            marker=Markers.TrackHeads.get(input.tracks_marker_type()),
            markersize=input.tracks_marks_size()*10,
            title=input.tracks_title()
        )
        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="svg", bbox_inches="tight")
                yield buffer.getvalue()



    @ui.bind_task_button(button_id="tnr_generate")
    @reactive.extended_task
    async def output_track_reconstruction_normalized(
        Spots_df,
        Tracks_df,
        condition,
        replicate,
        c_mode,
        only_one_color,
        lut_scaling_metric,
        smoothing_index,
        lw,
        background,
        grid,
        grid_style,
        mark_heads,
        marker,
        markersize,
        title
    ):
        
        # run sync plotting off the event loop
        def _build():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )

                local_Spots_df = Spots_df.copy(deep=True) if Spots_df is not None else pd.DataFrame()
                local_Tracks_df = Tracks_df.copy(deep=True) if Tracks_df is not None else pd.DataFrame()
                return VisualizeTracksNormalized(
                    Spots_df=local_Spots_df,
                    Tracks_df=local_Tracks_df,
                    condition=local_Tracks_df["Condition"].unique().tolist()[0] if "Condition" in local_Tracks_df.columns else condition,
                    replicate=replicate,
                    c_mode=c_mode,
                    only_one_color=only_one_color,
                    lut_scaling_metric=lut_scaling_metric,
                    smoothing_index=smoothing_index,
                    lw=lw,
                    background=background,
                    grid=grid,
                    grid_style=grid_style,
                    mark_heads=mark_heads,
                    marker=marker,
                    markersize=markersize,
                    title=title
                )

        # Either form is fine; pick one:
        return await asyncio.get_running_loop().run_in_executor(None, _build)
        # return await asyncio.to_thread(build)

    @reactive.Effect
    @reactive.event(input.tnr_generate, ignore_none=False)
    def _():
            
        output_track_reconstruction_normalized.cancel()

        req(
            S.SPOTSTATS.get() is not None and not S.SPOTSTATS.get().empty
            and S.TRACKSTATS.get() is not None and not S.TRACKSTATS.get().empty
            and "Condition" in S.SPOTSTATS.get().columns and "Replicate" in S.SPOTSTATS.get().columns
        )

        output_track_reconstruction_normalized(
            Spots_df=S.SPOTSTATS.get(),
            Tracks_df=S.TRACKSTATS.get(),
            condition=input.tracks_conditions(),
            replicate=input.tracks_replicates(),
            c_mode=input.tracks_color_mode(),
            only_one_color=input.tracks_only_one_color(),
            lut_scaling_metric=input.tracks_lut_scaling_metric(),
            smoothing_index=input.tracks_smoothing_index(),
            lw=input.tracks_line_width(),
            background=input.tracks_background(),
            grid=input.tracks_show_grid(),
            grid_style=input.tracks_grid_style(),
            mark_heads=input.tracks_mark_heads(),
            marker=Markers.TrackHeads.get(input.tracks_marker_type()),
            markersize=input.tracks_marks_size()*10,
            title=input.tracks_title()
        )

    @render.plot
    def track_reconstruction_normalized():
        return output_track_reconstruction_normalized.result()

    @render.download(filename=f"Normalized Track Reconstruction {date.today()}.svg")
    def tnr_download():
        req(
            S.SPOTSTATS.get() is not None and not S.SPOTSTATS.get().empty
            and S.TRACKSTATS.get() is not None and not S.TRACKSTATS.get().empty
            and "Condition" in S.SPOTSTATS.get().columns and "Replicate" in S.SPOTSTATS.get().columns
        )

        fig = VisualizeTracksNormalized(
            Spots_df=S.SPOTSTATS.get(),
            Tracks_df=S.TRACKSTATS.get(),
            condition=input.tracks_conditions(),
            replicate=input.tracks_replicates(),
            c_mode=input.tracks_color_mode(),
            only_one_color=input.tracks_only_one_color(),
            lut_scaling_metric=input.tracks_lut_scaling_metric(),
            smoothing_index=input.tracks_smoothing_index(),
            lw=input.tracks_line_width(),
            background=input.tracks_background(),
            grid=input.tracks_show_grid(),
            grid_style=input.tracks_grid_style(),
            mark_heads=input.tracks_mark_heads(),
            marker=Markers.TrackHeads.get(input.tracks_marker_type()),
            markersize=input.tracks_marks_size()*10,
            title=input.tracks_title()
        )
        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="svg", bbox_inches="tight")
                yield buffer.getvalue()

    @render.download(filename=f"Lut Map {date.today()}.svg")
    def download_lut_map_svg():
        req(
            S.SPOTSTATS.get() is not None and not S.SPOTSTATS.get().empty
            and S.TRACKSTATS.get() is not None and not S.TRACKSTATS.get().empty
            and "Condition" in S.SPOTSTATS.get().columns and "Replicate" in S.SPOTSTATS.get().columns
        )

        fig = GetLutMap(
            Spots_df=S.SPOTSTATS.get(),
            Tracks_df=S.TRACKSTATS.get(),
            c_mode=input.tracks_color_mode(),
            lut_scaling_metric=input.tracks_lut_scaling_metric(),
            units=S.UNITS.get(),
            _extend=input.tracks_lutmap_extend_edges()
        )
        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="svg", bbox_inches="tight")
                yield buffer.getvalue()
    
    
    
