import io
import asyncio
import warnings

from datetime import date
import pandas as pd

from shiny import render, reactive, ui, req
from utils import VisualizeTracksRealistics, VisualizeTracksNormalized, GetLutMap, Markers
from utils import Animated


import numpy as np
from io import BytesIO
from PIL import Image
import base64




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
    
    

    
    @output()
    @render.ui
    def replay_slider():
        req(S.FRAMESTATS.get() is not None and not S.FRAMESTATS.get().empty)
        num_frames = S.FRAMESTATS.get()["Frame"].unique().size
        return ui.input_slider(
            "frame_replay", "Replay",
            min=1, max=num_frames, value=1, step=1,
            animate={"interval": input.interval(), "loop": input.loop(),  # ~30 fps
                        "play_button": ui.input_action_button(id="play", label="▶", width="100%"),  # ~30 fps
                        "pause_button": ui.input_action_button(id="stop", label="❚❚", width="100%")},
            width="100%"
        )

    def set_frame(i: int):
        try:

            i = 1 if i < 1 else (S.FRAMESTATS.get()["Frame"].unique().size if i > S.FRAMESTATS.get()["Frame"].unique().size else i)
            session.send_input_message("frame_replay", {"value": i})
        except Exception as e:
            print(f"Error in set_frame: {e}")

    @reactive.Effect
    @reactive.event(input.prev)
    def _():
        try:
            ui.update_slider("frame_replay", value=input.frame_replay() - 1)
        except Exception:
            pass

    @reactive.Effect
    @reactive.event(input.next)
    def _():
        try:
            ui.update_slider("frame_replay", value=input.frame_replay() + 1)
        except Exception:
            pass

    @reactive.Effect
    @reactive.event(input.calculate_replay_animation)
    def _():
        try:
            req(S.FRAMESTATS.get() is not None and not S.FRAMESTATS.get().empty)

            print("Calculating replay animation...")

            STACK = Animated.create_image_stack(
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
                    title=input.tracks_title(),
                    units_time = S.UNITS.get()["Time point"],
                    units = S.UNITS.get()["X coordinate"] if S.UNITS.get()["X coordinate"] == S.UNITS.get()["Y coordinate"] else "",
                )  # (N,H,W,4)

            # Pre-encode frames to WebP data URLs (avoid file IO each render)
            def to_webp_data_url(arr: np.ndarray) -> str:
                im = Image.fromarray(arr, mode="RGBA")
                if (arr[:, :, 3] == 255).all():
                    im = im.convert("RGB")
                bio = BytesIO()
                im.save(bio, format="WEBP", quality=80, method=6)
                b64 = base64.b64encode(bio.getvalue()).decode("ascii")
                return f"data:image/webp;base64,{b64}"

            FRAME_URLS = [to_webp_data_url(STACK[i]) for i in range(S.FRAMESTATS.get()["Frame"].max())]

            S.REPLAY_ANIMATION.set(FRAME_URLS)
            print("Replay animation calculated.")

        except Exception as e:
            print(f"Error in calculate_replay_animation: {e}")


    @render.ui
    def viewer():
        try:
            req(S.REPLAY_ANIMATION.get() is not None)
            idx = int(input.frame_replay()) - 1
            src = S.REPLAY_ANIMATION.get()[idx]
            return ui.img({"src": src, "style": "display:block;"})
        except Exception as e:
            print(f"Error in viewer: {e}")


            
    # @render.ui
    # def viewer():
    #     req(S.FRAMESTATS.get() is not None and not S.FRAMESTATS.get().empty)
    #     try:
    #         idx = int(input.frame_replay()) - 1
    #         frame_urls = Animated.create_image_stack(
    #             Spots_df=S.SPOTSTATS.get(),
    #             Tracks_df=S.TRACKSTATS.get(),
    #             condition=input.tracks_conditions(),
    #             replicate=input.tracks_replicates(),
    #             c_mode=input.tracks_color_mode(),
    #             only_one_color=input.tracks_only_one_color(),
    #             lut_scaling_metric=input.tracks_lut_scaling_metric(),
    #             background=input.tracks_background(),
    #             smoothing_index=input.tracks_smoothing_index(),
    #             lw=input.tracks_line_width(),
    #             grid=input.tracks_show_grid(),
    #             mark_heads=input.tracks_mark_heads(),
    #             marker=Markers.TrackHeads.get(input.tracks_marker_type()),
    #             markersize=input.tracks_marks_size()*10,
    #             title=input.tracks_title(),
    #             units_time = S.UNITS.get()["Time point"],
    #             units = S.UNITS.get()["X coordinate"] if S.UNITS.get()["X coordinate"] == S.UNITS.get()["Y coordinate"] else "",
    #         )
    #         src = frame_urls[idx]
    #         return ui.img(
    #             # {"src": src, "width": 800, "height": 600, "style": "display:block;"}
    #             {"src": src, "style": "display:block;"}
    #         )
        # except Exception as e:
        #     print(f"Error in viewer: {e}")
