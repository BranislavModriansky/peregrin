import io
import asyncio
import warnings

from datetime import date
import pandas as pd

from shiny import render, reactive, ui, req
from utils import VisualizeTracksRealistics, VisualizeTracksNormalized, GetLutMap, Markers, frame_interval_ms
from utils import Animated


import numpy as np
from io import BytesIO
from PIL import Image
import imageio.v3 as iio
import base64




class MountTracks:

    def realistic_reconstruction(input, output, session, S):

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


    def polar_reconstruction(input, output, session, S):

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

    
    def lut_map(input, output, session, S):

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
        
        

    def animated_reconstruction(input, output, session, S):

        @output(id="replay_slider")
        @render.ui
        def replay_slider():
            req(S.FRAMESTATS.get() is not None and not S.FRAMESTATS.get().empty)
            req(input.tar_framerate() is not None)

            num_frames = S.FRAMESTATS.get()["Frame"].nunique()
            return ui.input_slider(
                "frame_replay", "Replay",
                min=1, max=num_frames, value=1, step=1,
                animate={
                    "interval": frame_interval_ms(input.tar_framerate()),
                    "loop": True,
                    "play_button": ui.input_action_button(id="play", label="▶", width="100%"),
                    "pause_button": ui.input_action_button(id="stop", label="❚❚", width="100%")
                },
                width="100%"
            )

        def set_frame(i: int):
            try:
                n = S.FRAMESTATS.get()["Frame"].nunique()
                i = 1 if i < 1 else (n if i > n else i)
                session.send_input_message("frame_replay", {"value": i})
            except Exception as e:
                print(f"Error in set_frame: {e}")

        @reactive.Effect
        @reactive.event(input.prev)
        def _prev():
            try:
                ui.update_slider("frame_replay", value=input.frame_replay() - 1)
            except Exception:
                pass

        @reactive.Effect
        @reactive.event(input.next)
        def _next():
            try:
                ui.update_slider("frame_replay", value=input.frame_replay() + 1)
            except Exception:
                pass

        # Bind the task correctly
        @ui.bind_task_button(button_id="tar_generate")
        @reactive.extended_task
        async def output_track_animated_reconstruction(
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
            title,
            dpi,
            units_time,
        ):
            def _build():
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                        category=UserWarning,
                    )
                    return Animated.create_image_stack(
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
                        title=title,
                        dpi=dpi,
                        units_time=units_time,
                    )
            return await asyncio.get_running_loop().run_in_executor(None, _build)

        @reactive.Effect
        @reactive.event(input.tar_generate, ignore_none=False)
        def _generate_stack():
            output_track_animated_reconstruction.cancel()
            req(S.FRAMESTATS.get() is not None and not S.FRAMESTATS.get().empty)
            output_track_animated_reconstruction(
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
                markersize=input.tracks_marks_size() * 10,
                title=input.tracks_title(),
                dpi=input.tar_dpi(),
                units_time=S.UNITS.get()["Time point"],
            )

        # Run only when the task completes
        @reactive.Effect
        def _get_stack_images():
            stack = output_track_animated_reconstruction.result()
            req(stack is not None and len(stack) == S.FRAMESTATS.get()["Frame"].nunique())
            # Optional: also check expected length against FRAMESTATS if you want
            # req(len(stack) == S.FRAMESTATS.get()["Frame"].nunique())

            def to_webp_data_url(arr: np.ndarray) -> str:
                im = Image.fromarray(arr, mode="RGBA")
                if (arr[:, :, 3] == 255).all():
                    im = im.convert("RGB")
                bio = BytesIO()
                im.save(bio, format="WEBP", quality=80, method=6)
                b64 = base64.b64encode(bio.getvalue()).decode("ascii")
                return f"data:image/webp;base64,{b64}"

            frame_urls = [to_webp_data_url(frame) for frame in stack]
            S.REPLAY_ANIMATION.set(frame_urls)

        @render.ui
        def viewer():
            # Do not catch req’s silent exception
            req(S.REPLAY_ANIMATION.get() is not None)
            req(input.frame_replay() is not None)

            frames = S.REPLAY_ANIMATION.get()
            # Clamp index to valid range
            idx = int(input.frame_replay()) - 1
            src = frames[idx]
            return ui.img(
                # {"src": src, "width": 800, "height": 600, "style": "display:block;"}
                {"src": src, "style": "display:block;"}
            )



        @render.download(filename=f"Animated Track Reconstruction {date.today()}.mp4")
        def tar_download():
            
            stack = output_track_animated_reconstruction.result()
            req(stack is not None and len(stack) == S.FRAMESTATS.get()["Frame"].nunique())

            rgb, out_params = Animated.save_image_stack_as_mp4(
                stack=stack,
                path=""
            )

            with io.BytesIO() as buffer:
                iio.imwrite(
                    buffer,
                    rgb,
                    extension=".mp4",
                    ffmpeg_params=out_params,
                    fps=input.tar_framerate(),
                )
                yield buffer.getvalue()
