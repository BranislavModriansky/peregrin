import io
import asyncio
import warnings

from datetime import date
import pandas as pd

from shiny import render, reactive, ui, req
from src.code import Markers, frame_interval_ms, ReconstructTracks, Values

import numpy as np
from io import BytesIO
from PIL import Image
import imageio.v3 as iio
import base64


# TODO: reduce code duplication
# TODO: enable the refresh button for categorical selections
# TODO: create uis for custom min/max lut scaling
# TODO: implement stock palettes selection



def MountTracks(input, output, session, S, noticequeue):

    @reactive.Effect
    def update_choices():

        @reactive.Effect
        @reactive.event(S.TRACKSTATS)
        def _():
            if S.TRACKSTATS.get() is None or S.TRACKSTATS.get().empty:
                return
            ui.update_selectize(id="conditions_tr", choices=S.TRACKSTATS.get()["Condition"].unique().tolist(), selected=S.TRACKSTATS.get()["Condition"].unique().tolist()[0] if S.TRACKSTATS.get()["Condition"].unique().tolist() else None)
            ui.update_selectize(id="replicates_tr", choices=S.TRACKSTATS.get()["Replicate"].unique().tolist(), selected=S.TRACKSTATS.get()["Replicate"].unique().tolist() if S.TRACKSTATS.get()["Replicate"].unique().tolist() else None)


        @reactive.Effect
        @reactive.event(input.conditions_reset_tr)
        def _():
            req(S.TINTERVALSTATS.get() is not None and not S.TINTERVALSTATS.get().empty)

            ui.update_selectize(
                id="conditions_tr",
                selected=[]
            ) if input.conditions_tr() else \
            ui.update_selectize(
                id="conditions_tr",
                selected=S.TINTERVALSTATS.get()["Condition"].unique().tolist()
            )
        
        @reactive.Effect
        @reactive.event(input.replicates_reset_tr)
        def _():
            req(S.TINTERVALSTATS.get() is not None and not S.TINTERVALSTATS.get().empty)

            ui.update_selectize(
                id="replicates_tr",
                selected=[]
            ) if input.replicates_tr() else \
            ui.update_selectize(
                id="replicates_tr",
                selected=S.TINTERVALSTATS.get()["Replicate"].unique().tolist()
            )

    @render.text
    def tracks_lutmap_auto_scale_info():
        if (
            S.SPOTSTATS.get() is None or S.SPOTSTATS.get().empty
            or S.TRACKSTATS.get() is None or S.TRACKSTATS.get().empty
        ): return "No data."
        
        stat = input.tracks_lut_scaling_metric()
        if stat == 'Speed instantaneous':
            stat = 'Distance'

        if stat in S.SPOTSTATS.get().columns:
            vals = S.SPOTSTATS.get()[stat]
        else:
            vals = S.TRACKSTATS.get()[stat]
        
        return f" min: {Values.RoundSigFigs(np.nanmin(vals))} - max: {Values.RoundSigFigs(np.nanmax(vals))} "


    def _reconstruct_tracks_kwargs() -> dict:
        """
        IMPORTANT: Call only from a reactive context (main thread).
        Do NOT call from inside an extended_task.
        """
        return dict(
            Spots_df=S.SPOTSTATS.get(),
            Tracks_df=S.TRACKSTATS.get(),
            conditions=input.conditions_tr(),
            replicates=input.replicates_tr(),
            c_mode=input.tracks_color_mode(),
            only_one_color=input.tracks_only_one_color(),
            lut_scaling_stat=input.tracks_lut_scaling_metric(),
            background=input.tracks_background(),
            smoothing_index=input.tracks_smoothing_index(),
            lw=input.tracks_line_width(),
            grid=input.tracks_show_grid(),
            gridstyle=input.tracks_grid_style(),
            mark_heads=input.tracks_mark_heads(),
            marker=Markers.TrackHeads.get(input.tracks_marker_type()),
            marker_size=input.tracks_marks_size() * 10,
            marker_color=input.tracks_marks_color(),
            title=input.tracks_title(),
            auto_lut_scaling=input.tracks_lutmap_scale_auto(), # TODO: not pass this arg and work with it as in the lut min max value setting, only in the forge (front-end of the app)
            lut_vmin=input.tracks_lutmap_scale_min(),
            lut_vmax=input.tracks_lutmap_scale_max(),
            use_stock_palette=input.tracks_use_stock_palette(), # TODO: not pass this arg and work with it as in the lut min max value setting, only in the forge (front-end of the app)
            stock_palette=input.tracks_stock_palette(),
            dpi=input.tar_dpi(),
            units_time=S.UNITS.get().get("Time point", "s"),
            noticequeue=noticequeue,
        )
    
    
    @ui.bind_task_button(button_id="trr_generate")
    @reactive.extended_task
    async def output_track_reconstruction_realistic(kwargs: dict):
        def _build(_kwargs=kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )
                return ReconstructTracks(**_kwargs).Realistic()

        return await asyncio.get_running_loop().run_in_executor(None, _build)

    @reactive.Effect
    @reactive.event(input.trr_generate, ignore_none=False)
    def _():
        output_track_reconstruction_realistic.cancel()

        req(
            S.SPOTSTATS.get() is not None and not S.SPOTSTATS.get().empty
            and S.TRACKSTATS.get() is not None and not S.TRACKSTATS.get().empty
            and "Condition" in S.SPOTSTATS.get().columns and "Replicate" in S.SPOTSTATS.get().columns
        )

        kwargs = _reconstruct_tracks_kwargs()

        # Optional: shallow copies so background thread doesn't race reactive updates
        if isinstance(kwargs.get("Spots_df"), pd.DataFrame):
            kwargs["Spots_df"] = kwargs["Spots_df"].copy(deep=False)
        if isinstance(kwargs.get("Tracks_df"), pd.DataFrame):
            kwargs["Tracks_df"] = kwargs["Tracks_df"].copy(deep=False)

        output_track_reconstruction_realistic(kwargs)

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

        kwargs = _reconstruct_tracks_kwargs()
        fig = ReconstructTracks(**kwargs).Realistic()

        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="svg", bbox_inches="tight")
                yield buffer.getvalue()


    @ui.bind_task_button(button_id="tnr_generate")
    @reactive.extended_task
    async def output_track_reconstruction_normalized(kwargs: dict):
        def _build(_kwargs=kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )
                return ReconstructTracks(**_kwargs).Normalized(_kwargs.get("Spots_df", None))

        return await asyncio.get_running_loop().run_in_executor(None, _build)

    @reactive.Effect
    @reactive.event(input.tnr_generate, ignore_none=False)
    def _():
        output_track_reconstruction_normalized.cancel()

        req(
            S.SPOTSTATS.get() is not None and not S.SPOTSTATS.get().empty
            and S.TRACKSTATS.get() is not None and not S.TRACKSTATS.get().empty
            and "Condition" in S.SPOTSTATS.get().columns and "Replicate" in S.SPOTSTATS.get().columns
        )

        kwargs = _reconstruct_tracks_kwargs()

        # Optional: shallow copies so background thread doesn't race reactive updates
        if isinstance(kwargs.get("Spots_df"), pd.DataFrame):
            kwargs["Spots_df"] = kwargs["Spots_df"].copy(deep=False)
        if isinstance(kwargs.get("Tracks_df"), pd.DataFrame):
            kwargs["Tracks_df"] = kwargs["Tracks_df"].copy(deep=False)

        output_track_reconstruction_normalized(kwargs)

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

        kwargs = _reconstruct_tracks_kwargs()
        fig = ReconstructTracks(**kwargs).Normalized(all_data=S.SPOTSTATS.get())

        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="svg", bbox_inches="tight")
                yield buffer.getvalue()


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


    @ui.bind_task_button(button_id="tar_generate")
    @reactive.extended_task
    async def output_track_reconstruction_animated(kwargs: dict):
        def _build(_kwargs=kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )
                return ReconstructTracks(**_kwargs).ImageStack(
                    dpi=_kwargs.get("dpi", 100),
                    units_time=_kwargs.get("units_time", "s"),
                )

        return await asyncio.get_running_loop().run_in_executor(None, _build)
    
    @reactive.Effect
    @reactive.event(input.tar_generate, ignore_none=False)
    def _():
        output_track_reconstruction_animated.cancel()

        req(
            S.SPOTSTATS.get() is not None and not S.SPOTSTATS.get().empty
            and S.TRACKSTATS.get() is not None and not S.TRACKSTATS.get().empty
            and "Condition" in S.SPOTSTATS.get().columns and "Replicate" in S.SPOTSTATS.get().columns
        )

        kwargs = _reconstruct_tracks_kwargs()
        output_track_reconstruction_animated(kwargs)

        

    @reactive.Effect
    def _get_stack_images():
        stack = output_track_reconstruction_animated.result()
        # Only require a non-empty stack
        req(stack is not None)

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
        idx = max(0, min(int(input.frame_replay()) - 1, len(frames) - 1))
        src = frames[idx]
        return ui.img(
            # {"src": src, "width": 800, "height": 600, "style": "display:block;"}
            {"src": src, "style": "display:block;"}
        )
    
    @render.download(filename=f"Animated Track Reconstruction {date.today()}.mp4")
    def tar_download():
        req(
            S.SPOTSTATS.get() is not None and not S.SPOTSTATS.get().empty
            and S.TRACKSTATS.get() is not None and not S.TRACKSTATS.get().empty
            and "Condition" in S.SPOTSTATS.get().columns and "Replicate" in S.SPOTSTATS.get().columns
        )

        kwargs = _reconstruct_tracks_kwargs()
        stack = output_track_reconstruction_animated.result()
        req(stack is not None)

        rgb, out_params = ReconstructTracks(**kwargs).SaveAnimation(
            stack=stack,
            path="",  # unused, kept for API compatibility
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


    @render.download(filename=f"Lut Map {date.today()}.svg")
    def download_lut_map_svg():
        req(
            S.SPOTSTATS.get() is not None and not S.SPOTSTATS.get().empty
            and S.TRACKSTATS.get() is not None and not S.TRACKSTATS.get().empty
            and "Condition" in S.SPOTSTATS.get().columns and "Replicate" in S.SPOTSTATS.get().columns
        )

        kwargs = _reconstruct_tracks_kwargs()
        fig = ReconstructTracks(**kwargs).GetLutMap(units=S.UNITS.get(), _extend=input.tracks_lutmap_extend_edges())

        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="svg", bbox_inches="tight")
                yield buffer.getvalue()

    