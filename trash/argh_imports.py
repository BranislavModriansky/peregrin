# app.py
from htmltools import HTML
from shiny import App, ui, render, reactive, req
import numpy as np
from io import BytesIO
from PIL import Image
import base64

# Extra deps for the new create_image_stack implementation
import io
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.colors as mcolors


spot_stats = pd.read_csv(r"C:\Users\modri\Desktop\Lab\Peregrin project\Data\Spot stats 2025-09-30-1.csv")
track_stats = pd.read_csv(r"C:\Users\modri\Desktop\Lab\Peregrin project\Data\Track stats 2025-09-30 1.csv")

# ---------- helpers ----------
def _get_cmap(name: str) -> mcolors.Colormap:
    """Minimal cmap resolver used by create_image_stack."""
    name = (name or "").lower()
    lut = {
        "greyscale lut": plt.cm.gist_gray,
        "reverse grayscale lut": plt.cm.gist_yarg,
        "jet lut": plt.cm.jet,
        "brg lut": plt.cm.brg,
        "cool lut": plt.cm.cool,
        "hot lut": plt.cm.hot,
        "inferno lut": plt.cm.inferno,
        "viridis lut": plt.cm.viridis,
        "plasma lut": plt.cm.plasma,
        "magma lut": plt.cm.magma,
        "cividis lut": plt.cm.cividis,
    }
    return lut.get(name, plt.cm.viridis)

# ---------- image stack builder (precompute with Matplotlib; playback uses raw RGBA) ----------
def create_image_stack(
    n_frames: int = 180,
    size: tuple[int, int] = (800, 450),
    *,
    # Optional real data inputs (wire these in your app as needed)
    Spots_df: pd.DataFrame | None = None,
    Tracks_df: pd.DataFrame | None = None,
    condition: str | None = None,
    replicate: str = "all",
    c_mode: str = "differentiate replicates",
    only_one_color: str = "blue",
    lut_scaling_metric: str = "Track displacement",
    background: str = "dark",
    smoothing_index: int | float = 0,
    lw: float = 1.0,
    units_time: str = "s",
    grid: bool = True,
    mark_heads: bool = False,
    marker: dict = {"symbol": "o", "fill": True},
    markersize: float = 5.0,
    title: str = "Track Visualization",
    frames_mode: str = "cumulative",  # 'cumulative' | 'per_frame'
    dpi: int = 150,
) -> np.ndarray:
    """
    Build a stack of frames from tracks, returning uint8 RGBA of shape (N, H, W, 4).
    If Spots_df/Tracks_df are not provided, falls back to a simple demo sine path.
    """

    W, H = size
    width_in = W / dpi
    height_in = H / dpi

    # If no real data supplied, generate a simple demo so the app still runs.
    if Spots_df is None or Tracks_df is None or condition is None:
        x = np.linspace(0, 2 * np.pi, 400)
        frames = []
        for i in range(n_frames):
            phase = 2 * np.pi * i / max(1, n_frames)
            y = 0.4 * np.sin(x + phase)

            fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
            # Background
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            # Draw line
            xs = (x - x.min()) / (x.max() - x.min())
            ys = (y - y.min()) / (y.max() - y.min())
            ax.plot(xs, ys, linewidth=2)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
            plt.close(fig)
            buf.seek(0)
            im = Image.open(buf).convert("RGBA")
            frames.append(np.asarray(im, dtype=np.uint8))
        return np.stack(frames, axis=0)

    # ---- Real data path (modified ImageStackTracksRealistics) ----
    Spots = Spots_df.copy()
    Tracks = Tracks_df.copy()

    required = ["Condition", "Replicate", "Track ID", "Time point", "X coordinate", "Y coordinate"]
    if any(col not in Spots.columns for col in required):
        # Shape-safe empty
        return np.zeros((0, H, W, 4), dtype=np.uint8)

    # Filter by condition/replicate
    if replicate == "all":
        Spots = Spots.loc[Spots["Condition"] == condition]
        Tracks = Tracks.loc[Tracks["Condition"] == condition]
    else:
        Spots = Spots.loc[(Spots["Condition"] == condition) & (Spots["Replicate"] == replicate)]
        Tracks = Tracks.loc[(Tracks["Condition"] == condition) & (Tracks["Replicate"] == replicate)]

    if Spots.empty:
        return np.zeros((0, H, W, 4), dtype=np.uint8)

    # Sort and index
    key_cols = ["Condition", "Replicate", "Track ID"]
    Spots = Spots.sort_values(key_cols + ["Time point"]).set_index(key_cols, drop=True)
    Tracks = Tracks.sort_values(key_cols).set_index(key_cols, drop=True)

    # Optional smoothing
    if isinstance(smoothing_index, (int, float)) and smoothing_index > 1:
        win = int(smoothing_index)
        Spots["X coordinate"] = Spots.groupby(level=key_cols)["X coordinate"].transform(
            lambda s: s.rolling(win, min_periods=1).mean()
        )
        Spots["Y coordinate"] = Spots.groupby(level=key_cols)["Y coordinate"].transform(
            lambda s: s.rolling(win, min_periods=1).mean()
        )

    # Colors
    rng = np.random.default_rng(42)

    def _rand_color_hex():
        return mcolors.to_hex(rng.random(3))

    def _rand_grey_hex():
        g = float(rng.random())
        return mcolors.to_hex((g, g, g))

    if c_mode in ["random colors", "random greys", "only-one-color"]:
        unique_tracks = Tracks.index.unique()
        if c_mode == "random colors":
            colors = [_rand_color_hex() for _ in range(len(unique_tracks))]
        elif c_mode == "random greys":
            colors = [_rand_grey_hex() for _ in range(len(unique_tracks))]
        else:
            colors = [only_one_color] * len(unique_tracks)
        track_to_color = dict(zip(unique_tracks, colors))
        Tracks["Track color"] = [track_to_color[idx] for idx in Tracks.index]

    elif c_mode == "differentiate replicates":
        Tracks["Track color"] = Tracks["Replicate color"] if "Replicate color" in Tracks.columns else "red"

    else:
        use_instantaneous = lut_scaling_metric == "Speed instantaneous"
        if lut_scaling_metric in Tracks.columns and not use_instantaneous:
            cmap = _get_cmap(c_mode)
            vmin = float(Tracks[lut_scaling_metric].min())
            vmax = float(Tracks[lut_scaling_metric].max())
            vmax = vmax if np.isfinite(vmax) and vmax > vmin else vmin + 1.0
            norm = plt.Normalize(vmin, vmax)
            Tracks["Track color"] = [mcolors.to_hex(cmap(norm(v))) for v in Tracks[lut_scaling_metric].to_numpy()]
        elif use_instantaneous:
            cmap = _get_cmap(c_mode)
            g = Spots.groupby(level=key_cols)
            d = np.sqrt((g["X coordinate"].diff()) ** 2 + (g["Y coordinate"].diff()) ** 2)
            speed_end = d  # distance per step; you can scale by time externally if needed
            vmax = float(np.nanmax(speed_end.to_numpy())) if np.isfinite(speed_end.to_numpy()).any() else 1.0
            norm = plt.Normalize(0.0, vmax if vmax > 0 else 1.0)
            Spots["Spot color"] = [
                mcolors.to_hex(_get_cmap(c_mode)(norm(v))) if np.isfinite(v) else mcolors.to_hex(_get_cmap(c_mode)(0.0))
                for v in speed_end.to_numpy()
            ]

    if not (lut_scaling_metric == "Speed instantaneous"):
        Spots = Spots.join(
            Tracks[["Track color"]],
            on=key_cols,
            how="left",
            validate="many_to_one",
        )

    # Axes limits fixed over time
    x_all = Spots["X coordinate"].to_numpy(dtype=float, copy=False)
    y_all = Spots["Y coordinate"].to_numpy(dtype=float, copy=False)
    xlim = (np.nanmin(x_all), np.nanmax(x_all))
    ylim = (np.nanmin(y_all), np.nanmax(y_all))

    # Background presets
    if background == "white":
        grid_color, face_color, grid_alpha, grid_ls = "gainsboro", "white", 0.5, "-." if grid else "None"
    elif background == "light":
        grid_color, face_color, grid_alpha, grid_ls = "silver", "lightgrey", 0.5, "-." if grid else "None"
    elif background == "mid":
        grid_color, face_color, grid_alpha, grid_ls = "silver", "darkgrey", 0.5, "-." if grid else "None"
    elif background == "dark":
        grid_color, face_color, grid_alpha, grid_ls = "grey", "dimgrey", 0.5, "-." if grid else "None"
    elif background == "black":
        grid_color, face_color, grid_alpha, grid_ls = "dimgrey", "black", 0.5, "-." if grid else "None"
    else:
        grid_color, face_color, grid_alpha, grid_ls = "gainsboro", "white", 0.5, "-." if grid else "None"

    # Time points
    time_points = np.unique(Spots["Time point"].to_numpy())
    time_points.sort()

    # If n_frames is provided but fewer time points exist, clamp.
    if n_frames is not None and len(time_points) > n_frames:
        time_points = time_points[:n_frames]

    frames: list[np.ndarray] = []

    for t in time_points:
        if frames_mode == "per_frame":
            Spots_t = Spots.loc[Spots["Time point"] == t]
        else:  # cumulative
            Spots_t = Spots.loc[Spots["Time point"] <= t]

        # Build line segments
        segments = []
        seg_colors = []

        if lut_scaling_metric == "Speed instantaneous" and "Spot color" in Spots_t.columns:
            for _, g in Spots_t.groupby(level=key_cols, sort=False):
                xy = g[["X coordinate", "Y coordinate"]].to_numpy(dtype=float, copy=False)
                if xy.shape[0] >= 2:
                    cols = g["Spot color"].astype(str).to_numpy()
                    for i in range(1, xy.shape[0]):
                        segments.append(xy[i - 1 : i + 1])
                        seg_colors.append(cols[i])
        else:
            for _, g in Spots_t.groupby(level=key_cols, sort=False):
                xy = g[["X coordinate", "Y coordinate"]].to_numpy(dtype=float, copy=False)
                if xy.shape[0] >= 2:
                    segments.append(xy)
                    seg_colors.append(g["Track color"].iloc[0] if "Track color" in g.columns else "red")

        # Render
        fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(f"X coordinate [{units_time}]")
        ax.set_ylabel(f"Y coordinate [{units_time}]")
        ax.set_title(f"{title} | t={t}")
        ax.set_facecolor(face_color)
        if grid:
            ax.grid(True, which="both", axis="both", color=grid_color, linestyle=grid_ls, linewidth=1, alpha=grid_alpha)
        else:
            ax.grid(False)

        ax.xaxis.set_major_locator(MultipleLocator(200))
        ax.yaxis.set_major_locator(MultipleLocator(200))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(MultipleLocator(50))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.tick_params(axis="both", which="major", labelsize=8)

        if segments:
            lc = LineCollection(segments, colors=seg_colors, linewidths=lw, zorder=10)
            ax.add_collection(lc)

        if mark_heads:
            ends = Spots_t.groupby(level=key_cols, sort=False).tail(1)
            if len(ends):
                xe = ends["X coordinate"].to_numpy(dtype=float, copy=False)
                ye = ends["Y coordinate"].to_numpy(dtype=float, copy=False)
                if lut_scaling_metric == "Speed instantaneous" and "Spot color" in ends.columns:
                    cols = ends["Spot color"].astype(str).to_numpy()
                else:
                    cols = ends["Track color"].astype(str).to_numpy() if "Track color" in ends.columns else np.array(["red"] * len(ends))
                m = np.isfinite(xe) & np.isfinite(ye)
                if m.any():
                    ax.scatter(
                        xe[m],
                        ye[m],
                        marker=marker.get("symbol", "o"),
                        s=markersize,
                        edgecolor=cols[m],
                        facecolor=cols[m] if marker.get("fill", True) else "none",
                        linewidths=lw,
                        zorder=12,
                    )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        im = Image.open(buf).convert("RGBA")
        frames.append(np.asarray(im, dtype=np.uint8))

    if not frames:
        return np.zeros((0, H, W, 4), dtype=np.uint8)

    stack = np.stack(frames, axis=0)
    # Ensure exact WxH; if backends add borders, resize safely.
    if stack.shape[1] != H or stack.shape[2] != W:
        # Resize each frame to requested size
        fixed = []
        for f in stack:
            fixed.append(np.asarray(Image.fromarray(f, mode="RGBA").resize((W, H), Image.BILINEAR), dtype=np.uint8))
        stack = np.stack(fixed, axis=0)
    return stack

# Precompute frames once
# You can pass real dataframes via kwargs later. Default call still works.
STACK = create_image_stack(n_frames=180, size=(400, 300),
                           Spots_df=spot_stats, Tracks_df=track_stats,
                           condition=track_stats["Condition"].iloc[0],
                           replicate=track_stats["Replicate"].iloc[0],
                           c_mode="jet lut", lut_scaling_metric="Speed instantaneous")  # (N,H,W,4)
N_FRAMES, HEIGHT, WIDTH, _ = STACK.shape

# Pre-encode frames to WebP data URLs (avoid file IO each render)
def to_webp_data_url(arr: np.ndarray) -> str:
    im = Image.fromarray(arr, mode="RGBA")
    if (arr[:, :, 3] == 255).all():
        im = im.convert("RGB")
    bio = BytesIO()
    im.save(bio, format="WEBP", quality=80, method=6)
    b64 = base64.b64encode(bio.getvalue()).decode("ascii")
    return f"data:image/webp;base64,{b64}"

FRAME_URLS = [to_webp_data_url(STACK[i]) for i in range(N_FRAMES)]

# ---------- UI ----------
app_ui = ui.page_fluid(
    ui.row(
        ui.tags.style(
            """
            #prev, #next {
                border: none !important;
                background: none !important;
                box-shadow: none !important;
                padding: 0.25rem 0.5rem !important;
                font-size: 3rem;
            }
            #prev:hover, #next:hover {
                color: #007bff;
                cursor: pointer;
            }
            #play, #stop {
                border: none !important;
                background: none !important;
                box-shadow: none !important;
                padding: 0.25rem 0.5rem !important;
                font-size: 2rem;
            }
            #play:hover, #stop:hover {
                color: dimgrey;
                cursor: pointer;
            }
            """
        ),
        ui.row(
            ui.column(
                10,
                ui.card(
                    ui.div(
                        ui.input_action_button("prev", "-"),
                        ui.output_ui("viewer"),
                        ui.input_action_button("next", "+"),
                        style="display:flex; justify-content:space-between; align-items:center;"
                    ),
                    ui.output_ui("replay_slider"),
                    ui.input_checkbox("loop", "Loop", True),
                    ui.input_numeric(
                        "interval",
                        "Time interval between consecutive frames (ms)",
                        value=200,
                        min=10,
                        max=1000,
                        step=1,
                    ),
                    full_screen=False,
                    width="100",
                    height=f"{HEIGHT + 250}px",
                ),
            )
        ),
    ),
)

# ---------- Server ----------
def server(input, output, session):

    @output()
    @render.ui
    def replay_slider():
        return ui.input_slider(
            "frame_replay",
            "Replay",
            min=1,
            max=N_FRAMES,
            value=1,
            step=1,
            animate={
                "interval": input.interval(),
                "loop": input.loop(),
                "play_button": ui.input_action_button(id="play", label="▶", width="100%"),
                "pause_button": ui.input_action_button(id="stop", label="❚❚", width="100%"),
            },
            width="100%",
        )

    def set_frame(i: int):
        i = 1 if i < 1 else (N_FRAMES if i > N_FRAMES else i)
        session.send_input_message("frame_replay", {"value": i})

    @reactive.Effect
    @reactive.event(input.prev)
    def _prev():
        ui.update_slider("frame_replay", value=input.frame_replay() - 1)

    @reactive.Effect
    @reactive.event(input.next)
    def _next():
        ui.update_slider("frame_replay", value=input.frame_replay() + 1)

    @render.ui
    def viewer():
        idx = int(input.frame_replay()) - 1
        src = FRAME_URLS[idx]
        return ui.img({"src": src, "width": WIDTH, "height": HEIGHT, "style": "display:block;"})

app = App(app_ui, server)

# Run:
# shiny run --reload app.py
