# app.py
from htmltools import HTML
from shiny import App, ui, render, reactive, req
import numpy as np
from io import BytesIO
from PIL import Image
import base64

# ---------- image stack builder (no Matplotlib during playback) ----------
def create_image_stack(n_frames: int = 180, size=(800, 450)) -> np.ndarray:
    """Return uint8 RGBA frames: shape (N, H, W, 4)."""
    W, H = size
    x = np.linspace(0, 2 * np.pi, 1000)
    frames = []
    for i in range(n_frames):
        phase = 2 * np.pi * i / n_frames
        y = 0.4 * np.sin(x + phase)
        img = Image.new("RGBA", (W, H), (255, 255, 255, 255))
        xs = ((x - x.min()) / (x.max() - x.min()) * (W - 1)).astype(int)
        ys = ((1 - (y - y.min()) / (y.max() - y.min())) * (H - 1)).astype(int)
        px = img.load()
        for X, Y in zip(xs, ys):
            for dy in (-1, 0, 1):
                yy = max(0, min(H - 1, Y + dy))
                px[X, yy] = (0, 0, 0, 255)
        frames.append(np.asarray(img, dtype=np.uint8))
    return np.stack(frames, axis=0)

# Precompute frames once
STACK = create_image_stack(n_frames=180, size=(400, 300))  # (N,H,W,4)
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
            ui.column(10,
                ui.card(
                    ui.div(
                        ui.input_action_button("prev", "-"),
                        ui.output_ui("viewer"),
                        ui.input_action_button("next", "+"),
                        style="display:flex; justify-content:space-between; align-items:center;"
                    ),
                    ui.output_ui("replay_slider"),
                    ui.input_checkbox("loop", "Loop", True),
                    ui.input_numeric("interval", "Time interval between consecutive frames (ms)", value=200, min=10, max=1000, step=1),
                    full_screen=False, width="100", height=f"{HEIGHT + 250}px"
                )
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
            "frame_replay", "Replay",
            min=1, max=N_FRAMES, value=1, step=1,
            animate={"interval": input.interval(), "loop": input.loop(),  # ~30 fps
                        "play_button": ui.input_action_button(id="play", label="▶", width="100%"),  # ~30 fps
                        "pause_button": ui.input_action_button(id="stop", label="❚❚", width="100%")},
            width="100%"
        )

    def set_frame(i: int):
        i = 1 if i < 1 else (N_FRAMES if i > N_FRAMES else i)
        session.send_input_message("frame_replay", {"value": i})

    # @reactive.Effect
    # @reactive.event(input.prev, input.next)
    # def _():

    @reactive.Effect
    @reactive.event(input.prev)
    def _():
        ui.update_slider("frame_replay", value=input.frame_replay() - 1)

    @reactive.Effect
    @reactive.event(input.next)
    def _():
        ui.update_slider("frame_replay", value=input.frame_replay() + 1)
            
    @render.ui
    def viewer():
        idx = int(input.frame_replay()) - 1
        src = FRAME_URLS[idx]
        return ui.img(
            {"src": src, "width": WIDTH, "height": HEIGHT, "style": "display:block;"}
        )

app = App(app_ui, server)

# Run:
# shiny run --reload app.py
