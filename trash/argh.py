from shiny import App, ui

MEDIA_DIR = r"C:\Users\modri\Desktop\Lab\Peregrin project\Shots"  # served at /media

app_ui = ui.page_fluid(
    ui.h3("Video player"),
    ui.tags.video(
        {"controls": True, "width": "640"},
        ui.tags.source(src="/media/Gating%20demo.mp4", type="video/mp4"),  # URL, not C:\ path
    ),
)

app = App(app_ui, None, static_assets={"/media": MEDIA_DIR})
