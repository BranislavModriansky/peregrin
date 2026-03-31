from pathlib import Path
from shiny import ui


styles_path = Path(__file__).parents[3] / "src" / "styles"


navpanel_inputs = ui.nav_panel(
    "Input Menu",
    # _ Buttons & input UIs _
    ui.div(
        ui.div(ui.output_ui("import_mode")),
        ui.div(ui.output_ui("buttons")),
        style="display:flex; align-items:center; justify-content:space-between; gap:12px;"
    ),
    ui.output_ui("input_panel"),
    ui.include_js(styles_path / "icons.js"),
    ui.include_js(styles_path / "file_preview_popup.js"),
)