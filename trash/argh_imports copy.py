from shiny import App, ui, render

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.navset_pill_list(
            ui.nav_panel("Filters",
                ui.input_slider("thresh", "Threshold", 0, 100, 50),
                ui.input_checkbox("smooth", "Smooth", True),
            ),
            ui.nav_panel("Style",
                ui.input_select("palette", "Palette", ["viridis","magma","plasma"]),
                ui.input_numeric("lw", "Line width", 1.5, min=0.1, step=0.1),
            ),
            id="side_nav",
        ),
        open="open",
    ),
    ui.card(
        ui.h4("Main content"),
        ui.output_text("which")
    ),
)

def server(input, output, session):
    @output
    @render.text
    def which():
        return f"Active sidebar tab: {input.side_nav()}"

app = App(app_ui, server)
