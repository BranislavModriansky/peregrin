from shiny import reactive, ui, render


def set_theme(input, output, session, S):

    @output()
    @render.ui
    def custom_theme_url():
        selected_theme = input.app_theme()
        try:
            selected_theme_base = selected_theme.split("-")[0]
        except Exception:
            selected_theme_base = selected_theme
        return [
            ui.include_css(f"peregrin_app/src/styles/{selected_theme_base}/{selected_theme}-Theme.css"),
            ui.tags.link(rel="stylesheet", href=f"{selected_theme_base}/{selected_theme_base}-Fonts.css") if selected_theme_base != "Sleek" else None
        ]
        
        