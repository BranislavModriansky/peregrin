from shiny import reactive, ui, render


def set_theme(input, output, session, S):

    @output()
    @render.ui
    def custom_theme_url():
        selected_theme = input.app_theme()
        return [
            ui.include_css(f"peregrin_app/src/design/{selected_theme}/{selected_theme}-Theme.css"),
            ui.tags.link(rel="stylesheet", href=f"{selected_theme}/{selected_theme}-Fonts.css") if selected_theme != "Sleek" else None
        ]
        
        