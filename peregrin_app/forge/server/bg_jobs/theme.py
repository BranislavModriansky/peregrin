from shiny import reactive, ui, render


def set_theme(input, output, session, S):

    @reactive.Effect
    def _():
        selected_theme = input.app_theme()
        
        if selected_theme == "Console-1":
            if not S.PLAY.get():
                ui.insert_nav_panel(
                    id="main_nav",
                    nav_panel=ui.nav_panel(
                        "Play",
                        ui.panel_well(
                            ui.markdown("#### **Playground**"),
                            ui.markdown("Play with the physics of the background."),
                            
                        )
                    ),
                    target="Visualisation",
                    position="after" 
                )
                S.PLAY.set(True)
        else:
            if S.PLAY.get():
                ui.remove_nav_panel(
                    id="main_nav",
                    target="Play"
                )
                S.PLAY.set(False)

    @output()
    @render.ui
    def custom_theme_url():
        styles = []

        selected_theme = input.app_theme()
        try:
            selected_theme_base = selected_theme.split("-")[0]
        except Exception:
            selected_theme_base = selected_theme

        styles.append(
            [
                ui.include_css(f"peregrin_app/src/styles/{selected_theme_base}/{selected_theme}-Theme.css"),
                ui.tags.link(rel="stylesheet", href=f"{selected_theme_base}/{selected_theme_base}-Fonts.css") if selected_theme_base != "Sleek" else None
            ]
        )

        styles.append(ui.include_js("peregrin_app/src/js/remove_js.js"))

        if selected_theme == "Console-1":
            styles.append(ui.include_js("peregrin_app/src/js/proton_grid.js"))
            
        if selected_theme == "Console-2":
            styles.append(ui.include_js("peregrin_app/src/js/tiles_grid.js"))
        
        return styles

