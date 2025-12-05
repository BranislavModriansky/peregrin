from shiny import reactive, ui, render
import json
from pathlib import Path


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
                        ui.div(
                            ui.panel_well(
                                ui.markdown("#### **Playground**"),
                                ui.markdown("Play with the physics of the background."),
                                ui.layout_columns(
                                    ui.card(
                                        ui.input_slider("js_mouse_force", "Cursor's Force", 0, 5, 0.75, step=0.05),
                                        ui.input_slider("js_influence_radius", "Influence Radius", 100, 1500, 750, step=50),
                                        ui.input_slider("js_spring_stiffness", "Spring Stiffness", 0.001, 0.2, 0.035, step=0.001),
                                        ui.input_slider("js_damping", "Damping", 0.1, 0.99, 0.75, step=0.01),  
                                    ),
                                    ui.card(
                                        ui.input_slider("js_mouse_lag", "Mouse Lag", 0.01, 1.0, 0.75, step=0.05),
                                        ui.input_slider("js_tangle_threshold", "Repell Radius", 1, 100, 10, step=1),
                                        ui.input_slider("js_glow_intensity", "Glow Intensity", 0, 10, 5.25, step=0.25),
                                        ui.input_slider("js_glow_power", "Glow Increase Curve", 1, 6, 3, step=1),
                                    )
                                ),
                                # Place the config output here in the body so it binds correctly.
                                # Hidden via style, but active due to suspend_when_hidden=False.
                                ui.output_ui("proton_grid_config", style="display: none;")
                            ),
                            style="height: 1500px;"
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

    # Manage background ON/OFF switch as a side-effect, not inside an output
    @reactive.Effect
    def _bg_switch_manager():
        selected_theme = input.app_theme()

        # Only show the switch for Console themes
        if selected_theme in ["Console-1", "Console-2"]:
            if not S.BG_SWITCH.get():
                # Insert the switch once, after #main_nav
                ui.insert_ui(
                    selector="#main_nav",
                    ui=ui.input_switch(
                        id="animated_bg",
                        label=None,
                        value=S.BG_ENABLED.get(),
                        width="30px",
                    ),
                    where="afterEnd",
                )
                S.BG_SWITCH.set(True)
        else:
            if S.BG_SWITCH.get():
                # Remove the switch when leaving Console themes
                ui.remove_ui(selector="#animated_bg")
                S.BG_SWITCH.set(False)


    # This output will re-render whenever any slider changes.
    # It emits a <script> that calls window.PeregrinGridUpdateConfig({...})
    # FIX: suspend_when_hidden=False ensures this runs even if the output container is invisible
    @output(suspend_when_hidden=False)
    @render.ui
    def proton_grid_config():
        # Only emit when Console-1 + Play tab inserted
        if not S.PLAY.get() or input.app_theme() != "Console-1":
            return ui.HTML("")

        try:
            cfg = {
                "MOUSE_FORCE": input.js_mouse_force(),
                "INFLUENCE_RADIUS": input.js_influence_radius(),
                "SPRING_STIFFNESS": input.js_spring_stiffness(),
                "GLOW_INTENSITY": input.js_glow_intensity(),
                "DAMPING": input.js_damping(),
                "MOUSE_LAG": input.js_mouse_lag(),
                "GLOW_POWER": input.js_glow_power(),
                "TANGLE_THRESHOLD": input.js_tangle_threshold(),
                # propagate background enabled state into JS
                "ENABLED": S.BG_ENABLED.get(),
            }
        except Exception:
            # Inputs might not be bound yet
            return ui.HTML("")

        # Use json.dumps to ensure valid JS syntax (e.g. True -> true)
        return ui.HTML(
            f"<script>window.PeregrinGridUpdateConfig && "
            f"window.PeregrinGridUpdateConfig({json.dumps(cfg)});</script>"
        )
    
    @reactive.Effect
    def _sync_bg_enabled_from_switch():
        try:
            enabled = input.animated_bg()
        except Exception:
            return
        if enabled != S.BG_ENABLED.get():
            S.BG_ENABLED.set(enabled)

    @output()
    @render.ui
    def custom_theme_url():
        styles = []

        # Go from this file up to project root by name instead of fixed parents
        # Current file: peregrin_app/forge/server/bg_jobs/theme.py
        root = Path(__file__).resolve()
        for ancestor in root.parents:
            if (ancestor / "src").is_dir():
                root = ancestor
                break

        selected_theme = input.app_theme()
        try:
            selected_theme_base = selected_theme.split("-")[0]
        except Exception:
            selected_theme_base = selected_theme

        styles.append(ui.include_js(f"{root}/src/js/remove_js.js"))

        styles.append(
            [
                ui.include_css(
                    f"{root}/src/styles/{selected_theme_base}/{selected_theme}-Theme.css"
                ),
                ui.tags.link(
                    rel="stylesheet",
                    href=f"{root}/src/styles/{selected_theme_base}/{selected_theme_base}-Fonts.css",
                )
                if selected_theme_base != "Shiny"
                else None,
            ]
        )

        # Only pure includes + output_ui here; no insert_ui/remove_ui side-effects.
        if selected_theme == "Console-1":
            # Removed ui.output_ui("proton_grid_config") from here as it's now in the Play panel
            if S.BG_ENABLED.get():
                styles.append(ui.include_js(f"{root}/src/js/proton_grid.js"))
            

        if selected_theme == "Console-2":
            if S.BG_ENABLED.get():
                styles.append(ui.include_js(f"{root}/src/js/tiles_grid.js"))

        return styles





