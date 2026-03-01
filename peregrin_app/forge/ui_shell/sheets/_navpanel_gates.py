from shiny import ui

# TODO: build a gating panel

navpanel_gates = ui.nav_panel(
    "Gating Panel",
    ui.layout_sidebar(
        ui.sidebar(
            "Gating_sidebar", 
            ui.input_checkbox("gating_params_inputs", "Inputs for gating params here", True)
        ), 
        ui.markdown(
            """ 
            Gates here
            """
        )
    )
)
