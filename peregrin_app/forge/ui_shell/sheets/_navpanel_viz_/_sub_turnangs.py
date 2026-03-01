from shiny import ui

from src.code import Metrics, Markers, Dyes, Modes



import warnings
from shiny._deprecated import ShinyDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message=r".*panel_well\(\) is deprecated\. Use shiny\.ui\.card\(\) instead\.",
    category=ShinyDeprecationWarning,
)



subpanel_turnangs = ui.nav_panel(
    "Turning angles",
    ui.panel_well(
        ui.markdown(
            """
            ### **Turning angles**
            ___
            """
        ),
        ui.accordion(
            ui.accordion_panel(
                "Data Categories",
                ui.row(
                    ui.column(6, ui.input_selectize(id="conditions_ta", label="Conditions:", choices=[], selected=[], multiple=True, options={"placeholder": "Select conditions"})),
                    ui.column(1, ui.input_action_button(id="conditions_reset_ta", label="🗘", class_="btn-noframe")),
                ),
                ui.row(
                    ui.column(4, ui.input_selectize(id="replicates_ta", label="Replicates:", choices=[], selected=[], multiple=True, options={"placeholder": "Select replicates"})),
                    ui.column(1, ui.input_action_button(id="replicates_reset_ta", label="🗘", class_="btn-noframe")),
                )
            ),
            ui.accordion_panel(
                "Compose",
                ui.row(
                    ui.div(
                        ui.div("Angle range [°]:", style="margin-top: 5px; padding-right: 15px;"),
                        ui.input_numeric("angle_range_ta", None, value=15, min=0, max=360, step=5, width="120px"),
                        style="display: flex;"
                    ),
                ),
                ui.br(),
                ui.input_text("title_ta", label=None, placeholder="Title me!", width="100%"),
            ),
            ui.accordion_panel(
                "Color",
                ui.input_selectize(id="cmap_ta", label="Color map:", choices=Dyes.QuantitativeCModes, selected="magma LUT", width="200px"),
            ),
            class_="accordion02"
        ),
        ui.br(),
        ui.input_task_button(id="generate_ta", label="Generate", class_="btn-secondary task-btn", width="100%")
    ),
    ui.br(),
    ui.card(
        ui.output_plot("plot_ta"),
        full_screen=True,
        height="1000px",
    ),
    ui.download_button("download_plot_ta", "Download", width="100%")
)
