from shiny import ui

from src.code import Metrics, Markers, Dyes, Modes



import warnings
from shiny._deprecated import ShinyDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message=r".*panel_well\(\) is deprecated\. Use shiny\.ui\.card\(\) instead\.",
    category=ShinyDeprecationWarning,
)




subpanel_series = ui.nav_panel(
    "Time series",
    ui.panel_well(
        ui.markdown(
            """
            ### **Time series**
            ___
            """
        ),

        ui.accordion(
            ui.accordion_panel(
                "Data Categories",
                ui.row(
                    ui.column(6, ui.input_selectize(id="conditions_ts", label="Conditions:", choices=[], selected=[], multiple=True, options={"placeholder": "Select conditions"})),
                    ui.column(1, ui.input_action_button(id="conditions_reset_ts", label="🗘", class_="btn-noframe")),
                ),
                ui.row(
                    ui.column(4, ui.input_selectize(id="replicates_ts", label="Replicates:", choices=[], selected=[], multiple=True, options={"placeholder": "Select replicates"})),
                    ui.column(1, ui.input_action_button(id="replicates_reset_ts", label="🗘", class_="btn-noframe")),
                )
            ),
            ui.accordion_panel(
                "Compose",
                ui.input_selectize(id="metric_ts", label="Metric:", choices=Metrics.Time, width="150px"),

                ui.br(),
                ui.input_text("title_ts", label=None, placeholder="Title me!", width="100%"),
            ),
            class_="accordion02"
        ),
        ui.br(),
        ui.div(" 🏗️ ", style="font-size: 360px")
    )
)
