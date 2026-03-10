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
                ui.row(
                    ui.column(3, ui.input_selectize(id="metric_ts", label="Metric:", choices=Metrics.Time, width="350px")),
                    ui.column(2, ui.input_selectize(id="statistic_ts", label="Statistic:", choices=['mean', 'median', 'min', 'max'], selected='mean', width="135px")),
                    ui.column(2, ui.input_selectize(id="dispertion_ts", label="Dispertion:", choices=['min-max', 'sd', 'sem', 'iqr', 'ci', 'none'], width="135px")),
                    ui.column(2, ui.input_selectize(id="level_ts", label="Aggregation level:", choices=['Condition', 'Replicate'], width="175px")),
                ),
                ui.accordion(
                    ui.accordion_panel(
                        "Chart Parameters",
                        ui.row(
                            ui.column(1, ui.input_numeric(id="fig_width_ts", label="Width (in):", value=8, min=1, max=20, step=1, width="95px")),
                            ui.column(1, ui.input_numeric(id="fig_height_ts", label="Height (in):", value=5, min=1, max=20, step=1, width="95px")),
                        ),
                        ui.input_selectize(id="xscale_ts", label="X-axis scale:", choices=['frame', 'time'], selected='frame', width="125px"),
                    ),
                    ui.accordion_panel(
                        "Color",
                        ui.row(
                            ui.column(2, ui.input_checkbox(id="stock_palette_ts", label="Use stock palette", value=True)),
                            ui.panel_conditional(
                                "input.stock_palette_ts == true",
                                ui.column(2, ui.input_selectize(id="palette_ts", label="Palette:", choices=Dyes.PaletteQualitativeMatplotlib, width="175px")),
                            )
                        )
                    )
                ),
                ui.br(),
                ui.input_text("title_ts", label=None, placeholder="Title me!", width="100%"),
            ),
            class_="accordion02"
        ),
        ui.br(),
        ui.input_task_button(id="generate_ts", label="Generate", class_="btn-secondary task-btn", width="100%"),
    ),
    ui.br(),
    ui.card(
        ui.output_plot("plot_ts"),
        full_screen=True,
        height="800px",
    ),
    ui.download_button("download_plot_ts", "Download", width="100%")
)
