from shiny import ui

from src.code import Metrics, Markers, Dyes, Modes



import warnings
from shiny._deprecated import ShinyDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message=r".*panel_well\(\) is deprecated\. Use shiny\.ui\.card\(\) instead\.",
    category=ShinyDeprecationWarning,
)




subpanel_flow = ui.nav_panel(
    "Motion Flow",
    ui.panel_well(
        ui.markdown(
            """
            ### **Motion flow**
            ___
            """
        ),
        ui.accordion(
            ui.accordion_panel(
                "Data Categories",
                ui.row(
                    ui.column(6, ui.input_selectize(id="conditions_mf", label="Conditions:", choices=[], selected=[], multiple=True, options={"placeholder": "Select conditions"})),
                    ui.column(1, ui.input_action_button(id="conditions_reset_mf", label="🗘", class_="btn-noframe")),
                ),
                ui.row(
                    ui.column(4, ui.input_selectize(id="replicates_mf", label="Replicates:", choices=[], selected=[], multiple=True, options={"placeholder": "Select replicates"})),
                    ui.column(1, ui.input_action_button(id="replicates_reset_mf", label="🗘", class_="btn-noframe")),
                )
            ),
            ui.accordion_panel(
                "Compose",
                ui.input_selectize(id="chart_type_mf", label="Chart type:", choices={"quiver": "Quiver plot", "stream": "Stream plot"}, selected="stream", multiple=False),
                ui.accordion(
                    ui.accordion_panel(
                        "Arrow parameters",
                        ui.row(
                            ui.column(2, ui.input_numeric(id="n_arrows_y_mf", label="Number of arrows (y):", value=10, min=1)),
                            ui.column(2, ui.input_numeric(id="n_arrows_x_mf", label="Number of arrows (x):", value=10, min=1)),
                            ui.panel_conditional(
                                "chart_type_mf == 'stream'",
                                ui.row(
                                    ui.column(2, ui.input_numeric(id="stream_density_mf", label="Stream density:", value=2, min=0.1, step=0.1)),
                                )
                            )
                        ),
                        ui.row(
                            ui.column(2, ui.input_numeric(id="min_arrow_size_mf", label="Min arrow size:", value=0, min=0, step=1)),
                            ui.column(2, ui.input_numeric(id="max_arrow_size_mf", label="Max arrow size:", value=5, min=0.1, step=1)),
                        ),
                        ui.row(
                            ui.column(2, ui.input_selectize(id="arrow_scaling_method_mf", label="Arrow scaling method:", 
                                                            choices={"density": "Density", "min": "Minimum", "max": "Maximum", "mean": "Mean", "median": "Median", "sum": "Sum", 
                                                                     "sd": "Standard Deviation", "add": "Add", "subtract": "Subtract", "multiply": "Multiply", "divide": "Divide"}, 
                                                                     selected="linear", multiple=False)),
                            ui.panel_conditional(
                                "['min', 'max', 'mean', 'median', 'sum', 'sd'].includes(arrow_scaling_method_mf)",
                                ui.column(2, ui.input_selectize(id="arrow_scale_by_mf", label="Scale arrow size by:", choices=[], multiple=False)),
                            ),
                            ui.panel_conditional(
                                "['add', 'subtract', 'multiply', 'divide'].includes(arrow_scaling_method_mf)",
                                ui.row(
                                    ui.column(2, ui.input_selectize(id="arrow_scale_by_a_mf", label="Value A:", choices=[], multiple=False)),
                                    ui.column(2, ui.input_selectize(id="arrow_scale_by_b_mf", label="Value B:", choices=[], multiple=False)),
                                ),
                            )
                        )
                    ),
                )
            ),
            ui.accordion_panel(
                "Color",
                ui.row(
                    ui.column(2, ui.input_selectize(id="color_cmap_mf", label="Colormap:", choices=Dyes.QuantitativeCModes, selected="viridis", multiple=False)),
                    ui.column(2, ui.input_selectize(id="color_by_mf", label="Color by:", choices=[], multiple=False)),
                )
            ),
            class_="accordion02"
        ),
        ui.br(),
        ui.input_task_button(id="generate_mf", label="Generate", class_="btn-secondary task-btn", width="100%")
    )

)