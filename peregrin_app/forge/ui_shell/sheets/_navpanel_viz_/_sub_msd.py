from shiny import ui

from src.code import Metrics, Markers, Dyes, Modes




import warnings
from shiny._deprecated import ShinyDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message=r".*panel_well\(\) is deprecated\. Use shiny\.ui\.card\(\) instead\.",
    category=ShinyDeprecationWarning,
)



subpanel_msd = ui.nav_panel(
    "Mean Squared Displacement",
    ui.panel_well(
        ui.markdown(
            """
            ### **Mean Squared Displacement**
            ___
            """
        ),
        ui.accordion(
            ui.accordion_panel(
                "Data Categories",
                ui.row(
                    ui.column(6, ui.input_selectize(id="conditions_msd", label="Conditions:", choices=[], selected=[], multiple=True, options={"placeholder": "Select conditions"})),
                    ui.column(1, ui.input_action_button(id="conditions_reset_msd", label="🗘", class_="btn-noframe")),
                ),
                ui.row(
                    ui.column(4, ui.input_selectize(id="replicates_msd", label="Replicates:", choices=[], selected=[], multiple=True, options={"placeholder": "Select replicates"})),
                    ui.column(1, ui.input_action_button(id="replicates_reset_msd", label="🗘", class_="btn-noframe")),
                    ui.column(4, ui.div(
                        ui.input_checkbox(id="replicates_group_msd", label="Group Replicates", value=True)),
                        style="margin-top: 38px; margin-left: -40px;"
                    )
                )
            ),
            ui.accordion_panel(
                "Compose",
                ui.accordion(
                    ui.accordion_panel(
                        'Line and Scatter',
                        ui.row(
                            ui.column(2, 
                                ui.input_selectize("statistic_msd", None, choices=['mean', 'median'], selected='mean', width="120px"),
                            ),
                            ui.column(2, 
                                ui.input_checkbox("line_show_msd", "Show line", True),
                            ),
                            ui.column(2,
                                ui.input_checkbox("scatter_show_msd", "Show scatter", False),          
                            )  
                        ),
                    ),
                    ui.accordion_panel(
                        'Linear fit',
                        ui.input_checkbox("fit_show_msd", "Show linear fit", True),
                    ),
                    ui.accordion_panel(
                        'Error band',
                        ui.row(
                            ui.column(2, 
                                ui.input_checkbox("error_band_show_msd", "Show error band", True)
                            ),
                            ui.column(2, 
                                ui.panel_conditional(
                                    "input.error_band_show_msd == true",
                                    ui.input_selectize("error_band_type_msd", None, choices=['sd', 'sem', 'min-max', 'CI95'], selected='sem', width="120px"),
                                )
                            )
                        )
                    ),
                    ui.accordion_panel(
                        'Decoratives',
                        ui.input_checkbox("grid_show_msd", "Show grid", True),
                    ),
                )
            ),
            ui.accordion_panel(
                'Color',
                ui.row(
                    ui.column(3, 
                        ui.input_selectize("c_mode_msd", "Color mode:", choices=['differentiate conditions', 'differentiate replicates', 'single color'], selected='differentiate conditions'),
                    ),
                    ui.column(2,
                        ui.panel_conditional(
                            "input.c_mode_msd == 'single color'",
                            ui.div(
                                ui.input_selectize("only_one_color_msd", "Color:", Dyes.Colors),
                                style="margin-left: 15px;"
                            )
                        ),
                        ui.panel_conditional(
                            "input.c_mode_msd != 'single color'",
                            ui.div(
                                ui.input_checkbox("palette_stock_msd", "Use stock palette", True),
                                style="margin-top: 38px; margin-left: 15px;"
                            )
                        )
                    ),
                    ui.column(2, 
                        ui.panel_conditional(
                            "input.c_mode_msd != 'single color' && input.palette_stock_msd == true",
                            ui.div(
                                ui.input_selectize("palette_stock_type_msd", "Palette:", Dyes.PaletteQualitativeMatplotlib),    
                                style="margin-left: -15px;"
                            )
                        )
                    ),
                )
            ),
            class_="accordion02",
        ),
        ui.br(),
        # TODO: create a class for text inputs for plot titling which makes it so thaat the inputed text as well as the placeholder are in the center of the window
        ui.row(ui.input_text(id="title_msd", label=None, placeholder="Title me!", width="100%"), style="margin-left: 1px; margin-right: 1px;"),
        ui.input_task_button(id="generate_msd", label="Generate", class_="btn-secondary task-btn", width="100%"),
    ),
    ui.br(),
    ui.card(
        ui.output_plot("plot_msd"),
        full_screen=True,
        height="800px",
    ),
    ui.download_button("download_plot_msd", "Download", width="100%")
)
