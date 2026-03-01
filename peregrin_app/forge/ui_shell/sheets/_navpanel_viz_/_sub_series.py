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
        
        ui.input_select("tch_plot", "Plot:", choices=["Scatter", "Line", "Error band"]),

        ui.accordion(
            ui.accordion_panel(
                "Dataset",
                ui.input_selectize("tch_condition", "Condition:", ["all"]),
                ui.panel_conditional(
                    "input.tch_condition != 'all'",
                    ui.input_selectize("tch_replicate", "Replicate:", ["all"]),
                    ui.panel_conditional(
                        "input.tch_replicate == 'all'",
                        ui.input_checkbox("time_separate_replicates", "Show replicates separately", False),
                    ),
                ),
            ),

            ui.accordion_panel(
                "Metric",
                ui.input_selectize("tch_metric", label=None, choices=Metrics.Time, selected="Straightness index mean"),
                ui.row(
                    ui.column(2,
                        ui.input_radio_buttons("tch_y_axis", "Y axis with", ["Absolute values", "Relative values"]),
                        style="margin-left: 10px;",
                    ), 
                    ui.column(3,
                        ui.input_selectize("tch_descriptive_stats", "Descriptive statistics:", ["mean", "median", "max", "min"], selected="mean", multiple=True),
                        style="margin-left: 30px;",
                    )
                )
            ),

            ui.accordion_panel(
                "Plot settings",

                ui.panel_conditional(
                    "input.tch_plot == 'Scatter'",

                    ui.accordion(
                        ui.accordion_panel(
                            "Central tendency",
                            ui.input_selectize("tch_scatter_central_tendency", label=None, choices=["Mean", "Median"], selected=["Median"]),
                        ),
                        ui.accordion_panel(
                            "Polynomial fit",
                            ui.row(
                                ui.column(3, ui.input_checkbox("tch_polynomial_fit", "Fit", True)),
                                ui.panel_conditional(
                                    "input.tch_polynomial_fit == true",
                                    ui.input_switch("tch_fit_best", "Automatic fit", True),
                                ),
                            ),
                            ui.panel_conditional(
                                "input.tch_polynomial_fit == true",
                                ui.panel_conditional(
                                    "input.tch_fit_best == false",
                                    ui.input_selectize("tch_force_fit", "Fit:", list(Modes.FitModel.keys())),
                                ),
                            ),
                        ),
                    ),
                ),

                ui.panel_conditional(
                    "input.tch_plot == 'Line'",
                    ui.input_selectize("tch_line_interpolation", "Interpolation:", choices=Modes.Interpolate),
                ),

                ui.panel_conditional(
                    "input.tch_plot == 'Error band'",
                    ui.input_selectize("tch_errorband_error", "Error:", Modes.ExtentError),
                    ui.input_selectize("tch_errorband_interpolation", "Interpolation:", Modes.Interpolate),
                ),
            ),

            ui.accordion_panel(
                "Aesthetics",

                ui.panel_conditional(
                    "input.tch_plot == 'Scatter'",

                    ui.accordion(
                        ui.accordion_panel(
                            "Coloring",
                            ui.input_selectize("tch_scatter_background", "Background:", Dyes.Background),
                            ui.input_selectize("tch_scatter_color_palette", "Color palette:", []),
                        ),

                        ui.accordion_panel(
                            "Elements",
                            # Bullet settings
                            ui.accordion(
                                ui.accordion_panel(
                                    "Bullets",
                                    ui.input_checkbox("tch_show_scatter", "Show scatter", True),
                                    ui.panel_conditional(
                                        "input.tch_show_scatter == true",
                                        ui.input_numeric("tch_bullet_opacity", "Opacity:", 0.5, min=0, max=1, step=0.1),
                                        ui.input_checkbox("tch_fill_bullets", "Fill bullets", True),
                                        ui.input_numeric("tch_bullet_size", "Bullet size:", 5, min=0, step=0.5),
                                        ui.panel_conditional(
                                            "input.tch_fill_bullets == true",
                                            ui.input_checkbox("tch_outline_bullets", "Outline bullets", False),
                                            ui.panel_conditional(
                                                "input.tch_outline_bullets == true",
                                                ui.input_selectize("tch_bullet_outline_color", "Outline color:", Dyes.Colors, selected="match"),
                                                ui.input_numeric("tch_bullet_outline_width", "Outline width:", 1, min=0, step=0.1),
                                            ),
                                        ),
                                        ui.panel_conditional(
                                            "input.tch_fill_bullets == false",
                                            ui.input_numeric("tch_bullet_stroke_width", "Stroke width:", 1, min=0, step=0.1),
                                        ),
                                    ),
                                ),
                            ),
                            # Line settings
                            ui.panel_conditional(
                                "input.tch_polynomial_fit == true",
                                ui.markdown("""  <br>  """),
                                ui.input_numeric("tch_scatter_line_width", "Line width:", 1, min=0, step=0.1),
                            ),
                        ),
                    ),
                ),

                ui.panel_conditional(
                    "input.tch_plot == 'Line'",
                    ui.accordion(
                        ui.accordion_panel(
                            "Coloring",
                            ui.input_selectize("tch_line_background", "Background:", Dyes.Background),
                            ui.input_selectize("tch_line_color_palette", "Color palette:", []),
                        ),
                        ui.accordion_panel(
                            "Elements",
                            # Line settings
                            ui.markdown("""  <p>  """),
                            ui.input_numeric("tch_line_line_width", "Line width:", 1, min=0, step=0.1),
                            # Bullets settings
                            ui.markdown("""  <br>  """),
                            ui.input_checkbox("tch_line_show_bullets", "Show bullets", False),
                            ui.panel_conditional(
                                "input.tch_line_show_bullets == true",
                                ui.input_numeric("tch_line_bullet_size", "Bullet size:", 1, min=0, step=0.5),
                            ),
                        ),
                    ),
                ),

                ui.panel_conditional(
                    "input.tch_plot == 'Error band'",
                    ui.accordion(
                        ui.accordion_panel(
                            "Coloring",
                            ui.input_selectize("tch_errorband_background", "Background:", Dyes.Background),
                            ui.input_selectize("tch_errorband_color_palette", "Color palette:", []),
                        ),
                    
                        ui.accordion_panel(
                            "Bands",
                            # Error band settings
                            ui.input_checkbox("tch_errorband_fill", "Fill area", True),
                            ui.panel_conditional(
                                "input.tch_errorband_fill == true",
                                ui.input_numeric("tch_errorband_fill_opacity", "Fill opacity:", 0.5, min=0, max=1, step=0.1),
                                ui.input_checkbox("tch_errorband_outline", "Outline area", False),
                            ),
                            ui.panel_conditional(
                                "input.tch_errorband_fill == true && input.tch_errorband_outline == true || input.tch_errorband_fill == false",
                                ui.input_numeric("tch_errorband_outline_width", "Outline width:", 1, min=0, step=0.1),
                                ui.input_selectize("tch_errorband_outline_color", "Outline color:", Dyes.Colors, selected="match"),
                            ),
                        ),
                        ui.accordion_panel(
                            "Lines",
                            ui.accordion(
                                ui.accordion_panel(
                                    "Mean",
                                    ui.input_checkbox("tch_errorband_show_mean", "Show mean", False),
                                    ui.panel_conditional(
                                        "input.tch_errorband_show_mean == true",
                                        ui.input_selectize("tch_errorband_mean_line_color", "Line color:", Dyes.Colors, selected="match"),
                                        ui.input_selectize("tch_errorband_mean_line_style", "Line style:", Dyes.LineStyle),
                                        ui.input_numeric("tch_errorband_mean_line_width", "Line width:", 1, min=0, step=0.1),
                                    ),
                                ),
                                ui.accordion_panel(
                                    "Median",
                                    ui.input_checkbox("tch_errorband_show_median", "Show median", False),
                                    ui.panel_conditional(
                                        "input.tch_errorband_show_median == true",
                                        ui.input_selectize("tch_errorband_median_line_color", "Line color:", Dyes.Colors, selected="match"),
                                        ui.input_selectize("tch_errorband_median_line_style", "Line style:", Dyes.LineStyle),
                                        ui.input_numeric("tch_errorband_median_line_width", "Line width:", 1, min=0, step=0.1),
                                    ),
                                ),
                                ui.accordion_panel(
                                    "Min",
                                    ui.input_checkbox("tch_errorband_show_min", "Show min", False),
                                    ui.panel_conditional(
                                        "input.tch_errorband_show_min == true",
                                        ui.input_selectize("tch_errorband_min_line_color", "Line color:", Dyes.Colors, selected="match"),
                                        ui.input_selectize("tch_errorband_min_line_style", "Line style:", Dyes.LineStyle),
                                        ui.input_numeric("tch_errorband_min_line_width", "Line width:", 1, min=0, step=0.1),
                                    )
                                ),
                                ui.accordion_panel(
                                    "Max",
                                    ui.input_checkbox("tch_errorband_show_max", "Show max", False),
                                    ui.panel_conditional(
                                        "input.tch_errorband_show_max == true",
                                        ui.input_selectize("tch_errorband_max_line_color", "Line color:", Dyes.Colors, selected="match"),
                                        ui.input_selectize("tch_errorband_max_line_style", "Line style:", Dyes.LineStyle),
                                        ui.input_numeric("tch_errorband_max_line_width", "Line width:", 1, min=0, step=0.1),
                                    )
                                )
                            )
                        )
                    )
                )
            ),
            class_="accordion02"
        ),
        ui.br(),
        ui.div(" 🏗️ ", style="font-size: 360px")
    )
)
