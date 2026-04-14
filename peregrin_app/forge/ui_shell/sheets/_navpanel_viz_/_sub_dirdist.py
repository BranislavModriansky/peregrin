from shiny import ui

from src.code import Metrics, Markers, Dyes, Modes



import warnings
from shiny._deprecated import ShinyDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message=r".*panel_well\(\) is deprecated\. Use shiny\.ui\.card\(\) instead\.",
    category=ShinyDeprecationWarning,
)



subpanel_dirdist = ui.nav_panel(
    "Directionality distribution",
    ui.panel_well(
        ui.markdown(
            """
            ### **Directionality distribution**
            ___
            """
        ),
        ui.accordion(
            ui.accordion_panel(
                "Data Categories & Settings",
                ui.row(
                    ui.column(5, ui.input_selectize(id="conditions_dd", label="Conditions:", choices=[], selected=[], multiple=True, options={"placeholder": "Select conditions"})),
                    ui.column(1, ui.input_action_button(id="conditions_reset_dd", label="🗘", class_="btn-noframe")),
                    ui.column(5, ui.input_selectize(id="replicates_dd", label="Replicates:", choices=[], selected=[], multiple=True, options={"placeholder": "Select replicates"})),
                    ui.column(1, ui.input_action_button(id="replicates_reset_dd", label="🗘", class_="btn-noframe")),
                ),
                ui.br(),
                ui.row(
                    ui.column(6, ui.input_radio_buttons(id="dd_normalization", label=None, choices={"globally": "Normalize globally", "locally": "Normalize to selected categories", "none": "No normalization"}, selected="globally")),
                    # ui.column(6, ui.input_selectize(id="dd_weight_by", label="Weights:", choices=[], selected=[], multiple=False)),
                )
            ),
            class_="accordion02"
        ),
    ),
    ui.br(),
    ui.layout_column_wrap(
        ui.card(
            ui.card_header("Rose Chart", class_="bg-secondary-css"),
            ui.panel_well(
                ui.accordion(
                    ui.accordion_panel(
                        "Compose",
                        ui.input_numeric("dd_rosechart_bins", "Number of bins:", value=24, min=2, step=1, width="125px"),
                        ui.row(
                            ui.input_selectize("dd_rosechart_alignment", "Align bins:", choices=["center", "edge"], selected="edge", width="115px"),
                            ui.input_numeric("dd_rosechart_gap", "Bin gaps:", value=0, min=0, max=10, step=0.1, width="100px"),
                        ),
                        ui.row(
                            ui.panel_conditional(
                                "input.dd_rosechart_line_r_labels == true",
                                ui.div(ui.markdown("Label Color:"), style="margin-left: 50px;"),
                                ui.div(ui.input_selectize("dd_rosechart_line_r_label_color", label=None, choices=Dyes.Colors, selected="#ffffff", width="175px"), style="margin-left: 10px; margin-top: -6px;"),
                                ui.div(ui.markdown("Position [°]:"), style="margin-left: 24px;"),
                                ui.div(ui.input_numeric("dd_rosechart_line_r_axis_position", label=None, value=75, min=0, max=360, step=1, width="75px"), style="margin-left: 10px; margin-top: -6px;"),
                            )
                        ),
                        ui.input_text("dd_rosechart_title", label=None, placeholder="Title me!", width="100%")
                    ),
                    ui.accordion_panel(
                        "Color",
                        ui.input_selectize(id="dd_rosechart_cmode", label="Color mode:", choices=["single color", "level-based", "n-tiles", "differentiate conditions", "differentiate replicates"], selected="n-tiles", width="215px"),
                        ui.panel_conditional(
                            "input.dd_rosechart_cmode == 'level-based'",
                            ui.input_numeric("dd_rosechart_levels", "Levels:", value=8, min=2, step=1, width="90px"),
                        ),
                        ui.panel_conditional(
                            "input.dd_rosechart_cmode == 'n-tiles'",
                            ui.row(
                                ui.input_numeric("dd_rosechart_ntiles", "n-tiles:", value=6, min=2, step=1, width="90px"),
                                ui.input_selectize("dd_rosechart_discretize", "Discretize:", choices=[], width="210px"),
                            )
                        ),
                        ui.panel_conditional(
                            "input.dd_rosechart_cmode == 'single color'",
                            ui.input_selectize("dd_rosechart_single_color", "Color:", Dyes.Colors, selected="#536267", width="150px"),
                        ),
                        ui.panel_conditional(
                            "['differentiate conditions', 'differentiate replicates'].includes(input.dd_rosechart_cmode)",
                            ui.input_checkbox("dd_rosechart_custom_colors", "Use custom colors", False),
                        ),
                        ui.panel_conditional(
                            "['level-based', 'n-tiles', 'differentiate conditions', 'differentiate replicates'].includes(input.dd_rosechart_cmode) && input.dd_rosechart_custom_colors == false",
                            ui.input_selectize("dd_rosechart_lut_map", "Select LUT map:", choices=Dyes.QuantitativeCModes, selected="plasma LUT", width="200px"),
                        ),
                        ui.row(
                            ui.column(4, ui.div(ui.input_checkbox("dd_rosechart_outline", "Outline bins", False), style="margin-left: 18px; margin-top: 12px;")),
                            ui.panel_conditional(
                                "input.dd_rosechart_outline == true",
                                ui.column(4, ui.input_selectize("dd_rosechart_outline_color", "Outline color:", Dyes.Colors, selected="#1a1a1a", width="210px")),
                                ui.column(4, ui.div(ui.input_numeric("dd_rosechart_outline_width", "Outline width:", value=2, min=0, step=0.5, width="120px"), style="margin-left: 18px;")),
                            )
                        ),
                        ui.column(4, ui.div(ui.input_selectize(id="dd_rosechart_bg_color", label="Background color:", choices=Dyes.Colors, selected="#3c4142", width="175px"), style="margin-left: -4px;"), offset=4),
                    ),
                    class_="accordion02"
                ),
                ui.br(),
                ui.input_task_button(id="generate_dd_rosechart", label="Generate", class_="btn-secondary task-btn", width="100%"),
            ),
            ui.br(),
            ui.output_plot("dd_plot_rosechart", height="500px"),
            ui.br(),
            ui.row(
                ui.column(6, ui.download_button("dd_rosechart_download_svg", "Download SVG", width="100%")),
                ui.column(6, ui.download_button("dd_rosechart_download_png", "Download PNG", width="100%"))
            )
        ),
        ui.card(
            ui.card_header("Gaussian KDE Colormesh", class_="bg-secondary-css"),
            ui.panel_well(
                ui.accordion(
                    ui.accordion_panel(
                        "Compose",
                        ui.row(
                            ui.input_numeric("dd_kde_colormesh_bins", "Number of bins:", value=720, min=2, step=1, width="150px"),
                            ui.input_numeric("dd_kde_colormesh_kappa", "Kappa:", value=25, min=0, step=1, width="115px"),
                        ),
                        ui.row(
                            ui.input_checkbox("dd_kde_colormesh_auto_scale_lut", "Auto scale LUT to min/max density", value=True),
                            ui.panel_conditional(
                                "input.dd_kde_colormesh_auto_scale_lut == true || input.dd_kde_colormesh_auto_scale_lut == false",
                                ui.div(ui.output_text_verbatim(id="dd_kde_colormesh_density_range", placeholder=True), style="margin-left: 20px; margin-right: 30px; margin-top: -6px;"),
                            ),
                        ),
                        ui.row(
                            ui.panel_conditional(
                                "input.dd_kde_colormesh_auto_scale_lut == false",
                                ui.div(ui.markdown("LUT map scale range:"), style="margin-left: 30px; margin-right: 10px; margin-top: 5px;"),
                                ui.div(ui.input_numeric("dd_kde_colormesh_lutmap_scale_min", None, 0, step=0.1, width="100px"), style="margin-left: 10px; margin-right: 10px;"),
                                ui.div(ui.input_numeric("dd_kde_colormesh_lutmap_scale_max", None, 1, step=0.1, width="100px"), style="margin-left: 10px; margin-right: 30px;")
                            )
                        ),
                        ui.input_checkbox(id="dd_kde_colormesh_theta_labels", label="Theta axis annotation", value=True),
                        ui.br(),
                        ui.input_text("dd_kde_colormesh_title", label=None, placeholder="Title me!", width="100%")
                    ),
                    ui.accordion_panel(
                        "Color",
                        ui.input_selectize(id="dd_kde_colormesh_lut_map", label="Select LUT map:", choices=Dyes.QuantitativeCModes, selected="plasma LUT", width="200px"),
                    ),
                    class_="accordion02"
                ),
                ui.br(),
                ui.input_task_button(id="generate_dd_kde_colormesh", label="Generate", class_="btn-secondary task-btn", width="100%"),
            ),
            ui.br(),
            ui.output_plot("dd_plot_kde_colormesh", height="500px"),
            ui.br(),
            ui.row(
                ui.column(6, ui.download_button("dd_kde_colormesh_download_svg", "Download SVG", width="100%")),
                ui.column(6, ui.download_button("dd_kde_colormesh_download_png", "Download PNG", width="100%"))
            )
        ),
        ui.card(
            ui.card_header("KDE Line Plot", class_="bg-secondary-css"),
            ui.panel_well(
                ui.accordion(
                    ui.accordion_panel(
                        "Compose",
                        ui.input_numeric(id="dd_kde_line_kappa", label="Kappa:", value=25, min=0, step=1, width="90px"),
                        ui.input_checkbox(id="dd_kde_line_mean", label="Display circular mean", value=True),
                        ui.input_checkbox(id="dd_kde_line_peak_direction_trend", label="Display peak dial", value=False),
                        ui.row(
                            ui.column(6,
                                ui.input_checkbox("dd_kde_line_theta_labels", "Theta axis annotation", value=True),
                                ui.input_checkbox("dd_kde_line_r_labels", "R axis annotation", value=True)
                            )
                        ),
                        ui.row(
                            ui.panel_conditional(
                                "input.dd_kde_line_r_labels == true",
                                ui.div(ui.markdown("Label Color:"), style="margin-left: 50px;"),
                                ui.div(ui.input_selectize("dd_kde_line_r_label_color", label=None, choices=Dyes.Colors, selected="#985e2b", width="175px"), style="margin-left: 10px; margin-top: -6px;"),
                                ui.div(ui.markdown("Position [°]:"), style="margin-left: 24px;"),
                                ui.div(ui.input_numeric("dd_kde_line_r_axis_position", label=None, value=75, min=0, max=360, step=1, width="75px"), style="margin-left: 10px; margin-top: -6px;"),
                            )
                        ),
                        ui.br(),
                        ui.input_text("dd_kde_line_title", label=None, placeholder="Title me!", width="100%")
                    ),
                    ui.accordion_panel(
                        "Color",
                        ui.row(
                            ui.column(4, ui.div(ui.input_checkbox(id="dd_kde_line_outline", label="Show line", value=True), style="margin-left: 18px; margin-top: 12px;")),
                            ui.panel_conditional(
                                "input.dd_kde_line_outline == true",
                                ui.column(4, ui.input_selectize(id="dd_kde_line_outline_color", label="Line color:", choices=Dyes.Colors, selected="#ffffff", width="175px")),
                                ui.column(4, ui.input_numeric(id="dd_kde_line_outline_width", label="Line width:", value=2, min=0.0, step=0.1, width="150px")),
                            ),
                        ),
                        ui.row(
                            ui.column(4, ui.div(ui.input_checkbox(id="dd_kde_line_fill", label="Fill KDE area", value=True), style="margin-left: 18px; margin-top: 12px;")),
                            ui.panel_conditional(
                                "input.dd_kde_line_fill == true",
                                ui.column(4, ui.input_selectize(id="dd_kde_line_fill_color", label="Fill color:", choices=Dyes.Colors, selected="#ffffff", width="175px")),
                                ui.column(4, ui.input_numeric(id="dd_kde_line_fill_alpha", label="Fill opacity:", value=0.3, min=0, max=1, step=0.1, width="150px"))
                            )
                        ),
                        ui.row(
                            ui.panel_conditional(
                                "input.dd_kde_line_mean == true",
                                ui.column(4, ui.input_selectize(id="dd_kde_line_mean_color", label="Mean dial color:", choices=Dyes.Colors, selected="#f7022a", width="175px"), offset=4),
                                ui.column(4, ui.input_numeric(id="dd_kde_line_mean_width", label="Mean dial width:", value=3, min=0.0, step=0.1, width="150px")),
                            )
                        ),
                        ui.row(
                            ui.panel_conditional(
                                "input.dd_kde_line_peak_direction_trend == true",
                                ui.column(4, ui.input_selectize(id="dd_kde_line_peak_direction_trend_color", label="Peak dial color:", choices=Dyes.Colors, selected="#1e488f", width="175px"), offset=4),
                                ui.column(4, ui.input_numeric(id="dd_kde_line_peak_direction_trend_width", label="Peak dial width:", value=3, min=0.0, step=0.1, width="150px")),
                            )
                        ),
                        ui.column(4, ui.div(ui.input_selectize(id="dd_kde_line_bg_color", label="Background color:", choices=Dyes.Colors, selected="#1e488f", width="175px"), style="margin-left: -4px;"), offset=4),
                    ),
                    class_="accordion02"
                ),
                ui.br(),
                ui.input_task_button(id="generate_dd_kde_line", label="Generate", class_="btn-secondary task-btn", width="100%"),
            ),
            ui.br(),
            ui.output_plot("dd_plot_kde_line", height="450px"),
            ui.br(),
            ui.row(
                ui.column(6, ui.download_button("dd_kde_line_download_svg", "Download SVG", width="100%")),
                ui.column(6, ui.download_button("dd_kde_line_download_png", "Download PNG", width="100%"))
            )
        ),
        width=1/2,
        class_="dd-cards-wrap"
    )
)
