from shiny import ui

from src.code import Metrics, Markers, Dyes, Modes




import warnings
from shiny._deprecated import ShinyDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message=r".*panel_well\(\) is deprecated\. Use shiny\.ui\.card\(\) instead\.",
    category=ShinyDeprecationWarning,
)




subpanel_superplots = ui.nav_panel(
    "Superplots",

    ui.panel_well(
        ui.markdown(
            """
            ### **Superplots**
            ___
            """
        ),

        ui.input_selectize(id="superplot_type", label="Plot:", choices=["Hybrid Superplots", "Superviolins"], selected="Hybrid Superplots", width="200px"),
         
        ui.accordion(
            ui.accordion_panel(
                "Data Categories",
                ui.row(
                    ui.column(6, ui.input_selectize(id="conditions_sp", label="Conditions:", choices=[], selected=[], multiple=True, options={"placeholder": "Select conditions"})),
                    ui.column(1, ui.input_action_button(id="conditions_reset_sp", label="🗘", class_="btn-noframe")),
                ),
                ui.row(
                    ui.column(4, ui.input_selectize(id="replicates_sp", label="Replicates:", choices=[], selected=[], multiple=True, options={"placeholder": "Select replicates"})),
                    ui.column(1, ui.input_action_button(id="replicates_reset_sp", label="🗘", class_="btn-noframe")),
                )
            ),
            ui.accordion_panel(
                "Compose",
                ui.row(
                    ui.column(2, ui.input_selectize(id="metric_sp", label=None, choices=[], width="200px")),
                    ui.column(2, ui.input_radio_buttons(id="sp_orientation", label=None, choices={"v": "Vertical orientation", "h": "Horizontal orientation"}, selected="v")),
                    ui.panel_conditional(
                        "input.superplot_type == 'Hybrid Superplots'",
                        ui.column(2, ui.div("Density normalization method:", style="margin-top: 5px; margin-left: 5px;")),
                        ui.column(2, ui.input_selectize(id="sp_density_norm", label=None, choices={"area": "Area", "count": "Count"}, selected="area", width="90px")),
                    ),
                ),
                ui.br(),

                ui.accordion(
                    ui.accordion_panel(
                        "Central tendency skeleton",
                        ui.row(
                            ui.column(2, ui.input_checkbox(id="sp_cond_mean", label="Show condition mean", value=True)),
                            ui.panel_conditional(
                                "input.sp_cond_mean == true",
                                ui.column(2, ui.input_selectize(id="sp_cond_mean_ls", label="Line style:", choices={'-': 'Solid', '--': 'Dashed', '-.': 'Dash-dot', ':': 'Dotted'}, selected='-', width="115px")),
                            )
                        ),
                        ui.row(
                            ui.column(2, ui.input_checkbox(id="sp_cond_median", label="Show condition median", value=True)),
                            ui.panel_conditional(
                                "input.sp_cond_median == true",
                                ui.column(2, ui.input_selectize(id="sp_cond_median_ls", label="Line style:", choices={'-': 'Solid', '--': 'Dashed', '-.': 'Dash-dot', ':': 'Dotted'}, selected='--', width="115px")),
                            )
                        ),
                        ui.row(
                            ui.column(2, ui.input_checkbox(id="sp_error", label="Show error", value=True)),
                            ui.panel_conditional(
                                "input.sp_error == true",
                                ui.column(1, ui.input_selectize(id="sp_error_type", label="Method:", choices=["sd", "sem", "ci"], selected="sd", width="75px")),
                                ui.panel_conditional(
                                    "input.sp_error_type == 'ci'",
                                    ui.column(1, ui.input_selectize(id="sp_error_statistic", label="CI Statistic:", choices=["mean", "median"], selected="mean", width="105px")),
                                    ui.column(2, ui.div(ui.input_numeric(id="sp_error_ci_level", label="Confidence level:", value=95, min=50, max=99.9, step=0.1, width="150px"), style="margin-left: 30px;")),
                                    ui.column(2, ui.input_numeric(id="sp_error_bootstraps", label="Bootstrap resamples:", value=1000, min=100, step=100, width="175px"))
                                )
                            )
                        ),
                    ),
                    
                    ui.accordion_panel(
                        "Replicate markers",
                        ui.panel_conditional(
                            "input.superplot_type == 'Hybrid Superplots'",
                            ui.row(
                                ui.column(2, ui.input_checkbox(id="sp_rep_means", label="Show replicate means", value=False)),
                                ui.panel_conditional(
                                    "input.sp_rep_means == true",
                                    ui.column(2, ui.input_numeric(id="sp_rep_mean_marker_size", label="Marker size:", value=80, min=0, step=1, width="115px")),
                                    ui.column(2, ui.input_numeric(id="sp_rep_mean_marker_alpha", label="Marker opacity:", value=1, min=0, max=1, step=0.1, width="125px")),
                                    ui.column(2, ui.input_numeric(id="sp_rep_mean_marker_outline_lw", label="Marker outline width:", value=0.75, min=0, step=0.05, width="175px")),
                                    ui.column(2, ui.input_checkbox(id="sp_rep_mean_marker_join", label="Join consecutive", value=True)),
                                    ui.panel_conditional(
                                        "input.sp_rep_mean_marker_join == true",
                                        ui.column(2, ui.input_selectize(id="sp_rep_mean_join_ls", label="Join line style", choices={'-': 'Solid', '--': 'Dashed', '-.': 'Dash-dot', ':': 'Dotted'}, selected='-', width="115px")),
                                    )
                                )
                            ),
                            ui.row(
                                ui.column(2, ui.input_checkbox(id="sp_rep_medians", label="Show replicate medians", value=True)),
                                ui.panel_conditional(
                                    "input.sp_rep_medians == true",
                                    ui.column(2, ui.input_numeric(id="sp_rep_median_marker_size", label="Marker size:", value=50, min=0, step=1, width="115px")),
                                    ui.column(2, ui.input_numeric(id="sp_rep_median_marker_alpha", label="Marker opacity:", value=1, min=0, max=1, step=0.1, width="125px")),
                                    ui.column(2, ui.input_numeric(id="sp_rep_median_marker_outline_lw", label="Marker outline width:", value=0.75, min=0, step=0.05, width="175px")),
                                    ui.column(2, ui.input_checkbox(id="sp_rep_median_marker_join", label="Join consecutive", value=True)),
                                    ui.panel_conditional(
                                        "input.sp_rep_median_marker_join == true",
                                        ui.column(2, ui.input_selectize(id="sp_rep_median_join_ls", label="Join line style", choices={'-': 'Solid', '--': 'Dashed', '-.': 'Dash-dot', ':': 'Dotted'}, selected='--', width="115px")),
                                    )
                                )
                            ),
                        ),
                        ui.panel_conditional(
                            "input.superplot_type == 'Superviolins'",
                            ui.row(
                                ui.column(2, ui.input_checkbox(id="sp_show_rep_markers", label="Show replicate markers", value=True)),
                                ui.panel_conditional(
                                    "input.sp_show_rep_markers == true",
                                    ui.input_selectize(id="sp_rep_center", label="Replicate markers:", choices={"mean": "Replicate means", "median": "Replicate medians"}, selected="median", width="180px"),
                                )
                            )
                        )
                    ),
                    
                    ui.accordion_panel(
                        "Violins",
                        ui.panel_conditional(
                            "input.superplot_type == 'Superviolins'",
                            ui.row(
                                ui.column(2, ui.input_checkbox(id="sp_violin_outline_1", label="Outline violins", value=True)),
                                ui.panel_conditional(
                                    "input.sp_violin_outline_1 == true",
                                    ui.column(2, ui.input_numeric(id="sp_violin_outline_lw_1", label="Violin outline width:", value=1, min=0, step=0.1, width="175px")),
                                ),
                                ui.column(2, ui.input_checkbox(id="sp_subviolins_outline_1", label="Outline sub-violins", value=False)),
                                ui.panel_conditional(
                                    "input.sp_subviolins_outline_1 == true",
                                    ui.column(2, ui.input_numeric(id="sp_subviolins_outline_lw_1", label="Sub-violin outline width:", value=0.5, min=0, step=0.1, width="195px")),
                                )
                            )
                        ),
                        ui.panel_conditional(
                            "input.superplot_type == 'Hybrid Superplots'",
                            ui.row(
                                ui.column(2, ui.input_checkbox(id="sp_show_violins_2", label="Show violins", value=True)),
                                ui.panel_conditional(
                                    "input.sp_show_violins_2 == true",
                                    ui.column(2, ui.input_checkbox(id="sp_violin_outline_2", label="Outline violins", value=True)),
                                    ui.panel_conditional(
                                        "input.sp_violin_outline_2 == true",
                                        ui.column(2, ui.input_numeric(id="sp_violin_outline_lw_2", label="Violin outline width:", value=1, min=0, step=0.1, width="175px")),
                                        ui.column(2, ui.input_selectize(id="sp_violin_outline_color_2", label="Outline color:", choices=Dyes.Colors, selected="#5170d7", width="175px")),
                                    ),
                                    ui.column(2, ui.input_selectize(id="sp_violin_fill_color_2", label="Violin fill color:", choices=Dyes.Colors, selected="#ffffff", width="175px")),
                                    ui.column(2, ui.input_numeric(id="sp_violin_fill_alpha_2", label="Violin opacity:", value=0.5, min=0, max=1, step=0.1, width="175px")),
                                )
                            )
                        )
                    ),

                    ui.accordion_panel(
                        "Scatter",
                        ui.panel_conditional(
                            "input.superplot_type == 'Superviolins'",
                            ui.markdown("*No scatter is plotted when plotting Superviolins*")
                        ),
                        ui.panel_conditional(
                            "input.superplot_type == 'Hybrid Superplots'",
                            ui.row(
                                ui.column(2, ui.input_checkbox(id="sp_show_scatter", label="Show scatter", value=True)),
                                ui.panel_conditional(
                                    "input.sp_show_scatter == true",
                                    ui.column(2, ui.input_selectize(id="sp_scatter_type", label="Scatter method:", choices={"sina": "Sinaplot", "swarm": "Swarmplot"}, selected="sina", width="135px")),
                                    ui.column(2, ui.input_numeric(id="sp_scatter_marker_size", label="Marker size:", value=5, min=0, step=1, width="115px")),
                                    ui.column(2, ui.input_numeric(id="sp_scatter_marker_alpha", label="Marker opacity:", value=0.8, min=0, max=1, step=0.1, width="125px")),
                                    ui.column(2, ui.input_numeric(id="sp_scatter_marker_outline_lw", label="Marker outline width:", value=0, min=0, step=0.5, width="175px")),
                                )
                            )
                        )
                    ),

                    ui.accordion_panel(
                        "KDE",
                        ui.panel_conditional(
                            "input.superplot_type == 'Superviolins'",
                            ui.markdown("*These setting do not apply to Superviolins*")
                        ),
                        ui.panel_conditional(
                            "input.superplot_type == 'Hybrid Superplots'",
                            ui.row(
                                ui.column(2, ui.input_checkbox(id="sp_show_kde", label="Show KDE", value=False)),
                                ui.panel_conditional(
                                    "input.sp_show_kde == true",
                                    ui.column(2, ui.input_checkbox(id="sp_kde_outline", label="KDE outline", value=False)),
                                    ui.panel_conditional(
                                        "input.sp_kde_outline == true",
                                        ui.column(2, ui.input_numeric(id="sp_kde_outline_lw", label="KDE line width:", value=1, min=0, step=0.1, width="150px"))
                                    ),
                                    ui.column(2, ui.input_checkbox(id="sp_kde_fill", label="Fill KDE area", value=True)),
                                    ui.panel_conditional(
                                        "input.sp_kde_fill == true",
                                        ui.column(2, ui.input_numeric(id="sp_kde_fill_alpha", label="KDE fill opacity:", value=0.5, min=0, max=1, step=0.1, width="175px")),
                                    )
                                )
                            )
                        )
                    ),

                    ui.accordion_panel(
                        "Color",
                        ui.row(
                            ui.column(2, ui.input_checkbox(id="sp_monochrome", label="Monochrome", value=False)),
                            ui.panel_conditional(
                                "input.sp_monochrome == false",
                                ui.column(2, ui.input_checkbox(id="stock_palette_sp", label="Use stock color palette", value=True)),
                                ui.panel_conditional(
                                    "input.stock_palette_sp == true",
                                    ui.column(
                                        2,
                                        ui.input_selectize(
                                            id="palette_sp",
                                            label="Color palette:",
                                            choices=Dyes.PaletteQualitativeMatplotlib,
                                            selected="tab10",
                                        ),
                                    ),
                                )
                            )
                        )
                    ),

                    ui.accordion_panel(
                        "Size",
                        ui.row(
                            ui.input_numeric(id="sp_fig_width", label="Width [in]:", value=12, min=1, step=0.5, width="150px"),
                            ui.input_numeric(id="sp_fig_height", label="Height [in]:", value=6, min=1, step=0.5, width="150px")
                        )
                    )
                ),
                ui.br(),
                ui.row(ui.input_text(id="sp_title", label=None, placeholder="Title:", width="100%"), style="margin-left: 1px; margin-right: 1px;")
            ),
            
            class_="accordion02"
        ),
        ui.br(),

        ui.panel_conditional(
            "input.superplot_type == 'Hybrid Superplots'",
            ui.input_task_button(id="sp_generate", label="Generate", class_="btn-secondary task-btn", width="100%")
        ),
        ui.panel_conditional(
            "input.superplot_type == 'Superviolins'",
            ui.input_task_button(id="vp_generate", label="Generate", class_="btn-secondary task-btn", width="100%")
        ),
    ),
    ui.br(),
    ui.panel_conditional(
        "input.superplot_type == 'Hybrid Superplots'",
        ui.output_ui("sp_plot_card"),
        ui.row(
            ui.column(6, ui.download_button(id="sp_download_svg", label="Download SVG", width="100%")),
            ui.column(6, ui.download_button(id="sp_download_png", label="Download PNG", width="100%"))
        )
    ),
    ui.panel_conditional(
        "input.superplot_type == 'Superviolins'",
        ui.output_ui("vp_plot_card"),
        ui.row(
            ui.column(6, ui.download_button(id="vp_download_svg", label="Download SVG", width="100%")),
            ui.column(6, ui.download_button(id="vp_download_png", label="Download PNG", width="100%"))
        )
    )
)
