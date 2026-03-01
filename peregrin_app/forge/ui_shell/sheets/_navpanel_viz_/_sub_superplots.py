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

        ui.input_selectize(id="superplot_type", label="Plot:", choices=["Swarms", "Violins"], selected="Swarms", width="110px"),
        # _ _ Swarmplot settings _ _
        # ui.panel_conditional(
        #     "input.superplot_type == 'Swarms'",
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
                ui.input_selectize(id="metric_sp", label=None, choices=Metrics.Track, selected="Straightness index", width="200px"),

                # TODO: ui.input_radio_buttons(id="axis_sp", label="Show categories on axis:", choices=["X", "Y"], selected="X", inline=True),
                ui.accordion(
                    ui.accordion_panel(
                        "Central tendency skeleton",
                        
                    ),
                    ui.accordion_panel(
                        "Color",
                        ui.input_checkbox(id="stock_palette_sp", label="Use stock color palette", value=False),
                        ui.panel_conditional(
                            "input.stock_palette_sp == true",
                            ui.input_selectize(id="palette_sp", label="Color palette:", choices=Dyes.PaletteQualitativeSeaborn + Dyes.PaletteQualitativeMatplotlib, selected=2),
                        )
                    ),
                    id="color_accordion_sp"
                ),
                ui.row(ui.input_text(id="sp_title", label=None, placeholder="Title me!", width="100%"), style="margin-left: 1px; margin-right: 1px;")
            ),
            ui.accordion_panel(
                "Elements & Specs",
                ui.accordion(
                    ui.accordion_panel(
                        "Swarms",
                        ui.input_checkbox(id="sp_show_swarms", label="Show swarms", value=True),
                        ui.panel_conditional(
                            "input.sp_show_swarms == true",
                            ui.input_numeric("sp_swarm_marker_size", "Dot size:", 1, min=0, step=0.5),
                            ui.input_numeric("sp_swarm_marker_alpha", "Dot opacity:", 0.5, min=0, max=1, step=0.1),
                            ui.input_selectize("sp_swarm_marker_outline", "Dot outline color:", Dyes.Colors, selected="black"),
                        ),
                    ),
                    ui.accordion_panel(
                        "Violins",
                        ui.input_checkbox(id="sp_show_violins", label="Show violins", value=True),
                        ui.panel_conditional(
                            "input.sp_show_violins == true",
                            ui.input_selectize("sp_violin_fill", "Fill color:", Dyes.Colors, selected="whitesmoke"),
                            ui.input_numeric("sp_violin_alpha", "Fill opacity:", 0.5, min=0, max=1, step=0.1),
                            ui.input_selectize("sp_violin_outline", "Outline color:", Dyes.Colors, selected="lightgrey"),
                            ui.input_numeric("sp_violin_outline_width", "Outline width:", 1, min=0, step=1),
                        ),
                    ),
                    ui.accordion_panel(
                        "KDE",
                        ui.input_checkbox(id="sp_show_kde", label="Show KDE", value=False),
                        ui.panel_conditional(
                            "input.sp_show_kde == true",
                            ui.input_numeric("sp_kde_line_width", "Outline width:", 1, min=0, step=0.1),
                            ui.input_checkbox("sp_kde_fill", "Fill area", False),
                            ui.panel_conditional(
                                "input.sp_kde_fill == true",
                                ui.input_numeric("sp_kde_fill_alpha", "Fill opacity:", 0.5, min=0, max=1, step=0.1),
                            ),
                            ui.input_numeric("sp_kde_bandwidth", "KDE bandwidth:", 0.75, min=0.1, step=0.1),
                        ),
                    ),
                    ui.accordion_panel(
                        "Skeleton",
                        ui.input_checkbox(id="sp_show_cond_mean", label="Show condition mean lines", value=False),
                        ui.input_checkbox(id="sp_show_cond_median", label="Show condition median lines", value=False),
                        ui.input_checkbox(id="sp_show_errbars", label="Show error bars", value=False),
                        ui.panel_conditional(
                            "input.sp_show_cond_mean == true && input.sp_show_cond_median == true",
                            ui.input_selectize("sp_set_as_primary", label="Set as primary:", choices=["mean", "median"], selected="mean"),
                        ),
                        ui.panel_conditional(
                            "input.sp_show_cond_mean == true",
                            ui.input_numeric(id="sp_mean_line_span", label="Mean line span length:", value=0.12, min=0, step=0.01),
                            ui.input_selectize(id="sp_mean_line_color", label="Mean line color:", choices=Dyes.Colors, selected="black"),
                        ),
                        ui.panel_conditional(
                            "input.sp_show_cond_median == true",
                            ui.input_numeric(id="sp_median_line_span", label="Median line span length:", value=0.08, min=0, step=0.01),
                            ui.input_selectize(id="sp_median_line_color", label="Median line color:", choices=Dyes.Colors, selected="darkblue"),
                        ),
                        ui.panel_conditional(
                            "input.sp_show_cond_mean == true || input.sp_show_cond_median == true",
                            ui.input_numeric(id="sp_lines_lw", label="Mean/Median Line width:", value=1, min=0, step=0.5),
                        ),
                        ui.panel_conditional(
                            "input.sp_show_errbars == true",
                            ui.input_numeric(id="sp_errorbar_capsize", label="Error bar cap size:", value=4, min=0, step=1),
                            ui.input_numeric(id="sp_errorbar_lw", label="Error bar line width:", value=1, min=0, step=0.5),
                            ui.input_selectize(id="sp_errorbar_color", label="Error bar color:", choices=Dyes.Colors, selected="black"),
                            ui.input_numeric(id="sp_errorbar_alpha", label="Error bar opacity:", value=1, min=0, max=1, step=0.1),
                        ),
                        ui.panel_conditional(
                            "input.sp_show_cond_means == false && input.sp_show_cond_medians == false && input.sp_show_errbars == false",
                            ui.markdown(
                                """
                                *Condition means/medians/error bars not enabled.*
                                """
                            )
                        ),
                    ),
                    ui.accordion_panel(
                        "Bullets",
                        ui.input_checkbox(id="sp_show_rep_means", label="Show replicate mean bullets", value=False),
                        ui.input_checkbox(id="sp_show_rep_medians", label="Show replicate median bullets", value=True),
                        ui.panel_conditional(
                            "input.sp_show_rep_means == true",
                            ui.input_numeric("sp_mean_bullet_size", "Mean bullet size:", 80, min=0, step=1),
                            ui.input_selectize("sp_mean_bullet_outline", "Mean bullet outline color:", Dyes.Colors, selected="black"),
                            ui.input_numeric("sp_mean_bullet_outline_width", "Mean bullet outline width:", 0.75, min=0, step=0.05),
                            ui.input_numeric("sp_mean_bullet_alpha", "Mean bullet opacity:", 1, min=0, max=1, step=0.1),
                        ),
                        ui.panel_conditional(
                            "input.sp_show_rep_medians == true",
                            ui.input_numeric("sp_median_bullet_size", "Median bullet size:", 50, min=0, step=1),
                            ui.input_selectize("sp_median_bullet_outline", "Median bullet outline color:", Dyes.Colors, selected="black"),
                            ui.input_numeric("sp_median_bullet_outline_width", "Median bullet outline width:", 0.75, min=0, step=0.05),
                            ui.input_numeric("sp_median_bullet_alpha", "Median bullet opacity:", 1, min=0, max=1, step=0.1),
                        ),
                    ),
                    ui.accordion_panel(
                        "Plot specs",
                        
                        ui.row(
                            ui.column(3,
                                ui.input_checkbox(id="sp_show_legend", label="Show legend", value=True),
                                ui.input_checkbox(id="sp_grid", label="Show grid", value=False),
                                ui.input_checkbox(id="sp_spine", label="Open axes top/right", value=True),
                            ),
                            ui.input_numeric(id="sp_fig_width", label="Fig width:", value=10, min=1, step=0.5),
                            ui.input_numeric(id="sp_fig_height", label="Fig height:", value=7, min=1, step=0.5)
                        ),
                    )
                ),


                
                
                # TODO: ui.input_checkbox(id="sp_flip", label="Flip axes", value=False),
                
            ),
            class_="accordion02"
        ),
        ui.markdown(""" <br> """),
        # ),

        # _ _ Violinplot settings _ _
        ui.panel_conditional(
            "input.superplot_type == 'Violins'",
            ui.accordion(
                ui.accordion_panel(
                    "Metric",
                    ui.input_selectize("vp_metric", label=None, choices=Metrics.Track, selected="Straightness index"),
                    # TODO: ui.input_radio_buttons("vp_y_axis", "Y axis with", ["Absolute values", "Relative values"]),
                ),
                ui.accordion_panel(
                    "General",
                    ui.input_radio_buttons(id="vp_replicate_bullets", label="Replicate bullets", choices=["Mean", "Median"], selected="Median"),
                    ui.input_radio_buttons(id="vp_skeleton_centre", label="Skeleton centre value", choices=["Mean", "Median"], selected="Median"),
                    ui.input_radio_buttons(id="vp_errorbars_method", label="Error bars method", choices=["SEM", "SD", "95% CI"], selected="SEM"),
                    ui.input_checkbox(id="vp_show_legend", label="Show legend", value=True),
                    ui.row(
                        ui.column(
                            6,
                            ui.input_numeric(id="vp_fig_width", label="Fig width:", value=12, min=1, step=0.5)
                        ),
                        ui.column(
                            6,
                            ui.input_numeric(id="vp_fig_height", label="Fig height:", value=5, min=1, step=0.5)
                        ),
                    ),
                ),
                ui.accordion_panel(
                    "Aesthetics",
                    ui.input_checkbox(id="vp_use_stock_palette", label="Use stock color palette", value=False),
                    ui.input_selectize(id="vp_palette", label="Color palette:", choices=Dyes.PaletteQualitativeSeaborn, selected="Accent"),
                    ui.accordion(
                        ui.accordion_panel(
                            "Violins",
                            ui.input_numeric(id="vp_violin_bandwidth", label="Violin band width:", value=0.8, min=0, step=0.1),
                            ui.input_numeric(id="vp_violin_outline_width", label="Total-violin outline width:", value=1, min=0, step=0.1),
                            ui.input_numeric(id="vp_subviolin_outline_width", label="Sub-violin outline width:", value=0, min=0, step=0.1),
                        ),
                        ui.accordion_panel(
                            "Replicate bullets",
                            ui.input_numeric(id="vp_bullet_size", label="Bullet size:", value=5, min=0, step=1),
                            ui.input_numeric(id="vp_bullet_outline_linewidth", label="Bullet outline width:", value=0.8, min=0, step=0.1),
                        ),
                        ui.accordion_panel(
                            "Error bars & skeleton",
                            ui.input_numeric(id="vp_errorbar_linewidth", label="Line width:", value=1, min=0, step=0.1),
                        )
                    )
                ),
                class_="accordion02"
            ),
            ui.markdown(""" <br> """),
            ui.row(ui.input_text(id="vp_title", label=None, placeholder="Title me!", width="100%"), style="margin-left: 1px; margin-right: 1px;"),
        ),
        
        ui.markdown(""" <br> """),
        ui.panel_conditional(
            "input.superplot_type == 'Swarms'",
            ui.input_task_button(id="sp_generate", label="Generate", class_="btn-secondary task-btn", width="100%")
        ),
        ui.panel_conditional(
            "input.superplot_type == 'Violins'",
            ui.input_task_button(id="vp_generate", label="Generate", class_="btn-secondary task-btn", width="100%")
        ),
        ui.br(),
        ui.div(" 🏗️ ", style="font-size: 360px"),
    ),
    # ui.markdown(""" <br> """),
    # ui.panel_conditional(
    #     "input.superplot_type == 'Swarms'",
    #     ui.output_ui("sp_plot_card"),
    #     ui.download_button(id="sp_download_svg", label="Download SVG", width="100%"),
    # ),
    # ui.panel_conditional(
    #     "input.superplot_type == 'Violins'",
    #     ui.output_ui("vp_plot_card"),
    #     ui.download_button(id="vp_download_svg", label="Download SVG", width="100%"),
    # )
)
