
import warnings
import shiny.ui as ui
from shiny._deprecated import ShinyDeprecationWarning

from utils import Customize, Metrics, Markers, Styles, Modes


warnings.filterwarnings(
    "ignore",
    message=r".*panel_well\(\) is deprecated\. Use shiny\.ui\.card\(\) instead\.",
    category=ShinyDeprecationWarning,
)


# _ _ _ _  UI DESIGN DEFINITION  _ _ _ _
app_ui = ui.page_sidebar(

    # _ _ _ SIDEBAR - DATA FILTERING _ _ _
    ui.sidebar(
        ui.tags.style(Customize.Accordion),
        ui.markdown("""  <p>  """),
        ui.output_ui(id="sidebar_label"),
        ui.input_action_button(id="append_threshold", label="Add threshold", class_="btn-primary", width="100%", disabled=True),
        ui.input_action_button(id="remove_threshold", label="Remove threshold", class_="btn-primary", width="100%", disabled=True),
        ui.output_ui(id="sidebar_accordion_placeholder"),
        ui.input_task_button(id="set_threshold", label="Set threshold", label_busy="Applying...", type="secondary", disabled=True),
        ui.markdown("<p style='line-height:0.1;'> <br> </p>"),
        ui.output_ui(id="threshold_info"),
        ui.download_button(id="download_threshold_info", label="Info SVG", width="100%", _class="space-x-2"),
        id="sidebar", open="closed", position="right", bg="f8f8f8",
    ),

    # _ _ _ PANEL NAVIGATION BAR _ _ _
    ui.navset_bar(

        # _ _ _ _ RAW DATA INPUT PANEL - APP INITIALIZATION _ _ _ _
        ui.nav_panel(
            "Input Menu",
            ui.div(
                {"id": "data-inputs"},

                # _ Buttons & input UIs _
                ui.input_action_button("add_input", "Add data input", class_="btn-primary"),
                ui.input_action_button("remove_input", "Remove data input", class_="btn-primary", disabled=True),
                ui.input_action_button("run", label="Run", class_="btn-secondary", disabled=True),
                # TODO - ui.input_action_button("reset", "Reset", class_="btn-danger"),
                # TODO - ui.input_action_button("input_help", "Show help"),
                ui.output_ui("initialize_loader1"),
                ui.markdown(" <br> "),

                # _ Label settings (secondary sidebar) _
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.div(
                            ui.markdown(""" <br><h4><b>  Label settings:  </h4></b> """), 
                            style="display: flex; flex-direction: column; justify-content: center; height: 100%; text-align: center;"
                        ),
                        ui.div(  
                            ui.tags.style(Customize.Link1),
                            ui.input_action_link("explain_auto_label", "What's Auto-label?", class_="plain-link"),
                            ui.input_checkbox("auto_label", "Auto-label", False),

                            ui.panel_conditional(
                                "input.run > 0",
                                ui.input_switch("write_replicate_labels", "Write replicate labels", False)
                            ),
                            ui.panel_conditional(
                                "input.write_replicate_labels == true",
                                ui.output_ui("replicate_labels_inputs")
                            ),

                            ui.panel_conditional(
                                "input.run > 0",
                                ui.input_switch("write_replicate_colors", "Set replicate colors", False)
                            ),
                            ui.panel_conditional(
                                "input.write_replicate_colors == true && input.run > 0",
                                ui.output_ui("replicate_colors_inputs")
                            ),

                            ui.panel_conditional(
                                "input.run > 0",
                                ui.input_switch("set_condition_order", "Set condition order", False)
                            ),
                            ui.panel_conditional(
                                "input.set_condition_order == true",
                                ui.tags.style(Customize.Ladder),
                                ui.output_ui("condition_order_ladder")
                            )
                        ), 
                        ui.input_task_button("write_values", label="Write", label_busy="Writing...", class_="btn-secondary", width="100%"), 
                        bg="#f8f8f8",
                    ), 
                    # File inputs
                    ui.div(
                        {"id": "input_file_container_1"},
                        ui.input_text(id=f"condition_label1", label=f"Label:", placeholder="Condition 1"),
                        ui.input_file(id=f"input_file1", label="Upload files:", placeholder="Drag and drop here!", multiple=True),
                        ui.markdown(""" <hr style="border: none; border-top: 1px dotted" /> """),
                    ), border=True, border_color="whitesmoke", bg="#fefefe",
                ),

                # _ Draggable accordion panel - columns selection _
                ui.tags.style(Customize.AccordionDraggable),
                ui.panel_absolute(
                    ui.card(
                        ui.accordion(
                            ui.accordion_panel(
                                "Select columns",
                                ui.input_selectize("select_id", "Track identifier:", ["e.g. TRACK_ID"]),
                                ui.input_selectize("select_t", "Time point:", ["e.g. POSITION_T"]),
                                ui.row(
                                    ui.column(6,
                                        ui.input_selectize(id="select_t_unit", label=None, choices=list(Metrics.Units.TimeUnits.keys()), selected="seconds"),
                                        style_="margin-bottom: 5px;",
                                    )
                                ),
                                ui.input_selectize("select_x", "X coordinate:", ["e.g. POSITION_X"]),
                                ui.input_selectize("select_y", "Y coordinate:", ["e.g. POSITION_Y"]),
                                ui.markdown("<span style='color:darkgrey; font-style:italic;'>You can drag me around!</span>"),
                            ), 
                            open=False, class_="custom-accordion"
                        )
                    ), 
                    width="350px", right="450px", top="130px", draggable=True, 
                    class_="elevated-panel", style_="z-index: 1000;"
                )
            )
        ),

        # _ _ _ _ PROCESSED DATA DISPLAY _ _ _ _
        ui.nav_panel(
            "Data Tables",

            # _ Input for already processed data _
            ui.row(
                ui.column(6, 
                    ui.markdown(
                        """ 
                        <p style='line-height:0.1;'> <br> </p>
                        <h4 style='margin: 0.5;'> 
                            Got previously processed data? </h4> 
                        <p style='color: #0171b7;'><i> 
                            Drop in <b>Spot Stats CSV</b> file here: </i></p>
                        """
                    ),
                    ui.input_file(id="already_processed_input", label=None, placeholder="Drag and drop here!", accept=[".csv"], multiple=False), 
                    offset=1
                )
            ), 
            ui.markdown(""" ___ """),

            # _ Data display _
            ui.layout_columns(
                ui.card(
                    ui.card_header("Spot stats", class_="bg-blue"),
                    ui.output_data_frame("render_spot_stats"),
                    ui.download_button("download_spot_stats", "Download CSV"),
                    full_screen=True, 
                ),
                ui.card(
                    ui.card_header("Track stats", class_="bg-light"),
                    ui.output_data_frame("render_track_stats"),
                    ui.download_button("download_track_stats", "Download CSV"),
                    full_screen=True, 
                ),
                ui.card(
                    ui.card_header("Frame stats", class_="bg-light"),
                    ui.output_data_frame("render_frame_stats"),
                    ui.download_button("download_frame_stats", "Download CSV"),
                    full_screen=True, 
                ),
            ),
            ui.output_ui("initialize_loader2"),
        ),
        
        # TODO - _ _ _ _ GATING _ _ _ _
        ui.nav_panel(
            "Gating Panel",
            ui.layout_sidebar(
                ui.sidebar(
                    "Gating_sidebar", 
                    ui.input_checkbox("gating_params_inputs", "Inputs for gating params here", True),
                    bg="#f8f8f8",
                ), 
                ui.markdown(
                    """ 
                    Gates here
                    """
                )
            ),
            
        ),

        # _ _ _ _ VISUALIZATION PANEL _ _ _ _
        ui.nav_panel(
            "Visualisation",
            ui.navset_pill_list(

                # _ _ _ TRACKS VISUALIZATION _ _ _
                ui.nav_panel(
                    "Tracks",

                    ui.panel_well( 
                        ui.markdown(   # TODO - annotate which libraries were used
                            """
                            #### **Track visualization**
                            *used libraries:*  `matplotlib`,..
                            <hr style="height: 4px; background-color: black; border: none" />
                            """
                        ),

                        # _ _ SETTINGS _ _
                        ui.input_selectize(id="track_reconstruction_method", label="Select reconstruction method:", choices=["Realistic", "Normalized"], selected="Realistic"),

                        ui.accordion(
                            ui.accordion_panel(
                                "Select condition/replicate",
                                ui.input_selectize("tracks_conditions", "Condition:", []),
                                ui.input_selectize("tracks_replicates", "Replicate:", ["all"]),
                            
                            ),
                            ui.accordion_panel(
                                "Tracks",
                                ui.input_numeric("tracks_smoothing_index", "Smoothing index:", 0),
                                ui.input_numeric("tracks_line_width", "Line width:", 0.85),
                            ),
                            ui.accordion_panel(
                                "Track heads",
                                ui.input_checkbox("tracks_mark_heads", "Mark track heads", True),
                                ui.panel_conditional(
                                    "input.tracks_mark_heads",
                                    ui.input_selectize("tracks_marker_type", "Marker:", list(Markers.TrackHeads.keys()), selected="circle-empty"),
                                    ui.input_numeric("tracks_marks_size", "Marker size:", 3, min=0),
                                ),
                            ),
                            ui.accordion_panel(
                                "Aesthetics",
                                ui.input_selectize("tracks_color_mode", "Color mode:", Styles.ColorMode),
                                ui.panel_conditional(
                                    "input.tracks_color_mode != 'random greys' && input.tracks_color_mode != 'random colors' && input.tracks_color_mode != 'only-one-color' && input.tracks_color_mode != 'differentiate replicates'",
                                    ui.input_selectize("tracks_lut_scaling_metric", "LUT scaling metric:", Metrics.Lut),
                                ),
                                ui.panel_conditional(
                                    "input.tracks_color_mode == 'only-one-color'",
                                    ui.input_selectize("tracks_only_one_color", "Color:", Styles.Color),
                                ),
                                ui.input_selectize("tracks_background", "Background:", Styles.Background),
                                ui.input_checkbox("tracks_show_grid", "Show grid", True),
                                ui.panel_conditional(
                                    "input.tracks_show_grid",
                                    ui.input_selectize("tracks_grid_style", "Grid style:", ["simple-1", "simple-2", "spindle", "radial", "dartboard-1", "dartboard-2"]),
                                ),
                                ui.panel_conditional(
                                    "input.tracks_color_mode != 'random greys' && input.tracks_color_mode != 'random colors' && input.tracks_color_mode != 'only-one-color' && input.tracks_color_mode != 'differentiate replicates'",
                                    ui.input_checkbox(id="tracks_lutmap_extend_edges", label="Sharp LUT scale edges", value=True)
                                ),
                            ),
                        ),

                        # _ Title and generate plot _
                        ui.markdown(""" <br> """),
                        ui.input_text(id="tracks_title", label=None, placeholder="Title me!"),
                        ui.panel_conditional(
                            "input.track_reconstruction_method == 'Realistic'",
                            ui.input_task_button(id="trr_generate", label="Generate", class_="btn-secondary", width="100%"),
                        ),
                        ui.panel_conditional(
                            "input.track_reconstruction_method == 'Normalized'",
                            ui.input_task_button(id="tnr_generate", label="Generate", class_="btn-secondary", width="100%"),
                        )
                    ),

                    # _ _ PLOT DISPLAYS AND DOWNLOADS _ _
                    ui.markdown(""" <br> """),
                    ui.panel_conditional(
                        "input.track_reconstruction_method == 'Realistic'",
                        ui.card(
                            ui.output_plot("track_reconstruction_realistic"),
                            full_screen=False,
                            height="800px",
                        )
                    ),
                    ui.panel_conditional(
                        "input.track_reconstruction_method == 'Normalized'",
                        ui.card(
                            ui.output_plot("track_reconstruction_normalized"),
                            full_screen=False,
                            height="800px",
                        )
                    ),
                    ui.panel_conditional(
                        "input.track_reconstruction_method == 'Realistic'",
                        ui.download_button("trr_download", "Download", width="100%"),
                    ),
                    ui.panel_conditional(
                        "input.track_reconstruction_method == 'Normalized'",
                        ui.download_button("tnr_download", "Download", width="100%"),
                    ),
                    ui.panel_conditional(
                        "input.tracks_color_mode != 'random greys' && input.tracks_color_mode != 'random colors' && input.tracks_color_mode != 'only-one-color' && input.tracks_color_mode != 'differentiate replicates'",
                        ui.markdown(""" <p></p> """),
                        ui.download_button(id="download_lut_map_svg", label="Download LUT Map SVG", width="100%"),
                    ),
                ),


                ui.nav_panel(
                    "Time charts",
                    ui.panel_well(
                        ui.markdown(
                            """
                            #### **Time series charts**
                            *made with*  `altair`
                            <hr style="height: 4px; background-color: black; border: none" />
                            """
                        ),
                        
                        ui.input_select("tch_plot", "Plot:", choices=["Scatter", "Line", "Error band"]),

                        ui.accordion(
                            ui.accordion_panel(
                                "Dataset",
                                ui.input_selectize("tch_condition", "Condition:", ["all", "not all"]),
                                ui.panel_conditional(
                                    "input.tch_condition != 'all'",
                                    ui.input_selectize("tch_replicate", "Replicate:", ["all", "not all"]),
                                    ui.panel_conditional(
                                        "input.tch_replicate == 'all'",
                                        ui.input_checkbox("time_separate_replicates", "Show replicates separately", False),
                                    ),
                                ),
                            ),

                            ui.accordion_panel(
                                "Metric",
                                ui.input_selectize("tch_metric", label=None, choices=Metrics.Time, selected="Confinement ratio mean"),
                                ui.input_radio_buttons("y_axis", "Y axis with", ["Absolute values", "Relative values"]),
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
                                            ui.input_selectize("tch_scatter_background", "Background:", Styles.Background),
                                            ui.input_selectize("tch_scatter_color_palette", "Color palette:", Styles.PaletteQualitative),
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
                                                                ui.input_selectize("tch_bullet_outline_color", "Outline color:", Styles.Color + ["match"], selected="match"),
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
                                            ui.input_selectize("tch_line_background", "Background:", Styles.Background),
                                            ui.input_selectize("tch_line_color_palette", "Color palette:", Styles.PaletteQualitative),
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
                                            ui.input_selectize("tch_errorband_background", "Background:", Styles.Background),
                                            ui.input_selectize("tch_errorband_color_palette", "Color palette:", Styles.PaletteQualitative),
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
                                                ui.input_selectize("tch_errorband_outline_color", "Outline color:", Styles.Color + ["match"], selected="match"),
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
                                                        ui.input_selectize("tch_errorband_mean_line_color", "Line color:", Styles.Color + ["match"], selected="match"),
                                                        ui.input_selectize("tch_errorband_mean_line_style", "Line style:", Styles.LineStyle),
                                                        ui.input_numeric("tch_errorband_mean_line_width", "Line width:", 1, min=0, step=0.1),
                                                    ),
                                                ),
                                                ui.accordion_panel(
                                                    "Median",
                                                    ui.input_checkbox("tch_errorband_show_median", "Show median", False),
                                                    ui.panel_conditional(
                                                        "input.tch_errorband_show_median == true",
                                                        ui.input_selectize("tch_errorband_median_line_color", "Line color:", Styles.Color + ["match"], selected="match"),
                                                        ui.input_selectize("tch_errorband_median_line_style", "Line style:", Styles.LineStyle),
                                                        ui.input_numeric("tch_errorband_median_line_width", "Line width:", 1, min=0, step=0.1),
                                                    ),
                                                ),
                                                ui.accordion_panel(
                                                    "Min",
                                                    ui.input_checkbox("tch_errorband_show_min", "Show min", False),
                                                    ui.panel_conditional(
                                                        "input.tch_errorband_show_min == true",
                                                        ui.input_selectize("tch_errorband_min_line_color", "Line color:", Styles.Color + ["match"], selected="match"),
                                                        ui.input_selectize("tch_errorband_min_line_style", "Line style:", Styles.LineStyle),
                                                        ui.input_numeric("tch_errorband_min_line_width", "Line width:", 1, min=0, step=0.1),
                                                    ),
                                                ),
                                                ui.accordion_panel(
                                                    "Max",
                                                    ui.input_checkbox("tch_errorband_show_max", "Show max", False),
                                                    ui.panel_conditional(
                                                        "input.tch_errorband_show_max == true",
                                                        ui.input_selectize("tch_errorband_max_line_color", "Line color:", Styles.Color + ["match"], selected="match"),
                                                        ui.input_selectize("tch_errorband_max_line_style", "Line style:", Styles.LineStyle),
                                                        ui.input_numeric("tch_errorband_max_line_width", "Line width:", 1, min=0, step=0.1),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                    ui.markdown(""" <p> """),
                    ui.card(
                        ui.output_plot("time_series_poly_fit_chart"),
                        ui.download_button("download_time_series_poly_fit_chart__html", "Download Time Series Poly Fit Chart HTML"),
                        ui.download_button("download_time_series_poly_fit_chart_svg", "Download Time Series Poly Fit Chart SVG"),
                    ),
                    # ... more cards for line chart and errorband
                ),

                ui.nav_panel(
                    "Superplots",

                    ui.panel_well(
                        ui.markdown(
                            """
                            #### **Superplots**
                            *made with*  `seaborn`
                            <hr style="height: 4px; background-color: black; border: none" />
                            """
                        ),

                        ui.input_selectize(id="superplot_type", label="Plot:", choices=["Swarms", "Violins"], selected="Swarms"),

                        ui.accordion(

                            ui.accordion_panel(
                                "Pre-sets",
                                ui.input_selectize("sp_preset", "Try a preset:", ["Swarms", "Swarms & Violins", "Violins & KDEs", "Swarms & Violins & KDEs"], selected="Swarms & Violins"),
                            ),
                            ui.accordion_panel(
                                "Metric",
                                ui.input_selectize("sp_metric", label=None, choices=Metrics.Track, selected="Confinement ratio"),
                                # TODO: ui.input_radio_buttons("sp_y_axis", "Y axis with", ["Absolute values", "Relative values"]),
                            ),
                            ui.accordion_panel(
                                "General",
                                ui.input_checkbox(id="sp_show_swarms", label="Show swarms", value=True),
                                ui.input_checkbox(id="sp_show_violins", label="Show violins", value=True),
                                ui.input_checkbox(id="sp_show_kde", label="Show KDE", value=False),
                                ui.panel_conditional(
                                    "input.sp_show_kde == true",
                                    ui.input_checkbox(id="sp_kde_legend", label="Show KDE legend", value=False),
                                ),
                                ui.input_checkbox(id="sp_show_cond_mean", label="Show condition means as lines", value=False),
                                ui.input_checkbox(id="sp_show_cond_median", label="Show condition medians as lines", value=False),
                                ui.input_checkbox(id="sp_show_errbars", label="Show error bars", value=False),
                                ui.input_checkbox(id="sp_show_rep_means", label="Show replicate mean bullets", value=False),
                                ui.input_checkbox(id="sp_show_rep_medians", label="Show replicate median bullets", value=True),
                                ui.input_checkbox(id="sp_show_legend", label="Show legend", value=True),
                                ui.input_checkbox(id="sp_grid", label="Show grid", value=False),
                                ui.input_checkbox(id="sp_spine", label="Open axes top/right", value=True),
                                # TODO: ui.input_checkbox(id="sp_flip", label="Flip axes", value=False),
                                ui.row(
                                    ui.column(
                                        6,
                                        ui.input_numeric(id="sp_fig_width", label="Fig width:", value=10, min=1, step=0.5)
                                    ),
                                    ui.column(
                                        6,
                                        ui.input_numeric(id="sp_fig_height", label="Fig height:", value=7, min=1, step=0.5)
                                    ),
                                ),
                            ),

                            ui.accordion_panel(
                                "Aesthetics",

                                ui.input_selectize(id="sp_palette", label="Color palette:", choices=Styles.PaletteQualitative, selected="tab10"),

                                ui.accordion(
                                    
                                    ui.accordion_panel(
                                        "Swarms",
                                        ui.panel_conditional(
                                            "input.sp_show_swarms == true",
                                            ui.input_numeric("sp_swarm_marker_size", "Dot size:", 1, min=0, step=0.5),
                                            ui.input_numeric("sp_swarm_marker_alpha", "Dot opacity:", 0.5, min=0, max=1, step=0.1),
                                            ui.input_selectize("sp_swarm_marker_outline", "Dot outline color:", Styles.Color, selected="black"),
                                        ),
                                        ui.panel_conditional(
                                            "input.sp_show_swarms == false",
                                            ui.markdown(
                                                """
                                                *Swarms not enabled.*
                                                """
                                            )
                                        )
                                    ),

                                    ui.accordion_panel(
                                        "Violins",
                                        ui.panel_conditional(
                                            "input.sp_show_violins == true",
                                            ui.input_selectize("sp_violin_fill", "Fill color:", Styles.Color, selected="whitesmoke"),
                                            ui.input_numeric("sp_violin_alpha", "Fill opacity:", 0.5, min=0, max=1, step=0.1),
                                            ui.input_selectize("sp_violin_outline", "Outline color:", Styles.Color, selected="lightgrey"),
                                            ui.input_numeric("sp_violin_outline_width", "Outline width:", 1, min=0, step=1),
                                        ),
                                        ui.panel_conditional(
                                            "input.sp_show_violins == false",
                                            ui.markdown(
                                                """
                                                *Violins not enabled.*
                                                """
                                            )
                                        ),
                                    ),

                                    ui.accordion_panel(
                                        "Kernel Density Estimate (KDE)",
                                        ui.markdown(
                                            """
                                            *KDEs are computed across data points of specific replicates in each condition, modeling the underlying data distribution* <br>
                                            """
                                        ),
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
                                        ui.panel_conditional(
                                            "input.sp_show_kde == false",
                                            ui.markdown(
                                                """
                                                *KDE not enabled.*
                                                """
                                            )
                                        ),
                                    ),

                                    ui.accordion_panel(
                                        "Lines and error bars",
                                        ui.panel_conditional(
                                            "input.sp_show_cond_mean == true && input.sp_show_cond_median == true",
                                            ui.input_selectize("sp_set_as_primary", label="Set as primary:", choices=["mean", "median"], selected="mean"),
                                        ),
                                        ui.panel_conditional(
                                            "input.sp_show_cond_mean == true",
                                            ui.input_numeric(id="sp_mean_line_span", label="Mean line span length:", value=0.12, min=0, step=0.01),
                                            ui.input_selectize(id="sp_mean_line_color", label="Mean line color:", choices=Styles.Color, selected="black"),
                                        ),
                                        ui.panel_conditional(
                                            "input.sp_show_cond_median == true",
                                            ui.input_numeric(id="sp_median_line_span", label="Median line span length:", value=0.08, min=0, step=0.01),
                                            ui.input_selectize(id="sp_median_line_color", label="Median line color:", choices=Styles.Color, selected="darkblue"),
                                        ),
                                        ui.panel_conditional(
                                            "input.sp_show_cond_mean == true || input.sp_show_cond_median == true",
                                            ui.input_numeric(id="sp_lines_lw", label="Mean/Median Line width:", value=1, min=0, step=0.5),
                                        ),
                                        ui.panel_conditional(
                                            "input.sp_show_errbars == true",
                                            ui.input_numeric(id="sp_errorbar_capsize", label="Error bar cap size:", value=4, min=0, step=1),
                                            ui.input_numeric(id="sp_errorbar_lw", label="Error bar line width:", value=1, min=0, step=0.5),
                                            ui.input_selectize(id="sp_errorbar_color", label="Error bar color:", choices=Styles.Color, selected="black"),
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
                                        ui.panel_conditional(
                                            "input.sp_show_rep_means == true",
                                            ui.input_numeric("sp_mean_bullet_size", "Mean bullet size:", 80, min=0, step=1),
                                            ui.input_selectize("sp_mean_bullet_outline", "Mean bullet outline color:", Styles.Color, selected="black"),
                                            ui.input_numeric("sp_mean_bullet_outline_width", "Mean bullet outline width:", 0.75, min=0, step=0.05),
                                            ui.input_numeric("sp_mean_bullet_alpha", "Mean bullet opacity:", 1, min=0, max=1, step=0.1),
                                        ),
                                        ui.panel_conditional(
                                            "input.sp_show_rep_medians == true",
                                            ui.input_numeric("sp_median_bullet_size", "Median bullet size:", 50, min=0, step=1),
                                            ui.input_selectize("sp_median_bullet_outline", "Median bullet outline color:", Styles.Color, selected="black"),
                                            ui.input_numeric("sp_median_bullet_outline_width", "Median bullet outline width:", 0.75, min=0, step=0.05),
                                            ui.input_numeric("sp_median_bullet_alpha", "Median bullet opacity:", 1, min=0, max=1, step=0.1),
                                        ),
                                        ui.panel_conditional(
                                            "input.sp_show_rep_means == false && input.sp_show_rep_medians == false",
                                            ui.markdown(
                                                """
                                                *Replicate means/medians not enabled.*
                                                """
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        ui.markdown(""" <br> """),
                        ui.input_text(id="sp_title", label=None, placeholder="Title me!"),

                        ui.markdown(""" <br> """),
                        ui.panel_conditional(
                            "input.superplot_type == 'Swarms'",
                            ui.input_task_button(id="sps_generate", label="Generate", class_="btn-secondary", width="100%")
                        ),
                        ui.panel_conditional(
                            "input.superplot_type == 'Violins'",
                            ui.input_task_button(id="spv_generate", label="Generate", class_="btn-secondary", width="100%")
                        )
                    ),
                    ui.markdown(""" <br> """),
                    ui.panel_conditional(
                        "input.superplot_type == 'Swarms'",
                        ui.output_ui("sps_plot_card"),
                        ui.download_button(id="sps_download_svg", label="Download SVG", width="100%"),
                    ),
                    ui.panel_conditional(
                        "input.superplot_type == 'Violins'",
                        ui.output_ui("spv_plot_card"),
                        ui.download_button(id="spv_download_svg", label="Download SVG", width="100%"),
                    )
                ),
                widths = (2, 10)
            ),
        ),
        # ui.nav_spacer(),
        # ui.nav_control(ui.input_dark_mode(mode="light")),
        title="Peregrin"
    ),
)