import warnings
import shiny.ui as ui
from shiny._deprecated import ShinyDeprecationWarning

from src.code import Metrics, Markers, Styles, Modes

from pathlib import Path




warnings.filterwarnings(
    "ignore",
    message=r".*panel_well\(\) is deprecated\. Use shiny\.ui\.card\(\) instead\.",
    category=ShinyDeprecationWarning,
)



# Add a section for enabling the theme/style selection (brutalistic, solemn, digital, silky)

# _ _ _ _  UI DESIGN DEFINITION  _ _ _ _
app_ui = ui.page_sidebar(
    
    # _ _ _ SIDEBAR - DATA FILTERING _ _ _
    ui.sidebar(    
        ui.markdown("""  <p>  """),
        ui.output_ui(id="sidebar_label"),
        ui.input_action_button(id="append_threshold", label="Add threshold", class_="btn-primary", width="100%", disabled=True),
        ui.input_action_button(id="remove_threshold", label="Remove threshold", class_="btn-primary", width="100%", disabled=True),
        ui.output_ui(id="sidebar_accordion_placeholder"),
        ui.input_task_button(id="set_threshold", label="Set threshold", label_busy="Applying...", class_="btn-secondary task-btn", disabled=True),
        ui.markdown("<p style='line-height:0.1;'> <br> </p>"),
        ui.output_ui(id="threshold_info"),
        ui.download_button(id="download_threshold_info", label="Info SVG", width="100%", _class="space-x-2"),
        id="sidebar", open="closed", position="right", width="300px"
    ),

    # _ _ _ _ PANEL NAVIGATION BAR _ _ _ _
    ui.navset_bar(

        ui.nav_panel(
            "About",
            ui.br(),
            ui.markdown(
                """ 
                    ##### A tool designed for processing and interpreting tracking data, offering a user-friendly interface built with [Py-Shiny](https://shiny.posit.co/py/).
                    *Import raw or processed data.* <br>
                    *Explore them by applying filters and generating insightful visualizations.* <br>
                    *Export results.* <br>
                """
            ),
        ),

        # _ _ _ _ RAW DATA INPUT PANEL - APP INITIALIZATION _ _ _ _
        ui.nav_panel(
            "Input Menu",
            ui.div(
                {"id": "data-inputs"},

                # _ Buttons & input UIs _
                ui.input_action_button("add_input", "Add data input", class_="btn-primary"),
                ui.input_action_button("remove_input", "Remove data input", class_="btn-primary", disabled=True),
                ui.output_ui("run_btn_ui"),
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
                        ui.input_checkbox("strip_data", "Strip data, only keeping necessary columns", True),
                        ui.div(  
                            # ui.tags.style(Customize.Link1),
                            ui.input_action_link("explain_auto_label", "What's Auto-label?", class_="plain-link"),
                            ui.input_checkbox("auto_label", "Auto-label", False),
                            
                            ui.output_ui("data_labeling_ui"),

                        ), 
                        ui.output_ui('task_btn_labeling_sidebar'),
                        width="300px",
                        id="labeling_sidebar",
                    ), 
                    # File inputs
                    ui.div(
                        {"id": "input_file_container_1"},
                        ui.input_text(id=f"condition_label1", label=f"Label:", placeholder="Condition 1"),
                        ui.input_file(id=f"input_file1", label="Upload files:", placeholder="Drag and drop here!", multiple=True),
                        ui.markdown(""" <hr style="border: none; border-top: 1px dotted" /> """),
                    ), 
                    id="labeling_sidebar_n_data_input_layout",
                ),

                # _ Draggable accordion panel - columns selection _
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
                                ui.markdown("<span style='color:darkgrey; font-style:italic;'>You can drag me around!</span>"), #TODO - define this style inside of the css and remove it from here
                            ), 
                            open=False, class_="draggable-accordion", id="draggable_column_selector_panel"
                        ),
                    ), 
                    width="350px", right="450px", top="130px", draggable=True, class_="elevated-panel"
                )
            )
        ),

        # _ _ _ _ PROCESSED DATA DISPLAY _ _ _ _
        ui.nav_panel(
            "Data Tables",

            # _ Input for already processed data _
            ui.row(
                ui.column(6, 
                    ui.markdown( #TODO - define this style inside of the css and remove it from here
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
                    # ui.card_header("Track stats", class_="bg-light"),
                    ui.card_header("Track stats", class_="bg-secondary-css"),
                    ui.output_data_frame("render_track_stats"),
                    ui.download_button("download_track_stats", "Download CSV"),
                    full_screen=True, 
                ),
                ui.card(
                    # ui.card_header("Frame stats", class_="bg-light"),
                    ui.card_header("Frame stats", class_="bg-secondary-css"),
                    ui.output_data_frame("render_frame_stats"),
                    ui.download_button("download_frame_stats", "Download CSV"),
                    full_screen=True, 
                ),
            ),
            ui.layout_columns(
                ui.card(
                    # ui.card_header("Time interval stats", class_="bg-light"),
                    ui.card_header("Time interval stats", class_="bg-secondary-css"),
                    ui.output_data_frame("render_time_interval_stats"),
                    ui.download_button("download_time_interval_stats", "Download CSV"),
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
                    # bg="#f8f8f8",
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

                # _ _ _ TRACK RECONSTRUCTION _ _ _
                ui.nav_panel(
                    "Track reconstruction",

                    ui.panel_well( 
                        ui.markdown(   # TODO - annotate which libraries were used
                            """
                            ### **Track reconstruction**
                            ___
                            """
                        ),

                        # _ _ SETTINGS _ _
                        ui.input_selectize(id="track_reconstruction_method", label="Reconstruction method:", choices=["Realistic", "Polar", "Animated"], selected="Animated", width="200px"),

                        ui.accordion(
                            ui.accordion_panel(
                                "Data Categories",
                                ui.row(
                                    ui.column(6, ui.input_selectize(id="conditions_tr", label="Conditions:", choices=[], multiple=True, options={"placeholder": "Select conditions"})),
                                    ui.column(1, ui.input_action_button(id="conditions_reset_tr", label="🗘", class_="btn-noframe")),
                                ),
                                ui.row(
                                    ui.column(4, ui.input_selectize(id="replicates_tr", label="Replicates:", choices=[], multiple=True, options={"placeholder": "Select replicates"})),
                                    ui.column(1, ui.input_action_button(id="replicates_reset_tr", label="🗘", class_="btn-noframe")),
                                )
                            ),
                            ui.accordion_panel(
                                "Compose",
                                ui.accordion(
                                    ui.accordion_panel(
                                        "Tracks",
                                        ui.row(
                                            ui.input_numeric("tracks_smoothing_index", "Smoothing index:", 0, width="170px"),
                                            ui.input_numeric("tracks_line_width", "Line width:", 0.85, width="120px")
                                        )
                                    ),
                                    ui.accordion_panel(
                                        "Head markers",
                                        ui.input_checkbox("tracks_mark_heads", "Mark track heads", True),
                                        ui.panel_conditional(
                                            "input.tracks_mark_heads",
                                            ui.row(
                                                ui.input_selectize("tracks_marker_type", "Marker:", list(Markers.TrackHeads.keys()), selected="circle-empty", width="200px"),
                                                ui.input_numeric("tracks_marks_size", "Marker size:", 3, min=0, width="130px")
                                            )
                                        ),
                                    ),
                                    ui.accordion_panel(
                                        "Decoratives",
                                        ui.input_checkbox("tracks_show_grid", "Show grid", True),
                                        ui.panel_conditional(
                                            "input.tracks_show_grid && input.track_reconstruction_method == 'Polar'",
                                            ui.input_selectize("tracks_grid_style", "Grid style:", ["simple-1", "simple-2", "spindle", "radial", "dartboard-1", "dartboard-2"], width="140px"),
                                        ),
                                    ),
                                )
                            ),
                            ui.accordion_panel(
                                "Color",
                                ui.row(
                                    ui.input_selectize("tracks_color_mode", "Color mode:", Styles.ColorMode, width="250px"),
                                    ui.panel_conditional(
                                        "!['single color', 'random colors', 'random greys', 'differentiate conditions', 'differentiate replicates'].includes(input.tracks_color_mode)",
                                        ui.input_selectize("tracks_lut_scaling_metric", "LUT scaling metric:", Metrics.Lut, width="230px"),
                                    ),
                                    ui.panel_conditional(
                                        "input.tracks_color_mode == 'single color'",
                                        ui.input_selectize("tracks_only_one_color", "Color:", list(Styles.Color.keys()), width="190px"),
                                    ),
                                    ui.input_selectize("tracks_background", "Background:", Styles.Background, width="120px"),
                                ),
                                ui.br(),
                                ui.row(
                                    ui.panel_conditional(
                                        "['differentiate conditions', 'differentiate replicates'].includes(input.tracks_color_mode)",
                                        ui.div(ui.input_checkbox("tracks_use_stock_palette", "Use stock palette", True), style="margin-left: 20px; margin-right: 10px; margin-top: 5px;"),
                                    ),
                                    ui.panel_conditional(
                                        "['differentiate conditions', 'differentiate replicates'].includes(input.tracks_color_mode) && input.tracks_use_stock_palette == true",
                                        ui.div(ui.input_selectize("tracks_stock_palette", None, Styles.PaletteQualitativeMatplotlib, width="200px"), style="margin-left: 10px; margin-right: 30px;"),
                                    ),
                                    ui.panel_conditional(
                                        "!['single color', 'random colors', 'random greys', 'differentiate conditions', 'differentiate replicates'].includes(input.tracks_color_mode)",
                                        ui.div(ui.input_checkbox("tracks_lutmap_scale_auto", "Auto scale LUT", True), style="margin-left: 20px; margin-right: 30px; margin-top: 5px;"),
                                    ),
                                    ui.panel_conditional(
                                        "!['single color', 'random colors', 'random greys', 'differentiate conditions', 'differentiate replicates'].includes(input.tracks_color_mode) && input.tracks_lutmap_scale_auto == true",
                                        ui.div(ui.output_text_verbatim(id="tracks_lutmap_auto_scale_info", placeholder=True), style="margin-left: 30px; margin-right: 30px;"),
                                    ),
                                    ui.panel_conditional(
                                        "!['single color', 'random colors', 'random greys', 'differentiate conditions', 'differentiate replicates'].includes(input.tracks_color_mode) && input.tracks_lutmap_scale_auto == false",
                                        ui.div(ui.markdown("LUT map scale range:"), style="margin-left: 30px; margin-right: 10px; margin-top: 5px;"),
                                        ui.div(ui.input_numeric("tracks_lutmap_scale_min", None, 0, width="100px"), style="margin-left: 10px; margin-right: 10px;"),
                                        ui.div(ui.input_numeric("tracks_lutmap_scale_max", None, 100, width="100px"), style="margin-left: 10px; margin-right: 30px;"),
                                    ),
                                    ui.panel_conditional(
                                        "!['single color', 'random colors', 'random greys', 'differentiate conditions', 'differentiate replicates'].includes(input.tracks_color_mode)",
                                        ui.div(ui.input_checkbox(id="tracks_lutmap_extend_edges", label="Sharp edged LUT scale", value=True), style="margin-left: 60px; margin-right: 20px; margin-top: 5px;")
                                    )
                                )
                            ),
                            class_="accordion02"
                        ),

                        # _ Title and generate plot _
                        ui.markdown(""" <br> """),
                        ui.row(ui.input_text(id="tracks_title", label=None, placeholder="Title me!", width="100%"), style="margin-left: 1px; margin-right: 1px;"),
                        ui.panel_conditional(
                            "input.track_reconstruction_method == 'Realistic'",
                            ui.input_task_button(id="trr_generate", label="Generate", class_="btn-secondary task-btn", width="100%"),
                        ),
                        ui.panel_conditional(
                            "input.track_reconstruction_method == 'Polar'",
                            ui.input_task_button(id="tnr_generate", label="Generate", class_="btn-secondary task-btn", width="100%"),
                        ),
                        ui.panel_conditional(
                            "input.track_reconstruction_method == 'Animated'",
                            ui.markdown(""" <p> </p> """),
                            ui.row(
                                ui.input_numeric("tar_dpi", "DPI resolution:", value=100, min=10, max=1000, step=10, width="180px"),
                                style="margin-left: 4px;"
                            ),
                            ui.markdown(""" <p> </p> """),
                            ui.input_task_button(id="tar_generate", label="Generate", class_="btn-secondary task-btn", width="100%"),
                        )
                    ),
                    # "#c0c4ca",

                    # _ _ PLOT DISPLAYS AND DOWNLOADS _ _
                    ui.markdown(""" <br> """),
                    ui.panel_conditional(
                        "input.track_reconstruction_method == 'Realistic'",
                        ui.card(
                            ui.output_plot("track_reconstruction_realistic"),
                            full_screen=False,
                            height="800px",
                        ),
                        ui.download_button("trr_download", "Download", width="100%")
                    ),
                    ui.panel_conditional(
                        "input.track_reconstruction_method == 'Polar'",
                        ui.card(
                            ui.output_plot("track_reconstruction_normalized"),
                            full_screen=False,
                            height="800px",
                        ),
                        ui.download_button("tnr_download", "Download", width="100%"),
                    ),
                    ui.panel_conditional(
                        "input.track_reconstruction_method == 'Animated'",
                        ui.card(
                            ui.div(
                                ui.input_action_button("prev", "-"),
                                ui.output_ui("viewer"),
                                ui.input_action_button("next", "+"),
                                style="display:flex; justify-content:space-between; align-items:center;"
                            ),
                            ui.output_ui("replay_slider"),
                            full_screen=False, width="100", height="800px"
                        ),
                        ui.markdown(""" <p></p> """),
                        ui.panel_well(
                            ui.accordion(
                                ui.accordion_panel(
                                    "Replay settings",
                                    ui.input_numeric("tar_framerate", "Frame rate (fps)", value=30, min=1, max=1000, step=1, width="140px"),
                                )
                            ),
                            ui.markdown(""" <p></p> """),
                            ui.download_button("tar_download", "Download video", width="100%"),
                        )
                    ),
                    ui.panel_conditional(
                        "!['single color', 'random colors', 'random greys'].includes(input.tracks_color_mode)",
                        ui.markdown(""" <p></p> """),
                        ui.download_button(id="download_lut_map_svg", label="Download LUT Map SVG", width="100%"),
                    ),
                ),

                # _ _ _ DIRECTIONALITY DISTRIBUTION _ _ _
                ui.nav_panel(
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
                                    ui.input_radio_buttons(id="dd_normalization", label=None, choices={"globally": "Normalize globally", "per condition": "Normalize per condition"}, selected="globally"),
                                    ui.column(
                                        1, 
                                        ui.input_checkbox(id="dd_add_weights", label="Add weights", value=False),
                                        offset=1,
                                    ),
                                    ui.panel_conditional(
                                        "input.dd_add_weights == true",
                                        ui.div(ui.input_selectize(id="dd_weight", label=None, choices=[], selected=None, width="200px"), style="margin-top: -5px;"),
                                    ),
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
                                        ""
                                    ),
                                    ui.accordion_panel(
                                        "Color",
                                        ""
                                    ),
                                    class_="accordion02"
                                ),
                                ui.br(),
                                ui.input_task_button(id="generate_dd_rosechart", label="Generate", class_="btn-secondary task-btn", width="100%"),
                            ),
                            ui.br(),
                            ui.output_plot("dd_plot_rosechart"),
                            ui.br(),
                            ui.download_button("download_dd_rosechart", "Download Rose Chart", width="100%")
                        ),
                        ui.card(
                            ui.card_header("Gaussian KDE Colormesh", class_="bg-secondary-css"),
                            ui.panel_well(
                                ui.accordion(
                                    ui.accordion_panel(
                                        "Compose",
                                        ui.input_text("dd_kde_colormesh_title", label=None, placeholder="Title me!", width="100%")
                                    ),
                                    ui.accordion_panel(
                                        "Color",
                                        ""
                                    ),
                                    class_="accordion02"
                                ),
                                ui.br(),
                                ui.input_task_button(id="generate_dd_kde_colormesh", label="Generate", class_="btn-secondary task-btn", width="100%"),
                            ),
                            ui.br(),
                            ui.output_plot("dd_plot_kde_colormesh"),
                            ui.download_button("download_dd_kde_colormesh", "Download Colormesh", width="100%")
                        ),
                        ui.card(
                            ui.card_header("KDE Line Plot", class_="bg-secondary-css"),
                            ui.panel_well(
                                ui.accordion(
                                    ui.accordion_panel(
                                        "Compose",
                                        ""
                                    ),
                                    ui.accordion_panel(
                                        "Color",
                                        ""
                                    ),
                                    class_="accordion02"
                                ),
                                ui.br(),
                                ui.input_task_button(id="generate_dd_kde_line", label="Generate", class_="btn-secondary task-btn", width="100%"),
                            ),
                            ui.br(),
                            ui.output_plot("dd_plot_kde_line"),
                            ui.download_button("download_dd_kde_line", "Download Line Plot", width="100%")
                        ),
                        ui.card(
                            ui.card_header("Overlay", class_="bg-secondary-css"),
                            ui.panel_well(
                                ui.accordion(
                                    ui.accordion_panel(
                                        "Compose",
                                        ""
                                    ),
                                    ui.accordion_panel(
                                        "Color",
                                        ""
                                    ),
                                    class_="accordion02"
                                ),
                                ui.br(),
                                ui.input_task_button(id="generate_dd_overlay", label="Generate", class_="btn-secondary task-btn", width="100%"),
                            ),
                            ui.br(),
                            ui.output_plot("dd_plot_overlay"),
                            ui.download_button("download_dd_overlay", "Download Overlay", width="100%")
                        ),
                        width=1/2,
                        class_="dd-cards-wrap"
                    )
                ),

                # _ _ _ MSD _ _ _
                ui.nav_panel(
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
                                        'Line',
                                        ui.input_checkbox("line_show_msd", "Show line", True),
                                    ),
                                    ui.accordion_panel(
                                        'Scatter',
                                        ui.input_checkbox("scatter_show_msd", "Show scatter", False),
                                    ),
                                    ui.accordion_panel(
                                        'Linear fit',
                                        ui.input_checkbox("fit_show_msd", "Show linear fit", True),
                                    ),
                                    ui.accordion_panel(
                                        'Error band',
                                        ui.row(
                                            ui.column(3, 
                                                ui.input_checkbox("error_band_show_msd", "Show error band", True)
                                            ),
                                            ui.column(3, 
                                                ui.panel_conditional(
                                                    "input.error_band_show_msd == true",
                                                    ui.input_selectize("error_band_type_msd", None, choices=['sem', 'sd'], selected='sem'),
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
                                                ui.input_selectize("only_one_color_msd", "Color:", list(Styles.Color.keys())),
                                                style="margin-left: 15px;"
                                            )
                                        ),
                                        ui.panel_conditional(
                                            "input.c_mode_msd != 'single color'",
                                            ui.div(
                                                ui.input_checkbox("palette_stock_msd", "Use stock palette", False),
                                                style="margin-top: 38px; margin-left: 15px;"
                                            ),
                                        )
                                    ),
                                    ui.column(2, 
                                        ui.panel_conditional(
                                            "input.c_mode_msd != 'single color' && input.palette_stock_msd == true",
                                            ui.div(
                                                ui.input_selectize("palette_stock_type_msd", "Palette:", Styles.PaletteQualitativeMatplotlib),    
                                                style="margin-left: -30px;"
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
                    ui.download_button("download_plot_msd", "Download", width="100%"),
                ),

                # _ _ _ TURNING ANGLES _ _ _
                ui.nav_panel(
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
                            class_="accordion02"
                        )
                    )
                ),

                # _ _ _ TIME SERIES _ _ _
                ui.nav_panel(
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
                                ui.input_selectize("tch_metric", label=None, choices=Metrics.Time, selected="Confinement ratio mean"),
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
                                            ui.input_selectize("tch_scatter_background", "Background:", Styles.Background),
                                            ui.input_selectize("tch_scatter_color_palette", "Color palette:", Styles.PaletteQualitativeAltair),
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
                                                                ui.input_selectize("tch_bullet_outline_color", "Outline color:", list(Styles.Color.keys()) + ["match"], selected="match"),
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
                                            ui.input_selectize("tch_line_color_palette", "Color palette:", Styles.PaletteQualitativeAltair),
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
                                            ui.input_selectize("tch_errorband_color_palette", "Color palette:", Styles.PaletteQualitativeAltair),
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
                                                ui.input_selectize("tch_errorband_outline_color", "Outline color:", list(Styles.Color.keys()) + ["match"], selected="match"),
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
                                                        ui.input_selectize("tch_errorband_mean_line_color", "Line color:", list(Styles.Color.keys()) + ["match"], selected="match"),
                                                        ui.input_selectize("tch_errorband_mean_line_style", "Line style:", Styles.LineStyle),
                                                        ui.input_numeric("tch_errorband_mean_line_width", "Line width:", 1, min=0, step=0.1),
                                                    ),
                                                ),
                                                ui.accordion_panel(
                                                    "Median",
                                                    ui.input_checkbox("tch_errorband_show_median", "Show median", False),
                                                    ui.panel_conditional(
                                                        "input.tch_errorband_show_median == true",
                                                        ui.input_selectize("tch_errorband_median_line_color", "Line color:", list(Styles.Color.keys()) + ["match"], selected="match"),
                                                        ui.input_selectize("tch_errorband_median_line_style", "Line style:", Styles.LineStyle),
                                                        ui.input_numeric("tch_errorband_median_line_width", "Line width:", 1, min=0, step=0.1),
                                                    ),
                                                ),
                                                ui.accordion_panel(
                                                    "Min",
                                                    ui.input_checkbox("tch_errorband_show_min", "Show min", False),
                                                    ui.panel_conditional(
                                                        "input.tch_errorband_show_min == true",
                                                        ui.input_selectize("tch_errorband_min_line_color", "Line color:", list(Styles.Color.keys()) + ["match"], selected="match"),
                                                        ui.input_selectize("tch_errorband_min_line_style", "Line style:", Styles.LineStyle),
                                                        ui.input_numeric("tch_errorband_min_line_width", "Line width:", 1, min=0, step=0.1),
                                                    )
                                                ),
                                                ui.accordion_panel(
                                                    "Max",
                                                    ui.input_checkbox("tch_errorband_show_max", "Show max", False),
                                                    ui.panel_conditional(
                                                        "input.tch_errorband_show_max == true",
                                                        ui.input_selectize("tch_errorband_max_line_color", "Line color:", list(Styles.Color.keys()) + ["match"], selected="match"),
                                                        ui.input_selectize("tch_errorband_max_line_style", "Line style:", Styles.LineStyle),
                                                        ui.input_numeric("tch_errorband_max_line_width", "Line width:", 1, min=0, step=0.1),
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            ),
                            class_="accordion02"
                        )
                    ),
                    ui.markdown(""" <p> """),
                    ui.card(
                        ui.output_plot("time_series_poly_fit_chart"),
                        ui.download_button("download_time_series_poly_fit_chart__html", "Download Time Series Poly Fit Chart HTML"),
                        ui.download_button("download_time_series_poly_fit_chart_svg", "Download Time Series Poly Fit Chart SVG"),
                    ),
                    # ... more cards for line chart and errorband
                ),

                # _ _ _ SUPERPLOTS _ _ _
                ui.nav_panel(
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
                        ui.panel_conditional(
                            "input.superplot_type == 'Swarms'",
                            ui.accordion(
                                ui.accordion_panel(
                                    "Pre-sets",
                                    ui.input_selectize("sp_preset", "Choose a preset:", ["Swarms", "Swarms & Violins", "Violins & KDEs", "Swarms & Violins & KDEs"], selected="Swarms & Violins", width="230px"),
                                ),
                                ui.accordion_panel(
                                    "Metric",
                                    ui.input_selectize("sp_metric", label=None, choices=Metrics.Track, selected="Confinement ratio", width="200px"),
                                    # TODO: ui.input_radio_buttons("sp_y_axis", "Y axis with", ["Absolute values", "Relative values"]),
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
                                                ui.input_selectize("sp_swarm_marker_outline", "Dot outline color:", list(Styles.Color.keys()), selected="black"),
                                            ),
                                        ),
                                        ui.accordion_panel(
                                            "Violins",
                                            ui.input_checkbox(id="sp_show_violins", label="Show violins", value=True),
                                            ui.panel_conditional(
                                                "input.sp_show_violins == true",
                                                ui.input_selectize("sp_violin_fill", "Fill color:", list(Styles.Color.keys()), selected="whitesmoke"),
                                                ui.input_numeric("sp_violin_alpha", "Fill opacity:", 0.5, min=0, max=1, step=0.1),
                                                ui.input_selectize("sp_violin_outline", "Outline color:", list(Styles.Color.keys()), selected="lightgrey"),
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
                                                ui.input_selectize(id="sp_mean_line_color", label="Mean line color:", choices=list(Styles.Color.keys()), selected="black"),
                                            ),
                                            ui.panel_conditional(
                                                "input.sp_show_cond_median == true",
                                                ui.input_numeric(id="sp_median_line_span", label="Median line span length:", value=0.08, min=0, step=0.01),
                                                ui.input_selectize(id="sp_median_line_color", label="Median line color:", choices=list(Styles.Color.keys()), selected="darkblue"),
                                            ),
                                            ui.panel_conditional(
                                                "input.sp_show_cond_mean == true || input.sp_show_cond_median == true",
                                                ui.input_numeric(id="sp_lines_lw", label="Mean/Median Line width:", value=1, min=0, step=0.5),
                                            ),
                                            ui.panel_conditional(
                                                "input.sp_show_errbars == true",
                                                ui.input_numeric(id="sp_errorbar_capsize", label="Error bar cap size:", value=4, min=0, step=1),
                                                ui.input_numeric(id="sp_errorbar_lw", label="Error bar line width:", value=1, min=0, step=0.5),
                                                ui.input_selectize(id="sp_errorbar_color", label="Error bar color:", choices=list(Styles.Color.keys()), selected="black"),
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
                                                ui.input_selectize("sp_mean_bullet_outline", "Mean bullet outline color:", list(Styles.Color.keys()), selected="black"),
                                                ui.input_numeric("sp_mean_bullet_outline_width", "Mean bullet outline width:", 0.75, min=0, step=0.05),
                                                ui.input_numeric("sp_mean_bullet_alpha", "Mean bullet opacity:", 1, min=0, max=1, step=0.1),
                                            ),
                                            ui.panel_conditional(
                                                "input.sp_show_rep_medians == true",
                                                ui.input_numeric("sp_median_bullet_size", "Median bullet size:", 50, min=0, step=1),
                                                ui.input_selectize("sp_median_bullet_outline", "Median bullet outline color:", list(Styles.Color.keys()), selected="black"),
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
                                ui.accordion_panel(
                                    "Color",
                                    ui.input_checkbox(id="sp_use_stock_palette", label="Use stock color palette", value=False),
                                    ui.panel_conditional(
                                        "input.sp_use_stock_palette == true",
                                        ui.input_selectize(id="sp_palette", label="Color palette:", choices=Styles.PaletteQualitativeSeaborn + Styles.PaletteQualitativeMatplotlib, selected=2),
                                    )
                                ),
                                class_="accordion02"
                            ),
                            ui.markdown(""" <br> """),
                            ui.row(ui.input_text(id="sp_title", label=None, placeholder="Title me!", width="100%"), style="margin-left: 1px; margin-right: 1px;")
                        ),

                        # _ _ Violinplot settings _ _
                        ui.panel_conditional(
                            "input.superplot_type == 'Violins'",
                            ui.accordion(
                                ui.accordion_panel(
                                    "Metric",
                                    ui.input_selectize("vp_metric", label=None, choices=Metrics.Track, selected="Confinement ratio"),
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
                                    ui.input_selectize(id="vp_palette", label="Color palette:", choices=Styles.PaletteQualitativeSeaborn, selected="Accent"),
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
                        )
                    ),
                    ui.markdown(""" <br> """),
                    ui.panel_conditional(
                        "input.superplot_type == 'Swarms'",
                        ui.output_ui("sp_plot_card"),
                        ui.download_button(id="sp_download_svg", label="Download SVG", width="100%"),
                    ),
                    ui.panel_conditional(
                        "input.superplot_type == 'Violins'",
                        ui.output_ui("vp_plot_card"),
                        ui.download_button(id="vp_download_svg", label="Download SVG", width="100%"),
                    )
                ),
                widths = (2, 10)
            ),
        ),

        ui.nav_spacer(),

        ui.nav_control(
            ui.input_selectize(
                id="app_theme",
                label=None,
                choices=["Shiny", "Console-0", "Console-1", "Console-2"],
                selected="Shiny",
                width="140px",
                options={"hideSelected": True, },
            ),
            ui.output_ui("custom_theme_url"),
        ),
        # title="Peregrin",
        title=ui.tags.span(
            ui.a(
            "Peregrin",
            href="https://github.com/BranislavModriansky/Peregrin/tree/main",
            class_="peregrin-logo",
            ),
            
        ),
        # title=ui.nav_panel
        
        id="main_nav",
        selected="Input Menu",
        
    ),
)