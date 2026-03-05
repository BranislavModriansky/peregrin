from shiny import ui

from src.code import Metrics, Markers, Dyes, Modes




import warnings
from shiny._deprecated import ShinyDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message=r".*panel_well\(\) is deprecated\. Use shiny\.ui\.card\(\) instead\.",
    category=ShinyDeprecationWarning,
)



subpanel_tracks = ui.nav_panel(
    "Track reconstruction",

    ui.panel_well( 
        ui.markdown(   # TODO - annotate which libraries were used
            """
            ### **Track reconstruction**
            ___
            """
        ),

        # _ _ SETTINGS _ _
        ui.input_selectize(id="track_reconstruction_method", label="Reconstruction method:", choices=["Realistic", "Normalized", "Animated"], selected="Normalized", width="200px"),

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
                                ui.input_selectize("tracks_marker_type", "Marker:", list(Markers.TrackHeads.keys()), selected="hexagon-empty", width="200px"),
                                ui.input_selectize("tracks_marks_color", "Marker Color:", Dyes.Colors, selected="match", width="150px"),
                                ui.input_numeric("tracks_marks_size", "Marker size:", 2, min=0, width="130px")
                            )
                        )
                    ),
                    ui.accordion_panel(
                        "Decoratives",
                        ui.row(
                            ui.column(2, ui.input_checkbox("tracks_show_grid", "Show grid", True)),
                            ui.panel_conditional(
                                "input.tracks_show_grid && input.track_reconstruction_method == 'Normalized'",
                                ui.input_selectize("tracks_grid_style", "Grid style:", ["simple-1", "simple-2", "spindle", "radial", "dartboard-1", "dartboard-2"], width="140px"),
                            ), 
                        ),
                        ui.row(
                            ui.column(2, ui.input_checkbox("tracks_annotate_r", "Annotate r axis", True)),
                            ui.panel_conditional(
                                "input.tracks_annotate_r && input.track_reconstruction_method == 'Normalized'",
                                ui.column(2, ui.input_selectize("tracks_r_annotstyle", "R axis annotation:", ["minimal", "detailed"], width="140px"))
                            )
                        ),
                        ui.column(2, ui.input_checkbox("tracks_annotate_theta", "Annotate theta axis", False)),                                        
                    ),
                )
            ),
            ui.accordion_panel(
                "Color",
                ui.row(
                    ui.input_selectize("tracks_color_mode", "Color mode:", Dyes.CModes, width="250px"),
                    ui.panel_conditional(
                        "!['single color', 'random colors', 'random greys', 'differentiate conditions', 'differentiate replicates'].includes(input.tracks_color_mode)",
                        ui.input_selectize(id="tracks_lut_scaling_metric", label="LUT scaling metric:", choices=[], multiple=False, width="230px"),
                    ),
                    ui.panel_conditional(
                        "input.tracks_color_mode == 'single color'",
                        ui.input_selectize("tracks_only_one_color", "Color:", Dyes.Colors, width="190px"),
                    ),
                    ui.input_selectize("tracks_background", "Background:", Dyes.Background, width="120px"),
                ),
                ui.br(),
                ui.row(
                    ui.panel_conditional(
                        "['differentiate conditions', 'differentiate replicates'].includes(input.tracks_color_mode)",
                        ui.div(ui.input_checkbox("tracks_use_stock_palette", "Use stock palette", True), style="margin-left: 20px; margin-right: 10px; margin-top: 5px;"),
                    ),
                    ui.panel_conditional(
                        "['differentiate conditions', 'differentiate replicates'].includes(input.tracks_color_mode) && input.tracks_use_stock_palette == true",
                        ui.div(ui.input_selectize("tracks_stock_palette", None, Dyes.PaletteQualitativeMatplotlib, width="200px"), style="margin-left: 10px; margin-right: 30px;"),
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
            "input.track_reconstruction_method == 'Normalized'",
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
        "input.track_reconstruction_method == 'Normalized'",
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
            full_screen=False, width="100%", height="800px"
        ),
        ui.markdown(""" <p></p> """),
        ui.panel_well(
            ui.accordion(
                ui.accordion_panel(
                    "Replay settings",
                    ui.input_numeric("tar_framerate", "Frame rate (fps)", value=1, min=0, max=100, step=1, width="140px"),
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
    )
)
