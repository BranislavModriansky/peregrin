from shiny import App, Inputs, Outputs, Session, render, reactive, req, ui
from shiny.types import FileInfo
from shinywidgets import render_plotly, render_altair, output_widget, render_widget
import shiny_sortable as sortable

# ensure local package imports work whether run as a script or as a module
import sys
from pathlib import Path
try:
    from utils import (
        DataLoader, GetInfoSVG,
        Spots, Tracks, Frames,
        Normalize_01, JoinByIndex,
        Metrics, Styles, Markers, Modes,
        Customize, FilenameFormatExample,
        VisualizeTracksRealistics, VisualizeTracksNormalized, GetLutMap,
        BeyondSwarms, SuperViolins,
        Debounce, Throttle
    )
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from utils import (
        DataLoader, GetInfoSVG,
        Metrics, Styles, Markers, Modes,
        Customize, FilenameFormatExample,
        VisualizeTracksRealistics, VisualizeTracksNormalized, GetLutMap,
        BeyondSwarms, SuperViolins,
        Debounce, Throttle
    )

import asyncio
import io
import warnings
import tempfile

import pandas as pd
import numpy as np
from html import escape
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns

from math import floor, ceil
from scipy.stats import gaussian_kde
from datetime import date

import warnings
from shiny._deprecated import ShinyDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message=r".*panel_well\(\) is deprecated\. Use shiny\.ui\.card\(\) instead\.",
    category=ShinyDeprecationWarning,
)

# _ _  CUSTOM SORTABLE UI COMPONENTS _ _
@sortable.make(updatable=True)
def ladder(inputID: str, items: list[str]):
    lis = [
        ui.tags.li(label, **{"data-id": label}, class_="p-2 mb-2 border rounded")
        for label in items
    ]
    return ui.tags.ul(*lis, id=inputID)



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
                                ui.input_selectize("sp_preset", "Choose a preset:", ["Bees", "Bass", "Bees n Bass", "Bass n Bows", "Bees n Bass n Bows"], selected="Bees"),
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



# --- Server logic skeleton ---

def server(input: Inputs, output: Outputs, session: Session):
    """
    Main server logic for the Peregrin UI application.
    Handles dynamic UI components and user interactions for threshold filtering and file input management.
    Args:
        input (Inputs): Reactive input object for user actions and UI events.
        output (Outputs): Reactive output object for rendering UI components.
        session (Session): Session object for sending messages and managing UI state.
    Features:
        - Dynamic creation and removal of threshold threshold panels (1D and 2D modes).
        - Toggle between 1D and 2D thresholding modes, updating UI accordingly.
        - Dynamic enabling/disabling of remove threshold/input buttons based on current state.
        - Dynamic creation and removal of file input/label pairs for user data upload.
        - Renders sidebar accordion UI with threshold panels and threshold settings.
        - Renders sidebar label indicating current thresholding mode.
        - Sends messages to the session to update UI controls as needed.
    Notes:
        - Uses reactive.Value for state management and reactive.effect for event-driven updates.
        - UI components are rendered using the `ui` and `render` modules.
        - Assumes existence of Metrics.SpotAndTrack and Modes.Thresholding for selectize choices.
    """


    # _ _ _ _ Input IDs for file inputs _ _ _ _
    INPUTS = reactive.Value(1)

    # _ _ _ _ Initialize memory for filtering (1D) and gating (2D) _ _ _ _
    THRESHOLDS = reactive.Value(None)
    THRESHOLDS_ID = reactive.Value(1)

    # _ _ _ _ Data frame placeholders _ _ _ _
    RAWDATA = reactive.Value(pd.DataFrame())
    UNFILTERED_SPOTSTATS = reactive.Value(pd.DataFrame())
    UNFILTERED_TRACKSTATS = reactive.Value(pd.DataFrame())
    UNFILTERED_FRAMESTATS = reactive.Value(pd.DataFrame())
    SPOTSTATS = reactive.Value(pd.DataFrame())
    TRACKSTATS = reactive.Value(pd.DataFrame())
    FRAMESTATS = reactive.Value(pd.DataFrame())

    UNITS = reactive.Value()



    # _ _ _ _ FILE INPUTS MANAGEMENT _ _ _ _

    @reactive.Effect
    @reactive.event(input.add_input)
    def add_input():
        id = INPUTS.get()
        INPUTS.set(id + 1)
        session.send_input_message("remove_input", {"disabled": id < 1})

    @reactive.Effect
    @reactive.event(input.remove_input)
    def remove_input():
        id = INPUTS.get()
        if id > 1:
            INPUTS.set(id - 1)
        if INPUTS.get() <= 1:

            session.send_input_message("remove_input", {"disabled": True})

    def _input_container_ui(id: int):
        return ui.div(
            {"id": f"input_file_container_{id}"},
            ui.input_text(
                id=f"condition_label{id}",
                label="Label:",
                placeholder=f"Condition {id}",
            ),
            ui.input_file(
                id=f"input_file{id}",
                label="Upload files:",
                placeholder="Drag and drop here!",
                multiple=True,
            ),
            ui.markdown("<hr style='border: none; border-top: 1px dotted' />"),
        )

    @reactive.effect
    @reactive.event(input.add_input)
    def _add_container():
        id = INPUTS.get()
        ui.insert_ui(
            ui=_input_container_ui(id),
            selector=f"#input_file_container_{id - 1}",
            where="afterEnd"
        )
 
    @reactive.effect
    @reactive.event(input.remove_input)
    def _remove_container():
        id = INPUTS.get()
        ui.insert_ui(
            ui.tags.script(
                f"Shiny.setInputValue('input_file{id+1}', null, {{priority:'event'}});"
                f"Shiny.setInputValue('condition_label{id+1}', '', {{priority:'event'}});"
                # Clear browser chooser if the element still exists
            ), 
            selector="body", 
            where="beforeEnd"
        )
        ui.remove_ui(
            selector=f"#input_file_container_{id+1}",
            multiple=True
        )
    
    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _



    # _ _ _ _ REQUIRED COLUMNS SPECIFICATION _ _ _ _

    @reactive.Effect
    def column_selection():
        ui.update_selectize(id="select_id", choices=["e.g. TRACK ID"])
        ui.update_selectize(id="select_t", choices=["e.g. POSITION T"])
        ui.update_selectize(id="select_x", choices=["e.g. POSITION X"])
        ui.update_selectize(id="select_y", choices=["e.g. POSITION Y"])

        for idx in range(1, INPUTS.get()+1):
            files = input[f"input_file{idx}"]()
            if files and isinstance(files, list) and len(files) > 0:
                try:
                    columns = DataLoader.GetColumns(files[0]["datapath"])

                    for sel in Metrics.LookFor.keys():
                        choice = DataLoader.FindMatchingColumn(columns, Metrics.LookFor[sel])
                        if choice is not None:
                            ui.update_selectize(sel, choices=columns, selected=choice)
                        else:
                            ui.update_selectize(sel, choices=columns, selected=columns[0] if columns else None)
                    break  # Use the first available slot
                except Exception as e:
                    continue

    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _


    # _ _ _ _ LABEL SETTINGS SIDEBAR _ _ _ _

    @reactive.Effect
    @reactive.event(input.explain_auto_label)
    def _():
        ui.modal_show(
            ui.modal(
                ui.markdown(
                    """
                    Switching on <b>Auto-label</b> will automatically derive the <code>Condition</code> and <code>Replicate</code> labels from the <b>text segments</b> between <b>#</b> symbols in the filenames of your uploaded files and assign them, respectively. <br><br>
                    """ 
                ),
                ui.markdown(
                    """ 
                    <span style='font-size: 18px;'>Desired filename format: <i>...#Condition#Replicate#...</i></span> <br>
                    """
                ),
                ui.output_data_frame("FilenameFormatExampleTable"),
                ui.markdown(
                    """ <br><br>
                    <span style='color: dimgrey;'><i>If you want to use this feature, make sure it is enabled before running the analysis.</i></span> <br>
                    """
                ),
            title="What's Auto-label?",
            easy_close=True,
            footer=None,
            # background_color="#2b2b2b"
            )
        )

    @output()
    @render.data_frame
    def FilenameFormatExampleTable():
        return FilenameFormatExample

    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _


    # _ _ _ _ CONDITION LABELS ORDER LADDER _ _ _ _

    @output(id="condition_order_ladder")
    @render.ui
    def condition_order_ladder():
        if TRACKSTATS.get() is not None:
            req(not TRACKSTATS.get().empty)

            items = TRACKSTATS.get()["Condition"].unique().tolist()

            if isinstance(items, list) and len(items) > 1:
                return ladder("order", items)
            elif isinstance(items, list) and len(items) == 1:
                return ui.markdown("*Only one condition present.*")
            else:
                return

    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _



    # _ _ _ _ WRITE DATA VALUES _ _ _ _

    @reactive.Effect
    @reactive.event(input.write_values)
    def _write_values():

        df_unfiltered_spots = UNFILTERED_SPOTSTATS.get().copy()
        df_unfiltered_tracks = UNFILTERED_TRACKSTATS.get().copy()
        df_unfiltered_frames = UNFILTERED_FRAMESTATS.get().copy()
        df_spots = SPOTSTATS.get().copy()
        df_tracks = TRACKSTATS.get().copy()
        df_frames = FRAMESTATS.get().copy()
        req(df is not None and not df.empty for df in [df_unfiltered_spots, df_unfiltered_tracks, df_unfiltered_frames, df_spots, df_tracks, df_frames])
        

        # REPLICATE LABELS AND COLORS
        @reactive.Effect
        def _replicate_values():
            req("Replicate" in df_tracks.columns)
            replicates = sorted(df_tracks["Replicate"].unique())
            if len(set(type(rep) for rep in replicates)) > 1:
                return

            @reactive.Effect
            def _replicate_labels():
                with reactive.isolate():
                    if input.write_replicate_labels() == False:
                        return

                    for idx, rep in enumerate(replicates):
                        
                        label = input[f"replicate_label{idx}"]() if isinstance(input[f"replicate_label{idx}"](), str) and input[f"replicate_label{idx}"]() != "" else rep
                        if label != str(rep):

                            df_unfiltered_spots.loc[df_unfiltered_spots["Replicate"] == rep, "Replicate"] = label
                            df_unfiltered_tracks.loc[df_unfiltered_tracks["Replicate"] == rep, "Replicate"] = label
                            df_unfiltered_frames.loc[df_unfiltered_frames["Replicate"] == rep, "Replicate"] = label
                            df_spots.loc[df_spots["Replicate"] == rep, "Replicate"] = label
                            df_tracks.loc[df_tracks["Replicate"] == rep, "Replicate"] = label
                            df_frames.loc[df_frames["Replicate"] == rep, "Replicate"] = label

                    UNFILTERED_SPOTSTATS.set(df_unfiltered_spots); UNFILTERED_TRACKSTATS.set(df_unfiltered_tracks); UNFILTERED_FRAMESTATS.set(df_unfiltered_frames);
                    SPOTSTATS.set(df_spots); TRACKSTATS.set(df_tracks); FRAMESTATS.set(df_frames)


            @reactive.Effect
            def _replicate_colors():
                with reactive.isolate():
                    print("Writing replicate colors...")
                    print(input.write_replicate_colors())

                    if input.write_replicate_colors():

                        for idx, rep in enumerate(replicates):
                            print(input[f"replicate_color{idx}"]())
                            color = input[f"replicate_color{idx}"]()
                            if color:
                                df_unfiltered_spots.loc[df_unfiltered_spots["Replicate"] == rep, "Replicate color"] = color
                                df_unfiltered_tracks.loc[df_unfiltered_tracks["Replicate"] == rep, "Replicate color"] = color
                                df_unfiltered_frames.loc[df_unfiltered_frames["Replicate"] == rep, "Replicate color"] = color
                                df_spots.loc[df_spots["Replicate"] == rep, "Replicate color"] = color
                                df_tracks.loc[df_tracks["Replicate"] == rep, "Replicate color"] = color
                                df_frames.loc[df_frames["Replicate"] == rep, "Replicate color"] = color

                    elif not input.write_replicate_colors():
                        print("Removing replicate colors...")
                        if "Replicate color" in df_unfiltered_spots.columns:
                            df_unfiltered_spots.drop(columns=["Replicate color"], inplace=True)
                        if "Replicate color" in df_unfiltered_tracks.columns:
                            df_unfiltered_tracks.drop(columns=["Replicate color"], inplace=True)
                        if "Replicate color" in df_unfiltered_frames.columns:
                            df_unfiltered_frames.drop(columns=["Replicate color"], inplace=True)
                        if "Replicate color" in df_spots.columns:
                            df_spots.drop(columns=["Replicate color"], inplace=True)
                        if "Replicate color" in df_tracks.columns:
                            df_tracks.drop(columns=["Replicate color"], inplace=True)
                        if "Replicate color" in df_frames.columns:
                            df_frames.drop(columns=["Replicate color"], inplace=True)

                    UNFILTERED_SPOTSTATS.set(df_unfiltered_spots); UNFILTERED_TRACKSTATS.set(df_unfiltered_tracks); UNFILTERED_FRAMESTATS.set(df_unfiltered_frames)
                    SPOTSTATS.set(df_spots); TRACKSTATS.set(df_tracks); FRAMESTATS.set(df_frames)



        # CONDITION ORDER
        @reactive.Effect
        def _condition_values():
            req("Condition" in df_tracks.columns)

            with reactive.isolate():
                if not input.set_condition_order():
                    return
                
                req(input.order() is not None and not len(input.order()) < 2)
                order = list(input.order())

                df_unfiltered_spots.sort_values("Condition", key=lambda x: x.map({v: i for i, v in enumerate(order)}), inplace=True)
                df_unfiltered_tracks.sort_values("Condition", key=lambda x: x.map({v: i for i, v in enumerate(order)}), inplace=True)
                df_unfiltered_frames.sort_values("Condition", key=lambda x: x.map({v: i for i, v in enumerate(order)}), inplace=True)
                df_spots.sort_values("Condition", key=lambda x: x.map({v: i for i, v in enumerate(order)}), inplace=True)
                df_tracks.sort_values("Condition", key=lambda x: x.map({v: i for i, v in enumerate(order)}), inplace=True)
                df_frames.sort_values("Condition", key=lambda x: x.map({v: i for i, v in enumerate(order)}), inplace=True)

                UNFILTERED_SPOTSTATS.set(df_unfiltered_spots); UNFILTERED_TRACKSTATS.set(df_unfiltered_tracks); UNFILTERED_FRAMESTATS.set(df_unfiltered_frames)
                SPOTSTATS.set(df_spots); TRACKSTATS.set(df_tracks); FRAMESTATS.set(df_frames)

    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _


    # _ _ _ _ ALREADY PROCESSED DATA INPUT _ _ _ _

    @reactive.Effect
    @reactive.event(input.already_processed_input)
    def load_processed_data():
        fileinfo = input.already_processed_input()
        try:
            df = DataLoader.GetDataFrame(fileinfo[0]["datapath"])

            UNFILTERED_SPOTSTATS.set(df)
            UNFILTERED_TRACKSTATS.set(Tracks(df))
            UNFILTERED_FRAMESTATS.set(Frames(df))
            SPOTSTATS.set(df)
            TRACKSTATS.set(Tracks(df))
            FRAMESTATS.set(Frames(df))

            THRESHOLDS.set({1: {"spots": UNFILTERED_SPOTSTATS.get(), "tracks": UNFILTERED_TRACKSTATS.get()}})
            ui.update_action_button(id="append_threshold", disabled=False)
            # (optional) show sidebar like in the raw path
            ui.update_sidebar(id="sidebar", show=True)

        except:
            pass

    
    # _ _ _ _ ALREADY PROCESSED DATA INPUT _ _ _ _

    @reactive.Effect
    @reactive.event(input.already_processed_input)
    def load_processed_data():
        fileinfo = input.already_processed_input()
        try:
            df = DataLoader.GetDataFrame(fileinfo[0]["datapath"])

            UNFILTERED_SPOTSTATS.set(df)
            UNFILTERED_TRACKSTATS.set(Tracks(df))
            UNFILTERED_FRAMESTATS.set(Frames(df))
            SPOTSTATS.set(df)
            TRACKSTATS.set(Tracks(df))
            FRAMESTATS.set(Frames(df))

            THRESHOLDS.set({1: {"spots": UNFILTERED_SPOTSTATS.get(), "tracks": UNFILTERED_TRACKSTATS.get()}})

            ui.update_action_button(id="append_threshold", disabled=False)
            
        except Exception as e:
            print(e)

    @reactive.extended_task
    async def loader2():
        with ui.Progress(min=0, max=20) as p:
            p.set(message="Initialization in progress")

            for i in range(1, 12):
                p.set(i, message="Initializing Peregrin...")
                await asyncio.sleep(0.12)
        pass

    @reactive.effect
    @reactive.event(input.already_processed_input, ignore_none=True)
    def initialize_loader2():
        return loader2()



    # _ _ _ _ RUNNING THE ANALYSIS _ _ _ _

    @reactive.Effect
    def enable_run_button():
        files_uploaded = [input[f"input_file{idx}"]() for idx in range(1, INPUTS.get()+1)]
        def is_busy(val):
            return isinstance(val, list) and len(val) > 0
        all_busy = all(is_busy(f) for f in files_uploaded)
        session.send_input_message("run", {"disabled": not all_busy})

    @reactive.Effect
    @reactive.event(input.run)
    def parsed_files():
        all_data = []

        for idx in range(1, INPUTS.get()+1):

            files = input[f"input_file{idx}"]()

            if input.auto_label():
                cond_label = files[0].get("name").split("#")[1] if len(files[0].get("name").split("#")) >= 2 else None #TODO: display an error if the file label is incorrect
            else:
                cond_label = input[f"condition_label{idx}"]()

            if not files:
                break

            for file_idx, fileinfo in enumerate(files, start=1):
                try:
                    df = DataLoader.GetDataFrame(fileinfo["datapath"])
                    print(df.head())
                    extracted = DataLoader.Extract(
                        df,
                        id_col=input.select_id(),
                        t_col=input.select_t(),
                        x_col=input.select_x(),
                        y_col=input.select_y(),
                        mirror_y=True,
                    )
                except: continue

                extracted["Condition"] = cond_label if cond_label else str(idx)
                extracted["Replicate"] = fileinfo.get("name").split("#")[2] if input.auto_label() and len(fileinfo.get("name").split("#")) >= 2 else str(file_idx)

                all_data.append(extracted)
                
        if all_data:
            all_data = pd.concat(all_data, axis=0)
            all_data["Time unit"] = Metrics.Units.TimeUnits.get(input.select_t_unit())
            RAWDATA.set(all_data)
            UNFILTERED_SPOTSTATS.set(Spots(RAWDATA.get()))
            UNFILTERED_TRACKSTATS.set(Tracks(RAWDATA.get()))
            UNFILTERED_FRAMESTATS.set(Frames(RAWDATA.get()))
            SPOTSTATS.set(UNFILTERED_SPOTSTATS.get())
            TRACKSTATS.set(UNFILTERED_TRACKSTATS.get())
            FRAMESTATS.set(UNFILTERED_FRAMESTATS.get())

            THRESHOLDS.set({1: {"spots": UNFILTERED_SPOTSTATS.get(), "tracks": UNFILTERED_TRACKSTATS.get()}})

            ui.update_sidebar(id="sidebar", show=True)
            ui.update_action_button(id="append_threshold", disabled=False)

        else:
            pass

    @reactive.extended_task
    async def loader1():
        with ui.Progress(min=0, max=12) as p:
            p.set(message="Initialization in progress")

            for i in range(0, 10):
                p.set(i, message="Initializing Peregrin...")
                await asyncio.sleep(0.04)
        pass

    @render.text
    @reactive.event(input.run, ignore_none=True)
    def initialize_loader1():
        return loader1()
        
    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _


    @reactive.Effect
    def _():

        # _ _ _ _ WRITE REPLICATE LABELS _ _ _ _

        @output(id="replicate_labels_inputs")
        @render.ui
        def replicate_labels_inputs():
            req(not UNFILTERED_SPOTSTATS.get().empty)
            
            replicates = sorted(UNFILTERED_SPOTSTATS.get()["Replicate"].unique())
            inputs = []
            for idx, rep in enumerate(replicates):
                inputs.append(
                    ui.input_text(
                        id=f"replicate_label{idx}",
                        label=None,
                        placeholder=f"Replicate {rep}"
                    )
                )
            return ui.div(*inputs)
    

        # _ _ _ _ SET REPLICATE COLORS _ _ _ _

        @output(id="replicate_colors_inputs")
        @render.ui
        def replicate_colors_inputs():

            req(not UNFILTERED_SPOTSTATS.get().empty)

            if "Replicate color" not in UNFILTERED_SPOTSTATS.get().columns:
                UNFILTERED_SPOTSTATS.get()["Replicate color"] = "#59a9d7"  # default color

            replicates = sorted(UNFILTERED_SPOTSTATS.get()["Replicate"].unique())
            items = []
            for idx, rep in enumerate(replicates):

                try:
                    value = UNFILTERED_SPOTSTATS.get().loc[UNFILTERED_SPOTSTATS.get()["Replicate"] == rep, "Replicate color"].iloc[0]
                except:
                    value = "#59a9d7"

                cid = f"replicate_color{idx}"
                items.append(
                    ui.div(
                        ui.tags.label(f"{rep}", **{"for": cid}),
                        ui.tags.input(
                            type="color",
                            id=cid,
                            value=value,
                            style="width:100%; height:2.2rem; padding:0; border:none;"
                        ),
                        ui.tags.script(
                            f"""
                            (function(){{
                            const el = document.getElementById('{cid}');
                            function send() {{
                                Shiny.setInputValue('{cid}', el.value, {{priority:'event'}});
                            }}
                            el.addEventListener('input', send);
                            // push initial so input.{cid}() is defined immediately
                            send();
                            }})();
                            """
                        ),
                        style="margin-bottom: 0.5rem;"
                    )
                )
            return ui.div(*items)

    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _


    # _ _ _ _ SET UNITS _ _ _ _

    @reactive.Effect
    def set_units():
        UNITS.set(Metrics.Units.SetUnits(t=Metrics.Units.TimeUnits.get(input.select_t_unit())))

    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _


    # _ _ _ _ SIDEBAR ACCORDION THRESHOLDS LAYOUT  _ _ _ _

    @output()
    @render.ui
    def sidebar_accordion_placeholder():
        return ui.accordion(
            ui.accordion_panel(
                "Settings",
                ui.input_numeric("bins", "Number of bins", value=25, min=1, step=1),
                ui.markdown("<p style='line-height:0.1;'> <br> </p>"),
            ),
            ui.accordion_panel(
                f"Threshold 1",
                ui.panel_well(
                    ui.input_selectize(f"threshold_property_1", "Property", choices=Metrics.Thresholding.Properties),
                    ui.input_selectize(f"threshold_type_1", "Threshold type", choices=Modes.Thresholding),
                    ui.panel_conditional(
                        f"input.threshold_type_1 == 'Quantile'",
                        ui.input_selectize(f"threshold_quantile_1", "Quantile", choices=[200, 100, 50, 25, 20, 10, 5, 4, 2], selected=100),
                    ),
                    ui.panel_conditional(
                        f"input.threshold_type_1 == 'Relative to...'",
                        ui.input_selectize(f"reference_value_1", "Reference value", choices=["Mean", "Median", "My own value"]),
                        ui.panel_conditional(
                            f"input.reference_value_1 == 'My own value'",
                            ui.input_numeric(f"my_own_value_1", "My own value", value=0, step=1)
                        ),
                    ),
                    ui.output_ui(f"manual_threshold_value_setting_placeholder_1"),
                    ui.output_ui(f"threshold_slider_placeholder_1"),
                    ui.output_plot(f"thresholding_histogram_placeholder_1"),
                ),
            ),
            id="threshold_accordion",
            open="Threshold 1",
        )
    
    def render_threshold_accordion_panel(id):
        return ui.accordion_panel(
            f"Threshold {id}",
            ui.panel_well(
                ui.input_selectize(f"threshold_property_{id}", "Property", choices=Metrics.Thresholding.Properties),
                ui.input_selectize(f"threshold_type_{id}", "Threshold type", choices=Modes.Thresholding),
                ui.panel_conditional(
                    f"input.threshold_type_{id} == 'Quantile'",
                    ui.input_selectize(f"threshold_quantile_{id}", "Quantile", choices=[200, 100, 50, 25, 20, 10, 5, 4, 2], selected=100),
                ),
                ui.panel_conditional(
                    f"input.threshold_type_{id} == 'Relative to...'",
                    ui.input_selectize(f"reference_value_{id}", "Reference value", choices=["Mean", "Median", "My own value"]),
                    ui.panel_conditional(
                        f"input.reference_value_{id} == 'My own value'",
                        ui.input_numeric(f"my_own_value_{id}", "My own value", value=0, step=1)
                    )
                ),
                ui.output_ui(f"manual_threshold_value_setting_placeholder_{id}"),
                ui.output_ui(f"threshold_slider_placeholder_{id}"),
                ui.output_plot(f"thresholding_histogram_placeholder_{id}"),
            )
        )

    
    @reactive.Effect
    @reactive.event(input.append_threshold)
    def append_threshold():
        thresholds = THRESHOLDS.get()
        if not thresholds:
            return

        THRESHOLDS_ID.set(THRESHOLDS_ID.get() + 1)
        thresholds |= {THRESHOLDS_ID.get(): thresholds.get(THRESHOLDS_ID.get() - 1)}
        THRESHOLDS.set(thresholds)

        if THRESHOLDS_ID.get() > 1:
            session.send_input_message("remove_threshold", {"disabled": False})

        ui.insert_accordion_panel(
            id="threshold_accordion",
            panel=render_threshold_accordion_panel(THRESHOLDS_ID.get()),
            position="after"
        ) 

    @reactive.Effect
    @reactive.event(input.remove_threshold)
    def remove_threshold():
        thresholds = THRESHOLDS.get()

        if (THRESHOLDS_ID.get() - 1) <= 1:
            session.send_input_message("remove_threshold", {"disabled": True})

        ui.remove_accordion_panel(
            id="threshold_accordion",
            target=f"Threshold {THRESHOLDS_ID.get()}"
        )

        del thresholds[THRESHOLDS_ID.get()]
        THRESHOLDS.set(thresholds)

        THRESHOLDS_ID.set(THRESHOLDS_ID.get() - 1)

    @reactive.Effect
    @reactive.event(input.run, input.already_processed_input)
    def refresh_sidebar():
        thresholds = THRESHOLDS.get()
        if not thresholds:
            return
        for id in range(1, THRESHOLDS_ID.get() + 1):
            if id not in list(thresholds.keys()):
                ui.remove_accordion_panel(
                    id="threshold_accordion",
                    target=f"Threshold {id}"
                )
        session.send_input_message("remove_threshold", {"disabled": True})
        THRESHOLDS_ID.set(1)
        
        

    # _ _ _ _ Thresholding helper functions _ _ _ _

    EPS = 1e-12

    def _nearly_equal_pair(a, b, eps=EPS):
        try:
            return abs(float(a[0]) - float(b[0])) <= eps and abs(float(a[1]) - float(b[1])) <= eps
        except Exception:
            return False

    def _is_whole_number(x) -> bool:
        try:
            fx = float(x)
        except Exception:
            return False
        return abs(fx - round(fx)) < EPS

    def _int_if_whole(x):
        # Return an int if x is effectively whole, otherwise return float
        if x is None:
            return None
        try:
            fx = float(x)
        except Exception:
            return x
        if _is_whole_number(fx):
            return int(round(fx))
        return fx

    def _format_numeric_pair(values):
        """
        Normalize `values` into a (low, high) numeric pair.

        Accepts:
        - scalar numbers (including numpy.float64) -> returns (v, v)
        - 1-length iterables -> returns (v, v)
        - 2+-length iterables -> returns (first, second) but ensures low<=high where possible
        - None or empty -> (None, None)
        """

        if values is None:
            return None, None

        # numpy scalar or python scalar
        if np.isscalar(values):
            v = values.item() if hasattr(values, "item") else float(values)
            v = _int_if_whole(v)
            return v, v

        # Try to coerce to list/sequence
        try:
            seq = list(values)
        except Exception:
            # Fallback: treat as scalar
            try:
                v = float(values)
                v = _int_if_whole(v)
                return v, v
            except Exception:
                return None, None

        if len(seq) == 0:
            return None, None
        if len(seq) == 1:
            v = seq[0]
            try:
                fv = float(v)
                fv = _int_if_whole(fv)
                return fv, fv
            except Exception:
                return v, v

        # len >= 2 -> take first two and try to ensure lo <= hi
        a, b = seq[0], seq[1]
        try:
            fa = float(a)
            fb = float(b)
            if fa <= fb:
                return _int_if_whole(fa), _int_if_whole(fb)
            else:
                return _int_if_whole(fb), _int_if_whole(fa)
        except Exception:
            return _int_if_whole(a), _int_if_whole(b)
    
    def _get_steps(highest):
        """
        Returns the step size for the slider based on the range.
        """
        if highest < 0.01:
            steps = 0.0001
        elif 0.01 <= highest < 0.1:
            steps = 0.001
        elif 0.1 <= highest < 1:
            steps = 0.01
        elif 1 <= highest < 10:
            steps = 0.1
        elif 10 <= highest < 1000:
            steps = 1
        elif 1000 <= highest < 100000:
            steps = 10
        elif 100000 < highest:
            steps = 100
        else:
            steps = 1
        return steps

    def _compute_reference_and_span(values_series: pd.Series, reference: str, my_value: float | None):
        """
        Returns (reference_value, max_delta) for the 'Relative to...' mode.
        max_delta is the farthest absolute distance from reference to any data point.
        """
        vals = values_series.dropna()
        if vals.empty:
            return 0.0, 0.0

        if reference == "Mean":
            ref = float(vals.mean())
        elif reference == "Median":
            ref = float(vals.median())
        elif reference == "My own value":
            ref = float(my_value) if isinstance(my_value, (int, float)) else 0.0
        else:
            ref = float(vals.mean())

        max_delta = float(np.max(np.abs(vals - ref)))
        return ref, max_delta

    def _get_threshold_value_params(
        spot_data: pd.DataFrame, 
        track_data: pd.DataFrame, 
        property_name: str, 
        threshold_type: str, 
        quantile: int = None,
        reference: str = None,
        reference_value: float = None
    ):
        
        if threshold_type == "Literal":
            if property_name in Metrics.Thresholding.SpotProperties:
                minimal = spot_data[property_name].min()
                maximal = spot_data[property_name].max()
            elif property_name in Metrics.Thresholding.TrackProperties:
                minimal = track_data[property_name].min()
                maximal = track_data[property_name].max()
            else:
                minimal, maximal = 0, 100

            steps = _get_steps(maximal)
            minimal, maximal = floor(minimal), ceil(maximal)

        elif threshold_type == "Normalized 0-1":
            minimal, maximal = 0, 1
            steps = 0.01

        elif threshold_type == "Quantile":
            minimal, maximal = 0, 100
            steps = 100/float(quantile)

        elif threshold_type == "Relative to...":
            if property_name in Metrics.Thresholding.SpotProperties:
                series = spot_data[property_name]
            elif property_name in Metrics.Thresholding.TrackProperties:
                series = track_data[property_name]
            else:
                series = pd.Series(dtype=float)

            # Compute reference and span
            reference_value, max_delta = _compute_reference_and_span(series, reference, reference_value)

            minimal = 0
            maximal = ceil(max_delta) if np.isfinite(max_delta) else 0
            steps = _get_steps(maximal)

        return minimal, maximal, steps, reference_value

    def filter_data(df, threshold: tuple, property: str, threshold_type: str, reference: str = None, reference_value: float = None):
        if df is None or df.empty:
            return df
        
        try:
            working_df = df[property].dropna()
        except Exception:
            return df
        
        _floor, _roof = threshold
        if (
            _floor is None or _roof is None
            or not isinstance(_floor, (int, float)) or not isinstance(_roof, (int, float))
        ):
            return working_df

        if threshold_type == "Literal":
            return working_df[(working_df >= _floor) & (working_df <= _roof)]

        elif threshold_type == "Normalized 0-1":
            normalized = Normalize_01(df, property)
            return normalized[(normalized >= _floor) & (normalized <= _roof)]

        elif threshold_type == "Quantile":
            
            q_floor, q_roof = _floor / 100, _roof / 100
            if not 0 <= q_floor <= 1 or not 0 <= q_roof <= 1:
                q_floor, q_roof = 0, 1

            lower_bound = np.quantile(working_df, q_floor)
            upper_bound = np.quantile(working_df, q_roof)
            return working_df[(working_df >= lower_bound) & (working_df <= upper_bound)]

        elif threshold_type == "Relative to...":
            # req(reference is not None)
            if reference is None:
                reference = 0.0
            ref, _ = _compute_reference_and_span(working_df, reference, reference_value)

            # print(f"Reference value: {ref}, Floor: {ref + _floor}, Roof: {ref + _roof}, -Floor: {ref - _floor}, -Roof: {ref - _roof}")

            return working_df[
                (working_df >= (ref + _floor)) 
                & (working_df <= (ref + _roof))
                | (working_df <= (ref - _floor)) 
                & (working_df >= (ref - _roof))    
            ]

        return df


    # _ _ _ _ INITIALIZING THRESHOLD CONTAINERS _ _ _ _

    @Debounce(1)
    @reactive.Calc
    def get_bins():
        return input.bins() if input.bins() is not None and input.bins() != 0 else 25

    def render_threshold_container(id, thresholds):
        
        @output(id=f"manual_threshold_value_setting_placeholder_{id}")
        @render.ui
        def manual_threshold_value_setting():
            
            data = thresholds.get(id)
            req(data is not None and data.get("spots") is not None and data.get("tracks") is not None)

            spot_data = data.get("spots")
            track_data = data.get("tracks")

            property_name = input[f"threshold_property_{id}"]()
            threshold_type = input[f"threshold_type_{id}"]()
            req(property_name and threshold_type)
            
            minimal, maximal, steps, _ = _get_threshold_value_params(
                spot_data=spot_data,
                track_data=track_data,
                property_name=property_name,
                threshold_type=threshold_type,
                quantile=input[f"threshold_quantile_{id}"](),
                reference=input[f"reference_value_{id}"](),
                reference_value=input[f"my_own_value_{id}"]()
            )
            
            v_lo, v_hi = _format_numeric_pair((minimal,maximal))
            min_fmt, max_fmt = _int_if_whole(minimal), _int_if_whole(maximal)

            return ui.row(
                ui.column(6, ui.input_numeric(
                    f"floor_threshold_value_{id}",
                    label="min",
                    value=v_lo,
                    min=min_fmt,
                    max=max_fmt,
                    step=steps
                )),
                ui.column(6, ui.input_numeric(
                    f"ceil_threshold_value_{id}",
                    label="max",
                    value=v_hi,
                    min=min_fmt,
                    max=max_fmt,
                    step=steps
                )),
            )

        @output(id=f"threshold_slider_placeholder_{id}")
        @render.ui
        def threshold_slider():
            
            data = thresholds.get(id)
            req(data is not None and data.get("spots") is not None and data.get("tracks") is not None)

            spot_data = data.get("spots")
            track_data = data.get("tracks")

            property_name = input[f"threshold_property_{id}"]()
            threshold_type = input[f"threshold_type_{id}"]()
            req(property_name and threshold_type)
            
            minimal, maximal, steps, _ = _get_threshold_value_params(
                spot_data=spot_data,
                track_data=track_data,
                property_name=property_name,
                threshold_type=threshold_type,
                quantile=input[f"threshold_quantile_{id}"](),
                reference=input[f"reference_value_{id}"](),
                reference_value=input[f"my_own_value_{id}"]()
            )

            return ui.input_slider(
                f"threshold_slider_{id}",
                label=None,
                min=minimal,
                max=maximal,
                value=(minimal,maximal),
                step=steps
            )

        @output(id=f"thresholding_histogram_placeholder_{id}")
        @render.plot
        def threshold_histogram():
            data = thresholds.get(id)
            req(data is not None and data.get("spots") is not None and data.get("tracks") is not None)

            if input[f"threshold_property_{id}"]() in Metrics.Thresholding.SpotProperties:
                data = data.get("spots")
            if input[f"threshold_property_{id}"]() in Metrics.Thresholding.TrackProperties:
                data = data.get("tracks")
            if data is None or data.empty:
                return
            
            property = input[f"threshold_property_{id}"]()
            threshold_type = input[f"threshold_type_{id}"]()
            try:
                slider_low_pct, slider_high_pct = input[f"threshold_slider_{id}"]()
            except Exception:
                return

            if threshold_type == "Literal":

                bins = get_bins()
                values = data[property].dropna()

                fig, ax = plt.subplots()
                n, bins, patches = ax.hist(values, bins=bins, density=False)

                # Color threshold
                for i in range(len(patches)):
                    if bins[i] < slider_low_pct or bins[i+1] > slider_high_pct:
                        patches[i].set_facecolor("grey")
                    else:
                        patches[i].set_facecolor("#337ab7")

                # Add KDE curve (scaled to match histogram)
                kde = gaussian_kde(values)
                x_kde = np.linspace(bins[0], bins[-1], 500)
                y_kde = kde(x_kde)
                # Scale KDE to histogram
                y_kde_scaled = y_kde * (n.max() / y_kde.max())
                ax.plot(x_kde, y_kde_scaled, color="black", linewidth=1.5)

                ax.set_xticks([])  # Remove x-axis ticks
                ax.set_yticks([])  # Remove y-axis ticks
                ax.spines[["top", "left", "right"]].set_visible(False)

                return fig
            
            if threshold_type == "Normalized 0-1":

                values = data[property].dropna()
                try:
                    normalized = (values - values.min()) / (values.max() - values.min())
                except ZeroDivisionError:
                    normalized = 0
                bins = input.bins() if input.bins() is not None else 25

                fig, ax = plt.subplots()
                n, bins, patches = ax.hist(normalized, bins=bins, density=False)

                # Color threshold
                for i in range(len(patches)):
                    if bins[i] < slider_low_pct or bins[i+1] > slider_high_pct:
                        patches[i].set_facecolor("grey")
                    else:
                        patches[i].set_facecolor("#337ab7")

                # Add KDE curve (scaled to match histogram)
                kde = gaussian_kde(normalized)
                x_kde = np.linspace(bins[0], bins[-1], 500)
                y_kde = kde(x_kde)
                # Scale KDE to histogram
                y_kde_scaled = y_kde * (n.max() / y_kde.max())
                ax.plot(x_kde, y_kde_scaled, color="black", linewidth=1.5)

                ax.set_xticks([])  # Remove x-axis ticks
                ax.set_yticks([])  # Remove y-axis ticks
                ax.spines[["top", "left", "right"]].set_visible(False)

                return fig

            if threshold_type == "Quantile":
                bins = input.bins() if input.bins() is not None else 25

                values = data[property].dropna()
                
                fig, ax = plt.subplots()
                n, bins, patches = ax.hist(values, bins=bins, density=False)

                # Get slider quantile values, 0-100 scale
                slider_low, slider_high = slider_low_pct / 100, slider_high_pct / 100

                if not 0 <= slider_low <= 1 or not 0 <= slider_high <= 1:
                    slider_low, slider_high = 0, 1

                # Convert slider percentiles to actual values
                lower_bound = np.quantile(values, slider_low)
                upper_bound = np.quantile(values, slider_high)

                # Color histogram based on slider quantile bounds
                for i in range(len(patches)):
                    bin_start, bin_end = bins[i], bins[i + 1]
                    if bin_end < lower_bound or bin_start > upper_bound:
                        patches[i].set_facecolor("grey")
                    else:
                        patches[i].set_facecolor("#337ab7")

                # KDE curve
                kde = gaussian_kde(values)
                x_kde = np.linspace(values.min(), values.max(), 500)
                y_kde = kde(x_kde)
                y_kde_scaled = y_kde * (n.max() / y_kde.max()) if y_kde.max() != 0 else y_kde
                ax.plot(x_kde, y_kde_scaled, color="black", linewidth=1.5)

                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines[["top", "left", "right"]].set_visible(False)
                return fig

            if threshold_type == "Relative to...":
                reference = input[f"reference_value_{id}"]()
                if reference == "Mean":
                    reference_value = float(data[property].dropna().mean())
                elif reference == "Median":
                    reference_value = float(data[property].dropna().median())
                elif reference == "My own value":
                    try:
                        mv = input[f"my_own_value_{id}"]() if input[f"my_own_value_{id}"]() is not None else 0.0
                        reference_value = float(mv) if isinstance(mv, (int, float)) else 0.0
                    except Exception:
                        reference_value = 0.0
                else:
                    return

                # Build histogram in "shifted" space (centered at 0 = reference)
                shifted = data[property].dropna() - reference_value
                bins = input.bins() if input.bins() is not None else 25

                fig, ax = plt.subplots()
                n, bins, patches = ax.hist(shifted, bins=bins, density=False)

                # Slider gives distances [low, high] away from the reference
                sel_low, sel_high = input[f"threshold_slider_{id}"]()
                # Normalize order just in case
                if sel_low > sel_high:
                    sel_low, sel_high = sel_high, sel_low

                # Utility: does [bin_start, bin_end] intersect either [-sel_high, -sel_low] or [sel_low, sel_high]?
                def _intersects_symmetric(b0, b1, a, b):
                    # interval A: [-b, -a], interval B: [a, b]
                    left_hit  = (b1 >= -b) and (b0 <= -a)
                    right_hit = (b1 >=  a) and (b0 <=  b)
                    return left_hit or right_hit

                # Color threshold bands: keep bars whose centers fall within the selected annulus
                for i in range(len(patches)):
                    bin_start, bin_end = bins[i], bins[i+1]
                    if _intersects_symmetric(bin_start, bin_end, sel_low, sel_high):
                        patches[i].set_facecolor("#337ab7")
                    else:
                        patches[i].set_facecolor("grey")

                # KDE on shifted values (optional but matches your style)
                kde = gaussian_kde(shifted)
                x_kde = np.linspace(bins[0], bins[-1], 500)
                y_kde = kde(x_kde)
                y_kde_scaled = y_kde * (n.max() / y_kde.max()) if y_kde.max() != 0 else y_kde
                ax.plot(x_kde, y_kde_scaled, color="black", linewidth=1.5)

                ax.axvline(0, linestyle="--", linewidth=1, color="black")


                ax.set_xticks([]); ax.set_yticks([])
                ax.spines[["top", "left", "right"]].set_visible(False)
                return fig

    @reactive.Effect
    @reactive.event(input.append_threshold, UNFILTERED_SPOTSTATS, UNFILTERED_TRACKSTATS)
    def render_threshold():
        threshold_id = THRESHOLDS_ID.get()
        with reactive.isolate():
            try:
                thresholds = THRESHOLDS.get()
            except Exception:
                return
        if not threshold_id or not thresholds:
            return

        render_threshold_container(threshold_id, thresholds)


    # _ _ _ _ SYNCING SLIDER/MANUAL-SETTING VALUES _ _ _ _

    def sync_threshold_values(id):

        @Debounce(50)
        @reactive.Effect
        @reactive.event(
            input[f"floor_threshold_value_{id}"],
            input[f"ceil_threshold_value_{id}"],
            )
        def sync_with_manual_threshold_value_setting():
            
            # Read without creating extra reactive deps
            with reactive.isolate():
                try:
                    slider_vals = input[f"threshold_slider_{id}"]()
                except Exception:
                    slider_vals = (None, None)
                try:
                    cur_floor = input[f"floor_threshold_value_{id}"]()
                    cur_ceil  = input[f"ceil_threshold_value_{id}"]()
                except Exception:
                    return

                # Validate + normalize
                if not isinstance(cur_floor, (int, float)) or not isinstance(cur_ceil, (int, float)):
                    return
                if cur_floor > cur_ceil:
                    cur_floor, cur_ceil = cur_ceil, cur_floor

                # Only push if changed
                existing = slider_vals if (isinstance(slider_vals, (tuple, list)) and len(slider_vals) == 2) else (None, None)
                if cur_floor != existing[0] or cur_ceil != existing[1]:
                    ui.update_slider(f"threshold_slider_{id}", value=(cur_floor, cur_ceil))

        @Debounce(50)
        @reactive.Effect
        @reactive.event(input[f"threshold_slider_{id}"])
        def sync_with_threshold_slider():
            
            # Read without creating extra reactive deps
            with reactive.isolate():
                try:
                    cur_floor = input[f"floor_threshold_value_{id}"]()
                    cur_ceil  = input[f"ceil_threshold_value_{id}"]()
                except Exception:
                    return
                try:
                    slider_vals = input[f"threshold_slider_{id}"]()
                except Exception:
                    slider_vals = (None, None)

                # Validate + normalize
                if not isinstance(slider_vals[0], (int, float)) or not isinstance(slider_vals[1], (int, float)):
                    return
                if slider_vals[0] > slider_vals[1]:
                    slider_vals = (slider_vals[1], slider_vals[0])


                # Only push if changed
                existing = (cur_floor, cur_ceil) if (isinstance(cur_floor, (int, float)) and isinstance(cur_ceil, (int, float))) else (None, None)
                if slider_vals != existing:
                    ui.update_numeric(f"floor_threshold_value_{id}", value=float(slider_vals[0]))
                    ui.update_numeric(f"ceil_threshold_value_{id}", value=float(slider_vals[1]))

    @Throttle(50)
    @reactive.Effect
    def sync_thresholds():
        for id in range(1, THRESHOLDS_ID.get()+1):
            sync_threshold_values(id)


    # _ _ _ _ UPDATING THRESHOLDS ON CHANGE _ _ _ _
    
    def update_thresholds_wired(id):

        @Debounce(50)
        @reactive.Effect
        @reactive.event(input[f"threshold_slider_{id}"])
        def pass_thresholded_data():
            thresholds = THRESHOLDS.get()

            data = thresholds.get(id)
            req(data is not None and data.get("spots") is not None and data.get("tracks") is not None)

            spot_data = data.get("spots")
            track_data = data.get("tracks")

            filter = filter_data(
                df=spot_data if input[f"threshold_property_{id}"]() in Metrics.Thresholding.SpotProperties else track_data,
                threshold=input[f"threshold_slider_{id}"](),
                property=input[f"threshold_property_{id}"](),
                threshold_type=input[f"threshold_type_{id}"](),
                reference=input[f"reference_value_{id}"](),
                reference_value=input[f"my_own_value_{id}"]()
            )

            spots_output = spot_data.loc[filter.index.intersection(spot_data.index)]
            tracks_output = track_data.loc[filter.index.intersection(track_data.index)]

            # print(f"Spots after thresholding: {len(spots_output)}")
            # print(f"Tracks after thresholding: {len(tracks_output)}")

            thresholds |= {id+1: {"spots": spots_output, "tracks": tracks_output}}
            THRESHOLDS.set(thresholds)

        @Debounce(50)
        @reactive.Effect
        @reactive.event(input[f"threshold_slider_{id}"])
        def update_next_threshold():
            """
            Updating the slider updates the manual threshold values setting as well as the filte histogram.
            """
            thresholds = THRESHOLDS.get()
            
            try:
                data = thresholds.get(id+1)
            except Exception:
                return
            req(data is not None and data.get("spots") is not None and data.get("tracks") is not None)

            spot_data = data.get("spots")
            track_data = data.get("tracks")

            property_name = input[f"threshold_property_{id+1}"]()
            threshold_type = input[f"threshold_type_{id+1}"]()
            req(property_name and threshold_type)
            
            minimal, maximal, steps, _ = _get_threshold_value_params(
                spot_data=spot_data,
                track_data=track_data,
                property_name=property_name,
                threshold_type=threshold_type,
                quantile=input[f"threshold_quantile_{id+1}"](),
                reference=input[f"reference_value_{id+1}"](),
                reference_value=input[f"my_own_value_{id+1}"]()
            )

            return ui.update_slider(
                f"threshold_slider_{id+1}",
                min=minimal,
                max=maximal,
                value=(minimal,maximal),
                step=steps
            )   

    @Debounce(100)
    @reactive.Effect
    def update_thresholds():
        try:
            thresholds = THRESHOLDS.get()
        except Exception:
            return
        if not thresholds:
            return
        
        for id in range(1, THRESHOLDS_ID.get()+1):
            update_thresholds_wired(id)
    

    # _ _ _ _ SETTING THE THRESHOLDS _ _ _ _

    @reactive.Effect
    @reactive.event(input.set_threshold)
    def threshold_data():
        try: 
            thresholds = THRESHOLDS.get()
            latest = thresholds.get(list(thresholds.keys())[-1])
            req(latest is not None and latest.get("spots") is not None and latest.get("tracks") is not None)

        except Exception:
            return
        
        spots_filtered = pd.DataFrame(latest.get("spots") if latest is not None and isinstance(latest, dict) else UNFILTERED_SPOTSTATS.get())
        tracks_filtered = pd.DataFrame(latest.get("tracks") if latest is not None and isinstance(latest, dict) else UNFILTERED_TRACKSTATS.get())
        
        SPOTSTATS.set(spots_filtered)
        TRACKSTATS.set(tracks_filtered)
        FRAMESTATS.set(Frames(spots_filtered) if spots_filtered is not None and not spots_filtered.empty else UNFILTERED_FRAMESTATS.get())

















    # _ _ _ _ Filtered info display _ _ _ _

    @output()
    @render.ui
    def threshold_info():

        try:
            blocks = []
            thresholds = THRESHOLDS.get()

            # iterate deterministically if keys are integers
            for t in sorted(thresholds.keys()):
                if t > THRESHOLDS_ID.get():
                    break
                try:
                    t_state = thresholds.get(t)
                    t_state_after = thresholds.get(t + 1)
                    
                    data = len(t_state.get("tracks"))
                    data_after = len(t_state_after.get("tracks")) if t_state_after else data
                    out = data - data_after
                    out_percent = round(out / data * 100) if data else 0

                    prop = input[f"threshold_property_{t}"]()
                    ftype = input[f"threshold_type_{t}"]()
                    if ftype == "Relative to...":
                        ref = input[f"reference_value_{t}"]()
                        if ref == "My own value":
                            ref_val = input[f"my_own_value_{t}"]()
                        else:
                            ref_val = ref
                        reference = f"<br>Reference: <br><i><b>{ref}</b> (<b>{ref_val}</b>)</i><br>" if not isinstance(ref_val, str) else f"<br>Reference: <br><i><b>{ref}</b></i><br>"
                    else:
                        reference =  ""
                    vals = input[f"threshold_slider_{t}"]()

                except Exception:
                    break

                blocks.append(
                    ui.markdown(
                        f"""
                        <div style="height:5px;"></div>
                            <hr style="border:0; border-top:1px solid #000000; margin:8px 0;">
                        <div style="height:5px;"></div>
                        <p style="margin-bottom:8px; margin-top:10px;">
                            <b><h5>Threshold {t}</h5></b>
                            Filtered out: <br>
                            <i><b>{out}</b> (<b>{out_percent}%</b>)</i>
                        </p>
                        <p style="margin-bottom:8px; margin-top:0px;">
                            Property: <br>
                            <i><b>{prop}</b></i> <br>
                            Filter: <br>
                            <i><b>{ftype}</b></i> <br>
                            Range: <br>
                            <i><b>{vals[0]}</b> - <b>{vals[1]}</b></i>
                            {reference}
                        """
                    )
                )

        except Exception:
            pass

        total_tracks = len(UNFILTERED_TRACKSTATS.get())
        filtered_tracks = len(thresholds.get(THRESHOLDS_ID.get()+1).get("tracks")) if thresholds and thresholds.get(THRESHOLDS_ID.get()+1) else total_tracks

        filtered_tracks_percent = (
            round(filtered_tracks / total_tracks * 100) if total_tracks else 0
        )

        # --- Header + summary block
        blocks.insert(0,
            ui.markdown(
                f"""
                <p style="margin-bottom:0px; margin-top:0px;">
                    <h4> <b> Info </b> </h4>
                </p>
                <p style="margin-bottom:8px; margin-top:12px;">
                    Cells in total: <br>
                    <i><b>{total_tracks}</b> <br></i>
                </p>
                <p style="margin-bottom:8px; margin-top:0px;">
                    In focus: <br>
                    <i><b>{filtered_tracks}</b> (<b>{filtered_tracks_percent}%</b>)</i>
                </p>
                """
            )
        )

        # Return a single well with all blocks as children
        return ui.panel_well(*blocks)



    def GetInfoSVG(*, width: int = 190, txt_color: str = "#000000") -> str:
        """
        Build an SVG 'Info' panel using current Shiny reactives.
        Works for both 1D and 2D thresholding like in your filter_info().
        """
        
        # ---------- helpers ----------
        pad = 16
        title_size = 18
        body_size = 14
        line_gap = 8
        section_gap = 14
        rule_gap = 5
        rule_color = "#000000"
        font_family = "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif"

        def addy(y, inc):  # move the cursor
            return y + inc

        def tspan(text, cls=None):
            if cls:
                return f"<tspan class='{cls}'>{escape(str(text))}</tspan>"
            return f"<tspan>{escape(str(text))}</tspan>"

        # ---------- totals ----------
        total_tracks = len(UNFILTERED_TRACKSTATS.get())
        filtered_tracks = len(TRACKSTATS.get())
        if total_tracks < 0:
            return ""
        if filtered_tracks < 0:
            filtered_tracks = total_tracks
        percent = 0 if total_tracks == 0 else round((filtered_tracks / total_tracks) * 100)

        # ---------- SVG header (height placeholder) ----------
        x = pad
        y = pad + title_size
        parts = [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='__HEIGHT__' "
            f"viewBox='0 0 {width} __HEIGHT__' role='img' aria-label='Info panel'>",
            "<style>.bold{font-weight:700}.ital{font-style:italic}</style>",
            f"<text x='{x}' y='{y}' font-family='{font_family}' font-size='{title_size}' "
            f"font-weight='700' fill='{txt_color}'>Info</text>",
            f"<g font-family='{font_family}' font-size='{body_size}' fill='{txt_color}'>"
        ]

        # ---------- header body ----------
        y = addy(y, section_gap + body_size)
        parts.append(f"<text x='{x}' y='{y}'>Cells in total:</text>")
        y = addy(y, body_size + line_gap)
        parts.append(f"<text x='{x}' y='{y}'>{tspan(total_tracks,'bold')}</text>")

        y = addy(y, section_gap + body_size)
        parts.append(f"<text x='{x}' y='{y}'>In focus:</text>")
        y = addy(y, body_size + line_gap)
        parts.append(
            f"<text x='{x}' y='{y}'>{tspan(filtered_tracks,'bold')} {tspan(f'({percent}%)','bold')}</text>"
        )

        # ---------- thresholds (read reactives exactly like your UI) ----------
        try:
            thresholds = THRESHOLDS.get()
            for t in sorted(thresholds.keys()):
                if t > THRESHOLDS_ID.get():
                    break
                try:
                    t_state = thresholds.get(t)
                    t_state_after = thresholds.get(t + 1)

                    data = len(t_state.get("tracks"))
                    data_after = len(t_state_after.get("tracks")) if t_state_after else data
                    out = data - data_after
                    out_percent = round(out / data * 100) if data else 0

                    prop = input[f"threshold_property_{t}"]()
                    ftype = input[f"threshold_type_{t}"]()
                    if ftype == "Relative to...":
                        ref = input[f"reference_value_{t}"]()
                        if ref == "My own value":
                            ref_val = input[f"my_own_value_{t}"]()
                        else:
                            ref_val = ref
                        reference = f"{ref} ({ref_val})" if not isinstance(ref_val, str) else f"{ref}"
                    else:
                        reference = ""

                    vmin, vmax = input[f"threshold_slider_{t}"]()
                except Exception:
                    break

                # hr
                y = addy(y, rule_gap + section_gap)
                parts.append(f"<line x1='{pad}' x2='{width-pad}' y1='{y}' y2='{y}' stroke='{rule_color}' stroke-width='1'/>")
                y = addy(y, rule_gap)

                # threshold header
                y = addy(y, body_size + line_gap)
                parts.append(f"<text x='{x}' y='{y}'>{tspan(f'Threshold {t}','bold')}</text>")

                # filtered out
                y = addy(y, body_size + line_gap)
                parts.append(
                    f"<text x='{x}' y='{y}'>Filtered out: "
                    f"{tspan(out,'ital bold')} {tspan(f'({out_percent}%)','ital bold')}</text>"
                )

                # property / threshold / range / reference
                y = addy(y, body_size + section_gap)
                parts.append(f"<text x='{x}' y='{y}'>Property:</text>")
                y = addy(y, body_size)
                parts.append(f"<text x='{x}' y='{y}'>{tspan(prop,'bold ital')}</text>")

                y = addy(y, body_size + section_gap)
                parts.append(f"<text x='{x}' y='{y}'>Filter:</text>")
                y = addy(y, body_size)
                parts.append(f"<text x='{x}' y='{y}'>{tspan(ftype,'bold ital')}</text>")

                y = addy(y, body_size + section_gap)
                parts.append(f"<text x='{x}' y='{y}'>Range:</text>")
                y = addy(y, body_size)
                parts.append(
                    f"<text x='{x}' y='{y}'>{tspan(vmin,'bold ital')} - {tspan(vmax,'bold ital')}</text>"
                )

                if reference:
                    y = addy(y, body_size + section_gap)
                    parts.append(f"<text x='{x}' y='{y}'>Reference:</text>")
                    y = addy(y, body_size)
                    parts.append(f"<text x='{x}' y='{y}'>{tspan(reference,'bold ital')}</text>")


        except Exception:
            pass

        # ---------- close and set height ----------
        parts.append("</g></svg>")
        height = y + pad
        svg = "".join(parts).replace("__HEIGHT__", str(height))
        return svg


    @render.download(filename=f"Threshold Info {date.today()}.svg", media_type="svg")
    def download_threshold_info():
        svg = GetInfoSVG()
        yield svg.encode("utf-8")

        














    # _ _ _ _ RENDERING DATA FRAMES _ _ _ _
    
    @render.data_frame
    def render_spot_stats():
        spot_stats = SPOTSTATS.get()
        if spot_stats is not None and not spot_stats.empty:
            return spot_stats
        else:
            pass

    @render.data_frame
    def render_track_stats():
        track_stats = TRACKSTATS.get()
        if track_stats is not None and not track_stats.empty:
            return track_stats
        else:
            pass

    @render.data_frame
    def render_frame_stats():
        frame_stats = FRAMESTATS.get()
        if frame_stats is not None and not frame_stats.empty:
            return frame_stats
        else:
            pass
    

    # _ _ _ _ DATAFRAME CSV DOWNLOADS _ _ _ _

    @render.download(filename=f"Spot stats {date.today()}.csv")
    def download_spot_stats():
        spot_stats = SPOTSTATS.get()
        if spot_stats is not None and not spot_stats.empty:
            with io.BytesIO() as buffer:
                spot_stats.to_csv(buffer, index=False)
                yield buffer.getvalue()
        else:
            pass

    @render.download(filename=f"Track stats {date.today()}.csv")
    def download_track_stats():
        track_stats = TRACKSTATS.get()
        if track_stats is not None and not track_stats.empty:
            with io.BytesIO() as buffer:
                track_stats.to_csv(buffer, index=False)
                yield buffer.getvalue()
        else:
            pass
    
    @render.download(filename=f"Frame stats {date.today()}.csv")
    def download_frame_stats():
        frame_stats = FRAMESTATS.get()
        if frame_stats is not None and not frame_stats.empty:
            with io.BytesIO() as buffer:
                frame_stats.to_csv(buffer, index=False)
                yield buffer.getvalue()
        else:
            pass



    @output()
    @render.text
    def sidebar_label():
        return ui.markdown(
            f""" <h3> <b>  Data threshold  </b> </h3> """
        )
    


    
    # @output(id="sps_plot_card")
    # @render.ui
    # def plot_card():

    #     req(input.sp_fig_height() is not None and input.sp_fig_width() is not None)
    #     fig_height, fig_width = input.sp_fig_height() * 96, input.sp_fig_width() * 96
    
    #     return ui.card(
    #         ui.div(
    #             # Make the *plot image* larger than the panel so scrolling kicks in
    #             ui.output_plot("swarmplot", width=f"{fig_width}px", height=f"{fig_height}px"),
    #             # ui.output_plot(id="swarmplot"),
    #             style=f"height: {fig_height}px; width: {fig_width}px; margin: auto",
    #             # style=f"overflow: auto;",
    #             class_="scroll-panel",
    #         ),
    #         full_screen=True, fill=False
    #     ), 



    # ======================= DATA VISUALIZATION =======================

    @reactive.Effect
    def update_choices():
        if TRACKSTATS.get() is None or TRACKSTATS.get().empty:
            return
        ui.update_selectize(id="tracks_conditions", choices=TRACKSTATS.get()["Condition"].unique().tolist())
        ui.update_selectize(id="tracks_replicates", choices=["all"] + TRACKSTATS.get()["Replicate"].unique().tolist())

    @reactive.Effect
    def get_preset():
        preset = input.sp_preset()

        ui.update_checkbox(f"sp_show_swarms", value=True if preset in ["Bees", "Bees n Bass", "Bees n Bass n Bows"] else False)
        ui.update_checkbox(f"sp_show_violins", value=True if preset in ["Bass", "Bees n Bass", "Bass n Bows", "Bees n Bass n Bows"] else False)
        ui.update_checkbox(f"sp_show_kde", value=True if preset in ["Bass n Bows", "Bees n Bass n Bows"] else False)
        ui.update_checkbox(f"sp_show_cond_mean", value=True if preset in ["Bass", "Bass n Bows", "Bees n Bass n Bows"] else False)
        ui.update_checkbox(f"sp_show_cond_median", value=True if preset in ["Bass", "Bees n Bass", "Bass n Bows", "Bees n Bass n Bows"] else False)
        ui.update_checkbox(f"sp_show_errbars", value=True if preset in ["Bass", "Bass n Bows", "Bees n Bass n Bows"] else False)
        ui.update_checkbox(f"sp_show_rep_medians", value=True if preset in ["Bees", "Bees n Bass", "Bees n Bass n Bows"] else False)

        if preset == "Bees":
            ui.update_numeric(f"sp_swarm_marker_alpha", value=1)
            ui.update_numeric(f"sp_swarm_marker_size", value=2.5)
            ui.update_selectize(f"sp_palette", selected="Set2")
            ui.update_numeric(f"sp_median_bullet_size", value=90)
        if preset == "Bass":
            ui.update_numeric(f"sp_violin_alpha", value=1)
            ui.update_selectize(f"sp_palette", selected="Pastel1")
            ui.update_selectize(f"sp_violin_fill", selected="sienna")
            ui.update_selectize(f"sp_violin_outline", selected="black")
            ui.update_numeric(f"sp_violin_outline_width", value=1)
        if preset == "Bees n Bass":
            ui.update_numeric(f"sp_swarm_marker_size", value=2.5)
            ui.update_numeric(f"sp_violin_alpha", value=0.65)
            ui.update_selectize(f"sp_palette", selected="Set2")
            ui.update_selectize(f"sp_violin_fill", selected="whitesmoke")
            ui.update_selectize(f"sp_violin_outline", selected="lightgrey")
            ui.update_numeric(f"sp_violin_outline_width", value=1)
            ui.update_numeric(f"sp_median_bullet_size", value=70)
        if preset == "Bass n Bows":
            ui.update_numeric(f"sp_violin_alpha", value=1)
            ui.update_selectize(f"sp_palette", selected="Pastel2")
            ui.update_selectize(f"sp_violin_fill", selected="lightgrey")
            ui.update_selectize(f"sp_violin_outline", selected="dimgrey")
            ui.update_numeric(f"sp_violin_outline_width", value=1)
            ui.update_numeric(f"sp_kde_alpha", value=0.5)
            ui.update_numeric(f"sp_kde_line_width", value=0)
            ui.update_checkbox(f"sp_kde_fill", value=True)
        if preset == "Bees n Bass n Bows":
            ui.update_numeric(f"sp_swarm_marker_size", value=1.5)
            ui.update_numeric(f"sp_swarm_marker_alpha", value=0.75)
            ui.update_numeric(f"sp_violin_alpha", value=0.5)
            ui.update_selectize(f"sp_palette", selected="tab10")
            ui.update_selectize(f"sp_violin_fill", selected="whitesmoke")
            ui.update_selectize(f"sp_violin_outline", selected="lightgrey")
            ui.update_numeric(f"sp_violin_outline_width", value=1)
            ui.update_numeric(f"sp_kde_alpha", value=0.75)
            ui.update_numeric(f"sp_kde_line_width", value=1)
            ui.update_checkbox(f"sp_kde_fill", value=False)
            ui.update_numeric(f"sp_median_bullet_size", value=70)
            ui.update_numeric(f"sp_mean_bullet_size", value=50)


    # _ _ _ _ SWARMPPLOT _ _ _ _ 

    @ui.bind_task_button(button_id="sps_generate")
    @reactive.extended_task
    async def output_swarmplot(
        df,
        metric,
        title,
        palette,
        show_swarm,
        swarm_size,
        swarm_outline_color,
        swarm_alpha,
        show_violin,
        violin_fill_color,
        violin_edge_color,
        violin_alpha,
        violin_outline_width,
        show_mean,
        mean_span,
        mean_color,
        show_median,
        median_span,
        median_color,
        line_width,
        show_error_bars,
        errorbar_capsize,
        errorbar_color,
        errorbar_lw,
        errorbar_alpha,
        show_mean_balls,
        mean_ball_size,
        mean_ball_outline_color,
        mean_ball_outline_width,
        mean_ball_alpha,
        show_median_balls,
        median_ball_size,
        median_ball_outline_color,
        median_ball_outline_width,
        median_ball_alpha,
        show_kde,
        kde_inset_width,
        kde_outline,
        kde_alpha,
        kde_fill,
        show_legend,
        show_grid,
        open_spine,
        plot_width,
        plot_height
    ):
        # run sync plotting off the event loop
        def build():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )

                local_df = df.copy(deep=True) if df is not None else pd.DataFrame()
                return BeyondSwarms(
                    df=local_df,
                    metric=metric,
                    title=title,
                    palette=palette,
                    show_swarm=show_swarm,
                    swarm_size=swarm_size,
                    swarm_outline_color=swarm_outline_color,
                    swarm_alpha=swarm_alpha,
                    show_violin=show_violin,
                    violin_fill_color=violin_fill_color,
                    violin_edge_color=violin_edge_color,
                    violin_alpha=violin_alpha,
                    violin_outline_width=violin_outline_width,
                    show_mean=show_mean,
                    mean_span=mean_span,
                    mean_color=mean_color,
                    show_median=show_median,
                    median_span=median_span,
                    median_color=median_color,
                    line_width=line_width,
                    show_error_bars=show_error_bars,
                    errorbar_capsize=errorbar_capsize,
                    errorbar_color=errorbar_color,
                    errorbar_lw=errorbar_lw,
                    errorbar_alpha=errorbar_alpha,
                    show_mean_balls=show_mean_balls,
                    mean_ball_size=mean_ball_size,
                    mean_ball_outline_color=mean_ball_outline_color,
                    mean_ball_outline_width=mean_ball_outline_width,
                    mean_ball_alpha=mean_ball_alpha,
                    show_median_balls=show_median_balls,
                    median_ball_size=median_ball_size,
                    median_ball_outline_color=median_ball_outline_color,
                    median_ball_outline_width=median_ball_outline_width,
                    median_ball_alpha=median_ball_alpha,
                    show_kde=show_kde,
                    kde_inset_width=kde_inset_width,
                    kde_outline=kde_outline,
                    kde_alpha=kde_alpha,
                    kde_fill=kde_fill,
                    show_legend=show_legend,
                    show_grid=show_grid,
                    open_spine=open_spine,
                    plot_width=plot_width,
                    plot_height=plot_height,
                )

        # Either form is fine; pick one:
        return await asyncio.get_running_loop().run_in_executor(None, build)
        # return await asyncio.to_thread(build)
    

    @reactive.Effect
    @reactive.event(input.sps_generate, ignore_none=False)
    def trigger_swarmplot():

        @reactive.Effect
        @reactive.event(input.sps_generate, ignore_none=False)
        def _():
            @output(id="sps_plot_card")
            @render.ui
            def plot_card():

                with reactive.isolate():
                    req(input.sp_fig_height() is not None and input.sp_fig_width() is not None)
                    fig_height, fig_width = input.sp_fig_height() * 96, input.sp_fig_width() * 96
            
                return ui.card(
                    ui.div(
                        # Make the *plot image* larger than the panel so scrolling kicks in
                        ui.output_plot("swarmplot", width=f"{fig_width}px", height=f"{fig_height}px"),
                        # ui.output_plot(id="swarmplot"),
                        style=f"height: {fig_height}px; width: {fig_width}px; margin: auto",
                        # style=f"overflow: auto;",
                        class_="scroll-panel",
                    ),
                    full_screen=True, fill=False
                ), 


        @reactive.Effect
        @reactive.event(input.sps_generate, ignore_none=False)
        def make_swarmplot():
            output_swarmplot.cancel()

            output_swarmplot(
                df=TRACKSTATS.get() if TRACKSTATS.get() is not None else pd.DataFrame(),
                metric=input.sp_metric(),
                title=input.sp_title(), 
                palette=input.sp_palette(),

                show_swarm=input.sp_show_swarms(),
                swarm_size=input.sp_swarm_marker_size(),
                swarm_outline_color=input.sp_swarm_marker_outline(),
                swarm_alpha=input.sp_swarm_marker_alpha() if 0.0 <= input.sp_swarm_marker_alpha() <= 1.0 else 1.0,

                show_violin=input.sp_show_violins(),
                violin_fill_color=input.sp_violin_fill(),
                violin_edge_color=input.sp_violin_outline(),
                violin_alpha=input.sp_violin_alpha() if 0.0 <= input.sp_violin_alpha() <= 1.0 else 1.0,
                violin_outline_width=input.sp_violin_outline_width(),

                show_mean=input.sp_show_cond_mean(),
                mean_span=input.sp_mean_line_span(),
                mean_color=input.sp_mean_line_color(),
                show_median=input.sp_show_cond_median(),
                median_span=input.sp_median_line_span(),
                median_color=input.sp_median_line_color(),
                line_width=input.sp_lines_lw(),
                show_error_bars=input.sp_show_errbars(),
                errorbar_capsize=input.sp_errorbar_capsize(),
                errorbar_color=input.sp_errorbar_color(),
                errorbar_lw=input.sp_errorbar_lw(),
                errorbar_alpha=input.sp_errorbar_alpha() if 0.0 <= input.sp_errorbar_alpha() <= 1.0 else 1.0,

                show_mean_balls=input.sp_show_rep_means(),
                mean_ball_size=input.sp_mean_bullet_size(),
                mean_ball_outline_color=input.sp_mean_bullet_outline(),
                mean_ball_outline_width=input.sp_mean_bullet_outline_width(),
                mean_ball_alpha=input.sp_mean_bullet_alpha() if 0.0 <= input.sp_mean_bullet_alpha() <= 1.0 else 1.0,
                show_median_balls=input.sp_show_rep_medians(),
                median_ball_size=input.sp_median_bullet_size(),
                median_ball_outline_color=input.sp_median_bullet_outline(),
                median_ball_outline_width=input.sp_median_bullet_outline_width(),
                median_ball_alpha=input.sp_median_bullet_alpha() if 0.0 <= input.sp_median_bullet_alpha() <= 1.0 else 1.0,

                show_kde=input.sp_show_kde(),
                kde_inset_width=input.sp_kde_bandwidth(),
                kde_outline=input.sp_kde_line_width(),
                kde_alpha=input.sp_kde_fill_alpha() if 0.0 <= input.sp_kde_fill_alpha() <= 1.0 else 1.0,
                kde_fill=input.sp_kde_fill(),

                show_legend=input.sp_show_legend(),
                show_grid=input.sp_grid(),
                open_spine=input.sp_spine(),
                
                plot_width=input.sp_fig_width(),
                plot_height=input.sp_fig_height(),
            )

    # @output(id="swarmplot")
    @render.plot
    def swarmplot():
        # Only update when output_swarmplot task completes (not reactively)
        return output_swarmplot.result()

    @render.download(filename=f"Swarmplot {date.today()}.svg")
    def download_swarmplot_svg():
        if TRACKSTATS.get() is None or TRACKSTATS.get().empty: return
        fig = BeyondSwarms(
            df=TRACKSTATS.get(),
            metric=input.sp_metric(),
            title=input.sp_title(),
            palette=input.sp_palette(),

            show_swarm=input.sp_show_swarms(),
            swarm_size=input.sp_swarm_marker_size(),
            swarm_outline_color=input.sp_swarm_marker_outline(),
            swarm_alpha=input.sp_swarm_marker_alpha(),

            show_violin=input.sp_show_violins(),
            violin_fill_color=input.sp_violin_fill(),
            violin_edge_color=input.sp_violin_outline(),
            violin_alpha=input.sp_violin_alpha(),
            violin_outline_width=input.sp_violin_outline_width(),

            show_mean=input.sp_show_cond_mean(),
            mean_span=input.sp_mean_line_span(),
            mean_color=input.sp_mean_line_color(),
            show_median=input.sp_show_cond_median(),
            median_span=input.sp_median_line_span(),
            median_color=input.sp_median_line_color(),
            line_width=input.sp_lines_lw(),
            show_error_bars=input.sp_show_errbars(),
            errorbar_capsize=input.sp_errorbar_capsize(),
            errorbar_color=input.sp_errorbar_color(),
            errorbar_lw=input.sp_errorbar_lw(),
            errorbar_alpha=input.sp_errorbar_alpha(),

            show_mean_balls=input.sp_show_rep_means(),
            mean_ball_size=input.sp_mean_bullet_size(),
            mean_ball_outline_color=input.sp_mean_bullet_outline(),
            mean_ball_outline_width=input.sp_mean_bullet_outline_width(),
            mean_ball_alpha=input.sp_mean_bullet_alpha(),
            show_median_balls=input.sp_show_rep_medians(),
            median_ball_size=input.sp_median_bullet_size(),
            median_ball_outline_color=input.sp_median_bullet_outline(),
            median_ball_outline_width=input.sp_median_bullet_outline_width(),
            median_ball_alpha=input.sp_median_bullet_alpha(),

            show_kde=input.sp_show_kde(),
            kde_inset_width=input.sp_kde_bandwidth(),
            kde_outline=input.sp_kde_line_width(),
            kde_alpha=input.sp_kde_fill_alpha(),
            kde_fill=input.sp_kde_fill(),

            show_legend=input.sp_show_legend(),
            show_grid=input.sp_grid(),
            open_spine=input.sp_spine(),
            
            plot_width=input.sp_fig_width(),
            plot_height=input.sp_fig_height(),
        )
        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="svg", bbox_inches="tight")
                yield buffer.getvalue()


    
    # _ _ _ _ SUPERVIOLINPLOT _ _ _ _ 

    @ui.bind_task_button(button_id="sps_generate")
    @reactive.extended_task
    async def output_superviolinplot(
        df,
        metric,
        title,
        palette,
        show_swarm,
        swarm_size,
        swarm_outline_color,
        swarm_alpha,
        show_violin,
        violin_fill_color,
        violin_edge_color,
        violin_alpha,
        violin_outline_width,
        show_mean,
        mean_span,
        mean_color,
        show_median,
        median_span,
        median_color,
        line_width,
        show_error_bars,
        errorbar_capsize,
        errorbar_color,
        errorbar_lw,
        errorbar_alpha,
        show_mean_balls,
        mean_ball_size,
        mean_ball_outline_color,
        mean_ball_outline_width,
        mean_ball_alpha,
        show_median_balls,
        median_ball_size,
        median_ball_outline_color,
        median_ball_outline_width,
        median_ball_alpha,
        show_kde,
        kde_inset_width,
        kde_outline,
        kde_alpha,
        kde_fill,
        show_legend,
        show_grid,
        open_spine,
        plot_width,
        plot_height
    ):
        # run sync plotting off the event loop
        def build():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )

                return SuperViolins(
                    df=df,
                    metric=metric,
                    title=title,
                    units=UNITS.get() if UNITS.get() is not None else "",
                    centre_val="mean",
                    middle_vals="mean",
                    error_bars="SEM",
                    total_width=0.8,
                    outline_lw=1,
                    dataframe=False,
                    sep_lw=0,
                    bullet_lw=0.75,
                    errorbar_lw=1,
                    bullet_size=50,
                    palette="Accent",
                    use_my_colors=True,
                    show_legend=True,
                    plot_width=12,
                    plot_height=5
                )

        # Either form is fine; pick one:
        return await asyncio.get_running_loop().run_in_executor(None, build)
        # return await asyncio.to_thread(build)
    

    @reactive.Effect
    @reactive.event(input.sps_generate, ignore_none=False)
    def trigger_swarmplot():

        @reactive.Effect
        @reactive.event(input.sps_generate, ignore_none=False)
        def _():
            @output(id="sps_plot_card")
            @render.ui
            def plot_card():

                with reactive.isolate():
                    req(input.sp_fig_height() is not None and input.sp_fig_width() is not None)
                    fig_height, fig_width = input.sp_fig_height() * 96, input.sp_fig_width() * 96
            
                return ui.card(
                    ui.div(
                        # Make the *plot image* larger than the panel so scrolling kicks in
                        ui.output_plot("swarmplot", width=f"{fig_width}px", height=f"{fig_height}px"),
                        # ui.output_plot(id="swarmplot"),
                        style=f"height: {fig_height}px; width: {fig_width}px; margin: auto",
                        # style=f"overflow: auto;",
                        class_="scroll-panel",
                    ),
                    full_screen=True, fill=False
                ), 


        @reactive.Effect
        @reactive.event(input.sps_generate, ignore_none=False)
        def make_swarmplot():
            output_swarmplot.cancel()

            output_swarmplot(
                df=TRACKSTATS.get() if TRACKSTATS.get() is not None else pd.DataFrame(),
                metric=input.sp_metric(),
                title=input.sp_title(), 
                palette=input.sp_palette(),

                show_swarm=input.sp_show_swarms(),
                swarm_size=input.sp_swarm_marker_size(),
                swarm_outline_color=input.sp_swarm_marker_outline(),
                swarm_alpha=input.sp_swarm_marker_alpha() if 0.0 <= input.sp_swarm_marker_alpha() <= 1.0 else 1.0,

                show_violin=input.sp_show_violins(),
                violin_fill_color=input.sp_violin_fill(),
                violin_edge_color=input.sp_violin_outline(),
                violin_alpha=input.sp_violin_alpha() if 0.0 <= input.sp_violin_alpha() <= 1.0 else 1.0,
                violin_outline_width=input.sp_violin_outline_width(),

                show_mean=input.sp_show_cond_mean(),
                mean_span=input.sp_mean_line_span(),
                mean_color=input.sp_mean_line_color(),
                show_median=input.sp_show_cond_median(),
                median_span=input.sp_median_line_span(),
                median_color=input.sp_median_line_color(),
                line_width=input.sp_lines_lw(),
                show_error_bars=input.sp_show_errbars(),
                errorbar_capsize=input.sp_errorbar_capsize(),
                errorbar_color=input.sp_errorbar_color(),
                errorbar_lw=input.sp_errorbar_lw(),
                errorbar_alpha=input.sp_errorbar_alpha() if 0.0 <= input.sp_errorbar_alpha() <= 1.0 else 1.0,

                show_mean_balls=input.sp_show_rep_means(),
                mean_ball_size=input.sp_mean_bullet_size(),
                mean_ball_outline_color=input.sp_mean_bullet_outline(),
                mean_ball_outline_width=input.sp_mean_bullet_outline_width(),
                mean_ball_alpha=input.sp_mean_bullet_alpha() if 0.0 <= input.sp_mean_bullet_alpha() <= 1.0 else 1.0,
                show_median_balls=input.sp_show_rep_medians(),
                median_ball_size=input.sp_median_bullet_size(),
                median_ball_outline_color=input.sp_median_bullet_outline(),
                median_ball_outline_width=input.sp_median_bullet_outline_width(),
                median_ball_alpha=input.sp_median_bullet_alpha() if 0.0 <= input.sp_median_bullet_alpha() <= 1.0 else 1.0,

                show_kde=input.sp_show_kde(),
                kde_inset_width=input.sp_kde_bandwidth(),
                kde_outline=input.sp_kde_line_width(),
                kde_alpha=input.sp_kde_fill_alpha() if 0.0 <= input.sp_kde_fill_alpha() <= 1.0 else 1.0,
                kde_fill=input.sp_kde_fill(),

                show_legend=input.sp_show_legend(),
                show_grid=input.sp_grid(),
                open_spine=input.sp_spine(),
                
                plot_width=input.sp_fig_width(),
                plot_height=input.sp_fig_height(),
            )

    # @output(id="swarmplot")
    @render.plot
    def swarmplot():
        # Only update when output_swarmplot task completes (not reactively)
        return output_swarmplot.result()

    @render.download(filename=f"Swarmplot {date.today()}.svg")
    def download_swarmplot_svg():
        if TRACKSTATS.get() is None or TRACKSTATS.get().empty: return
        fig = BeyondSwarms.SwarmPlot(
            df=TRACKSTATS.get(),
            metric=input.sp_metric(),
            title=input.sp_title(),
            palette=input.sp_palette(),

            show_swarm=input.sp_show_swarms(),
            swarm_size=input.sp_swarm_marker_size(),
            swarm_outline_color=input.sp_swarm_marker_outline(),
            swarm_alpha=input.sp_swarm_marker_alpha(),

            show_violin=input.sp_show_violins(),
            violin_fill_color=input.sp_violin_fill(),
            violin_edge_color=input.sp_violin_outline(),
            violin_alpha=input.sp_violin_alpha(),
            violin_outline_width=input.sp_violin_outline_width(),

            show_mean=input.sp_show_cond_mean(),
            mean_span=input.sp_mean_line_span(),
            mean_color=input.sp_mean_line_color(),
            show_median=input.sp_show_cond_median(),
            median_span=input.sp_median_line_span(),
            median_color=input.sp_median_line_color(),
            line_width=input.sp_lines_lw(),
            show_error_bars=input.sp_show_errbars(),
            errorbar_capsize=input.sp_errorbar_capsize(),
            errorbar_color=input.sp_errorbar_color(),
            errorbar_lw=input.sp_errorbar_lw(),
            errorbar_alpha=input.sp_errorbar_alpha(),

            show_mean_balls=input.sp_show_rep_means(),
            mean_ball_size=input.sp_mean_bullet_size(),
            mean_ball_outline_color=input.sp_mean_bullet_outline(),
            mean_ball_outline_width=input.sp_mean_bullet_outline_width(),
            mean_ball_alpha=input.sp_mean_bullet_alpha(),
            show_median_balls=input.sp_show_rep_medians(),
            median_ball_size=input.sp_median_bullet_size(),
            median_ball_outline_color=input.sp_median_bullet_outline(),
            median_ball_outline_width=input.sp_median_bullet_outline_width(),
            median_ball_alpha=input.sp_median_bullet_alpha(),

            show_kde=input.sp_show_kde(),
            kde_inset_width=input.sp_kde_bandwidth(),
            kde_outline=input.sp_kde_line_width(),
            kde_alpha=input.sp_kde_fill_alpha(),
            kde_fill=input.sp_kde_fill(),

            show_legend=input.sp_show_legend(),
            show_grid=input.sp_grid(),
            open_spine=input.sp_spine(),
            
            plot_width=input.sp_fig_width(),
            plot_height=input.sp_fig_height(),
        )
        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="svg", bbox_inches="tight")
                yield buffer.getvalue()
    
    
    

    # _ _ _ _ TRACK VISUALIZATION _ _ _ _

    @ui.bind_task_button(button_id="trr_generate")
    @reactive.extended_task
    async def output_track_reconstruction_realistic(
        Spots_df,
        Tracks_df,
        condition,
        replicate,
        c_mode,
        only_one_color,
        lut_scaling_metric,
        background,
        smoothing_index,
        lw,
        grid,
        mark_heads,
        marker,
        markersize,
        title
    ):
        
        # run sync plotting off the event loop
        def _build():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )

                return VisualizeTracksRealistics(
                    Spots_df=Spots_df,
                    Tracks_df=Tracks_df,
                    condition=condition,
                    replicate=replicate,
                    c_mode=c_mode,
                    only_one_color=only_one_color,
                    lut_scaling_metric=lut_scaling_metric,
                    background=background,
                    smoothing_index=smoothing_index,
                    lw=lw,
                    grid=grid,
                    mark_heads=mark_heads,
                    marker=marker,
                    markersize=markersize,
                    title=title
                )

        # Either form is fine; pick one:
        return await asyncio.get_running_loop().run_in_executor(None, _build)
        # return await asyncio.to_thread(_build)
    
    @reactive.Effect
    @reactive.event(input.trr_generate, ignore_none=False)
    def _():
            
        output_track_reconstruction_realistic.cancel()

        req(
            SPOTSTATS.get() is not None and not SPOTSTATS.get().empty 
            and TRACKSTATS.get() is not None and not TRACKSTATS.get().empty
            and "Condition" in SPOTSTATS.get().columns and "Replicate" in SPOTSTATS.get().columns
        )

        output_track_reconstruction_realistic(
            Spots_df=SPOTSTATS.get(),
            Tracks_df=TRACKSTATS.get(),
            condition=input.tracks_conditions(),
            replicate=input.tracks_replicates(),
            c_mode=input.tracks_color_mode(),
            only_one_color=input.tracks_only_one_color(),
            lut_scaling_metric=input.tracks_lut_scaling_metric(),
            background=input.tracks_background(),
            smoothing_index=input.tracks_smoothing_index(),
            lw=input.tracks_line_width(),
            grid=input.tracks_show_grid(),
            mark_heads=input.tracks_mark_heads(),
            marker=Markers.TrackHeads.get(input.tracks_marker_type()),
            markersize=input.tracks_marks_size()*10,
            title=input.tracks_title()
        )

    @render.plot
    def track_reconstruction_realistic():
        return output_track_reconstruction_realistic.result()

    @render.download(filename=f"Realistic Track Reconstruction {date.today()}.svg")
    def trr_download():
        req(
            SPOTSTATS.get() is not None and not SPOTSTATS.get().empty 
            and TRACKSTATS.get() is not None and not TRACKSTATS.get().empty
            and "Condition" in SPOTSTATS.get().columns and "Replicate" in SPOTSTATS.get().columns
        )

        fig = VisualizeTracksRealistics(
            Spots_df=SPOTSTATS.get(),
            Tracks_df=TRACKSTATS.get(),
            condition=input.tracks_conditions(),
            replicate=input.tracks_replicates(),
            c_mode=input.tracks_color_mode(),
            only_one_color=input.tracks_only_one_color(),
            lut_scaling_metric=input.tracks_lut_scaling_metric(),
            background=input.tracks_background(),
            smoothing_index=input.tracks_smoothing_index(),
            lw=input.tracks_line_width(),
            grid=input.tracks_show_grid(),
            mark_heads=input.tracks_mark_heads(),
            marker=Markers.TrackHeads.get(input.tracks_marker_type()),
            markersize=input.tracks_marks_size()*10,
            title=input.tracks_title()
        )
        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="svg", bbox_inches="tight")
                yield buffer.getvalue()



    @ui.bind_task_button(button_id="tnr_generate")
    @reactive.extended_task
    async def output_track_reconstruction_normalized(
        Spots_df,
        Tracks_df,
        condition,
        replicate,
        c_mode,
        only_one_color,
        lut_scaling_metric,
        smoothing_index,
        lw,
        background,
        grid,
        grid_style,
        mark_heads,
        marker,
        markersize,
        title
    ):
        
        # run sync plotting off the event loop
        def _build():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )

                local_Spots_df = Spots_df.copy(deep=True) if Spots_df is not None else pd.DataFrame()
                local_Tracks_df = Tracks_df.copy(deep=True) if Tracks_df is not None else pd.DataFrame()
                return VisualizeTracksNormalized(
                    Spots_df=local_Spots_df,
                    Tracks_df=local_Tracks_df,
                    condition=local_Tracks_df["Condition"].unique().tolist()[0] if "Condition" in local_Tracks_df.columns else condition,
                    replicate=replicate,
                    c_mode=c_mode,
                    only_one_color=only_one_color,
                    lut_scaling_metric=lut_scaling_metric,
                    smoothing_index=smoothing_index,
                    lw=lw,
                    background=background,
                    grid=grid,
                    grid_style=grid_style,
                    mark_heads=mark_heads,
                    marker=marker,
                    markersize=markersize,
                    title=title
                )

        # Either form is fine; pick one:
        return await asyncio.get_running_loop().run_in_executor(None, _build)
        # return await asyncio.to_thread(build)

    @reactive.Effect
    @reactive.event(input.tnr_generate, ignore_none=False)
    def _():
            
        output_track_reconstruction_normalized.cancel()

        req(
            SPOTSTATS.get() is not None and not SPOTSTATS.get().empty 
            and TRACKSTATS.get() is not None and not TRACKSTATS.get().empty
            and "Condition" in SPOTSTATS.get().columns and "Replicate" in SPOTSTATS.get().columns
        )

        output_track_reconstruction_normalized(
            Spots_df=SPOTSTATS.get(),
            Tracks_df=TRACKSTATS.get(),
            condition=input.tracks_conditions(),
            replicate=input.tracks_replicates(),
            c_mode=input.tracks_color_mode(),
            only_one_color=input.tracks_only_one_color(),
            lut_scaling_metric=input.tracks_lut_scaling_metric(),
            smoothing_index=input.tracks_smoothing_index(),
            lw=input.tracks_line_width(),
            background=input.tracks_background(),
            grid=input.tracks_show_grid(),
            grid_style=input.tracks_grid_style(),
            mark_heads=input.tracks_mark_heads(),
            marker=Markers.TrackHeads.get(input.tracks_marker_type()),
            markersize=input.tracks_marks_size()*10,
            title=input.tracks_title()
        )

    @render.plot
    def track_reconstruction_normalized():
        return output_track_reconstruction_normalized.result()

    @render.download(filename=f"Normalized Track Reconstruction {date.today()}.svg")
    def tnr_download():
        req(
            SPOTSTATS.get() is not None and not SPOTSTATS.get().empty 
            and TRACKSTATS.get() is not None and not TRACKSTATS.get().empty
            and "Condition" in SPOTSTATS.get().columns and "Replicate" in SPOTSTATS.get().columns
        )

        fig = VisualizeTracksNormalized(
            Spots_df=SPOTSTATS.get(),
            Tracks_df=TRACKSTATS.get(),
            condition=input.tracks_conditions(),
            replicate=input.tracks_replicates(),
            c_mode=input.tracks_color_mode(),
            only_one_color=input.tracks_only_one_color(),
            lut_scaling_metric=input.tracks_lut_scaling_metric(),
            smoothing_index=input.tracks_smoothing_index(),
            lw=input.tracks_line_width(),
            background=input.tracks_background(),
            grid=input.tracks_show_grid(),
            grid_style=input.tracks_grid_style(),
            mark_heads=input.tracks_mark_heads(),
            marker=Markers.TrackHeads.get(input.tracks_marker_type()),
            markersize=input.tracks_marks_size()*10,
            title=input.tracks_title()
        )
        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="svg", bbox_inches="tight")
                yield buffer.getvalue()

    @render.download(filename=f"Lut Map {date.today()}.svg")
    def download_lut_map_svg():
        req(
            SPOTSTATS.get() is not None and not SPOTSTATS.get().empty 
            and TRACKSTATS.get() is not None and not TRACKSTATS.get().empty
            and "Condition" in SPOTSTATS.get().columns and "Replicate" in SPOTSTATS.get().columns
        )

        fig = GetLutMap(
            Spots_df=SPOTSTATS.get(),
            Tracks_df=TRACKSTATS.get(),
            c_mode=input.tracks_color_mode(),
            lut_scaling_metric=input.tracks_lut_scaling_metric(),
            units=UNITS.get(),
            _extend=input.tracks_lutmap_extend_edges()
        )
        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="svg", bbox_inches="tight")
                yield buffer.getvalue()
    
    
    

    
# _ _ _ _ _ APP LAUNCHER _ _ _ _ _
app = App(app_ui, server)

# TODO - Keep the Track UID column in the dataframes when downloaded

# TODO - Track visualization plot with a slider;
#      - metrics for lut scaling would include SPOTSTATS metrics:
#      - e.g. speed: track point = n;   assigned distance value = 15 (colors for each distance value assigned based on the min. max. value for the current dataset)
#                    track point = n+1; assigned distance value = 20
#                    the line between n and n+1 would be coloured inn a gradient from the color at distance value 15 to the color at the distance value 20


# TODO - define pre-sets for plot settings, so that the user can try out different looks easily
# TODO - Add a button to reset all thresholding settings to default

# TODO - Keep all the raw data (columns) - rather format them (stripping of _ and have them not all caps)
# TODO - Make the 2D filtering logic work on the same logic as does the D filtering logic
# TODO - Make it possible to save/load threshold configurations
# TODO - Find a way to program all the functions so that functions do not refresh/re-render unnecessarily on just any reactive action
# TODO - Time point definition
# TODO - Make it possible for the user to title their charts
# TODO - Mean directional change rate
# TODO - Select which p-tests should be shown in the superplot chart
# TODO - P-test
# TODO - Again add rendered text showing the total number of cells in the input and the number of output cells
# TODO - Option to download a simple legend showing how much data was filtered out and how so
# TODO - input_selectize("Plot:"... with options "Polar/Normalized" or "Cartesian/Raw"
# TODO - Differentiate between frame(s) annotations and time annotations

# TODO - Potentially add a nav tab for interacting with the data and set the formatting to be right and ready for processing

# TODO - VERY IMPORTANT
#      - Must have an option to download the whole app settings together with the data 

# TODO - add ui where the user can define the order of the conditions

# TODO - implement a log scale for thresholding. Ideally a logarithmic scale that:
#      - first checks whether the data has 0 values or negative values
#      - chooses the log calculation accordingly
#      - then sets an automated code which finds out the best possible setting of the log scale function so that after it is applied to the data, its distribution always ends up in a normal gaussian distributian
