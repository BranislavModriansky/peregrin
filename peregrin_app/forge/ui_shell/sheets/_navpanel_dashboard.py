from shiny import ui



card_spots = ui.card(
    ui.card_header(
        ui.div(
            ui.span("Spot stats", style="font-size:18px; white-space:nowrap;"), 
            ui.div(
                ui.div("Max decimals:", style="font-weight: normal; white-space:nowrap;"),
                ui.input_numeric(
                    id="decimal_places",
                    label=None,
                    value=5,
                    min=0,
                    step=1,
                    width="60px",
                    update_on="blur"
                ),
                ui.div("Significant figures:", style="margin-left:12px; font-weight: normal; white-space:nowrap;"),
                ui.input_numeric(
                    id="significant_figures",
                    label=None,
                    value=5,
                    min=1,
                    step=1,
                    width="60px",
                    update_on="blur"
                ), style="display:flex; align-items:center; justify-content:center; flex-wrap:wrap; gap:8px;"
            ), style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; width:100%; gap:12px;"
        ), class_="bg-blue dataframe-card-header"
    ),
    ui.div(
        ui.div(
            ui.output_ui("spots_summary"),
            class_="df-summary"
        ),
        ui.div(
            ui.output_ui("spots_summaries"),
            class_="column-summaries"
        ),
        class_="layout"
    ),
    ui.output_data_frame("spot_stats"),
    ui.div(
        ui.div("Show table", style="width: 95px; margin-left: 60px;"),
        ui.div(ui.input_switch("show_spot_stats_tbl", None, True, width="10px"), style="margin-top: 0px;"),
        ui.div("Show summaries", style="width: 145px; margin-left: 60px;"),
        ui.div(ui.input_switch("show_spot_stats_sums", None, True, width="10px"), style="margin-top: 0px;"),
        style="display: flex; align-items: right; justify-content: right; flex-wrap: wrap; margin-bottom: -20px; margin-right: 20px;"
    ),
    ui.download_button("download_spot_stats", "Download CSV"),
    full_screen=True
),


card_tracks = ui.card(
    ui.card_header(ui.span("Track stats", style="font-size:18px;"), class_="bg-secondary-css dataframe-card-header"),
    ui.div(
        ui.div(
            ui.output_ui("tracks_summary"),
            class_="df-summary"
        ),
        ui.div(
            ui.output_ui("tracks_summaries"),
            class_="column-summaries"
        ),
        class_="layout"
    ),
    ui.output_data_frame("track_stats"),
    ui.div(
        ui.div("Show table", style="width: 95px; margin-left: 60px;"),
        ui.div(ui.input_switch("show_track_stats_tbl", None, True, width="10px"), style="margin-top: 0px;"),
        ui.div("Show summaries", style="width: 145px; margin-left: 60px;"),
        ui.div(ui.input_switch("show_track_stats_sums", None, False, width="10px"), style="margin-top: 0px;"),
        style="display: flex; align-items: right; justify-content: right; flex-wrap: wrap; margin-bottom: -20px; margin-right: 20px;"
    ),
    ui.download_button("download_track_stats", "Download CSV"),
    full_screen=True
)


card_frames = ui.card(
    ui.card_header(ui.span("Frame stats", style="font-size:18px;"), class_="bg-secondary-css dataframe-card-header"),
    ui.div(
        ui.div(
            ui.output_ui("frames_summary"),
            class_="df-summary"
        ),
        ui.div(
            ui.output_ui("frame_summaries"),
            class_="column-summaries"
        ),
        class_="layout"
    ),
    ui.output_data_frame("frame_stats"),
    ui.div(
        ui.div("Show table", style="width: 95px; margin-left: 60px;"),
        ui.div(ui.input_switch("show_frame_stats_tbl", None, True, width="10px"), style="margin-top: 0px;"),
        ui.div("Show summaries", style="width: 145px; margin-left: 60px;"),
        ui.div(ui.input_switch("show_frame_stats_sums", None, False, width="10px"), style="margin-top: 0px;"),
        style="display: flex; align-items: right; justify-content: right; flex-wrap: wrap; margin-bottom: -20px; margin-right: 20px;"
    ),
    ui.download_button("download_frame_stats", "Download CSV"),
    full_screen=True
)


card_tintervals = ui.card(
    ui.card_header(ui.span("Time interval stats", style="font-size:18px;"), class_="bg-secondary-css dataframe-card-header", style="padding-top: 11.5px; padding-bottom: 11px;"),
    ui.div(
        ui.div(
            ui.output_ui("tintervals_summary"),
            class_="df-summary"
        ),
        ui.div(
            ui.output_ui("tinterval_summaries"),
            class_="column-summaries"
        ),
        class_="layout"
    ),
    ui.output_data_frame("tinterval_stats"),
    ui.div(
        ui.div("Show table", style="width: 95px; margin-left: 60px;"),
        ui.div(ui.input_switch("show_tinterval_stats_tbl", None, True, width="10px"), style="margin-top: 0px;"),
        ui.div("Show summaries", style="width: 145px; margin-left: 60px;"),
        ui.div(ui.input_switch("show_tinterval_stats_sums", None, False, width="10px"), style="margin-top: 0px;"),
        style="display: flex; align-items: right; justify-content: right; flex-wrap: wrap; margin-bottom: -20px; margin-right: 20px;"
    ),
    ui.download_button("download_tinterval_stats", "Download CSV"),
    full_screen=True
)

navpanel_dashboard = ui.nav_panel(
    
    "Dashboard",
    
    # _ Data display _
    ui.layout_columns(
        card_spots,
        card_tracks,
        card_frames
    ),
    ui.layout_columns(
        card_tintervals
    )
)