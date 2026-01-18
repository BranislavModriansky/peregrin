import shiny.ui as ui
from shiny import reactive, render, req
from peregrin_app.src.code._handlers._scheduling import DebounceCalc, DebounceEffect
from src.code import Metrics, Modes, is_empty


    

def mount_thresholds_build(input, output, session, S):

    # ________ SIDEBAR ACCORDION THRESHOLDS LAYOUT  ________

    @output()
    @render.ui
    def sidebar_accordion_placeholder():

        if all(not is_empty(df) 
               for df in [S.SPOTSTATS.get(), 
                          S.TRACKSTATS.get(), 
                          S.FRAMESTATS.get(), 
                          S.TINTERVALSTATS.get()]):
            
            threshold_1 = [
                ui.panel_well(
                    ui.input_selectize(f"threshold_property_1", "Property", choices=S.SPOTSTATS_COLUMNS.get() + S.TRACKSTATS_COLUMNS.get(), selected='Track displacement'),
                    ui.input_selectize(f"threshold_type_1", "Threshold type", choices=Modes.Thresholding),
                    ui.panel_conditional(
                        f"input.threshold_type_1 == 'N-tile'",
                        ui.input_selectize(f"threshold_ntile_1", "N-tile", choices=[200, 100, 50, 25, 20, 10, 5, 4, 2], selected=100),
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
            ]

        else:

            threshold_1 = [""]

        return ui.accordion(
            ui.accordion_panel(
                "Settings",
                ui.input_numeric("bins", "Number of bins", value=15, min=1, step=1),
                ui.markdown("<p style='line-height:0.1;'> <br> </p>"),
            ),

            ui.accordion_panel(
                f"Threshold 1",
                *threshold_1
            ),
            id="threshold_accordion",
            open="Threshold 1",
        )
    
    def render_threshold_accordion_panel(id):
        return ui.accordion_panel(
            f"Threshold {id}",
            ui.panel_well(
                ui.input_selectize(f"threshold_property_{id}", "Property", choices=S.SPOTSTATS_COLUMNS.get() + S.TRACKSTATS_COLUMNS.get(), selected='Track displacement'),
                ui.input_selectize(f"threshold_type_{id}", "Threshold type", choices=Modes.Thresholding),
                ui.panel_conditional(
                    f"input.threshold_type_{id} == 'N-tile'",
                    ui.input_selectize(f"threshold_ntile_{id}", "N-tile", choices=[200, 100, 50, 25, 20, 10, 5, 4, 2], selected=100),
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
    

    # ________ SIDEBAR LABEL ________

    @output()
    @render.text
    def sidebar_label():
        return ui.markdown(
            f""" <h3> <b>  Data threshold  </b> </h3> """
        )
    

    # ________ ADD / REMOVE THRESHOLDS BUTTONS ________

    MIN_THRESHOLDS = 1
    MAX_THRESHOLDS = 8

    def _sync_threshold_buttons(count: int):
        ui.update_action_button("remove_threshold", disabled=(count <= MIN_THRESHOLDS))
        ui.update_action_button("append_threshold", disabled=(count >= MAX_THRESHOLDS))

    @reactive.Effect
    @reactive.event(input.append_threshold)
    def _append_threshold():
        thresholds = S.THRESHOLDS_ID.get()

        # clamp + guard
        if thresholds >= MAX_THRESHOLDS:
            _sync_threshold_buttons(thresholds)
            return

        ui.insert_accordion_panel(
            id="threshold_accordion",
            panel=render_threshold_accordion_panel(thresholds + 1),
            position="after"
        )

        S.THRESHOLDS_ID.set(thresholds + 1)
        _sync_threshold_buttons(thresholds + 1)

    @reactive.Effect
    @reactive.event(input.remove_threshold)
    def _remove_threshold():
        thresholds = S.THRESHOLDS_ID.get()

        # clamp + guard (do not remove last)
        if thresholds <= MIN_THRESHOLDS:
            _sync_threshold_buttons(thresholds)
            return

        ui.remove_accordion_panel(
            id="threshold_accordion",
            target=f"Threshold {thresholds}"
        )

        S.THRESHOLDS_ID.set(thresholds - 1)
        _sync_threshold_buttons(thresholds - 1)

    @reactive.Effect
    def initialize_append():
        if all(not is_empty(df) 
            for df in [S.SPOTSTATS.get(), 
                        S.TRACKSTATS.get(), 
                        S.FRAMESTATS.get(), 
                        S.TINTERVALSTATS.get()]):
            _sync_threshold_buttons(S.THRESHOLDS_ID.get())


