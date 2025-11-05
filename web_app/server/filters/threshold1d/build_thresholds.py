import shiny.ui as ui
from shiny import reactive, render
from utils import Metrics, Modes


    

def mount_thresholds_build(input, output, session, S):

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
    

    @output()
    @render.text
    def sidebar_label():
        return ui.markdown(
            f""" <h3> <b>  Data threshold  </b> </h3> """
        )

    
    @reactive.Effect
    @reactive.event(input.append_threshold)
    def append_threshold():
        thresholds = S.THRESHOLDS.get()
        if not thresholds:
            return

        S.THRESHOLDS_ID.set(S.THRESHOLDS_ID.get() + 1)
        thresholds |= {S.THRESHOLDS_ID.get(): thresholds.get(S.THRESHOLDS_ID.get() - 1)}
        S.THRESHOLDS.set(thresholds)

        if S.THRESHOLDS_ID.get() > 1:
            session.send_input_message("remove_threshold", {"disabled": False})

        ui.insert_accordion_panel(
            id="threshold_accordion",
            panel=render_threshold_accordion_panel(S.THRESHOLDS_ID.get()),
            position="after"
        ) 

    @reactive.Effect
    @reactive.event(input.remove_threshold)
    def remove_threshold():
        thresholds = S.THRESHOLDS.get()

        if (S.THRESHOLDS_ID.get() - 1) <= 1:
            session.send_input_message("remove_threshold", {"disabled": True})

        ui.remove_accordion_panel(
            id="threshold_accordion",
            target=f"Threshold {S.THRESHOLDS_ID.get()}"
        )

        del thresholds[S.THRESHOLDS_ID.get()]
        S.THRESHOLDS.set(thresholds)

        S.THRESHOLDS_ID.set(S.THRESHOLDS_ID.get() - 1)

    @reactive.Effect
    @reactive.event(input.run, input.already_processed_input)
    def refresh_sidebar():
        thresholds = S.THRESHOLDS.get()
        if not thresholds:
            return
        for id in range(1, S.THRESHOLDS_ID.get() + 1):
            if id not in list(thresholds.keys()):
                ui.remove_accordion_panel(
                    id="threshold_accordion",
                    target=f"Threshold {id}"
                )
        session.send_input_message("remove_threshold", {"disabled": True})
        S.THRESHOLDS_ID.set(1)
        