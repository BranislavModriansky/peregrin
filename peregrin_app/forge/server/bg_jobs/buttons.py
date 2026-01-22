import time
from shiny import reactive, ui, render


def mount_buttons(input, output, session, S, noticequeue):

    @output 
    @render.ui
    def import_mode():

        match S.IMPORT_MODE.get():
            case "raw":
                
                try: ui.remove_ui("import_mode_btn", session=session)
                except Exception: pass
                
                return [
                    ui.input_action_button("import_mode_btn", "Import processed data", class_="btn-tertiary-css"),
                    ui.input_action_button("add_input", "Add data input", class_="btn-primary"),
                    ui.input_action_button("remove_input", "Remove data input", class_="btn-primary", disabled=True),
                    ui.output_ui("run_btn_ui"),
                ]
                
            case "processed":
                
                try: ui.remove_ui("import_mode_btn", session=session)
                except Exception: pass

                # Wrap button and input in a flex container to keep them in one row
                return ui.div(
                    ui.div(
                        ui.input_action_button("import_mode_btn", "Import raw data", class_="btn-tertiary-css"),
                        style="margin-top: -32px;"
                    ),
                    ui.div(
                        ui.div(
                            ui.markdown("""<span style='color: #0171b7; white-space:nowrap;'><i>Drop in <b>Spot Stats CSV</b>: </i></span>"""),
                            style="margin-right: 15px; margin-top: -12px"
                        ),
                        ui.input_file(id="already_processed_input", label=None, placeholder="Drag & drop CSV", accept=[".csv"], multiple=False),
                        style="display: flex; align-items: center; margin-left: 35px;"
                    ),
                    style="display: flex; align-items: center;"
                )
                
    @reactive.effect
    def switch_import_mode():

        try:
            @reactive.effect
            @reactive.event(input.import_mode_btn)
            def _():
                time.sleep(0.02)  # small delay to avoid double triggering
                match S.IMPORT_MODE.get():
                    case "raw":
                        S.IMPORT_MODE.set("processed")
                    case "processed":
                        S.IMPORT_MODE.set("raw")
            
        except Exception:
            pass
