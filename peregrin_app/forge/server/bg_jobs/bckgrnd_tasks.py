import time
from shiny import reactive, ui, render


def mount_tasks(input, output, session, S, noticequeue):

    @reactive.Effect
    def update_choices():
        if S.TRACKSTATS.get() is None or S.TRACKSTATS.get().empty:
            return
        ui.update_selectize(id="tracks_conditions", choices=S.TRACKSTATS.get()["Condition"].unique().tolist())
        ui.update_selectize(id="tracks_replicates", choices=["all"] + S.TRACKSTATS.get()["Replicate"].unique().tolist())

    
    @output 
    @render.ui
    def import_mode():

        print("IMPORT MODE:", S.IMPORT_MODE.get())

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

                return ui.input_action_button("import_mode_btn", "Import raw data", class_="btn-tertiary-css")
                
    @reactive.effect
    def switch_import_mode():

        try:
            @reactive.effect
            @reactive.event(input.import_mode_btn)
            def _():
                time.sleep(0.025)  # small delay to avoid double triggering
                match S.IMPORT_MODE.get():
                    case "raw":
                        S.IMPORT_MODE.set("processed")
                    case "processed":
                        S.IMPORT_MODE.set("raw")
            
        except Exception:
            pass