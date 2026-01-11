from unittest import case
import pandas as pd
import shiny.ui as ui
from shiny import reactive, render   
from src.code import DataLoader, Spots, Tracks, Frames, TimeIntervals, Metrics

# from utils import emit_warning




def mount_data_input(input, output, session, S, noticequeue):


    @output
    @render.ui
    def input_panel():

        if S.IMPORT_MODE.get() == "raw":
            return [
                ui.div(
                    {"id": "data-inputs"},

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
            ]
        
    # _ _ _ _ RAW DATA INPUT CONTAINERS CONTROL _ _ _ _

    @reactive.Effect
    @reactive.event(input.add_input)
    def add_input():
        id = S.INPUTS.get()
        # emit_warning(report="Warning", message=f"Missing colors in {1} data: {2}. Generating random colors instead.")
        S.INPUTS.set(id + 1)
        session.send_input_message("remove_input", {"disabled": id < 1})

    @reactive.Effect
    @reactive.event(input.remove_input)
    def remove_input():
        id = S.INPUTS.get()
        if id > 1:
            S.INPUTS.set(id - 1)
        if S.INPUTS.get() <= 1:
            session.send_input_message("remove_input", {"disabled": True})

    def _input_container_ui(id: int):
        return ui.div(
            {"id": f"input_file_container_{id}"},
            ui.input_text(
                id=f"condition_label{id}",
                label="Label:",
                placeholder=f"Condition {id}"
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
        id = S.INPUTS.get()
        ui.insert_ui(
            ui=_input_container_ui(id),
            selector=f"#input_file_container_{id - 1}",
            where="afterEnd"
        )
 
    @reactive.effect
    @reactive.event(input.remove_input)
    def _remove_container():
        id = S.INPUTS.get()
        ui.insert_ui(
            ui.tags.script(
                # Clear browser chooser if the element still exists
                f"Shiny.setInputValue('input_file{id+1}', null, {{priority:'event'}});"
                f"Shiny.setInputValue('condition_label{id+1}', '', {{priority:'event'}});"
            ), 
            selector="body", 
            where="beforeEnd"
        )
        ui.remove_ui(
            selector=f"#input_file_container_{id+1}",
            multiple=True
        )



    # _ _ _ _ RUN - COMPUTE RAW INPUT _ _ _ _

    
    @reactive.Effect
    def run_btn_toggle():
        files_uploaded = [input[f"input_file{idx}"]() for idx in range(1, S.INPUTS.get()+1)]
        def is_busy(val):
            return isinstance(val, list) and len(val) > 0
        all_busy = all(is_busy(f) for f in files_uploaded)
        if all_busy:
            S.READYTORUN.set(True)
        else:
            S.READYTORUN.set(False)

    @output()
    @render.ui
    def run_btn_ui():
        if S.READYTORUN.get():
            ui.update_accordion(id="draggable_column_selector_panel", show=True)
            return ui.input_task_button("run", label="Run", class_="btn-secondary task-btn")
        else:
            ui.update_accordion(id="draggable_column_selector_panel", show=False)
            return ui.input_action_button("run0", label="Run", class_="btn-secondary task-btn-mask", disabled=True)

    @reactive.Effect
    @reactive.event(input.run)
    def parsed_files():
        all_data = []

        for idx in range(1, S.INPUTS.get()+1):

            files = input[f"input_file{idx}"]()

            if input.auto_label():
                cond_label = files[0].get("name").split("#")[1] if len(files[0].get("name").split("#")) >= 2 else None #TODO: display an error if the file label is incorrect
            else:
                cond_label = input[f"condition_label{idx}"]()

            if not files:
                break

            for file_idx, fileinfo in enumerate(files, start=1):
                try:
                    df = DataLoader.GetDataFrame(fileinfo["datapath"], noticequeue=noticequeue)

                    if input.strip_data():
                        extracted = DataLoader.ExtractStripped(
                            df,
                            id_col=input.select_id(),
                            t_col=input.select_t(),
                            x_col=input.select_x(),
                            y_col=input.select_y(),
                            mirror_y=True,
                        )
                    else:
                        extracted = DataLoader.ExtractFull(
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
            S.RAWDATA.set(all_data)
            S.UNFILTERED_SPOTSTATS.set(Spots(all_data))
            S.UNFILTERED_TRACKSTATS.set(Tracks(all_data))
            S.UNFILTERED_TRACKSTATS.set(Tracks(all_data))
            S.UNFILTERED_FRAMESTATS.set(Frames(all_data))
            S.UNFILTERED_TINTERVALSTATS.set(TimeIntervals(all_data)())
            S.SPOTSTATS.set(S.UNFILTERED_SPOTSTATS.get())
            S.TRACKSTATS.set(S.UNFILTERED_TRACKSTATS.get())
            S.FRAMESTATS.set(S.UNFILTERED_FRAMESTATS.get())
            S.TINTERVALSTATS.set(S.UNFILTERED_TINTERVALSTATS.get())

            S.THRESHOLDS.set({1: {"spots": S.UNFILTERED_SPOTSTATS.get(), "tracks": S.UNFILTERED_TRACKSTATS.get()}})

            ui.update_sidebar(id="sidebar", show=True)
            ui.update_action_button(id="append_threshold", disabled=False)

        else:
            pass



    # _ _ _ _ PROCESSED DATA INPUT _ _ _ _

    @reactive.Effect
    @reactive.event(input.already_processed_input)
    def load_processed_data():
        fileinfo = input.already_processed_input()
        try:
            df = DataLoader.GetDataFrame(fileinfo[0]["datapath"])

            S.UNFILTERED_SPOTSTATS.set(df)
            S.UNFILTERED_TRACKSTATS.set(Tracks(df))
            S.UNFILTERED_FRAMESTATS.set(Frames(df))
            S.UNFILTERED_TINTERVALSTATS.set(TimeIntervals(df)())
            S.SPOTSTATS.set(df)
            S.TRACKSTATS.set(Tracks(df))
            S.FRAMESTATS.set(Frames(df))
            S.TINTERVALSTATS.set(TimeIntervals(df)())
            S.THRESHOLDS.set({1: {"spots": S.UNFILTERED_SPOTSTATS.get(), "tracks": S.UNFILTERED_TRACKSTATS.get()}})

            ui.update_action_button(id="append_threshold", disabled=False)
            
        except Exception as e:
            print(e)

