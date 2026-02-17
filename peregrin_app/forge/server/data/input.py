import traceback
from unittest import case
import pandas as pd
import shiny.ui as ui
from shiny import reactive, render, req
from src.code import DataLoader, Stats, Metrics, Level, is_empty




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
                                ui.markdown(""" <h2><b>  pre-Run Settings:  </h2></b> """), 
                                style="display: flex; flex-direction: column; justify-content: center; height: 100%; text-align: center; margin-top: 15px; margin-bottom: -10px;"
                            ),
                            ui.input_checkbox("strip_data", "Strip data, only keeping necessary columns", True),
                            ui.div(  
                                ui.input_action_link("explain_auto_label", "What's Auto-label?", class_="plain-link"),
                                ui.input_checkbox("auto_label", "Auto-label", False),
                                ui.br(),
                                ui.div("Compute:", style="font-size: 18px; font-weight: 550; margin-bottom: 12px; margin-left: 2px;"),
                                ui.div(
                                    ui.input_checkbox("compute_spotstats", "Spotstats", True),
                                    ui.input_checkbox_group("compute_data", None, ["Trackstats", "Framestats", "TimeIntervals"], inline=False, selected=["Trackstats", "Framestats", "TimeIntervals"]),
                                    style="margin-left: 10px;"
                                ),
                                ui.div(
                                    ui.markdown(""" <h5><b> Statistical Configuration: </h5></b> """), 
                                    style="display: flex; flex-direction: column; height: 100%; margin-top: 30px; margin-bottom: 10px; margin-left: 0px;"
                                ),
                                ui.div(
                                    ui.div("Confidence interval statistic:", style="margin-top: 5px; margin-right: 5px;"),
                                    ui.tooltip(
                                        ui.input_selectize("ci_statistic", None, ["mean", "median"], width="110px"),
                                        "The statistic for which the confidence interval will be computed. The mean is appropariate for normally distributed data, while the median is more robust to outliers and skewed, non-normal distributions.",
                                        placement="right",
                                    ),
                                    style="display: flex; gap: 5px; margin-left: 10px;"
                                ),
                                ui.div(
                                    ui.div("Confidence level (%):", style="margin-top: 5px; margin-right: 5px;"),
                                    ui.tooltip(
                                        ui.input_numeric("ci_confidence", None, 95, min=50, max=99.9, step=0.5, width="80px"),
                                        "Confidence level defines the percentage of confidence intervals that are expected to contain the true parameter value. A common choice is 95%, which means that if the same population is sampled multiple times and confidence intervals are computed for each sample, 95% of those intervals are expected to contain the true parameter value.",
                                        placement="right",
                                    ),
                                    style="display: flex; gap: 5px; margin-left: 10px;"
                                ),
                                ui.div(
                                    ui.div("Confidence interval resamples:", style="margin-top: 5px; margin-right: 5px;"),
                                    ui.tooltip(  
                                        ui.input_numeric("ci_resamples", None, 1000, min=100, step=100, width="95px"),
                                        "To ensure a higher performance during exploratory analysis, the number of resamples is set to 1000 as default. A larger amount of resamples ensures more accurate confidence intervals, but also takes longer to compute. For high quality results, ≥9999 resamples is recommended.",   
                                        placement="right",
                                    ),
                                    style="display: flex; gap: 5px; margin-left: 10px;"
                                ),
                                ui.markdown(""" <hr style="border: none; border-top: 1px solid; opacity: 0.125; margin-top: 15px; margin-bottom: 0px;" /> """),
                                ui.output_ui("data_labeling_ui"),
                            ), 
                            ui.output_ui('task_btn_labeling_sidebar'),
                            width="420px",
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
        
    # _ _ _ _ STATISTICAL CONFIGURATION _ _ _ _

    @reactive.Effect
    @reactive.event(input.ci_statistic, input.ci_confidence, input.ci_resamples)
    def _():
        req(not is_empty(S.SPOTSTATS.get()), not is_empty(S.TRACKSTATS.get()), not is_empty(S.FRAMESTATS.get()), not is_empty(S.TINTERVALSTATS.get()))

        Stats.BOOTSTRAP_RESAMPLES = input.ci_resamples()
        Stats.CONFIDENCE_LEVEL = input.ci_confidence()
        Stats.CI_STATISTIC = input.ci_statistic()
        

    # _ _ _ _ STABILIZE DATA INPUT PANEL _ _ _ _

    @reactive.calc
    def stabilize_input():
        ui.update_action_button(id="import_mode_btn", disabled=True)

        
    # _ _ _ _ RAW DATA INPUT CONTAINERS CONTROL _ _ _ _

    @reactive.Effect
    @reactive.event(input.add_input)
    def add_input():
        id = S.INPUTS.get()
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



    # _ _ _ _ RAW DATA INPUT => RUN -> COMPUTE _ _ _ _

    def is_busy(val):
            return isinstance(val, list) and len(val) > 0
    
    @reactive.Effect
    def run_btn_toggle():
        files_uploaded = [input[f"input_file{idx}"]() for idx in range(1, S.INPUTS.get()+1)]
        
        if all(is_busy(f) for f in files_uploaded):
            S.READYTORUN.set(True)
        else:
            S.READYTORUN.set(False)

        if any(is_busy(f) for f in files_uploaded):
            stabilize_input()

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

            if not files:
                break

            if input.auto_label():
                cond_label = files[0].get("name").split("#")[1] if len(files[0].get("name").split("#")) >= 2 else None #TODO: display an error if the file label is incorrect
            else:
                cond_label = input[f"condition_label{idx}"]()

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

            with reactive.isolate():
                Stats.B_RESAMPLES = input.ci_resamples()

                stats = Stats(noticequeue=noticequeue)

                # Compute spotstats for all data; spotstats are necessary and will be computed in any case
                Spots = stats.Spots(all_data)

                # Compute desired stats based on user selection in the sidebar
                Tracks, Frames, TimeIntervals = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                if "Trackstats" in input.compute_data():
                    Tracks = stats.Tracks(all_data)
                if "Framestats" in input.compute_data():
                    Frames = stats.Frames(all_data)
                if "TimeIntervals" in input.compute_data():
                    TimeIntervals = stats.TimeIntervals(all_data)

            S.UNFILTERED_SPOTSTATS.set(Spots)
            S.UNFILTERED_TRACKSTATS.set(Tracks)
            S.UNFILTERED_FRAMESTATS.set(Frames)
            S.UNFILTERED_TINTERVALSTATS.set(TimeIntervals)

            S.SPOTSTATS.set(Spots)
            S.TRACKSTATS.set(Tracks)
            S.FRAMESTATS.set(Frames)
            S.TINTERVALSTATS.set(TimeIntervals)

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
            stabilize_input()

            df = DataLoader.GetDataFrame(fileinfo[0]["datapath"], noticequeue=noticequeue)

            stats = Stats(noticequeue=noticequeue)

            Tracks, Frames, TimeIntervals =  \
                stats.Tracks(df),            \
                stats.Frames(df),            \
                stats.TimeIntervals(df)
            
            S.UNFILTERED_SPOTSTATS.set(df)
            S.UNFILTERED_TRACKSTATS.set(Tracks)
            S.UNFILTERED_FRAMESTATS.set(Frames)
            S.UNFILTERED_TINTERVALSTATS.set(TimeIntervals)

            S.SPOTSTATS.set(df)
            S.TRACKSTATS.set(Tracks)
            S.FRAMESTATS.set(Frames)
            S.TINTERVALSTATS.set(TimeIntervals)
            
            ui.update_action_button(id="append_threshold", disabled=False)
            
        except Exception as e:
            error_trace = traceback.format_exc()
            noticequeue.Report(Level.error, "Error in loading processed data", error_trace)

