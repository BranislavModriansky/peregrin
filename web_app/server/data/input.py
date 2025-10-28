import pandas as pd
import shiny.ui as ui
from shiny import reactive   
from utils import DataLoader, Spots, Tracks, Frames



def mount_data_input(input, output, session, S):


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
    def enable_run_button():
        files_uploaded = [input[f"input_file{idx}"]() for idx in range(1, S.INPUTS.get()+1)]
        def is_busy(val):
            return isinstance(val, list) and len(val) > 0
        all_busy = all(is_busy(f) for f in files_uploaded)
        session.send_input_message("run", {"disabled": not all_busy})

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
                    df = DataLoader.GetDataFrame(fileinfo["datapath"])
                    # print(df.head())
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
            S.SPOTSTATS.set(S.UNFILTERED_SPOTSTATS.get())
            S.TRACKSTATS.set(S.UNFILTERED_TRACKSTATS.get())
            S.FRAMESTATS.set(S.UNFILTERED_FRAMESTATS.get())

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
            S.SPOTSTATS.set(df)
            S.TRACKSTATS.set(Tracks(df))
            S.FRAMESTATS.set(Frames(df))

            S.THRESHOLDS.set({1: {"spots": S.UNFILTERED_SPOTSTATS.get(), "tracks": S.UNFILTERED_TRACKSTATS.get()}})

            ui.update_action_button(id="append_threshold", disabled=False)
            
        except Exception as e:
            print(e)

    