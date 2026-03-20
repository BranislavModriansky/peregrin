import shiny.ui as ui
from shiny import reactive, req, render

import pandas as pd

from ...ui_shell import make_sortable_ui
from src.code import DataLoader, Metrics, FilenameFormatExample, is_empty, Reporter, Level, BaseDataInventory, DebounceCalc


def mount_data_labeling(input, output, session, S, noticequeue):

    dataloader = DataLoader(noticequeue=noticequeue)

    @output()
    @render.ui
    def task_btn_labeling_sidebar():
        if input.run() > 0:
            return ui.input_task_button("write_values", label="Set labels", label_busy="Labeling...", class_="btn-secondary task-btn", width="100%"), 

    @output()
    @render.ui
    def data_labeling_ui():
        if input.run() > 0:
            return [
                ui.div(ui.markdown(""" <br><h4><b>  Labeling settings:  </h4></b> """), 
                       style="display: flex; flex-direction: column; justify-content: center; height: 100%; text-align: center; "),

                ui.input_switch("write_replicate_colors", "Set replicate colors", False),
                ui.output_ui("replicate_colors_inputs"),

                ui.input_switch("write_condition_colors", "Set condition colors", False),
                ui.output_ui("condition_colors_inputs"),

                ui.input_switch("set_condition_order", "Set condition order", False),
                ui.output_ui("condition_order_ladder"),

                ui.input_switch("write_replicate_labels", "Re-write replicate labels", False),
                ui.output_ui("replicate_labels_inputs"),
            ]


    # _ _ _ _ RAW DATA INPUT (X, Y, T, ID) COLUMNS SPECIFICATION CONTROL _ _ _ _

    @reactive.Effect
    def column_selection():
        ui.update_selectize(id="select_id", choices=["e.g. TRACK ID"])
        ui.update_selectize(id="select_t", choices=["e.g. POSITION T"])
        ui.update_selectize(id="select_x", choices=["e.g. POSITION X"])
        ui.update_selectize(id="select_y", choices=["e.g. POSITION Y"])

        for idx in range(1, S.INPUTS.get()+1):
            files = input[f"input_file{idx}"]()
            if files and isinstance(files, list) and len(files) > 0:
                try:
                    columns = dataloader.GetColumns(files[0]["datapath"])

                    for sel in Metrics.LookFor.keys():
                        choice = dataloader.FindMatchingColumn(columns, Metrics.LookFor[sel])
                        if choice is not None:
                            ui.update_selectize(sel, choices=columns, selected=choice)
                        else:
                            ui.update_selectize(sel, choices=columns, selected=columns[0] if columns else None)
                    break
                except Exception as e:
                    Reporter(Level.error, f"An error occurred while processing file {files[0]['datapath']}: {str(e)}", trace=traceback.format_exc())

    # _ _ SET UNITS _ _ 
    @reactive.Effect
    def set_units():
        S.UNITS.set(Metrics.Units.SetUnits(t=Metrics.Units.TimeUnits.get(input.select_t_unit())))


    # _ _ _ _ LABELING SETTINGS ON INPUT SIDEBAR _ _ _ _

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
                    <span style='color: dimgrey;'><i>If you want to use this feature, make sure it is enabled <b>before</b> hitting run.</i></span> <br>
                    """
                ),
            title="What's Auto-label?",
            easy_close=True,
            footer=None,
            # background_color="#2b2b2b"
            class_="auto-labeling-modal"
            )
        )

    # _ Filename format example for Auto-labeling _
    @output()
    @render.data_frame
    def FilenameFormatExampleTable():
        return FilenameFormatExample



    # _ _ _ _ SORTABLE UI LADDER - CONDITION ORDER _ _ _ _

    @output(id="condition_order_ladder")
    @render.ui
    def condition_order_ladder():
        if not input.set_condition_order():
            return
        
        if S.TRACKSTATS.get() is not None:
            req(not S.TRACKSTATS.get().empty)

            items = S.TRACKSTATS.get()["Condition"].unique().tolist()

            if isinstance(items, list) and len(items) > 1:
                return make_sortable_ui(inputID="order", items=items)
            elif isinstance(items, list) and len(items) == 1:
                return ui.markdown("*Only one condition present.*")
            else:
                return



    # _ _ _ _ REPLICATE LABELS AND COLORS INPUT UIs _ _ _ _

    @reactive.Effect
    def replicate_labels_and_colors_inputs():

        # _ Replicate colors ui _
        @output(id="replicate_colors_inputs")
        @render.ui
        def replicate_colors_inputs():
            if not input.write_replicate_colors():
                return
            req(not is_empty(S.UNFILTERED_SPOTSTATS.get()))

            if "Replicate color" not in S.UNFILTERED_SPOTSTATS.get().columns:
                S.UNFILTERED_SPOTSTATS.get()["Replicate color"] = "#59a9d7"  # default color

            replicates = sorted(S.UNFILTERED_SPOTSTATS.get()["Replicate"].unique())
            items = []
            for idx, rep in enumerate(replicates):
                try:
                    value = S.UNFILTERED_SPOTSTATS.get().loc[S.UNFILTERED_SPOTSTATS.get()["Replicate"] == rep, "Replicate color"].iloc[0]
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
        

        @output(id="condition_colors_inputs")
        @render.ui
        def condition_colors_inputs():
            if not input.write_condition_colors():
                return
            
            req(not is_empty(S.UNFILTERED_SPOTSTATS.get()))

            if "Condition color" not in S.UNFILTERED_SPOTSTATS.get().columns:
                S.UNFILTERED_SPOTSTATS.get()["Condition color"] = "#59a9d7"  # default color

            conditions = sorted(S.UNFILTERED_SPOTSTATS.get()["Condition"].unique())
            items = []
            for idx, cond in enumerate(conditions):
                try:
                    value = S.UNFILTERED_SPOTSTATS.get().loc[S.UNFILTERED_SPOTSTATS.get()["Condition"] == cond, "Condition color"].iloc[0]
                except:
                    value = "#59a9d7"

                cid = f"condition_color{idx}"
                items.append(
                    ui.div(
                        ui.tags.label(f"{cond}", **{"for": cid}),
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

        # _ Replicate labels input ui _
        @output(id="replicate_labels_inputs")
        @render.ui
        def replicate_labels_inputs():
            if not input.write_replicate_labels():
                return
            
            req(not is_empty(S.UNFILTERED_SPOTSTATS.get()))
            replicates = sorted(S.UNFILTERED_SPOTSTATS.get()["Replicate"].unique())
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



    # _ _ _ _ WRITING LABELING VALUES _ _ _ _

    @reactive.Effect
    @reactive.event(input.write_values, ignore_init=True)
    def _write_values():

        with reactive.isolate():
            dataframes = dict(
                UNFILTERED_SPOTSTATS      = S.UNFILTERED_SPOTSTATS.get().copy(), 
                UNFILTERED_TRACKSTATS     = S.UNFILTERED_TRACKSTATS.get().copy(), 
                UNFILTERED_FRAMESTATS     = S.UNFILTERED_FRAMESTATS.get().copy(), 
                UNFILTERED_TINTERVALSTATS = S.UNFILTERED_TINTERVALSTATS.get().copy(), 
                SPOTSTATS                 = S.SPOTSTATS.get().copy(),
                TRACKSTATS                = S.TRACKSTATS.get().copy(),
                FRAMESTATS                = S.FRAMESTATS.get().copy(),
                TINTERVALSTATS            = S.TINTERVALSTATS.get().copy()
            )

            req(all(not is_empty(df) for df in dataframes.values()))

        # ── Replicate colors ──
        if input.write_replicate_colors():
            replicates = sorted(dataframes["UNFILTERED_TRACKSTATS"]["Replicate"].unique())
            for idx, rep in enumerate(replicates):
                color = input[f"replicate_color{idx}"]()
                if color:
                    for df in dataframes.values():
                        if "Replicate" in df.columns:
                            df.loc[df["Replicate"] == rep, "Replicate color"] = color
        else:
            for df in dataframes.values():
                if "Replicate color" in df.columns:
                    df.drop(columns=["Replicate color"], inplace=True)

        # ── Condition colors ──
        if input.write_condition_colors():
            conditions = sorted(dataframes["UNFILTERED_TRACKSTATS"]["Condition"].unique())
            for idx, cond in enumerate(conditions):
                color = input[f"condition_color{idx}"]()
                if color:
                    for df in dataframes.values():
                        if "Condition" in df.columns:
                            df.loc[df["Condition"] == cond, "Condition color"] = color
        else:
            for df in dataframes.values():
                if "Condition color" in df.columns:
                    df.drop(columns=["Condition color"], inplace=True)

        # ── Replicate labels ──
        if input.write_replicate_labels():
            replicates = sorted(dataframes["UNFILTERED_TRACKSTATS"]["Replicate"].unique())
            for idx, rep in enumerate(replicates):
                label = input[f"replicate_label{idx}"]() if isinstance(input[f"replicate_label{idx}"](), str) and input[f"replicate_label{idx}"]() != "" else rep
                if label != str(rep):
                    for df in dataframes.values():
                        if "Replicate" in df.columns:
                            df.loc[df["Replicate"] == rep, "Replicate"] = label

        # ── Condition order ──
        if input.set_condition_order():
            selected_order = input.order()
            if selected_order is not None and len(selected_order) >= 2:
                order = list(selected_order)
                rank = {cond: i for i, cond in enumerate(order)}

                for key, df in dataframes.items():
                    if "Condition" in df.columns:
                        dataframes[key] = df.sort_values(
                            by="Condition",
                            key=lambda col: col.map(rank).fillna(len(rank)),
                            kind="stable",
                        )

                
        # ── Push all changes at once ──
        for key in dataframes:
            getattr(S, key).set(dataframes[key])


    @DebounceCalc(3)
    @reactive.calc
    def _feed_data():
        BaseDataInventory.Spots = S.UNFILTERED_SPOTSTATS.get()
        BaseDataInventory.Tracks = S.UNFILTERED_TRACKSTATS.get()
        BaseDataInventory.Frames = S.UNFILTERED_FRAMESTATS.get()
        BaseDataInventory.TimeIntervals = S.UNFILTERED_TINTERVALSTATS.get()


    @reactive.Effect()
    @reactive.event(
        S.UNFILTERED_SPOTSTATS, S.UNFILTERED_TRACKSTATS, 
        S.UNFILTERED_FRAMESTATS, S.UNFILTERED_TINTERVALSTATS, 
        ignore_init=True
    )
    def feed_data():

        req(all(not is_empty(df) for df in [S.UNFILTERED_SPOTSTATS.get(), BaseDataInventory.Spots,
                                            S.UNFILTERED_TRACKSTATS.get(), BaseDataInventory.Tracks,
                                            S.UNFILTERED_FRAMESTATS.get(), BaseDataInventory.Frames,
                                            S.UNFILTERED_TINTERVALSTATS.get(), BaseDataInventory.TimeIntervals]))

        if all(r_df.equals(b_df) for r_df, b_df in [
              (S.UNFILTERED_SPOTSTATS.get(), BaseDataInventory.Spots),
              (S.UNFILTERED_TRACKSTATS.get(), BaseDataInventory.Tracks),
              (S.UNFILTERED_FRAMESTATS.get(), BaseDataInventory.Frames),
              (S.UNFILTERED_TINTERVALSTATS.get(), BaseDataInventory.TimeIntervals)]
        ): _feed_data()