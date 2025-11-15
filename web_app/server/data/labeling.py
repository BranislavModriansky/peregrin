import shiny.ui as ui
from shiny import reactive, req, render
from ui import make_sortable_ui
from utils import DataLoader, Metrics, FilenameFormatExample
from utils import Customize


def mount_data_labeling(input, output, session, S):

    @output()
    @render.ui
    def data_labeling_ui():
        if input.run() > 0:
            return [
                ui.input_switch("write_replicate_colors", "Set replicate colors", False),
                ui.output_ui("replicate_colors_inputs"),

                ui.input_switch("write_condition_colors", "Set condition colors", False),
                ui.output_ui("condition_colors_inputs"),

                ui.input_switch("set_condition_order", "Set condition order", False),
                ui.tags.style(Customize.Ladder),
                ui.output_ui("condition_order_ladder"),

                ui.input_switch("write_replicate_labels", "Write replicate labels", False),
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
                    columns = DataLoader.GetColumns(files[0]["datapath"])

                    for sel in Metrics.LookFor.keys():
                        choice = DataLoader.FindMatchingColumn(columns, Metrics.LookFor[sel])
                        if choice is not None:
                            ui.update_selectize(sel, choices=columns, selected=choice)
                        else:
                            ui.update_selectize(sel, choices=columns, selected=columns[0] if columns else None)
                    break
                except Exception as e:
                    continue

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
                    <span style='color: dimgrey;'><i>If you want to use this feature, make sure it is enabled before running the analysis.</i></span> <br>
                    """
                ),
            title="What's Auto-label?",
            easy_close=True,
            footer=None,
            # background_color="#2b2b2b"
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
            req(not S.UNFILTERED_SPOTSTATS.get().empty)

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
            
            req(not S.UNFILTERED_SPOTSTATS.get().empty)

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
            
            req(not S.UNFILTERED_SPOTSTATS.get().empty)
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
    @reactive.event(input.write_values)
    def _write_values():

        df_unfiltered_spots = S.UNFILTERED_SPOTSTATS.get().copy()
        df_unfiltered_tracks = S.UNFILTERED_TRACKSTATS.get().copy()
        df_unfiltered_frames = S.UNFILTERED_FRAMESTATS.get().copy()
        df_spots = S.SPOTSTATS.get().copy()
        df_tracks = S.TRACKSTATS.get().copy()
        df_frames = S.FRAMESTATS.get().copy()
        req(df is not None and not df.empty for df in [df_unfiltered_spots, df_unfiltered_tracks, df_unfiltered_frames, df_spots, df_tracks, df_frames])
        

        # _ Replicate Labels and Colors _
        @reactive.Effect
        def _replicate_values():
            req("Replicate" in df_tracks.columns)
            replicates = sorted(df_tracks["Replicate"].unique())
            if len(set(type(rep) for rep in replicates)) > 1:
                return

            @reactive.Effect
            def _replicate_colors():
                with reactive.isolate():
                    # print("Writing replicate colors...")
                    # print(input.write_replicate_colors())

                    if input.write_replicate_colors():

                        for idx, rep in enumerate(replicates):
                            # print(input[f"replicate_color{idx}"]())
                            color = input[f"replicate_color{idx}"]()
                            if color:
                                df_unfiltered_spots.loc[df_unfiltered_spots["Replicate"] == rep, "Replicate color"] = color
                                df_unfiltered_tracks.loc[df_unfiltered_tracks["Replicate"] == rep, "Replicate color"] = color
                                df_unfiltered_frames.loc[df_unfiltered_frames["Replicate"] == rep, "Replicate color"] = color
                                df_spots.loc[df_spots["Replicate"] == rep, "Replicate color"] = color
                                df_tracks.loc[df_tracks["Replicate"] == rep, "Replicate color"] = color
                                df_frames.loc[df_frames["Replicate"] == rep, "Replicate color"] = color

                    elif not input.write_replicate_colors():
                        # print("Removing replicate colors...")
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

                    if not df_unfiltered_spots.equals(S.UNFILTERED_SPOTSTATS.get()): S.UNFILTERED_SPOTSTATS.set(df_unfiltered_spots)
                    if not df_unfiltered_tracks.equals(S.UNFILTERED_TRACKSTATS.get()): S.UNFILTERED_TRACKSTATS.set(df_unfiltered_tracks)
                    if not df_unfiltered_frames.equals(S.UNFILTERED_FRAMESTATS.get()): S.UNFILTERED_FRAMESTATS.set(df_unfiltered_frames)
                    if not df_spots.equals(S.SPOTSTATS.get()): S.SPOTSTATS.set(df_spots)
                    if not df_tracks.equals(S.TRACKSTATS.get()): S.TRACKSTATS.set(df_tracks)
                    if not df_frames.equals(S.FRAMESTATS.get()): S.FRAMESTATS.set(df_frames)


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

                    if not df_unfiltered_spots.equals(S.UNFILTERED_SPOTSTATS.get()): S.UNFILTERED_SPOTSTATS.set(df_unfiltered_spots)
                    if not df_unfiltered_tracks.equals(S.UNFILTERED_TRACKSTATS.get()): S.UNFILTERED_TRACKSTATS.set(df_unfiltered_tracks)
                    if not df_unfiltered_frames.equals(S.UNFILTERED_FRAMESTATS.get()): S.UNFILTERED_FRAMESTATS.set(df_unfiltered_frames)
                    if not df_spots.equals(S.SPOTSTATS.get()): S.SPOTSTATS.set(df_spots)
                    if not df_tracks.equals(S.TRACKSTATS.get()): S.TRACKSTATS.set(df_tracks)
                    if not df_frames.equals(S.FRAMESTATS.get()): S.FRAMESTATS.set(df_frames)




        # _ Condition Colors and Order _
        @reactive.Effect
        def _condition_values():
            req("Condition" in df_tracks.columns)

            @reactive.Effect
            def _condition_colors():
                with reactive.isolate():
                    conditions = sorted(df_tracks["Condition"].unique())
                    if len(set(type(cond) for cond in conditions)) > 1:
                        raise Exception("Inconsistent condition types.")

                    if input.write_condition_colors():

                        for idx, cond in enumerate(conditions):
                            # print(input[f"replicate_color{idx}"]())
                            color = input[f"condition_color{idx}"]()
                            if color:
                                df_unfiltered_spots.loc[df_unfiltered_spots["Condition"] == cond, "Condition color"] = color
                                df_unfiltered_tracks.loc[df_unfiltered_tracks["Condition"] == cond, "Condition color"] = color
                                df_unfiltered_frames.loc[df_unfiltered_frames["Condition"] == cond, "Condition color"] = color
                                df_spots.loc[df_spots["Condition"] == cond, "Condition color"] = color
                                df_tracks.loc[df_tracks["Condition"] == cond, "Condition color"] = color
                                df_frames.loc[df_frames["Condition"] == cond, "Condition color"] = color

                    elif not input.write_condition_colors():
                        # print("Removing condition colors...")
                        if "Condition color" in df_unfiltered_spots.columns:
                            df_unfiltered_spots.drop(columns=["Condition color"], inplace=True)
                        if "Condition color" in df_unfiltered_tracks.columns:
                            df_unfiltered_tracks.drop(columns=["Condition color"], inplace=True)
                        if "Condition color" in df_unfiltered_frames.columns:
                            df_unfiltered_frames.drop(columns=["Condition color"], inplace=True)
                        if "Condition color" in df_spots.columns:
                            df_spots.drop(columns=["Condition color"], inplace=True)
                        if "Condition color" in df_tracks.columns:
                            df_tracks.drop(columns=["Condition color"], inplace=True)
                        if "Condition color" in df_frames.columns:
                            df_frames.drop(columns=["Condition color"], inplace=True)

                    if not df_unfiltered_spots.equals(S.UNFILTERED_SPOTSTATS.get()): S.UNFILTERED_SPOTSTATS.set(df_unfiltered_spots)
                    if not df_unfiltered_tracks.equals(S.UNFILTERED_TRACKSTATS.get()): S.UNFILTERED_TRACKSTATS.set(df_unfiltered_tracks)
                    if not df_unfiltered_frames.equals(S.UNFILTERED_FRAMESTATS.get()): S.UNFILTERED_FRAMESTATS.set(df_unfiltered_frames)
                    if not df_spots.equals(S.SPOTSTATS.get()): S.SPOTSTATS.set(df_spots)
                    if not df_tracks.equals(S.TRACKSTATS.get()): S.TRACKSTATS.set(df_tracks)
                    if not df_frames.equals(S.FRAMESTATS.get()): S.FRAMESTATS.set(df_frames)


            @reactive.Effect
            def _condition_order():
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

                    if not df_unfiltered_spots.equals(S.UNFILTERED_SPOTSTATS.get()): S.UNFILTERED_SPOTSTATS.set(df_unfiltered_spots)
                    if not df_unfiltered_tracks.equals(S.UNFILTERED_TRACKSTATS.get()): S.UNFILTERED_TRACKSTATS.set(df_unfiltered_tracks)
                    if not df_unfiltered_frames.equals(S.UNFILTERED_FRAMESTATS.get()): S.UNFILTERED_FRAMESTATS.set(df_unfiltered_frames)
                    if not df_spots.equals(S.SPOTSTATS.get()): S.SPOTSTATS.set(df_spots)
                    if not df_tracks.equals(S.TRACKSTATS.get()): S.TRACKSTATS.set(df_tracks)
                    if not df_frames.equals(S.FRAMESTATS.get()): S.FRAMESTATS.set(df_frames)

            