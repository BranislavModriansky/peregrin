from shiny import reactive, ui  


def mount_tasks(S):

    @reactive.Effect
    def update_choices():
        if S.TRACKSTATS.get() is None or S.TRACKSTATS.get().empty:
            return
        ui.update_selectize(id="tracks_conditions", choices=S.TRACKSTATS.get()["Condition"].unique().tolist())
        ui.update_selectize(id="tracks_replicates", choices=["all"] + S.TRACKSTATS.get()["Replicate"].unique().tolist())
