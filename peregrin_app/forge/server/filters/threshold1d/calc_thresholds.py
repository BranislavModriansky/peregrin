import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import shiny.ui as ui
from shiny import render, reactive, req, ui

from src.code import Frames, TimeIntervals, Inventory1D, Filter1D, DebounceCalc, ThrottleCalc, DebounceEffect, is_empty



def mount_thresholds_calc(input, output, session, S, noticequeue):

    Filter1D.noticequeue = noticequeue

    at_idx = reactive.Value()

    @DebounceCalc(0.5)
    @reactive.calc
    def get_bins():
        return input.bins() if input.bins() is not None and input.bins() != 0 else 15
    
    
    @reactive.Effect
    @reactive.event(S.UNFILTERED_SPOTSTATS, S.UNFILTERED_TRACKSTATS)
    def initialize_inventory():

        Inventory1D.spot_data = S.UNFILTERED_SPOTSTATS.get()
        Inventory1D.track_data = S.UNFILTERED_TRACKSTATS.get()

        threshold_ids = S.THRESHOLDS_ID.get()

        Inventory1D.id_idx = np.arange(threshold_ids)

        idxes = threshold_ids + 1

        property = [None] * idxes
        filter = [None] * idxes
        selection = [None] * idxes

        for idx in range(idxes):
            id = idx + 1

            try:
                property[idx] = input[f"threshold_property_{id}"]()
                ttype = input[f"threshold_type_{id}"]()

                match ttype:

                    case "Literal":
                        filter[idx] = ttype, None

                    case "Normalized 0-1":
                        filter[idx] = ttype, None

                    case "N-tile":
                        filter[idx] = ttype, input[f"threshold_ntile_{id}"]()

                    case "Relative to...":

                        ref = input[f"reference_value_{id}"]()

                        if ref == "My own value":
                            ref = input[f"my_own_value_{id}"]()

                        filter[idx] = ttype, ref

                selection[idx] = input[f"threshold_slider_{id}"]()

            except Exception:
                pass

            if property[idx] is None:
                property[idx] = 'Track displacement'

            if filter[idx] is None:
                filter[idx] = ('Literal', None)

        Inventory1D.property = property
        Inventory1D.filter = filter
        Inventory1D.selection = selection

        Filter1D().Initialize()

        S.THRESHOLDS.set({
            idx: {
                "property": Inventory1D.property[idx],
                "filter": Inventory1D.filter[idx],
                "selection": Inventory1D.selection[idx],
                "mask": Inventory1D.mask[idx],
                "series": Inventory1D.series[idx],
                "ambit": Inventory1D.ambit[idx],
            } for idx in Inventory1D.id_idx
        })



    def render_threshold_controls(inventory, idx, id, min=0, max=100, step=1):
        
        @output(id=f"manual_threshold_value_setting_placeholder_{id}")
        @render.ui
        def manual_threshold_value_setting():

            return ui.row(
                ui.column(6, ui.input_numeric(
                    f"floor_threshold_value_{id}",
                    label="min",
                    value=min,
                    min=min,
                    max=max,
                    step=step
                )),
                ui.column(6, ui.input_numeric(
                    f"ceil_threshold_value_{id}",
                    label="max",
                    value=max,
                    min=min,
                    max=max,
                    step=step
                )),
            )

        @output(id=f"threshold_slider_placeholder_{id}")
        @render.ui
        def threshold_slider():

            return ui.input_slider(
                f"threshold_slider_{id}",
                label=None,
                min=min,
                max=max,
                value=(min, max),
                step=step
            )
        

    def update_threshold_controls(inventory, idx, id):

        min, max, step = inventory["ambit"]
        selection = inventory["selection"]

        ui.update_slider(
            f"threshold_slider_{id}",
            min=min,
            max=max,
            step=step,
            value=selection
        )

        ui.update_numeric(
            f"floor_threshold_value_{id}",
            min=min,
            max=max,
            step=step,
            value=selection[0]
        )

        ui.update_numeric(
            f"ceil_threshold_value_{id}",
            min=min,
            max=max,
            step=step,
            value=selection[1]
        )


    # def render_threshold_hist(inventory, idx, id):
    @reactive.Effect
    def _():
        for idx in Inventory1D.id_idx:
            id = idx + 1

            @output(id=f"thresholding_histogram_placeholder_{id}")
            @render.plot
            def threshold_histogram():

                inventory = S.THRESHOLDS.get()[idx]

                print("Rendering histogram for threshold ID: ", id, " at index: ", idx)

                _color = 'black' if input.app_theme() == "Shiny" else 'white'
                _marker_color = '#337ab7' if input.app_theme() == "Shiny" else '#a15c5c'

                series = inventory["series"]

                print(f"filter: {inventory['filter']}")
                print(f"selection: {inventory['selection']}")
                req(not is_empty(series, details=True))

                req(not None in [
                    inventory["filter"][0], 
                    inventory["selection"]
                ])

                print("Series length:", len(series))

                print(inventory["filter"])
                filter = inventory["filter"]
                
                bottom, top = inventory["selection"]
                
                bins = get_bins()

                fig, ax = plt.subplots()
                    
                # if filter is "Normalized 0-1":
                #     try: series = (series - series.min()) / (series.max() - series.min())
                #     except ZeroDivisionError: series = 0

                n, bins, patches = ax.hist(series, bins=bins, density=False)

                relative = False

                match filter[0]:

                    case "Literal":
                        relative = False

                    case "Normalized 0-1":
                        relative = False

                    case "N-tile":
                        bottom, top = bottom/100, top/100
                        if not 0 <= bottom <= 1 or not 0 <= top <= 1:
                            bottom, top = 0, 1

                        # Convert n-tile values to actual data values
                        bottom = np.quantile(series, bottom)
                        top = np.quantile(series, top)

                        relative = False

                    case "Relative to...":
                        relative = True
                        reference = filter[1]
                        try:
                            bottom, top = sorted([abs(bottom), abs(top)])
                        except Exception:
                            bottom, top = 0, 0


                for i in range(len(patches)):

                    if filter[0] == "Relative to...":
                        bounds = Filter1D.intersects_symmetric(i, bins, bottom, top, reference)
                    else:
                        bounds = Filter1D.bin_bounds(i, bins, bottom, top)

                    if bounds:
                        patches[i].set_facecolor(_marker_color)
                    else:
                        patches[i].set_facecolor("grey")

                    

                kde = gaussian_kde(series)
                x_kde = np.linspace(bins[0], bins[-1], 500)
                y_kde = kde(x_kde)
                y_kde_scaled = y_kde * (n.max() / y_kde.max()) if y_kde.max() != 0 else y_kde
                
                ax.plot(x_kde, y_kde_scaled, color=_color, linewidth=1.5)

                if relative: 
                    ax.axvline(reference, linestyle="--", linewidth=1, color=_color)

                ax.set_xticks([]); ax.set_yticks([])  
                ax.spines[["top", "left", "right"]].set_visible(False)

                fig.set_facecolor('none')
                ax.set_facecolor('none')
                
                return fig
            

    @reactive.Effect
    @reactive.event(input.append_threshold, S.UNFILTERED_SPOTSTATS, S.UNFILTERED_TRACKSTATS)
    def render_threshold():
        
        id = S.THRESHOLDS_ID.get()
        idx = id - 1

        req(idx in S.THRESHOLDS.get().keys())

        inventory = S.THRESHOLDS.get()[idx]

        try:
            ambit = inventory["ambit"]
            if ambit is None:
                return
        except Exception:
            return

        render_threshold_controls(inventory, idx, id, *ambit)
        # render_threshold_hist(inventory, idx, id)


    @reactive.calc
    def update_thresholds():

        req(at_idx.is_set())
        idx = at_idx.get()

        inventory = S.THRESHOLDS.get()[idx]

        # render_threshold_hist(inventory, idx, idx + 1)
        update_threshold_controls(inventory, idx, idx + 1)




    @reactive.effect
    @reactive.event(input.add_input)
    def _():

        print("____________________________________")
        print("Thresholds inventory:")
        print(S.THRESHOLDS.get())
        print(S.THRESHOLDS.get()[S.THRESHOLDS_ID.get()-1])
        print(S.THRESHOLDS.get()[S.THRESHOLDS_ID.get()-1]["ambit"])


#     # _ _ _ _ SYNCING SLIDER/MANUAL-SETTING VALUES _ _ _ _

    # def sync_threshold_values(id):

    #     @DebounceEffect(0.5)
    #     @reactive.Effect
    #     @reactive.event(
    #         input[f"floor_threshold_value_{id}"],
    #         input[f"ceil_threshold_value_{id}"],
    #         )
    #     def sync_with_manual_threshold_value_setting():
            
    #         # Read without creating extra reactive deps
    #         with reactive.isolate():
    #             try:
    #                 slider_vals = input[f"threshold_slider_{id}"]()
    #             except Exception:
    #                 slider_vals = (None, None)
    #             try:
    #                 cur_floor = input[f"floor_threshold_value_{id}"]()
    #                 cur_ceil  = input[f"ceil_threshold_value_{id}"]()
    #             except Exception:
    #                 return

    #             # Validate + normalize
    #             if not isinstance(cur_floor, (int, float)) or not isinstance(cur_ceil, (int, float)):
    #                 return
    #             if cur_floor > cur_ceil:
    #                 cur_floor, cur_ceil = cur_ceil, cur_floor

    #             # Only push if changed
    #             existing = slider_vals if (isinstance(slider_vals, (tuple, list)) and len(slider_vals) == 2) else (None, None)
    #             if cur_floor != existing[0] or cur_ceil != existing[1]:
    #                 ui.update_slider(f"threshold_slider_{id}", value=(cur_floor, cur_ceil))

    #     @DebounceEffect(0.5)
    #     @reactive.Effect
    #     @reactive.event(input[f"threshold_slider_{id}"])
    #     def sync_with_threshold_slider():
            
    #         # Read without creating extra reactive deps
    #         with reactive.isolate():
    #             try:
    #                 cur_floor = input[f"floor_threshold_value_{id}"]()
    #                 cur_ceil  = input[f"ceil_threshold_value_{id}"]()
    #             except Exception:
    #                 return
    #             try:
    #                 slider_vals = input[f"threshold_slider_{id}"]()
    #             except Exception:
    #                 slider_vals = (None, None)

    #             # Validate + normalize
    #             if not isinstance(slider_vals[0], (int, float)) or not isinstance(slider_vals[1], (int, float)):
    #                 return
    #             if slider_vals[0] > slider_vals[1]:
    #                 slider_vals = (slider_vals[1], slider_vals[0])


    #             # Only push if changed
    #             existing = (cur_floor, cur_ceil) if (isinstance(cur_floor, (int, float)) and isinstance(cur_ceil, (int, float))) else (None, None)
    #             if slider_vals != existing:
    #                 ui.update_numeric(f"floor_threshold_value_{id}", value=float(slider_vals[0]))
    #                 ui.update_numeric(f"ceil_threshold_value_{id}", value=float(slider_vals[1]))

    # # @ThrottleCalc(50)
    # @reactive.Effect
    # def sync_thresholds():
    #     for id in range(1, S.THRESHOLDS_ID.get()+1):
    #         sync_threshold_values(id)


    # _ _ _ _ UPDATING THRESHOLDS ON CHANGE _ _ _ _
    
#     def update_thresholds_wired(id, min, max, step):

#         @DebounceEffect(0.5)
#         @reactive.Effect
#         @reactive.event(input[f"threshold_slider_{id}"])
#         def pass_thresholded_data():

#             with reactive.isolate():
#                 thresholds = S.THRESHOLDS.get()

#                 data = thresholds.get(id)
#                 req(data is not None and data.get("spots") is not None and data.get("tracks") is not None)

#                 spot_data = data.get("spots")
#                 track_data = data.get("tracks")

#                 filter = Filter1D().filter_data(
#                     df=spot_data if input[f"threshold_property_{id}"]() in S.SPOTSTATS_COLUMNS.get() else track_data,
#                     threshold=input[f"threshold_slider_{id}"](),
#                     property=input[f"threshold_property_{id}"](),
#                     threshold_type=input[f"threshold_type_{id}"](),
#                     reference=input[f"reference_value_{id}"](),
#                     reference_value=input[f"my_own_value_{id}"]()
#                 )

#                 spots_output = spot_data.loc[filter.index.intersection(spot_data.index)]
#                 tracks_output = track_data.loc[filter.index.intersection(track_data.index)]

#                 thresholds |= {id+1: {"spots": spots_output, "tracks": tracks_output}}
#                 S.THRESHOLDS.set(thresholds)

# # --------------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------------

#         @reactive.Effect
#         @reactive.event(
#             input[f"threshold_property_{id}"],
#             input[f"threshold_type_{id}"],
#                 input[f"threshold_ntile_{id}"],
#                 input[f"reference_value_{id}"],
#                 input[f"my_own_value_{id}"],
#         )
#         def _():

#             print("__________________________________________________________")
#             print(f"Updating threshold ID {id}")
#             print(f"New min: {min}, max: {max}, step: {step}")
            
#             ui.update_slider(
#                 f"threshold_slider_{id}",
#                 min=min,
#                 max=max,
#                 step=step
#             )

#             ui.update_numeric(
#                 f"floor_threshold_value_{id}",
#                 min=min,
#                 max=max,
#                 step=step
#             )

#             ui.update_numeric(
#                 f"ceil_threshold_value_{id}",
#                 min=min,
#                 max=max,
#                 step=step
#             )

#         @DebounceEffect(0.5)
#         @reactive.Effect
#         @reactive.event(input[f"threshold_slider_{id}"])
#         def update_next_threshold():
#             """
#             Updating the slider updates the manual threshold values setting as well as the filte histogram.
#             """

#             with reactive.isolate():
#                 thresholds = S.THRESHOLDS.get()
                
#                 try:
#                     data = thresholds.get(id+1)
#                 except Exception:
#                     return

#                 req(data is not None and data.get("spots") is not None and data.get("tracks") is not None)

#                 property_name = input[f"threshold_property_{id+1}"]()
#                 threshold_type = input[f"threshold_type_{id+1}"]()
#                 req(property_name and threshold_type)

#                 if property_name in S.SPOTSTATS_COLUMNS.get():
#                     data = data.get("spots")
#                 else:
#                     data = data.get("tracks")
                
#                 new_min, new_max, steps, _ = Filter1D().get_threshold_params(
#                     data=data,
#                     property_name=property_name,
#                     threshold_type=threshold_type,
#                     quantile=input[f"threshold_ntile_{id+1}"](),
#                     reference=input[f"reference_value_{id+1}"](),
#                     reference_value=input[f"my_own_value_{id+1}"]()
#                 )

#                 # Preserve existing selection if present; clamp to new bounds
#                 try:
#                     cur_low, cur_high = input[f"threshold_slider_{id+1}"]()
#                 except Exception:
#                     cur_low, cur_high = (None, None)

#                 if not isinstance(cur_low, (int, float)) or not isinstance(cur_high, (int, float)):
#                     new_low, new_high = (new_min, new_max)
#                 else:
#                     if cur_low > cur_high:
#                         cur_low, cur_high = cur_high, cur_low
#                     new_low = max(new_min, cur_low)
#                     new_high = min(new_max, cur_high)
#                     if new_low > new_high:
#                         new_low, new_high = (new_min, new_max)

#                 ui.update_slider(
#                     f"threshold_slider_{id+1}",
#                     min=new_min,
#                     max=new_max,
#                     value=(new_low, new_high),
#                     step=steps
#                 )

#                 # Keep numeric inputs in sync without resetting user selection
#                 ui.update_numeric(
#                     f"floor_threshold_value_{id+1}",
#                     min=new_min,
#                     max=new_max,
#                     value=float(new_low)
#                 )
#                 ui.update_numeric(
#                     f"ceil_threshold_value_{id+1}",
#                     min=new_min,
#                     max=new_max,
#                     value=float(new_high)
#                 )

                

    # @DebounceEffect(0.5)
    # @reactive.Effect
    # def update_thresholds():
    #     try:
    #         thresholds = S.THRESHOLDS.get()
    #     except Exception:
    #         return
    #     if not thresholds:
    #         return
        
    #     for id in range(1, S.THRESHOLDS_ID.get()+1):
    #         update_thresholds_wired(id)




    



    @reactive.Effect
    def _():

        Inventory1D.id_idx = np.arange(S.THRESHOLDS_ID.get())

        for idx in Inventory1D.id_idx:

            id = idx + 1

            @reactive.Effect
            @reactive.event(
                input[f"threshold_property_{id}"],
                input[f"threshold_type_{id}"],
                    input[f"threshold_ntile_{id}"],
                    input[f"reference_value_{id}"],
                    input[f"my_own_value_{id}"],
                input[f"threshold_slider_{id}"],
            )
            def _():

                p = input[f"threshold_property_{id}"]()
                t = input[f"threshold_type_{id}"]()
                s = input[f"threshold_slider_{id}"]()

                match t:
                    case "Literal":
                        f = t, None
                    case "Normalized 0-1":
                        f = t, None

                    case "N-tile":
                        f = t, input[f"threshold_ntile_{id}"]()

                    case "Relative to...":

                        ref = input[f"reference_value_{id}"]()

                        if ref == "My own value":
                            ref = input[f"my_own_value_{id}"]()

                        f = t, ref

                Inventory1D.property[idx] = p
                Inventory1D.filter[idx] = f
                Inventory1D.selection[idx] = s

                Filter1D().Downstream(idx)

                S.THRESHOLDS.set({
                    idx: {
                        "property": Inventory1D.property[idx],
                        "filter": Inventory1D.filter[idx],
                        "selection": Inventory1D.selection[idx],
                        "mask": Inventory1D.mask[idx],
                        "series": Inventory1D.series[idx],
                        "ambit": Inventory1D.ambit[idx],
                    } for idx in Inventory1D.id_idx
                })
                
                at_idx.unset()
                at_idx.set(idx)

                update_thresholds()

                # print("__________________________________________________________")
                # print(f"Syncing threshold ID {id}...")

                # print("------------------------------------")
                # print(f"Inventory1D.property: {Inventory1D.property}")
                # print(f"Inventory1D.property[{idx}]: {Inventory1D.property[idx]}")
                # print(f"Property: {input[f'threshold_property_{id}']()}")
                # print("------------------------------------")
                # print(f"all masks: {Inventory1D.mask}")
                # print(f"Mask: {Inventory1D.mask[idx]}")
                # print("------------------------------------")
                # print(f"all filters: {Inventory1D.filter}")
                # print(f"Filter: {Inventory1D.filter[idx]}")
                # print("------------------------------------")
                # print(f"all selections: {Inventory1D.selection}")
                # print(f"Selection: {Inventory1D.selection[idx]}")
                # print("------------------------------------")
                # print(f"all series: {Inventory1D.series}")
                # print(f"Series: {Inventory1D.series[idx]}")
                # print("------------------------------------")
                # print(f"all ambit: {Inventory1D.ambit}")
                # print(f"Ambit: {Inventory1D.ambit[idx]}")
                # print("------------------------------------")
                # print(" ")
                # print(" ")

        # @reactive.Effect
        # def _():
        #     """
        #     Update the thresholds storage after any filtering change.
        #     """

        #     print("Updating thresholds storage...")
        #     print("Ambit:", Inventory1D.ambit)

        #     ambit = Inventory1D.ambit[idx]
        #     req(ambit is not None)

        #     update_thresholds_wired(id, *ambit)

            


    @reactive.Effect
    @reactive.event(input.remove_threshold)
    def _():

        last_idx = S.THRESHOLDS_ID.get() - 1

        Filter1D().PopLast()

        S.THRESHOLDS.set(S.THRESHOLDS.get().pop(last_idx, None))


    

    # _ _ _ _ SETTING THE THRESHOLDS _ _ _ _

    # @reactive.Effect
    # @reactive.event(input.set_threshold)
    # def threshold_data():
    #     try: 
    #         thresholds = S.THRESHOLDS.get()
    #         latest = thresholds.get(list(thresholds.keys())[-1])
    #         req(latest is not None and latest.get("spots") is not None and latest.get("tracks") is not None)
    #     except Exception:
    #         return

    #     spots_filtered = pd.DataFrame(
    #         latest.get("spots") if latest is not None and isinstance(latest, dict) else S.UNFILTERED_SPOTSTATS.get()
    #     )
    #     tracks_filtered = pd.DataFrame(
    #         latest.get("tracks") if latest is not None and isinstance(latest, dict) else S.UNFILTERED_TRACKSTATS.get()
    #     )
    #     frames_filtered = Frames(spots_filtered)
    #     tintervals_filtered = TimeIntervals(spots_filtered)()

    #     # --- keep labeling-derived columns (e.g., colors) even if threshold snapshots were created before labeling ---
    #     color_cols = ("Replicate color", "Condition color")
        
    #     base_spots = S.UNFILTERED_SPOTSTATS.get()
    #     base_tracks = S.UNFILTERED_TRACKSTATS.get()
    #     base_frames = S.UNFILTERED_FRAMESTATS.get()
    #     base_tintervals = S.UNFILTERED_TINTERVALSTATS.get()

    #     req(df.empty and df is None for df in [base_spots, base_tracks, base_frames, base_tintervals])
    #     for c in color_cols:
    #         try: spots_filtered[c] = base_spots.reindex(spots_filtered.index)[c].to_numpy()
    #         except Exception: pass
    #         try: tracks_filtered[c] = base_tracks.reindex(tracks_filtered.index)[c].to_numpy()
    #         except Exception: pass
    #         try: frames_filtered[c] = base_frames.reindex(frames_filtered.index)[c].to_numpy()
    #         except Exception: pass
    #         try: tintervals_filtered[c] = base_tintervals.reindex(tintervals_filtered.index)[c].to_numpy()
    #         except Exception: pass
        

    #     S.SPOTSTATS.set(spots_filtered)
    #     S.TRACKSTATS.set(tracks_filtered)
    #     S.FRAMESTATS.set(frames_filtered)
    #     S.TINTERVALSTATS.set(tintervals_filtered)



