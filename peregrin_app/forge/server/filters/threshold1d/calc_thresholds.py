import time
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import shiny.ui as ui
from shiny import render, reactive, req, ui

from peregrin_app.src.code._general import clock
from src.code import Frames, TimeIntervals, Inventory1D, Filter1D, DebounceCalc, is_empty



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

        thresholds_id = S.THRESHOLDS_ID.get()
        idxes = thresholds_id + 1

        Inventory1D.id_idx = np.arange(idxes)

        property = [None] * thresholds_id
        filter = [None] * thresholds_id
        selection = [None] * thresholds_id

        for idx in range(thresholds_id):
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

        try:
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

        except Exception:
            pass



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
        
        # Sync slider to numeric inputs
        @reactive.effect
        @reactive.event(input[f"floor_threshold_value_{id}"], input[f"ceil_threshold_value_{id}"])
        def _sync_slider_from_numeric():
            floor_val = input[f"floor_threshold_value_{id}"]()
            ceil_val = input[f"ceil_threshold_value_{id}"]()
            
            if floor_val is not None and ceil_val is not None:
                with reactive.isolate():
                    current_slider = input[f"threshold_slider_{id}"]()
                    if current_slider != (floor_val, ceil_val):
                        ui.update_slider(
                            f"threshold_slider_{id}",
                            value=(floor_val, ceil_val)
                        )
        
        # Sync numeric inputs to slider
        @reactive.effect
        @reactive.event(input[f"threshold_slider_{id}"])
        def _sync_numeric_from_slider():
            slider_val = input[f"threshold_slider_{id}"]()
            
            if slider_val is not None:
                with reactive.isolate():
                    current_floor = input[f"floor_threshold_value_{id}"]()
                    current_ceil = input[f"ceil_threshold_value_{id}"]()
                    
                    if current_floor != slider_val[0]:
                        ui.update_numeric(
                            f"floor_threshold_value_{id}",
                            value=slider_val[0]
                        )
                    
                    if current_ceil != slider_val[1]:
                        ui.update_numeric(
                            f"ceil_threshold_value_{id}",
                            value=slider_val[1]
                        )

    def update_threshold_controls(inventory, idx, id):

        min, max, step = inventory["ambit"]
        selection = inventory["selection"]

        with reactive.isolate():

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


    @reactive.calc
    def update_histogram():
        idx = at_idx.get()

        if idx is not None:
            
            id = idx + 1

            @output(id=f"thresholding_histogram_placeholder_{id}")
            @render.plot
            def threshold_histogram():

                inventory = S.THRESHOLDS.get()[idx]

                _color = 'black' if input.app_theme() == "Shiny" else 'white'
                _marker_color = '#337ab7' if input.app_theme() == "Shiny" else '#a15c5c'

                series = inventory["series"]

                req(not is_empty(series))

                req(not None in [
                    inventory["filter"][0], 
                    inventory["selection"]
                ])

                filter = inventory["filter"]
                
                bottom, top = inventory["selection"]
                
                bins = get_bins()

                fig, ax = plt.subplots()
                    
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

    

    @reactive.calc
    def render_threshold():

        id = S.THRESHOLDS_ID.get()
        idx = id - 1

        inventory = S.THRESHOLDS.get()
        inventory = inventory.get(idx)

        if inventory is None or any(v is None for v in inventory.values()):
            return
        
        render_threshold_controls(inventory, idx, id, *inventory["ambit"])
        update_histogram()


    @reactive.Effect
    @reactive.event(S.UNFILTERED_SPOTSTATS, S.UNFILTERED_TRACKSTATS)
    def _():
        if is_empty(S.UNFILTERED_SPOTSTATS.get()) and is_empty(S.UNFILTERED_TRACKSTATS.get()):
            return
        
        if S._init_thresh.get():
            render_threshold()
            S._init_thresh.set(False)


    @reactive.Effect
    @reactive.event(input.append_threshold)
    def _():
        render_threshold()



    @reactive.calc
    def update_thresholds():

        req(at_idx.is_set())
        idx = at_idx.get()

        inventory = S.THRESHOLDS.get()[idx]

        update_threshold_controls(inventory, idx, idx + 1)
        update_histogram()


    @reactive.Effect
    def _():

        print(f"input.remove_threshold(): {input.remove_threshold()}")
        print(f"len(Inventory1D.id_idx): {len(Inventory1D.id_idx)}")
        print(f"S.THRESHOLDS_ID.get(): {S.THRESHOLDS_ID.get()}")

        if (input.remove_threshold() > 0
            and len(Inventory1D.id_idx) > S.THRESHOLDS_ID.get()):

            Inventory1D.id_idx = np.arange(S.THRESHOLDS_ID.get() + 1)

            return
        
        Inventory1D.id_idx = np.arange(S.THRESHOLDS_ID.get() + 1)

        for idx in Inventory1D.id_idx[:-1]:
            id = idx + 1

            @reactive.Effect
            @reactive.event(
                input[f"threshold_property_{id}"],
                input[f"threshold_type_{id}"],
                    input[f"threshold_ntile_{id}"],
                    input[f"reference_value_{id}"],
                    input[f"my_own_value_{id}"],
                input[f"threshold_slider_{id}"],
                input.app_theme,
            )
            def _():

                if idx not in Inventory1D.id_idx[:-1]:
                    return

                if None in [
                    input[f"threshold_property_{id}"](),
                    input[f"threshold_type_{id}"](),
                    input[f"threshold_slider_{id}"](),
                ]: return

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
                print(f"Updating threshold idx {idx}:")
                print(f"  - Inventory1D.property = {Inventory1D.property}")
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
                
                at_idx.set(idx)

                update_thresholds()



    @reactive.Effect
    @reactive.event(input.remove_threshold)
    def _():

        with reactive.isolate():

            Filter1D().PopLast()

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
            


    

    # _ _ _ _ SETTING THE THRESHOLDS _ _ _ _

    @reactive.Effect
    @reactive.event(input.set_threshold)
    def threshold_data():
    
        try:

            spotstats, trackstats, framestats, tintervalstats = Filter1D().Apply()
            # Set all stats at once to trigger reactive updates
            S.SPOTSTATS.set(spotstats)
            S.TRACKSTATS.set(trackstats)
            S.FRAMESTATS.set(framestats)
            S.TINTERVALSTATS.set(tintervalstats)

            print(f"Thresholds set successfully:")
            print(f"  - Spots: {len(spotstats)} rows")
            print(f"  - Tracks: {len(trackstats)} rows")
            print(f"  - Frames: {len(framestats)} rows")
            print(f"  - Intervals: {len(tintervalstats)} rows")

        except Exception as e:
            print(f"Error setting thresholds: {e}")
            traceback.print_exc()



