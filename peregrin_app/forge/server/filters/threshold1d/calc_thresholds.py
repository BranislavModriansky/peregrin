import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import shiny.ui as ui
from shiny import render, reactive, req, ui
from src.code import Frames, TimeIntervals, Metrics, Threshold, DebounceCalc, ThrottleCalc, DebounceEffect


THRESH = Threshold(eps=1e-12)


def mount_thresholds_calc(input, output, session, S):

    @DebounceCalc(0.5)
    @reactive.calc
    def get_bins():
        return input.bins() if input.bins() is not None and input.bins() != 0 else 15


    def render_threshold_container(id, thresholds):
        
        @output(id=f"manual_threshold_value_setting_placeholder_{id}")
        @render.ui
        def manual_threshold_value_setting():

            filt_data = thresholds.get(id)
            req(filt_data is not None and filt_data.get("spots") is not None and filt_data.get("tracks") is not None)

            

            property_name = input[f"threshold_property_{id}"]()
            threshold_type = input[f"threshold_type_{id}"]()
            req(property_name and threshold_type)

            if property_name in S.SPOTSTATS_COLUMNS.get():
                data = thresholds.get(id).get("spots")
                filt_data = filt_data.get("spots")
            else:
                data = thresholds.get(id).get("tracks")
                filt_data = filt_data.get("tracks")

            global_min, global_max = THRESH.get_range(data, input[f"threshold_property_{id}"]())
            
            local_min, local_max, steps, _ = THRESH.get_threshold_params(
                data=filt_data,
                property_name=property_name,
                threshold_type=threshold_type,
                quantile=input[f"threshold_quantile_{id}"](),
                reference=input[f"reference_value_{id}"](),
                reference_value=input[f"my_own_value_{id}"]()
            )
            
            return ui.row(
                ui.column(6, ui.input_numeric(
                    f"floor_threshold_value_{id}",
                    label="min",
                    value=local_min,
                    min=global_min,
                    max=global_max,
                    step=steps
                )),
                ui.column(6, ui.input_numeric(
                    f"ceil_threshold_value_{id}",
                    label="max",
                    value=local_max,
                    min=global_min,
                    max=global_max,
                    step=steps
                )),
            )

        @output(id=f"threshold_slider_placeholder_{id}")
        @render.ui
        def threshold_slider():
            
            filt_data = thresholds.get(id)
            req(filt_data is not None and filt_data.get("spots") is not None and filt_data.get("tracks") is not None)

            property_name = input[f"threshold_property_{id}"]()
            threshold_type = input[f"threshold_type_{id}"]()
            req(property_name and threshold_type)

            if property_name in S.SPOTSTATS_COLUMNS.get():
                data = thresholds.get(id).get("spots")
                filt_data = filt_data.get("spots")
            else:
                data = thresholds.get(id).get("tracks")
                filt_data = filt_data.get("tracks")

            global_min, global_max = THRESH.get_range(data, property_name)
            
            local_min, local_max, steps, _ = THRESH.get_threshold_params(
                data=filt_data,
                property_name=property_name,
                threshold_type=threshold_type,
                quantile=input[f"threshold_quantile_{id}"](),
                reference=input[f"reference_value_{id}"](),
                reference_value=input[f"my_own_value_{id}"]()
            )

            return ui.input_slider(
                f"threshold_slider_{id}",
                label=None,
                min=global_min,
                max=global_max,
                value=(local_min, local_max),
                step=steps
            )
        
        @output(id=f"thresholding_histogram_placeholder_{id}")
        @render.plot
        def threshold_histogram():

            _color = 'black' if input.app_theme() == "Shiny" else 'white'
            _marker_color = '#337ab7' if input.app_theme() == "Shiny" else '#a15c5c'

            data = thresholds.get(id)
            req(data is not None and data.get("spots") is not None and data.get("tracks") is not None)

            if input[f"threshold_property_{id}"]() in S.SPOTSTATS_COLUMNS.get():
                data = data.get("spots")
            else:
                data = data.get("tracks")

            req(data is not None and not data.empty)
            
            property = input[f"threshold_property_{id}"]()
            threshold_type = input[f"threshold_type_{id}"]()
            try:
                slider_low_pct, slider_high_pct = input[f"threshold_slider_{id}"]()
            except Exception:
                return
            
            bins = get_bins()

            fig, ax = plt.subplots()

            if threshold_type == "Literal":
                
                values = data[property].dropna()
                n, bins, patches = ax.hist(values, bins=bins, density=False)

                # Color threshold
                for i in range(len(patches)):
                    if bins[i] < slider_low_pct or bins[i+1] > slider_high_pct:
                        patches[i].set_facecolor("grey")
                    else:
                        patches[i].set_facecolor(_marker_color)

                # Add KDE curve (scaled to match histogram)
                kde = gaussian_kde(values)
                x_kde = np.linspace(bins[0], bins[-1], 500)
                y_kde = kde(x_kde)
                # Scale KDE to histogram
                y_kde_scaled = y_kde * (n.max() / y_kde.max())
                ax.plot(x_kde, y_kde_scaled, color=_color, linewidth=1.5)

                ax.set_xticks([])  # Remove x-axis ticks
                ax.set_yticks([])  # Remove y-axis ticks
                ax.spines[["top", "left", "right"]].set_visible(False)

            
            if threshold_type == "Normalized 0-1":

                values = data[property].dropna()
                try:
                    normalized = (values - values.min()) / (values.max() - values.min())
                except ZeroDivisionError:
                    normalized = 0

                
                n, bins, patches = ax.hist(normalized, bins=bins, density=False)

                # Color threshold
                for i in range(len(patches)):
                    if bins[i] < slider_low_pct or bins[i+1] > slider_high_pct:
                        patches[i].set_facecolor("grey")
                    else:
                        patches[i].set_facecolor(_marker_color)

                # Add KDE curve (scaled to match histogram)
                kde = gaussian_kde(normalized)
                x_kde = np.linspace(bins[0], bins[-1], 500)
                y_kde = kde(x_kde)
                # Scale KDE to histogram
                y_kde_scaled = y_kde * (n.max() / y_kde.max())
                ax.plot(x_kde, y_kde_scaled, color=_color, linewidth=1.5)

                ax.set_xticks([])  # Remove x-axis ticks
                ax.set_yticks([])  # Remove y-axis ticks
                ax.spines[["top", "left", "right"]].set_visible(False)


            if threshold_type == "Quantile":

                values = data[property].dropna()
                n, bins, patches = ax.hist(values, bins=bins, density=False)

                # Get slider quantile values, 0-100 scale
                slider_low, slider_high = slider_low_pct / 100, slider_high_pct / 100

                if not 0 <= slider_low <= 1 or not 0 <= slider_high <= 1:
                    slider_low, slider_high = 0, 1

                # Convert slider percentiles to actual values
                lower_bound = np.quantile(values, slider_low)
                upper_bound = np.quantile(values, slider_high)

                # Color histogram based on slider quantile bounds
                for i in range(len(patches)):
                    bin_start, bin_end = bins[i], bins[i + 1]
                    if bin_end < lower_bound or bin_start > upper_bound:
                        patches[i].set_facecolor("grey")
                    else:
                        patches[i].set_facecolor(_marker_color)

                # KDE curve
                kde = gaussian_kde(values)
                x_kde = np.linspace(values.min(), values.max(), 500)
                y_kde = kde(x_kde)
                y_kde_scaled = y_kde * (n.max() / y_kde.max()) if y_kde.max() != 0 else y_kde
                ax.plot(x_kde, y_kde_scaled, color=_color, linewidth=1.5)

                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines[["top", "left", "right"]].set_visible(False)
                

            if threshold_type == "Relative to...":
                reference = input[f"reference_value_{id}"]()
                if reference == "Mean":
                    reference_value = float(data[property].dropna().mean())
                elif reference == "Median":
                    reference_value = float(data[property].dropna().median())
                elif reference == "My own value":
                    try:
                        mv = input[f"my_own_value_{id}"]() if input[f"my_own_value_{id}"]() is not None else 0.0
                        reference_value = float(mv) if isinstance(mv, (int, float)) else 0.0
                    except Exception:
                        reference_value = 0.0
                else:
                    return

                # Build histogram in "shifted" space (centered at 0 = reference)
                shifted = data[property].dropna() - reference_value
                n, bins, patches = ax.hist(shifted, bins=bins, density=False)

                # Slider gives distances [low, high] away from the reference
                sel_low, sel_high = input[f"threshold_slider_{id}"]()
                # Normalize order just in case
                if sel_low > sel_high:
                    sel_low, sel_high = sel_high, sel_low

                # Utility: does [bin_start, bin_end] intersect either [-sel_high, -sel_low] or [sel_low, sel_high]?
                def _intersects_symmetric(b0, b1, a, b):
                    # interval A: [-b, -a], interval B: [a, b]
                    left_hit  = (b1 >= -b) and (b0 <= -a)
                    right_hit = (b1 >=  a) and (b0 <=  b)
                    return left_hit or right_hit

                # Color threshold bands: keep bars whose centers fall within the selected annulus
                for i in range(len(patches)):
                    bin_start, bin_end = bins[i], bins[i+1]
                    if _intersects_symmetric(bin_start, bin_end, sel_low, sel_high):
                        patches[i].set_facecolor(_marker_color)
                    else:
                        patches[i].set_facecolor("grey")

                # KDE on shifted values (optional but matches your style)
                kde = gaussian_kde(shifted)
                x_kde = np.linspace(bins[0], bins[-1], 500)
                y_kde = kde(x_kde)
                y_kde_scaled = y_kde * (n.max() / y_kde.max()) if y_kde.max() != 0 else y_kde
                ax.plot(x_kde, y_kde_scaled, color=_color, linewidth=1.5)

                ax.axvline(0, linestyle="--", linewidth=1, color=_color)

                ax.set_xticks([]); ax.set_yticks([])
                ax.spines[["top", "left", "right"]].set_visible(False)

            fig.set_facecolor('none')
            ax.set_facecolor('none')
            
            return fig

    @reactive.Effect
    @reactive.event(input.append_threshold, S.UNFILTERED_SPOTSTATS, S.UNFILTERED_TRACKSTATS)
    def render_threshold():
        threshold_id = S.THRESHOLDS_ID.get()
        
        with reactive.isolate():
            try:
                thresholds = S.THRESHOLDS.get()
            except Exception:
                return
        if not threshold_id or not thresholds:
            return

        render_threshold_container(threshold_id, thresholds)


    # _ _ _ _ SYNCING SLIDER/MANUAL-SETTING VALUES _ _ _ _

    def sync_threshold_values(id):

        @DebounceEffect(0.5)
        @reactive.Effect
        @reactive.event(
            input[f"floor_threshold_value_{id}"],
            input[f"ceil_threshold_value_{id}"],
            )
        def sync_with_manual_threshold_value_setting():
            
            # Read without creating extra reactive deps
            with reactive.isolate():
                try:
                    slider_vals = input[f"threshold_slider_{id}"]()
                except Exception:
                    slider_vals = (None, None)
                try:
                    cur_floor = input[f"floor_threshold_value_{id}"]()
                    cur_ceil  = input[f"ceil_threshold_value_{id}"]()
                except Exception:
                    return

                # Validate + normalize
                if not isinstance(cur_floor, (int, float)) or not isinstance(cur_ceil, (int, float)):
                    return
                if cur_floor > cur_ceil:
                    cur_floor, cur_ceil = cur_ceil, cur_floor

                # Only push if changed
                existing = slider_vals if (isinstance(slider_vals, (tuple, list)) and len(slider_vals) == 2) else (None, None)
                if cur_floor != existing[0] or cur_ceil != existing[1]:
                    ui.update_slider(f"threshold_slider_{id}", value=(cur_floor, cur_ceil))

        @DebounceEffect(0.5)
        @reactive.Effect
        @reactive.event(input[f"threshold_slider_{id}"])
        def sync_with_threshold_slider():
            
            # Read without creating extra reactive deps
            with reactive.isolate():
                try:
                    cur_floor = input[f"floor_threshold_value_{id}"]()
                    cur_ceil  = input[f"ceil_threshold_value_{id}"]()
                except Exception:
                    return
                try:
                    slider_vals = input[f"threshold_slider_{id}"]()
                except Exception:
                    slider_vals = (None, None)

                # Validate + normalize
                if not isinstance(slider_vals[0], (int, float)) or not isinstance(slider_vals[1], (int, float)):
                    return
                if slider_vals[0] > slider_vals[1]:
                    slider_vals = (slider_vals[1], slider_vals[0])


                # Only push if changed
                existing = (cur_floor, cur_ceil) if (isinstance(cur_floor, (int, float)) and isinstance(cur_ceil, (int, float))) else (None, None)
                if slider_vals != existing:
                    ui.update_numeric(f"floor_threshold_value_{id}", value=float(slider_vals[0]))
                    ui.update_numeric(f"ceil_threshold_value_{id}", value=float(slider_vals[1]))

    # @ThrottleCalc(50)
    @reactive.Effect
    def sync_thresholds():
        for id in range(1, S.THRESHOLDS_ID.get()+1):
            sync_threshold_values(id)


    # _ _ _ _ UPDATING THRESHOLDS ON CHANGE _ _ _ _
    
    def update_thresholds_wired(id):

        @DebounceEffect(0.5)
        @reactive.Effect
        @reactive.event(input[f"threshold_slider_{id}"])
        def pass_thresholded_data():

            with reactive.isolate():
                thresholds = S.THRESHOLDS.get()

                data = thresholds.get(id)
                req(data is not None and data.get("spots") is not None and data.get("tracks") is not None)

                spot_data = data.get("spots")
                track_data = data.get("tracks")

                filter = THRESH.filter_data(
                    df=spot_data if input[f"threshold_property_{id}"]() in S.SPOTSTATS_COLUMNS.get() else track_data,
                    threshold=input[f"threshold_slider_{id}"](),
                    property=input[f"threshold_property_{id}"](),
                    threshold_type=input[f"threshold_type_{id}"](),
                    reference=input[f"reference_value_{id}"](),
                    reference_value=input[f"my_own_value_{id}"]()
                )

                spots_output = spot_data.loc[filter.index.intersection(spot_data.index)]
                tracks_output = track_data.loc[filter.index.intersection(track_data.index)]

                thresholds |= {id+1: {"spots": spots_output, "tracks": tracks_output}}
                S.THRESHOLDS.set(thresholds)

# --------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------

        @DebounceEffect(0.5)
        @reactive.Effect
        @reactive.event(input[f"threshold_slider_{id}"])
        def update_next_threshold():
            """
            Updating the slider updates the manual threshold values setting as well as the filte histogram.
            """

            with reactive.isolate():
                thresholds = S.THRESHOLDS.get()
                
                try:
                    data = thresholds.get(id+1)
                except Exception:
                    return

                req(data is not None and data.get("spots") is not None and data.get("tracks") is not None)

                property_name = input[f"threshold_property_{id+1}"]()
                threshold_type = input[f"threshold_type_{id+1}"]()
                req(property_name and threshold_type)

                if property_name in S.SPOTSTATS_COLUMNS.get():
                    data = data.get("spots")
                else:
                    data = data.get("tracks")
                
                new_min, new_max, steps, _ = THRESH.get_threshold_params(
                    data=data,
                    property_name=property_name,
                    threshold_type=threshold_type,
                    quantile=input[f"threshold_quantile_{id+1}"](),
                    reference=input[f"reference_value_{id+1}"](),
                    reference_value=input[f"my_own_value_{id+1}"]()
                )

                # Preserve existing selection if present; clamp to new bounds
                try:
                    cur_low, cur_high = input[f"threshold_slider_{id+1}"]()
                except Exception:
                    cur_low, cur_high = (None, None)

                if not isinstance(cur_low, (int, float)) or not isinstance(cur_high, (int, float)):
                    new_low, new_high = (new_min, new_max)
                else:
                    if cur_low > cur_high:
                        cur_low, cur_high = cur_high, cur_low
                    new_low = max(new_min, cur_low)
                    new_high = min(new_max, cur_high)
                    if new_low > new_high:
                        new_low, new_high = (new_min, new_max)

                ui.update_slider(
                    f"threshold_slider_{id+1}",
                    min=new_min,
                    max=new_max,
                    value=(new_low, new_high),
                    step=steps
                )

                # Keep numeric inputs in sync without resetting user selection
                ui.update_numeric(
                    f"floor_threshold_value_{id+1}",
                    min=new_min,
                    max=new_max,
                    value=float(new_low)
                )
                ui.update_numeric(
                    f"ceil_threshold_value_{id+1}",
                    min=new_min,
                    max=new_max,
                    value=float(new_high)
                )

                

    @DebounceEffect(0.5)
    @reactive.Effect
    def update_thresholds():
        try:
            thresholds = S.THRESHOLDS.get()
        except Exception:
            return
        if not thresholds:
            return
        
        for id in range(1, S.THRESHOLDS_ID.get()+1):
            update_thresholds_wired(id)
    

    # _ _ _ _ SETTING THE THRESHOLDS _ _ _ _

    @reactive.Effect
    @reactive.event(input.set_threshold)
    def threshold_data():
        try: 
            thresholds = S.THRESHOLDS.get()
            latest = thresholds.get(list(thresholds.keys())[-1])
            req(latest is not None and latest.get("spots") is not None and latest.get("tracks") is not None)
        except Exception:
            return

        spots_filtered = pd.DataFrame(
            latest.get("spots") if latest is not None and isinstance(latest, dict) else S.UNFILTERED_SPOTSTATS.get()
        )
        tracks_filtered = pd.DataFrame(
            latest.get("tracks") if latest is not None and isinstance(latest, dict) else S.UNFILTERED_TRACKSTATS.get()
        )
        frames_filtered = Frames(spots_filtered)
        tintervals_filtered = TimeIntervals(spots_filtered)()

        # --- keep labeling-derived columns (e.g., colors) even if threshold snapshots were created before labeling ---
        color_cols = ("Replicate color", "Condition color")
        
        base_spots = S.UNFILTERED_SPOTSTATS.get()
        base_tracks = S.UNFILTERED_TRACKSTATS.get()
        base_frames = S.UNFILTERED_FRAMESTATS.get()
        base_tintervals = S.UNFILTERED_TINTERVALSTATS.get()

        req(df.empty and df is None for df in [base_spots, base_tracks, base_frames, base_tintervals])
        for c in color_cols:
            try: spots_filtered[c] = base_spots.reindex(spots_filtered.index)[c].to_numpy()
            except Exception: pass
            try: tracks_filtered[c] = base_tracks.reindex(tracks_filtered.index)[c].to_numpy()
            except Exception: pass
            try: frames_filtered[c] = base_frames.reindex(frames_filtered.index)[c].to_numpy()
            except Exception: pass
            try: tintervals_filtered[c] = base_tintervals.reindex(tintervals_filtered.index)[c].to_numpy()
            except Exception: pass
        

        S.SPOTSTATS.set(spots_filtered)
        S.TRACKSTATS.set(tracks_filtered)
        S.FRAMESTATS.set(frames_filtered)
        S.TINTERVALSTATS.set(tintervals_filtered)



