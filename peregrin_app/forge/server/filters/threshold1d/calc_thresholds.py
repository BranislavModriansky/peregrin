import time
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import shiny.ui as ui
from shiny import render, reactive, req, ui


from peregrin_app.src.code._handlers._reports import Level
from src.code import Inventory1D, Filter1D, DebounceCalc, is_empty


def mount_thresholds_calc(input, output, session, S, noticequeue):

    Filter1D.noticequeue = noticequeue

    at_idx = reactive.Value()

    def _ensure_inventory_len(n: int) -> None:
        """
        Ensure Inventory1D's list-like fields are exactly length n.
        Prevents IndexError when id_idx changes (append/remove thresholds).
        """
        for name in ("property", "filter", "selection", "mask", "series", "ambit"):
            cur = getattr(Inventory1D, name)

            # Normalize to python list
            if isinstance(cur, np.ndarray):
                cur = cur.tolist()
            elif not isinstance(cur, list):
                cur = list(cur) if cur is not None else []

            if len(cur) < n:
                cur.extend([None] * (n - len(cur)))
            elif len(cur) > n:
                cur = cur[:n]

            setattr(Inventory1D, name, cur)

    def _sync_id_idx_from_state() -> None:
        n = int(S.THRESHOLDS_ID.get()) + 1  # +1 for safe_end slot
        Inventory1D.id_idx = np.arange(n, dtype=int)
        _ensure_inventory_len(n)

    def _build_thresholds_payload() -> dict:
        """
        Build S.THRESHOLDS payload safely, assuming Inventory1D lists are synced.
        """
        _ensure_inventory_len(len(Inventory1D.id_idx))
        payload = {}
        for idx in Inventory1D.id_idx:
            idx = int(idx)
            payload[idx] = {
                "property": Inventory1D.property[idx],
                "filter": Inventory1D.filter[idx],
                "selection": Inventory1D.selection[idx],
                "mask": Inventory1D.mask[idx],
                "series": Inventory1D.series[idx],
                "ambit": Inventory1D.ambit[idx],
            }
        return payload

    @DebounceCalc(0.5)
    @reactive.calc
    def get_bins():
        return input.bins() if input.bins() is not None and input.bins() != 0 else 15
        

    @reactive.Effect
    @reactive.event(S.UNFILTERED_SPOTSTATS, S.UNFILTERED_TRACKSTATS)
    def initialize_inventory():

        Inventory1D.spot_data = S.UNFILTERED_SPOTSTATS.get()
        Inventory1D.track_data = S.UNFILTERED_TRACKSTATS.get()

        _sync_id_idx_from_state()

        thresholds_id = int(S.THRESHOLDS_ID.get())
        idxes = thresholds_id + 1  # includes safe_end slot

        # Allocate lists to match id_idx length
        property = [None] * idxes
        filter = [None] * idxes
        selection = [None] * idxes

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
                    case "Percentile":
                        filter[idx] = ttype, 100
                    case "Relative to...":
                        ref = input[f"reference_value_{id}"]()
                        if ref == "My own value":
                            ref = input[f"my_own_value_{id}"]()
                        filter[idx] = ttype, ref

                selection[idx] = (input[f"floor_threshold_value_{id}"](), input[f"ceil_threshold_value_{id}"]())

            except Exception:
                pass

            if property[idx] is None:
                property[idx] = "Track displacement"
            if filter[idx] is None:
                filter[idx] = ("Literal", None)

        # Reasonable defaults for safe_end slot (will be overwritten by Filter1D._safe_end anyway)
        if property[-1] is None:
            property[-1] = property[-2] if idxes >= 2 and property[-2] is not None else "Track displacement"
        if filter[-1] is None:
            filter[-1] = filter[-2] if idxes >= 2 and filter[-2] is not None else ("Literal", None)

        Inventory1D.property = property
        Inventory1D.filter = filter
        Inventory1D.selection = selection
        _ensure_inventory_len(len(Inventory1D.id_idx))

        Filter1D().Initialize()

        try:
            S.THRESHOLDS.set(_build_thresholds_payload())
        except Exception:
            pass

    def render_threshold_controls(id, min=0, max=100, step=1):
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

    @reactive.calc
    def update_threshold_controls():

        start = at_idx.get()

        for idx in range(start, len(Inventory1D.id_idx) - 1):

            id = idx + 1

            inventory = S.THRESHOLDS.get()[idx]

            req(not inventory is None 
                and not any(v is None for v in inventory.values()))

            min, max, step = inventory["ambit"]

            floor, ceil = input[f"floor_threshold_value_{id}"](), input[f"ceil_threshold_value_{id}"]()

            req(not None in [floor, ceil])

            if floor < min or floor > max:
                ui.update_numeric(
                    f"floor_threshold_value_{id}",
                    min=min,
                    max=max,
                    step=step,
                    value=min
                )
            else:
                ui.update_numeric(
                    f"floor_threshold_value_{id}",
                    min=min,
                    max=max,
                    step=step
                )

            if ceil < min or ceil > max:
                ui.update_numeric(
                    f"ceil_threshold_value_{id}",
                    min=min,
                    max=max,
                    step=step,
                    value=max
                )
            else:
                ui.update_numeric(
                    f"ceil_threshold_value_{id}",
                    min=min,
                    max=max,
                    step=step
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

                _color = 'black' if input.app_theme() == "light" else 'white'
                _marker_color = '#337ab7'

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

                    case "Percentile":
                        bottom, top = bottom/100, top/100
                        if not 0 <= bottom <= 1 or not 0 <= top <= 1:
                            bottom, top = 0, 1

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
        current_inventory = inventory.get(idx)

        req(not current_inventory is None 
            and not any(v is None for v in current_inventory.values()))

        render_threshold_controls(id, *current_inventory["ambit"])
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

    @DebounceCalc(1)
    @reactive.calc
    def update_thresholds():
        update_threshold_controls()
        update_histogram()
            

    @reactive.Effect
    def _():

        # Always sync sizes; remove/add effects can run in different order.
        _sync_id_idx_from_state()

        for idx in Inventory1D.id_idx[:-1]:
            id = idx + 1

            @reactive.Effect
            @reactive.event(
                input[f"threshold_property_{id}"],
                input[f"threshold_type_{id}"],
                input[f"reference_value_{id}"],
                input[f"my_own_value_{id}"],
                input[f"floor_threshold_value_{id}"],
                input[f"ceil_threshold_value_{id}"],
                input.app_theme,
            )
            def _(idx=idx, id=id):
                idx = int(idx)

                if idx not in Inventory1D.id_idx[:-1]:
                    return

                req(not None in [
                    input[f"threshold_property_{id}"](),
                    input[f"threshold_type_{id}"](),
                    input[f"floor_threshold_value_{id}"](),
                    input[f"ceil_threshold_value_{id}"](),
                ])

                _ensure_inventory_len(len(Inventory1D.id_idx))

                p = input[f"threshold_property_{id}"]()
                t = input[f"threshold_type_{id}"]()
                s = (input[f"floor_threshold_value_{id}"](), input[f"ceil_threshold_value_{id}"]())

                match t:
                    case "Literal":
                        f = t, None
                    case "Normalized 0-1":
                        f = t, None
                    case "Percentile":
                        f = t, 100
                    case "Relative to...":
                        ref = input[f"reference_value_{id}"]()
                        if ref == "My own value":
                            ref = input[f"my_own_value_{id}"]()
                        f = t, ref

                Inventory1D.property[idx] = p
                Inventory1D.filter[idx] = f
                Inventory1D.selection[idx] = s

                Filter1D().Downstream(idx)

                S.THRESHOLDS.set(_build_thresholds_payload())

                at_idx.set(idx)
                update_thresholds()


    @reactive.Effect
    @reactive.event(input.remove_threshold)
    def _():

        with reactive.isolate():
            Filter1D().PopLast()

            # Keep id_idx + lists consistent with state
            _sync_id_idx_from_state()

            S.THRESHOLDS.set(_build_thresholds_payload())


    # _ _ _ _ SETTING THE THRESHOLDS _ _ _ _

    @reactive.Effect
    @reactive.event(input.set_threshold)
    def threshold_data():

        try:
            spotstats, trackstats, framestats, tintervalstats = Filter1D().Apply()

            S.SPOTSTATS.set(spotstats)
            S.TRACKSTATS.set(trackstats)
            S.FRAMESTATS.set(framestats)
            S.TINTERVALSTATS.set(tintervalstats)

        except Exception as e:
            noticequeue.Report(
                Level.error,
                "Error applying threshold(s).",
                traceback.format_exc(),
            )


