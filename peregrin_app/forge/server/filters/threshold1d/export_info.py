import pandas as pd
import shiny.ui as ui
from shiny import render
from html import escape
from datetime import date



def mount_thresholds_info_export(input, output, session, S):

    @output()
    @render.ui
    def threshold_info():
        return None

        try:
            blocks = []
            thresholds = S.THRESHOLDS.get()

            # iterate deterministically if keys are integers
            for t in sorted(thresholds.keys()):
                if t > S.THRESHOLDS_ID.get():
                    break
                try:
                    t_state = thresholds.get(t)
                    t_state_after = thresholds.get(t + 1)
                    
                    data = len(t_state.get("tracks"))
                    data_after = len(t_state_after.get("tracks")) if t_state_after else data
                    out = data - data_after
                    out_percent = round(out / data * 100) if data else 0

                    prop = input[f"threshold_property_{t}"]()
                    ftype = input[f"threshold_type_{t}"]()
                    if ftype == "Relative to...":
                        ref = input[f"reference_value_{t}"]()
                        if ref == "My own value":
                            ref_val = input[f"my_own_value_{t}"]()
                        else:
                            ref_val = ref
                        reference = f"<br>Reference: <br><i><b>{ref}</b> (<b>{ref_val}</b>)</i><br>" if not isinstance(ref_val, str) else f"<br>Reference: <br><i><b>{ref}</b></i><br>"
                    else:
                        reference =  ""
                    vals = input[f"threshold_slider_{t}"]()

                except Exception:
                    break

                blocks.append(
                    ui.markdown(
                        f"""
                        <div style="height:5px;"></div>
                            <hr style="border:0; border-top:1px solid #000000; margin:8px 0;">
                        <div style="height:5px;"></div>
                        <p style="margin-bottom:8px; margin-top:10px;">
                            <b><h5>Threshold {t}</h5></b>
                            Filtered out: <br>
                            <i><b>{out}</b> (<b>{out_percent}%</b>)</i>
                        </p>
                        <p style="margin-bottom:8px; margin-top:0px;">
                            Property: <br>
                            <i><b>{prop}</b></i> <br>
                            Filter: <br>
                            <i><b>{ftype}</b></i> <br>
                            Range: <br>
                            <i><b>{vals[0]}</b> - <b>{vals[1]}</b></i>
                            {reference}
                        """
                    )
                )

        except Exception:
            pass

        total_tracks = len(S.UNFILTERED_TRACKSTATS.get())
        filtered_tracks = len(thresholds.get(S.THRESHOLDS_ID.get()+1).get("tracks")) if thresholds and thresholds.get(S.THRESHOLDS_ID.get()+1) else total_tracks

        filtered_tracks_percent = (
            round(filtered_tracks / total_tracks * 100) if total_tracks else 0
        )

        # --- Header + summary block
        blocks.insert(0,
            ui.markdown(
                f"""
                <p style="margin-bottom:0px; margin-top:0px;">
                    <h4> <b> Info </b> </h4>
                </p>
                <p style="margin-bottom:8px; margin-top:12px;">
                    Cells in total: <br>
                    <i><b>{total_tracks}</b> <br></i>
                </p>
                <p style="margin-bottom:8px; margin-top:0px;">
                    In focus: <br>
                    <i><b>{filtered_tracks}</b> (<b>{filtered_tracks_percent}%</b>)</i>
                </p>
                """
            )
        )

        # Return a single well with all blocks as children
        return ui.panel_well(*blocks)



    def GetInfoSVG(*, width: int = 190, txt_color: str = "#000000") -> str:
        """
        Build an SVG 'Info' panel using current Shiny reactives.
        Works for both 1D and 2D thresholding like in your filter_info().
        """
        
        # ---------- helpers ----------
        pad = 16
        title_size = 18
        body_size = 14
        line_gap = 8
        section_gap = 14
        rule_gap = 5
        rule_color = "#000000"
        font_family = "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif"

        def addy(y, inc):  # move the cursor
            return y + inc

        def tspan(text, cls=None):
            if cls:
                return f"<tspan class='{cls}'>{escape(str(text))}</tspan>"
            return f"<tspan>{escape(str(text))}</tspan>"

        # ---------- totals ----------
        total_tracks = len(S.UNFILTERED_TRACKSTATS.get())
        filtered_tracks = len(S.TRACKSTATS.get())
        if total_tracks < 0:
            return ""
        if filtered_tracks < 0:
            filtered_tracks = total_tracks
        percent = 0 if total_tracks == 0 else round((filtered_tracks / total_tracks) * 100)

        # ---------- SVG header (height placeholder) ----------
        x = pad
        y = pad + title_size
        parts = [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='__HEIGHT__' "
            f"viewBox='0 0 {width} __HEIGHT__' role='img' aria-label='Info panel'>",
            "<style>.bold{font-weight:700}.ital{font-style:italic}</style>",
            f"<text x='{x}' y='{y}' font-family='{font_family}' font-size='{title_size}' "
            f"font-weight='700' fill='{txt_color}'>Info</text>",
            f"<g font-family='{font_family}' font-size='{body_size}' fill='{txt_color}'>"
        ]

        # ---------- header body ----------
        y = addy(y, section_gap + body_size)
        parts.append(f"<text x='{x}' y='{y}'>Cells in total:</text>")
        y = addy(y, body_size + line_gap)
        parts.append(f"<text x='{x}' y='{y}'>{tspan(total_tracks,'bold')}</text>")

        y = addy(y, section_gap + body_size)
        parts.append(f"<text x='{x}' y='{y}'>In focus:</text>")
        y = addy(y, body_size + line_gap)
        parts.append(
            f"<text x='{x}' y='{y}'>{tspan(filtered_tracks,'bold')} {tspan(f'({percent}%)','bold')}</text>"
        )

        # ---------- thresholds (read reactives exactly like your UI) ----------
        try:
            thresholds = S.THRESHOLDS.get()
            for t in sorted(thresholds.keys()):
                if t > S.THRESHOLDS_ID.get():
                    break
                try:
                    t_state = thresholds.get(t)
                    t_state_after = thresholds.get(t + 1)

                    data = len(t_state.get("tracks"))
                    data_after = len(t_state_after.get("tracks")) if t_state_after else data
                    out = data - data_after
                    out_percent = round(out / data * 100) if data else 0

                    prop = input[f"threshold_property_{t}"]()
                    ftype = input[f"threshold_type_{t}"]()
                    if ftype == "Relative to...":
                        ref = input[f"reference_value_{t}"]()
                        if ref == "My own value":
                            ref_val = input[f"my_own_value_{t}"]()
                        else:
                            ref_val = ref
                        reference = f"{ref} ({ref_val})" if not isinstance(ref_val, str) else f"{ref}"
                    else:
                        reference = ""

                    vmin, vmax = input[f"threshold_slider_{t}"]()
                except Exception:
                    break

                # hr
                y = addy(y, rule_gap + section_gap)
                parts.append(f"<line x1='{pad}' x2='{width-pad}' y1='{y}' y2='{y}' stroke='{rule_color}' stroke-width='1'/>")
                y = addy(y, rule_gap)

                # threshold header
                y = addy(y, body_size + line_gap)
                parts.append(f"<text x='{x}' y='{y}'>{tspan(f'Threshold {t}','bold')}</text>")

                # filtered out
                y = addy(y, body_size + line_gap)
                parts.append(
                    f"<text x='{x}' y='{y}'>Filtered out: "
                    f"{tspan(out,'ital bold')} {tspan(f'({out_percent}%)','ital bold')}</text>"
                )

                # property / threshold / range / reference
                y = addy(y, body_size + section_gap)
                parts.append(f"<text x='{x}' y='{y}'>Property:</text>")
                y = addy(y, body_size)
                parts.append(f"<text x='{x}' y='{y}'>{tspan(prop,'bold ital')}</text>")

                y = addy(y, body_size + section_gap)
                parts.append(f"<text x='{x}' y='{y}'>Filter:</text>")
                y = addy(y, body_size)
                parts.append(f"<text x='{x}' y='{y}'>{tspan(ftype,'bold ital')}</text>")

                y = addy(y, body_size + section_gap)
                parts.append(f"<text x='{x}' y='{y}'>Range:</text>")
                y = addy(y, body_size)
                parts.append(
                    f"<text x='{x}' y='{y}'>{tspan(vmin,'bold ital')} - {tspan(vmax,'bold ital')}</text>"
                )

                if reference:
                    y = addy(y, body_size + section_gap)
                    parts.append(f"<text x='{x}' y='{y}'>Reference:</text>")
                    y = addy(y, body_size)
                    parts.append(f"<text x='{x}' y='{y}'>{tspan(reference,'bold ital')}</text>")


        except Exception:
            pass

        # ---------- close and set height ----------
        parts.append("</g></svg>")
        height = y + pad
        svg = "".join(parts).replace("__HEIGHT__", str(height))
        return svg


    @render.download(filename=f"Threshold Info {date.today()}.svg", media_type="svg")
    def download_threshold_info():
        svg = GetInfoSVG()
        yield svg.encode("utf-8")

        
