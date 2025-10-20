from math import ceil, floor

@staticmethod
def GetInfoSVG(
    *,
    total_tracks: int,
    filtered_tracks: int,
    threshold_list: list[int],
    threshold_dimension: str,               # "1D" or "2D"
    thresholds_state: dict,                 # dict of threshold index -> state with "tracks"/"spots"
    props: dict = None,                     # for 1D: {t: property_name}
    ftypes: dict = None,                    # for 1D: {t: filter_type}
    refs: dict = None,                      # for 1D: {t: reference_label}
    ref_vals: dict = None,                  # for 1D: {t: reference_value}
    values: dict = None,                    # for 1D: {t: (min,max)}
    propsX: dict = None,                    # for 2D: {t: propX}
    propsY: dict = None,                    # for 2D: {t: propY}
    txt_color: str = "#000000",
    width: int = 180,
    font_family: str = "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif",
) -> str:
    """
    Generate an SVG info panel similar to the Shiny UI.

    Arguments:
    - total_tracks: int
    - filtered_tracks: int
    - threshold_list: list of ints (active thresholds)
    - threshold_dimension: "1D" or "2D"
    - thresholds_state: dict of t -> state {"tracks": ..., "spots": ...}
    - props, ftypes, refs, ref_vals, values: dicts keyed by threshold index (for 1D mode)
    - propsX, propsY: dicts keyed by threshold index (for 2D mode)
    """
    if total_tracks < 0:
        return ""
    if filtered_tracks < 0:
        filtered_tracks = total_tracks

    in_focus_pct = 0 if total_tracks == 0 else round(filtered_tracks / total_tracks * 100)

    pad = 16
    title_size = 20
    h2_size = 16
    body_size = 14
    hr_thickness = 1
    gap_line = 6
    gap_section = 14
    gap_rule = 14

    def lh(size): return size + 4

    y = pad + lh(title_size) - 4
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="1000" viewBox="0 0 {width} 1000">',
        f'<text x="{pad}" y="{y}" font-family="{font_family}" font-size="{title_size}" font-weight="700" fill="{txt_color}">Info</text>'
    ]

    # Cells in total
    y += gap_section + lh(body_size)
    svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" fill="{txt_color}" font-family="{font_family}">Cells in total:</text>')
    y += lh(body_size)
    svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-weight="700" fill="{txt_color}" font-family="{font_family}">{total_tracks}</text>')

    # In focus
    y += gap_section + lh(body_size)
    svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" fill="{txt_color}" font-family="{font_family}">In focus:</text>')
    y += lh(body_size)
    svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-weight="700" fill="{txt_color}" font-family="{font_family}">{filtered_tracks} ({in_focus_pct}%)</text>')

    # Threshold blocks
    for t in sorted(thresholds_state.keys()):
        if t not in threshold_list:
            continue

        t_state = thresholds_state.get(t)
        t_state_after = thresholds_state.get(t + 1)
        data = len(t_state.get("tracks", []))
        data_after = len(t_state_after.get("tracks", [])) if t_state_after else data
        out = data - data_after
        out_percent = round(out / data * 100) if data else 0

        # Divider
        y += gap_rule
        svg.append(f'<line x1="{pad}" x2="{width-pad}" y1="{y}" y2="{y}" stroke="{txt_color}" stroke-opacity="0.4" stroke-width="{hr_thickness}" />')
        y += gap_rule + lh(h2_size) - 4

        svg.append(f'<text x="{pad}" y="{y}" font-size="{h2_size}" font-weight="700" font-family="{font_family}" fill="{txt_color}">Threshold {t+1}</text>')

        # Filtered out
        y += gap_section + lh(body_size)
        svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-family="{font_family}" fill="{txt_color}">Filtered out:</text>')
        y += lh(body_size)
        svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-weight="700" font-family="{font_family}" fill="{txt_color}">{out} ({out_percent}%)</text>')

        y += gap_section

        if threshold_dimension == "1D":
            prop = props[t]
            ftype = ftypes[t]
            val_min, val_max = values[t]
            ref = refs.get(t) if refs else None
            ref_val = ref_vals.get(t) if ref_vals else None

            # Property
            svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-family="{font_family}" fill="{txt_color}">Property:</text>')
            y += lh(body_size)
            svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-style="italic" font-weight="600" font-family="{font_family}" fill="{txt_color}">{prop}</text>')

            # Filter
            y += lh(body_size)
            svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-family="{font_family}" fill="{txt_color}">Filter:</text>')
            y += lh(body_size)
            svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-style="italic" font-weight="600" font-family="{font_family}" fill="{txt_color}">{ftype}</text>')

            # Range
            y += lh(body_size)
            svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-family="{font_family}" fill="{txt_color}">Range:</text>')
            y += lh(body_size)
            svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-style="italic" font-weight="600" font-family="{font_family}" fill="{txt_color}">{val_min} - {val_max}</text>')

            # Reference if available
            if ftype == "Relative to..." and ref:
                y += lh(body_size)
                svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-family="{font_family}" fill="{txt_color}">Reference:</text>')
                y += lh(body_size)
                ref_text = f"{ref} ({ref_val})" if ref_val is not None else ref
                svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-style="italic" font-weight="600" font-family="{font_family}" fill="{txt_color}">{ref_text}</text>')

        elif threshold_dimension == "2D":
            propX = propsX[t]
            propY = propsY[t]

            try:
                track_data = t_state_after.get("tracks")
                spot_data = t_state_after.get("spots")
            except Exception:
                track_data = t_state.get("tracks")
                spot_data = t_state.get("spots")

            dataX = track_data.get(propX, []) if isinstance(track_data, dict) else []
            dataY = track_data.get(propY, []) if isinstance(track_data, dict) else []

            if propX == "Confinement ratio":
                minX, maxX = f"{min(dataX):.2f}", f"{ceil(max(dataX)):.2f}"
            else:
                minX, maxX = floor(min(dataX)), ceil(max(dataX))
            if propY == "Confinement ratio":
                minY, maxY = f"{min(dataY):.2f}", f"{ceil(max(dataY)):.2f}"
            else:
                minY, maxY = floor(min(dataY)), ceil(max(dataY))

            # Properties
            svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-family="{font_family}" fill="{txt_color}">Properties:</text>')

            # X
            y += lh(body_size)
            svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-style="italic" font-weight="600" font-family="{font_family}" fill="{txt_color}">{propX}</text>')
            y += lh(body_size)
            svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-family="{font_family}" fill="{txt_color}">Range:</text>')
            svg.append(f'<text x="{pad+60}" y="{y}" font-size="{body_size}" font-family="{font_family}" fill="{txt_color}">{minX} - {maxX}</text>')

            # Y
            y += gap_line + lh(body_size)
            svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-style="italic" font-weight="600" font-family="{font_family}" fill="{txt_color}">{propY}</text>')
            y += lh(body_size)
            svg.append(f'<text x="{pad}" y="{y}" font-size="{body_size}" font-family="{font_family}" fill="{txt_color}">Range:</text>')
            svg.append(f'<text x="{pad+60}" y="{y}" font-size="{body_size}" font-family="{font_family}" fill="{txt_color}">{minY} - {maxY}</text>')

    svg.append("</svg>")
    return "\n".join(svg)
