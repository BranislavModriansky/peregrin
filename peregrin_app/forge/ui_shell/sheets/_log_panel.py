from shiny import ui


log_panel_ui = ui.tags.div(
        ui.tags.div(
            ui.tags.span(
                "▲ peregrin log",
                id="log-header-label",
                style=(
                    "font: 600 10px/1 'Fira Code','Cascadia Code','Consolas',monospace;"
                    "letter-spacing:0.14em; color:#adaeb2;"
                ),
            ),
            id="log-header",
            style=(
                "height:26px; display:flex; align-items:center; padding:0 14px;"
                "cursor:pointer; user-select:none; border-bottom:1px solid #49494b;"
                "flex-shrink:0;"
            ),
            onclick=(
                "const p=this.parentElement; const lbl=this.querySelector('#log-header-label');"
                "if(p.dataset.open==='true'){"
                "  p.style.height='26px'; p.dataset.open='false'; lbl.textContent='▲ peregrin log';"
                "}else{"
                "  p.style.height='286px'; p.dataset.open='true'; lbl.textContent='▼ peregrin log';"
                "}"
            ),
        ),
        ui.tags.div(
            ui.output_ui("log_output_content"),
            id="log-output",
            style=(
                "height:260px; overflow-y:auto; padding:8px 12px; box-sizing:border-box;"
                "font:12px/1.6 'Fira Code','Cascadia Code','Consolas',monospace;"
                "color:#e1e2e4; white-space:pre-wrap; word-break:break-all;"
            ),
        ),
        id="log-panel",
        style=(
            "position:fixed; bottom:0; left:0; width:600px; height:26px;"
            "overflow:hidden; background:#252525; border-radius:0 4px 0 0;"
            "z-index:9999; transition:height 0.22s ease;"
            "scrollbar-color: #49494b #252525; scrollbar-width: thin;"
        ),
        **{"data-open": "false"},
    )