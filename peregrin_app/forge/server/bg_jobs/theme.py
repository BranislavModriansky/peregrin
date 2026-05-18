from pathlib import Path
from shiny import reactive, render, ui


THEME_CONTROL_SPECS = {
    "light": [
        {"input_id": "theme_light_sidebar_bg", "label": "Sidebar background", "css_var": "--color-sidebar-bg", "default": "#f8f8f8"},
        {"input_id": "theme_light_primary", "label": "Primary text", "css_var": "--color-primary", "default": "#151414"},
        {"input_id": "theme_light_secondary", "label": "Secondary accent", "css_var": "--color-secondary", "default": "#474a4b"},
        {"input_id": "theme_light_hover", "label": "Hover color", "css_var": "--color-hover", "default": "#68656a"},
        {"input_id": "theme_light_zero", "label": "Muted nav text", "css_var": "--color-zero", "default": "#948593"},
        {"input_id": "theme_light_link", "label": "Link color", "css_var": "--color-link", "default": "#006ea9"},
        {"input_id": "theme_light_text", "label": "Body text", "css_var": "--color-text", "default": "#212529"},
        {"input_id": "theme_light_subtext", "label": "Subtext", "css_var": "--color-subtext", "default": "#6c757d"},
        {"input_id": "theme_light_border_soft", "label": "Soft border", "css_var": "--color-border-soft", "default": "#dddddd"},
        {"input_id": "theme_light_border_light", "label": "Header border", "css_var": "--color-border-light", "default": "#dee2e6"},
        {"input_id": "theme_light_item_bg", "label": "Card/item background", "css_var": "--color-item-bg", "default": "#f8f9fa"},
        {"input_id": "theme_light_item_hover_bg", "label": "Card/item hover", "css_var": "--color-item-hover-bg", "default": "#e9ecef"},
    ],
    "dark": [
        {"input_id": "theme_dark_surface_0", "label": "Base background", "css_var": "--surface-0", "default": "#121314"},
        {"input_id": "theme_dark_surface_1", "label": "Sidebar background", "css_var": "--surface-1", "default": "#181616"},
        {"input_id": "theme_dark_surface_3", "label": "Main surface", "css_var": "--surface-3", "default": "#1e1f21"},
        {"input_id": "theme_dark_surface_4", "label": "Raised surface", "css_var": "--surface-4", "default": "#26282d"},
        {"input_id": "theme_dark_surface_6", "label": "Border surface", "css_var": "--surface-6", "default": "#3c3f45"},
        {"input_id": "theme_dark_text_primary", "label": "Primary text", "css_var": "--text-primary", "default": "#f7f7f7"},
        {"input_id": "theme_dark_text_secondary", "label": "Secondary text", "css_var": "--text-secondary", "default": "#dbe5ea"},
        {"input_id": "theme_dark_text_muted", "label": "Muted text", "css_var": "--text-muted", "default": "#6f7476"},
        {"input_id": "theme_dark_accent_link", "label": "Link accent", "css_var": "--accent-link", "default": "#006ea9"},
        {"input_id": "theme_dark_accent_active", "label": "Active accent", "css_var": "--accent-active", "default": "#58a9d6"},
        {"input_id": "theme_dark_table_head", "label": "Table header", "css_var": "--table-head-bg", "default": "#9b9b9b"},
        {"input_id": "theme_dark_popup_border", "label": "Popup border", "css_var": "--popup-border", "default": "#435d7b"},
    ],
}


def _color_input(input_id: str, label: str, value: str):
    value_id = f"{input_id}_value"

    return ui.div(
        ui.tags.label(label, **{"for": input_id}, class_="theme-color-label"),
        ui.div(
            ui.tags.input(
                type="color",
                id=input_id,
                value=value,
                class_="theme-color-input",
                style="width:100%; height:2.4rem; padding:0; border:none; background:transparent;"
            ),
            ui.tags.code(value, id=value_id, class_="theme-color-value"),
            class_="theme-color-row"
        ),
        ui.tags.script(
            f"""
            (function() {{
                const el = document.getElementById('{input_id}');
                const valueEl = document.getElementById('{value_id}');
                if (!el) return;

                function send() {{
                    if (valueEl) valueEl.textContent = el.value;
                    Shiny.setInputValue('{input_id}', el.value, {{ priority: 'event' }});
                }}

                el.addEventListener('input', send);
                send();
            }})();
            """
        ),
        class_="theme-color-control"
    )


def set_theme(input, output, session, S):
    theme_values = {
        theme_name: reactive.Value(
            {spec["css_var"]: spec["default"] for spec in specs}
        )
        for theme_name, specs in THEME_CONTROL_SPECS.items()
    }

    def register_color_sync(theme_name: str, input_id: str, css_var: str):
        @reactive.Effect
        def _():
            try:
                value = input[input_id]()
            except Exception:
                return

            if not isinstance(value, str) or not value:
                return

            current = theme_values[theme_name].get()
            if current.get(css_var) == value:
                return

            updated = dict(current)
            updated[css_var] = value
            theme_values[theme_name].set(updated)

    for theme_name, specs in THEME_CONTROL_SPECS.items():
        for spec in specs:
            register_color_sync(theme_name, spec["input_id"], spec["css_var"])

    @output(id="theme_css_injector")
    @render.ui
    def theme_css_injector():
        theme = input.app_theme() or "light"
        return ui.include_css(Path(__file__).parents[3] / f"src/styles/{theme}-theme.css")
        # return ui.include_css(Path(__file__).parents[3] / f"src/styles/style.css")

    @output(id="theme_css_overrides")
    @render.ui
    def theme_css_overrides():
        theme = input.app_theme() or "light"
        current = theme_values[theme].get()

        declarations = "\n".join(
            f"  {css_var}: {color};"
            for css_var, color in current.items()
        )

        return ui.tags.style(
            f"""
            :root {{
            {declarations}
            }}
            """
        )

    @output(id="theme_customization_controls")
    @render.ui
    def theme_customization_controls():
        theme = input.app_theme() or "light"
        specs = THEME_CONTROL_SPECS[theme]
        current = theme_values[theme].get()

        return ui.div(
            ui.div(
                ui.tags.div("Theme editor", class_="theme-controls-heading"),
                ui.tags.div(f"Editing: {theme.title()} theme", class_="theme-controls-subheading"),
                class_="theme-controls-header"
            ),
            ui.div(
                *[
                    _color_input(
                        input_id=spec["input_id"],
                        label=spec["label"],
                        value=current.get(spec["css_var"], spec["default"])
                    )
                    for spec in specs
                ],
                class_="theme-control-grid"
            ),
            class_="theme-controls-wrap"
        )

    @output(id="repo_tree")
    @render.ui
    def _():
        url = f"https://raw.githubusercontent.com/BranislavModriansky/peregrin/main/media/animation%20{input.app_theme()}.gif"
        return ui.HTML(f'<img src="{url}" alt="animation" style="width:35%;">')