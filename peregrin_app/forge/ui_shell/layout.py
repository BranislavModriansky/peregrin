from pathlib import Path
from shiny import ui
from .sheets import *


import warnings
from shiny._deprecated import ShinyDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message=r".*panel_well\(\) is deprecated\. Use shiny\.ui\.card\(\) instead\.",
    category=ShinyDeprecationWarning,
)


theme_css = Path(__file__).parents[2] / f"src/styles/theme.css"
custom_theme_js = Path(__file__).parents[2] / f"src/styles/custom_theme.js"


# UI design definitions

app_ui = ui.page_sidebar(

    sidebar,
    
    # Panel navigation
    ui.navset_bar(

        navpanel_about,
        navpanel_inputs,
        navpanel_dashboard,
        navpanel_viz,
       
        ui.nav_spacer(),

        ui.nav_control(
            ui.include_js(custom_theme_js),
            ui.input_action_button(
                id="customize_theme",
                label="Customize theme",
                style="margin-right: 1.5rem; margin-top: -0.5rem",
                class_="btn-customizetheme"
            )
        ),

        ui.nav_control(
            ui.include_css(theme_css),
            ui.output_ui("theme_css_injector"),
            ui.output_ui("theme_css_overrides"),
            ui.div(
                ui.output_ui("theme_customization_controls"),
                id="theme_customization_controls_mount",
                style="display:none;"
            ),
            ui.input_dark_mode(id="app_theme"),
            log_panel_ui
        ),
        
        title=ui.tags.span(
            ui.a(
                "Peregrin",
                href="https://github.com/BranislavModriansky/Peregrin/tree/main",
                target="_blank",
                rel="noopener noreferrer",
                class_="peregrin-logo"
            )
        ),
        
        id="main_nav",

        selected="Input Menu",
        
    )
)