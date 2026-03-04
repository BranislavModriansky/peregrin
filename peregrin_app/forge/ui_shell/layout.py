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
            ui.output_ui("theme_css_injector"),
            ui.input_dark_mode(id="app_theme", mode="dark")
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