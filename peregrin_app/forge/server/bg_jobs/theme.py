from shiny import reactive, ui, render
import json
from pathlib import Path




def set_theme(input, output, session, S):

    @output(id="theme_css_injector")
    @render.ui
    def theme_css_injector():
        print(f"Selected theme: {input.app_theme()}")
        theme = input.app_theme()
        return ui.include_css(Path(__file__).parents[3] / f"src/styles/{theme}-theme.css")




