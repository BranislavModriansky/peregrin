from shiny import reactive, ui, render
import json
from pathlib import Path




def set_theme(input, output, session, S):

    @output(id="theme_css_injector")
    @render.ui
    def theme_css_injector():
        theme = input.app_theme()
        return ui.include_css(Path(__file__).parents[3] / f"src/styles/{theme}-theme.css")




    @output(id="repo_tree")
    @render.ui
    def _():
        # try:
        url = f"https://raw.githubusercontent.com/BranislavModriansky/peregrin/main/media/animation%20{input.app_theme()}.gif"
        return ui.HTML(f'<img src="{url}" alt="animation" style="width:35%;">')
        
        # except: pass
