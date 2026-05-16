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
        return ui.tags.img(
            src=f"https://github.com/BranislavModriansky/peregrin/tree/main/media/animation {input.app_theme()}.gif",
            alt="Animated network visualization",
            style="""
                display: block;
                width: 600px;
                max-width: 100%;
                height: auto;
                margin: 1rem 0 2rem 0;
                border-radius: 8px;
            """
        )
        
        # except: pass
