from shiny import reactive, ui, render
import json
from pathlib import Path


def set_theme(input, output, session, S):

    @reactive.Effect
    def _():
        print(f"Selected theme: {input.app_theme()}")
        theme = input.app_theme()
        ui.include_css(f"peregrin_app/src/styles/{theme}-theme.css"),
