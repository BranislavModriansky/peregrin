from shiny import ui, render, reactive
from shiny.types import ImgData

from pathlib import Path



def mount_references(input, output, session, S, noticequeue):

    dir = Path(__file__).resolve().parents[3] / 'src' / 'media'

    @output()
    @render.image
    def colormap_showcase():
        theme = input.app_theme()
        img: ImgData = {"src": str(dir / f"cmaps_theme_{theme}.svg")}
        return img
    

    @output()
    @render.image
    def colors_showcase():
        theme = input.app_theme()
        img: ImgData = {"src": str(dir / f"colors_theme_{theme}.svg")}
        return img
