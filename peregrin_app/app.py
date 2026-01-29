from pathlib import Path
from shiny import App
from forge import app_ui
from forge import Server


ROOT = Path(__file__).resolve().parent

app = App(app_ui, Server, static_assets=ROOT / "src" / "styles")

# TODO: make two options for chart downloads - an svg with no background and a png
#       -> the buttons can have width 50% each and be side by side

# TODO: MAKE A PACKAGE OUT OF THE PLOTTING FUNCTIONS + other useful utility functions

# TODO: Add a checkbox control to enable/disable y-mirroring when loading data 

# TODO: remodel the dictionary utilization so that keywords are actually hex codes instead of color names (because when inputing a dictionary into a selectize/other inputs, the items would be shown, and the keywords would be registered automatically - in this case there is no need to use list(dict.keys())) method nd later dict.get(key) method to get the values

# BUG - Violinplots show inappropriate y-axis labels

# TODO - make an option to convert the time units (e.g. user inputs a df in seconds but wants to see minutes/hours in the graphs and calculations)
# TODO - fix your shit, look into the calculations of parameters inside of the _params script and make sure they are correct.
#      - Switch the "metric" annotations to "parameter" or "value" as that is correct
#      - Make sure all parameter names are consistent and the logic is global: not Speed mean, Direction mean (rad), but Speed mean, Direction (rad) mean
# TODO - Keep all the raw data (columns) - rather format them (stripping of _ and have them not all caps)
# TODO - Make it possible to save/load threshold configurations
# TODO - Time point definition
# TODO - Mean directional change rate
# TODO - Again add rendered text showing the total number of cells in the input and the number of output cells
# TODO - Option to download a simple legend showing how much data was filtered out and how so
# TODO - input_selectize for track reconstruction, renam polar to normalized / normalized start and realistic to raw or some better synonym to realistic
# TODO - Differentiate between frame(s) annotations and time annotations

# TODO - VERY IMPORTANT
#      - Must have an option to download the whole app settings together with the data 