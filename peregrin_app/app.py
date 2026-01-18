from pathlib import Path
from shiny import App
from forge import app_ui
from forge import Server


ROOT = Path(__file__).resolve().parent

app = App(app_ui, Server, static_assets=ROOT / "src" / "styles")

#TODO: add details about the condition and replicate plotted to the file name, and perhaps a title too. 

# TODO: Fix the already processesd data input and perhaps move it to the Input menu tab

# TODO: Move title input boxes to be part of the plot settings modals instead of being always visible

# TODO: remodel the dictionary utilization so that keywords are actually hex codes instead of color names (because when inputing a dictionary into a selectize/other inputs, the items would be shown, and the keywords would be registered automatically - in this case there is no need to use list(dict.keys())) method nd later dict.get(key) method to get the values

# BUG - Setting a threshold removes assigned colors from the data
# BUG - Violinplots show inappropriate y-axis labels
# TODO - Finalize the Time interval implementation everywhere - filtering!!!

# TODO - make an option to convert the time units (e.g. user inputs a df in seconds but wants to see minutes/hours in the graphs and calculations)
 
# TODO - use / utilize np.nan values where data is not able to be calculated instead of 0 values and be sure it doesnt get into way of calculations and graph generations

# TODO - fix your shit, look into the calculations of parameters inside of the _params script and make sure they are correct.
#      - Switch the "metric" annotations to "parameter" or "value" as that is correct
#      - Make sure all parameter names are consistent and the logic is global: not Speed mean, Direction mean (rad), but Speed mean, Direction (rad) mean

# TODO - Add a button to reset all thresholding settings to default

# TODO - Keep all the raw data (columns) - rather format them (stripping of _ and have them not all caps)
# TODO - Make the 2D filtering logic work on the same logic as does the D filtering logic
# TODO - Make it possible to save/load threshold configurations
# TODO - Find a way to program all the functions so that functions do not refresh/re-render unnecessarily on just any reactive action
# TODO - Time point definition
# TODO - Make it possible for the user to title their charts
# TODO - Mean directional change rate
# TODO - Select which p-tests should be shown in the superplot chart
# TODO - P-test
# TODO - Again add rendered text showing the total number of cells in the input and the number of output cells
# TODO - Option to download a simple legend showing how much data was filtered out and how so
# TODO - input_selectize("Plot:"... with options "Polar/Normalized" or "Cartesian/Raw"
# TODO - Differentiate between frame(s) annotations and time annotations

# TODO - Potentially add a nav tab for interacting with the data and set the formatting to be right and ready for processing

# TODO - VERY IMPORTANT
#      - Must have an option to download the whole app settings together with the data 

# TODO - add ui where the user can define the order of the conditions

# TODO - implement a log scale for thresholding. Ideally a logarithmic scale that:
#      - first checks whether the data has 0 values or negative values
#      - chooses the log calculation accordingly
#      - then sets an automated code which finds out the best possible setting of the log scale function so that after it is applied to the data, its distribution always ends up in a normal gaussian distributian
