from shiny import App
from web_app.ui.layout import app_ui
from web_app.server import server



app = App(app_ui, server)




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
