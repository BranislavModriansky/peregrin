from pathlib import Path
from shiny import App
from forge import app_ui, Server

app = App(app_ui, Server)

# !IMPORTANT! Helge Hecht meeting:
# Chceck whether all source code is separated properly
# Divide the UI layout code into multiple objects = navpanels and even subdivide navpanels
# Use conda for installing packages into the environment, conda doesnt see pip installments
# Package all source code code into a peregrin library and use that inside of the shiny framework
# Templates for making a package in the temas chat with Helge Hecht




# TODO: !!!! ADD PLOT DOWNLOADS AS SVGs AND PNGs !!!!!



# TODO: make two options for chart downloads - an svg with no background and a png
#       -> the buttons can have width 50% each and be side by side

# TODO: MAKE A PACKAGE OUT OF THE PLOTTING FUNCTIONS + other useful utility functions

# TODO: Add a checkbox control to enable/disable y-mirroring when loading data 

# TODO - make an option to convert the time units (e.g. user inputs a df in seconds but wants to see minutes/hours in the graphs and calculations)

# TODO - Make it possible to save/load threshold configurations
# TODO - Mean directional change rate
# TODO - Again add rendered text showing the total number of cells in the input and the number of output cells
# TODO - Option to download a simple legend showing how much data was filtered out and how so

# TODO - VERY IMPORTANT
#      - Must have an option to download the whole app settings together with the data 

# TODO - Add a "reset" button to reset all settings to default values

# TODO - Add 2D filtering (gating) (with js) panel to filter out cells based on two parameters at once (e.g. mean speed vs. mean directional change rate)
#      - Options for coloring the points inn the 2D plot:
#          - a single color option
#          - LUT map by density (e.g. using a 2D histogram)
#          - LUT map by a parameter (which can be indepent of the two parameters being plotted)
#      - Plotting the kernel density estimation of the two parameters:
#          Kernel density estimation parameters will be defined by the user (e.g. kernel size and sigma).
#          Options for visualization of the kernel density estimation:
#          - A basic heatmap of the kernel density estimation (pixels have asssigned intensity values based on the kernel density estimation)
#          - Contours of the kernel density estimation split into levels/sub-ranges (number of levels/sub-ranges would be defined by the user)
#          - Colormesh of the kernel density (user defines the bin x/y parameters for the colormesh)
#          In each case of kernel density estimation visualization:
#             => range will be fixed -> set to 0-1
#             => the user will select the desired cmap
#             => the user will be able to apply mathematical transformations (e.g. linear, log, etc.)
#             => the user will be able to set the kernel density estimation parameters (e.g. kernel size and sigma), which in effect will control the density resolution = blur
#             => LUT map aplication by a parameter will be disabled 
#      - Have a kernel density estimate on the sides of the plot showing the distribution of each parameter (similar to a seaborn jointplot)
#      - Make it possible to save/load gating configurations
#      - Hatch the selected area https://stackoverflow.com/questions/41664850/hatch-area-using-pcolormesh-in-basemap
#      - https://sabopy.com/py/matplotlib-17/
#      - https://stackoverflow.com/questions/61938702/pcolormesh-secondary-axis-ticks-missing-when-set-as-in
#      - https://www.geeksforgeeks.org/python/matplotlib-axes-axes-pcolormesh-in-python/
