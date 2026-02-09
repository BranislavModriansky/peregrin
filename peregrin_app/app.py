from pathlib import Path
from shiny import App
from forge import app_ui, Server

app = App(app_ui, Server)

# TODO: make two options for chart downloads - an svg with no background and a png
#       -> the buttons can have width 50% each and be side by side

# TODO: add more lut maps. All reasonable from: 
#       https://matplotlib.org/stable/tutorials/colors/colormaps.html
#       cmaps = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Grays', 'Grays_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 
#       'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 
#       'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 
#       'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'berlin', 'berlin_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 
#       'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_grey_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 
#       'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gist_yerg', 'gist_yerg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'grey', 'grey_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 
#       'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'managua', 'managua_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 
#       'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 
#       'twilight_shifted_r', 'vanimo', 'vanimo_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r']

# TODO: MAKE A PACKAGE OUT OF THE PLOTTING FUNCTIONS + other useful utility functions

# TODO: Add a checkbox control to enable/disable y-mirroring when loading data 

# TODO: remodel the dictionary utilization so that keywords are actually hex codes instead of color names (because when inputing a dictionary into a selectize/other inputs, the items would be shown, and the keywords would be registered automatically - in this case there is no need to use list(dict.keys())) method nd later dict.get(key) method to get the values

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