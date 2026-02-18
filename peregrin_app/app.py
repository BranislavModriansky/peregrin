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

# TODO - make an option to convert the time units (e.g. user inputs a df in seconds but wants to see minutes/hours in the graphs and calculations)

# TODO - Make it possible to save/load threshold configurations
# TODO - Time point definition
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
