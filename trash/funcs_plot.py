import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
import matplotlib.lines as mlines
from scipy.stats import gaussian_kde
from scipy.signal import savgol_filter
from scipy.stats import mannwhitneyu
from peregrin.scripts import PlotParams
import seaborn as sns
from itertools import combinations
import altair as alt
import plotly.graph_objects as go
import pandas as pd





_cell = '🦠'

_random = ['👨🏽‍🦽‍➡️', '🌪️', '🚀', '👨‍🦯‍➡️', '🏇🏼', '🛩️', '🚁', '🚂', '✈️', '🛳️', '🪂', '🚡', '💩', '🚕', '🚓', '🛒', '🐌', '💸']

_farm = ['🐂', '🐃', '🐄', '🐏', '🐖', '🐎', '🦃', '🫏', '🐇']

_safari = ['🦍', '🐅', '🐆', '🦒', '🦘', '🦓', '🐂']

_insects = ['🪰', '🦗', '🦟', '🐞', '🐜', '🐝', '🦋', '🐛']

_birds = ['🐦', '🦜', '🦆', '🦅', '🦉', '🦩', '🦚', '🦃']

_forest = ['🦌', '🦫', '🦦', '🦔', '🦇', '🦉', '🦅', '🦆', '🐢', '🐍', '🦎', '🐌']

_aquarium = ['🐠', '🐟', '🐢', '🏊🏻‍♀️', '🐡', '🦈', '🐙', '🐬', '🦭', '🐋', '🪼', '🦑', '🦐']

_scaled = {
    '0-5': '🪦',
    '5-10': '🌳',
    '10-20': '🐌',
    '20-30': '👨🏽‍🦽‍➡️',
    '30-40': '👩🏼‍🦼‍➡️',
    '40-50': '⛵',
    '50-60': '🚎',
    '60-70': '🐆',
    '70-80': '🌪️',
    '80-90': '🚀',
    '90-100': '🦸🏼'
}

_trains = {
    '0-20': '🚂',
    '20-40': '🚝',
    '40-60': '🚈',
    '60-80': '🚄',
    '80-100': '🚅'
    }





def _generate_random_color():
    """
    Generate a random color in hexadecimal format.

    """

    r = np.random.randint(0, 255)   # Red LED intensity
    g = np.random.randint(0, 255)   # Green LED intensity
    b = np.random.randint(0, 255)   # Blue LED intensity

    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def _generate_random_grey():
    """
    Generate a random grey color in hexadecimal format.

    """

    n = np.random.randint(0, 240)  # All LED intensities

    return '#{:02x}{:02x}{:02x}'.format(n, n, n)


def _make_cmap(elements, cmap):
    """
    Generate a qualitative colormap for a given list of elements.

    """

    n = len(elements)   # Number of elements in the dictionary
    if n == 0:          # Return an empty list if there are no elements
        return []       
    
    cmap = plt.get_cmap(cmap)                                   # Get the colormap
    colors = [mcolors.to_hex(cmap(i / n)) for i in range(n)]    # Generate a color for each element

    return colors


def _get_cmap(c_mode):
    """
    Get a colormap according to the selected color mode.

    """

    if c_mode == 'greyscale LUT':
        return plt.cm.gist_yarg
    elif c_mode == 'jet LUT':
        return plt.cm.jet
    elif c_mode == 'brg LUT':
        return plt.cm.brg
    elif c_mode == 'hot LUT':
        return plt.cm.hot
    elif c_mode == 'gnuplot LUT':
        return plt.cm.gnuplot
    elif c_mode == 'viridis LUT':
        return plt.cm.viridis
    elif c_mode == 'rainbow LUT':
        return plt.cm.rainbow
    elif c_mode == 'turbo LUT':
        return plt.cm.turbo
    elif c_mode == 'nipy_spectral LUT':
        return plt.cm.nipy_spectral
    elif c_mode == 'gist_ncar LUT':
        return plt.cm.gist_ncar
    else:
        return None


def _assign_marker(value, markers):
    """
    Qualitatively map a metric's percentile value to a symbol.

    """

    lut = []    # Initialize a list to store the ranges and corresponding symbols

    for key, val in markers.items():                # Iterate through the markers dictionary
        low, high = map(float, key.split('-'))      # Split the key into low and high values
        lut.append((low, high, val))                # Append the range and symbol to the list

    for low, high, symbol in lut:               # Return the symbol for the range that contains the given value
        if low <= value < high:                  # Check if the value falls within the range
            return symbol
    
    return list(markers.items())[-1][-1]            # Return the last symbol for thr 100th percentile (which is not included in the ranges)


def _get_markers(markers):
    """
    Get the markers according to the selected mode.

    """

    if markers == 'cell':
        return _cell
    elif markers == 'scaled':
        return _scaled
    elif markers == 'trains':
        return _trains
    elif markers == 'random':
        return _random
    elif markers == 'farm':
        return _farm
    elif markers == 'safari':
        return _safari
    elif markers == 'insects':
        return _insects
    elif markers == 'birds':
        return _birds
    elif markers == 'forest':
        return _forest
    elif markers == 'aquarium':
        return _aquarium




def migration_directions_with_kde_plus_mean(df, metric, subject, scaling_metric, cmap_normalization_metric, cmap, threshold, title_size2):	

    # Recognizing the presence of a threshold
    if threshold == None:
        threshold = '_no_threshold'
    else:
        threshold = '_' + threshold

    df_mean_direction = df[metric]

    # Prepare for KDE plot
    x_kde = np.cos(df_mean_direction)
    y_kde = np.sin(df_mean_direction)
    kde = gaussian_kde([x_kde, y_kde])

    # Define the grid for evaluation
    theta_kde = np.linspace(0, 2 * np.pi, 360)
    x_grid = np.cos(theta_kde)
    y_grid = np.sin(theta_kde)

    # Evaluate the KDE on the grid and normalize
    z_kde = kde.evaluate([x_grid, y_grid])
    z_kde = z_kde / z_kde.max() * 0.5  # Normalize to fit within the radial limit

    # Calculate the mean direction
    mean_direction = np.arctan2(np.mean(y_kde), np.mean(x_kde))

    # Start plotting
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'polar': True})

    # Plot KDE
    ax.plot(theta_kde, z_kde, label='Circular KDE', color='None', zorder=5)
    ax.fill(theta_kde, z_kde, alpha=0.25, color='#1b5a9e', zorder=5)

    # Directional Arrows
    scaling_max = df[scaling_metric].max()

    # Normalization of the color map
    if cmap_normalization_metric == None:
        norm = mcolors.Normalize(vmin=0, vmax=1)
    else:
        normalization_min = df[cmap_normalization_metric].min()
        normalization_max = df[cmap_normalization_metric].max()
        norm = mcolors.Normalize(vmin=normalization_min, vmax=normalization_max)

    # Row itteration
    for _, row in df.iterrows():
        scaling_metrics = row[scaling_metric]
        mean_direction_rad = row[metric]
        arrow_length = scaling_metrics / scaling_max

        if cmap_normalization_metric == None:
            color = cmap(norm(arrow_length))
        else:
            color = cmap(norm(row[cmap_normalization_metric]))

        if arrow_length == 0:
            continue  # Skip if the arrow length is zero

        # Dynamically adjust the head size based on arrow_length
        scaling_factor = 1 / arrow_length if arrow_length != 0 else 1
        head_width = 0.011 * scaling_factor
        head_length = 0.013

        ax.arrow(mean_direction_rad, 0, 0, arrow_length, color=color, linewidth=0.75, 
                head_width=head_width, head_length=head_length, zorder=4)

    # Plot the dashed line in the mean direction
    ax.plot([mean_direction, mean_direction], [0, 1], linestyle='--', color='darkslateblue', alpha=0.93, linewidth=2.5, zorder=6)

    # Hide the polar plot frame (spines) but keep the grid visible
    ax.spines['polar'].set_visible(False)
    # Customize grid lines (if needed)
    ax.grid(True, 'major', color='#C6C6C6', linestyle='-', linewidth=0.5, zorder=0)

    # Access and customize the radial grid lines
    radial_lines = ax.get_xgridlines()
    for i, line in enumerate(radial_lines):
        if i % 2 == 0:  # Customize every other radial grid line
            line.set_linestyle('--')
            line.set_color('#E6E6E6')
            line.set_linewidth(0.5)

    radial_lines = ax.get_ygridlines()
    for i, line in enumerate(radial_lines):
        line.set_linestyle('--')
        line.set_color('#E6E6E6')
        line.set_linewidth(0.5)

    # Customize the appearance of the polar plot
    ax.set_title(f'Mean Direction of Travel\nwith Kernel Density Estimate\n$\it{{{subject}}}$', fontsize=title_size2)
    ax.set_yticklabels([])  # Remove radial labels
    ax.set_xticklabels([])  # Remove angular labels

    return plt.gcf()

def donut(df, ax, outer_radius, inner_radius, kde_bw):
    # Extend the data circularly to account for wrap-around at 0 and 2*pi
    extended_data = np.concatenate([df - 2 * np.pi, df, df + 2 * np.pi])

    # Create a grid of theta values (angles)
    theta_grid = np.linspace(0, 2 * np.pi, 360)  # 360 points over full circle
    
    # Create a grid of radii
    r_grid = np.linspace(inner_radius, outer_radius, 100)  # Radius from inner to outer edge
    
    # Compute KDE values for the extended data
    kde = gaussian_kde(extended_data, bw_method=kde_bw)
    kde_values = kde.evaluate(theta_grid)  # Evaluate KDE on the regular theta grid
    
    # Repeat KDE values across radii to create the heatmap data
    kde_values = np.tile(kde_values, (r_grid.size, 1))
    
    # Normalize KDE values for consistent color mapping
    norm = Normalize(vmin=kde_values.min(), vmax=kde_values.max())
    
    # Create the meshgrid for the polar plot
    theta_mesh, r_mesh = np.meshgrid(theta_grid, r_grid)
    
    # Remove polar grid lines and labels
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)  # Hide the outer frame

    return theta_mesh, r_mesh, kde_values, norm

def df_gaussian_donut(df, metric, subject, heatmap, weight, threshold, title_size2, label_size, figtext_color, figtext_size):

    # Recognizing the presence of a threshold
    if threshold == None:
        threshold = '_no_threshold'
    else:
        threshold = '_' + threshold

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    diameter=2
    width_ratio=0.3
    kde_bw=0.1

    df=df[metric]

    # Calculate radius and width from the diameter
    outer_radius = diameter / 2
    width = width_ratio * outer_radius
    inner_radius = outer_radius - width
    
    theta_mesh, r_mesh, kde_values, norm = donut(df, ax, outer_radius, inner_radius, kde_bw)
    
    # Set title and figure text
    ax.set_title(f'Heatmap of Migration Direction\n({subject})', pad=20, ha='center', fontsize=title_size2)
    
    # Add a colorbar
    cbar = plt.colorbar(ax.pcolormesh(theta_mesh, r_mesh, kde_values, shading='gouraud', cmap=heatmap, norm=norm), ax=ax, fraction=0.04, orientation='horizontal', pad=0.1)
    cbar.set_ticks([])
    cbar.outline.set_visible(False)  # Remove outline
    
    # Add min and max labels below the colorbar
    cbar.ax.text(0.05, -0.4, 'min', va='center', ha='center', color='black', transform=cbar.ax.transAxes, fontsize=9)
    cbar.ax.text(0.95, -0.4, 'max', va='center', ha='center', color='black', transform=cbar.ax.transAxes, fontsize=9)

    # Add the density label below the min and max labels
    cbar.set_label('Density', labelpad=10, fontsize=label_size)
    

    return plt.gcf()

    # try to normalize the heatmap colors to the absolute 0 (not min of the kde values) and to the max of the kde values




# =========================================================================================================================================================================================================================================================================================================================================================================================================
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# True track visualization functions


def Visualize_tracks_plotly(
    Spots_df: pd.DataFrame,
    Tracks_df: pd.DataFrame,
    condition: None,
    replicate: None,
    c_mode: str,
    only_one_color: str,
    lut_scaling_metric: str,
    background: str,
    smoothing_index: float,
    lw: float,
    show_tracks: bool,
    let_me_look_at_these: tuple,
    I_just_wanna_be_normal: bool,
    metric_dictionary: dict,
    end_track_markers: bool,
    marker_size: float,
    markers: None
):
    if not show_tracks:
        lw = 0

    # Convert types
    try: condition = int(condition) if condition != 'all' else condition
    except: pass
    try: replicate = int(replicate) if replicate != 'all' else replicate
    except: pass

    let_me_look_at_these = list(let_me_look_at_these)

    # Filter Spots_df
    if condition == 'all':
        filtered_spots = Spots_df.copy()
    elif replicate == 'all':
        filtered_spots = Spots_df[Spots_df['Condition'] == condition]
    else:
        filtered_spots = Spots_df[
            (Spots_df['Condition'] == condition) &
            (Spots_df['Replicate'] == replicate)
        ]

    # Sort once, but only if all columns exist
    sort_cols = ['Condition', 'Replicate', 'Track ID', 'Position T']
    missing_cols = [col for col in sort_cols if col not in filtered_spots.columns]
    if not missing_cols:
        filtered_spots = filtered_spots.sort_values(by=sort_cols)
    # else: skip sorting if columns are missing

    # Smoothing
    if smoothing_index > 1:
        def smooth(g):
            g['X coordinate'] = g['X coordinate'].rolling(smoothing_index, min_periods=1).mean()
            g['Y coordinate'] = g['Y coordinate'].rolling(smoothing_index, min_periods=1).mean()
            return g
        filtered_spots = filtered_spots.groupby(['Condition', 'Replicate', 'Track ID'], group_keys=False).apply(smooth)

    # Color mapping
    np.random.seed(42)
    if c_mode in ['random colors', 'random greys', 'only-one-color']:
        color_vals = {
            'random colors': [_generate_random_color() for _ in range(len(Tracks_df))],
            'random greys': [_generate_random_grey() for _ in range(len(Tracks_df))],
            'only-one-color': [only_one_color for _ in range(len(Tracks_df))]
        }[c_mode]
        Tracks_df['Color'] = color_vals
        color_map = Tracks_df.set_index('Track ID')['Color'].to_dict()
        filtered_spots['Color'] = filtered_spots['Track ID'].map(color_map)
    elif c_mode in ['differentiate conditions', 'differentiate replicates']:
        val_column = 'Condition' if c_mode == 'differentiate conditions' else 'Replicate'
        cmap = plt.get_cmap('Set1')
        unique_vals = filtered_spots[val_column].unique()
        val_to_color = {
            val: mcolors.to_hex(cmap(i % cmap.N))
            for i, val in enumerate(sorted(unique_vals))
        }
        filtered_spots['Color'] = filtered_spots[val_column].map(val_to_color)
    else:
        colormap = _get_cmap(c_mode)
        metric_min = filtered_spots[lut_scaling_metric].min()
        metric_max = filtered_spots[lut_scaling_metric].max()
        norm = plt.Normalize(metric_min, metric_max)
        filtered_spots['Color'] = filtered_spots[lut_scaling_metric].map(lambda v: mcolors.to_hex(colormap(norm(v))))

    # Tick & layout
    x_min, x_max = filtered_spots['X coordinate'].min(), filtered_spots['X coordinate'].max()
    y_min, y_max = filtered_spots['Y coordinate'].min(), filtered_spots['Y coordinate'].max()
    x_ticks = np.arange(x_min, x_max, 200)
    y_ticks = np.arange(y_min, y_max, 200)
    grid_color = 'gainsboro' if background == 'light' else 'silver'
    face_color = 'white' if background == 'light' else 'darkgrey'

    # Group and render
    fig = go.Figure()
    for (cond, repl, track), group_df in filtered_spots.groupby(['Condition', 'Replicate', 'Track ID']):
        color = group_df['Color'].iloc[0]
        hover_dict = {
            metric_dictionary[m]: Tracks_df.loc[
                (Tracks_df['Condition'] == cond) &
                (Tracks_df['Replicate'] == repl) &
                (Tracks_df['Track ID'] == track)
            ][m].values[0]
            for m in let_me_look_at_these if m in Tracks_df.columns
        }
        hover_text = "<br>".join(f"{k}: {v}" for k, v in hover_dict.items())

        fig.add_trace(go.Scatter(
            x=group_df['X coordinate'],
            y=group_df['Y coordinate'],
            mode='lines',
            line=dict(color=color, width=lw),
            hoverinfo='text',
            hovertext=hover_text,
            showlegend=False
        ))

        if end_track_markers:
            if I_just_wanna_be_normal:
                fig.add_trace(go.Scatter(
                    x=[group_df['X coordinate'].iloc[-1]],
                    y=[group_df['Y coordinate'].iloc[-1]],
                    mode='markers',
                    marker=dict(symbol=markers, size=marker_size, color=color),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            else:
                if markers == 'cell':
                    symbol = _cell
                elif markers in ['scaled', 'trains']:
                    track_len = Tracks_df.loc[
                        (Tracks_df['Condition'] == cond) &
                        (Tracks_df['Replicate'] == repl) &
                        (Tracks_df['Track ID'] == track)
                    ]['Track length'].values[0]
                    min_len, max_len = Tracks_df['Track length'].min(), Tracks_df['Track length'].max()
                    percentile = 100 * (track_len - min_len) / (max_len - min_len)
                    symbol = _assign_marker(percentile, _get_markers(markers))
                elif markers in ['random','farm','safari','insects','birds','forest','aquarium']:
                    symbol = np.random.choice(_get_markers(markers))
                else:
                    symbol = ''

                fig.add_trace(go.Scatter(
                    x=[group_df['X coordinate'].iloc[-1]],
                    y=[group_df['Y coordinate'].iloc[-1]],
                    mode='text',
                    text=symbol,
                    textposition='middle center',
                    textfont=dict(size=marker_size),
                    showlegend=False,
                    hoverinfo='skip'
                ))

    # Final layout
    fig.update_layout(
        xaxis=dict(title='Position X [microns]', range=[x_min, x_max], tickvals=x_ticks, ticktext=[f"{t:.0f}" for t in x_ticks], gridcolor=grid_color),
        yaxis=dict(title='Position Y [microns]', range=[y_min, y_max], tickvals=y_ticks, ticktext=[f"{t:.0f}" for t in y_ticks], gridcolor=grid_color),
        plot_bgcolor=face_color,
        width=960, height=720,
        title=dict(text="Track Visualization", x=0.5, font=dict(size=16)),
        showlegend=False
    )

    return fig




def Visualize_tracks_matplotlib(Spots_df:pd.DataFrame, Tracks_df:pd.DataFrame, condition:None, replicate:None, c_mode:str, only_one_color:str, lut_scaling_metric:str, background:str, smoothing_index:float, lw:float, show_tracks:bool, grid:bool, arrows:bool, arrowsize:int):

    if show_tracks:
        pass
    else:
        lw=0


    if condition == None or replicate == None:
        pass
    else:
        try:
            condition = int(condition)
        except ValueError or TypeError:
            pass
        try:
            replicate = int(replicate)
        except ValueError or TypeError:
            pass


    if 'level_0' in Tracks_df.columns:
        Tracks_df.drop(columns=['level_0'], inplace=True)
    
    Tracks_df.reset_index(drop=False, inplace=True)


    sort_cols = ['Condition', 'Replicate', 'Track ID', 'Time point']
    missing_cols = [col for col in sort_cols if col not in Spots_df.columns]
    if not missing_cols:
        if condition == 'all':
            Spots_df = Spots_df.sort_values(by=sort_cols)
        elif condition != 'all' and replicate == 'all':
            Spots_df = Spots_df[Spots_df['Condition'] == condition].sort_values(by=sort_cols)
        elif condition != 'all' and replicate != 'all':
            Spots_df = Spots_df[(Spots_df['Condition'] == condition) & (Spots_df['Replicate'] == replicate)].sort_values(by=sort_cols)
    # else: skip sorting if columns are missing


    if background =='light':
        grid_color = 'gainsboro'
        face_color = 'white'
        grid_alpha = 0.5
        if grid:
            grid_lines = '-.'
        else:
            grid_lines = 'None'

    else:
        grid_color = 'silver'
        face_color = 'darkgrey'
        grid_alpha = 0.75
        if grid:
            grid_lines = '-.'
        else:
            grid_lines = 'None'
    

    np.random.seed(42)  # For reproducibility

    if c_mode in ['random colors', 'random greys', 'only-one-color']:
        colormap = None

        if c_mode in ['random colors']:
            track_colors = [_generate_random_color() for _ in range(len(Tracks_df))]
        elif c_mode in ['random greys']:
            track_colors = [_generate_random_grey() for _ in range(len(Tracks_df))]
        else:
            track_colors = [only_one_color for _ in range(len(Tracks_df))]
        
        color_map_direct = dict(zip(Tracks_df['Track ID'], track_colors))
        Tracks_df['Color'] = Tracks_df['Track ID'].map(color_map_direct)

    elif c_mode in ['differentiate conditions', 'differentiate replicates']:
        if c_mode == 'differentiate conditions':
            colormap = plt.get_cmap('Set1')  # Use qualitative colormap
            unique_vals = Spots_df['Condition'].unique()
            val_column = 'Condition'
        else:
            colormap = plt.get_cmap('Set1')
            unique_vals = Spots_df['Replicate'].unique()
            val_column = 'Replicate'

        # Assign colors to each unique category
        val_to_color = {
            val: colormap(i % colormap.N)  # Wrap around if more values than colors
            for i, val in enumerate(sorted(unique_vals))
        }
        # Map those colors to the tracks
        Tracks_df['Color'] = Tracks_df[val_column].map(val_to_color)

    else:
        colormap = _get_cmap(c_mode)
        metric_min = Spots_df[lut_scaling_metric].min()
        metric_max = Spots_df[lut_scaling_metric].max()


    fig, ax = plt.subplots(figsize=(13, 10))

    x_min = Spots_df['X coordinate'].min()
    x_max = Spots_df['X coordinate'].max()
    y_min = Spots_df['Y coordinate'].min()
    y_max = Spots_df['Y coordinate'].max()

    ax.set_aspect('1', adjustable='box')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X coordinate [microns]')
    ax.set_ylabel('Y coordinate [microns]')
    ax.set_title('Track Visualization', fontsize=12)
    ax.set_facecolor(face_color)
    ax.grid(True, which='both', axis='both', color=grid_color, linestyle=grid_lines, linewidth=1, alpha=grid_alpha)

    # Manually set the major tick locations and labels
    x_ticks_major = np.arange(x_min, x_max, 200)  # Adjust the step size as needed
    y_ticks_major = np.arange(y_min, y_max, 200)  # Adjust the step size as needed
    ax.set_xticks(x_ticks_major)
    ax.set_yticks(y_ticks_major)
    ax.set_xticklabels([f'{tick:.0f}' for tick in x_ticks_major])
    ax.set_yticklabels([f'{tick:.0f}' for tick in y_ticks_major])

    # Enable minor ticks and set their locations
    ax.minorticks_on()
    x_ticks_minor = np.arange(x_min, x_max, 50)  # Minor ticks every 50 microns
    y_ticks_minor = np.arange(y_min, y_max, 50)  # Minor ticks every 50 microns
    ax.set_xticks(x_ticks_minor, minor=True)
    ax.set_yticks(y_ticks_minor, minor=True)
    ax.tick_params(axis='both', which='major', labelsize=8)


    Spots_grouped = Spots_df.groupby(['Condition', 'Replicate', 'Track ID'])
    Tracks_df.set_index(['Condition', 'Replicate', 'Track ID'], inplace=True)

    fig = go.Figure()

    for (cond, repl, track), group_df in Spots_grouped:
        """
        - group_keys is a tuple like: ('Condition_A', 'Rep1', 'Track_001')
        - group_df is the actual dataframe for that group
        
        """

        track_row = Tracks_df.loc[(cond, repl, track)]
        track_row['Condition'] = cond
        track_row['Replicate'] = repl
        track_row['Track ID'] = track


        if colormap is not None and c_mode in ['differentiate conditions', 'differentiate replicates']:
            key = track_row[val_column]  # val_column is either 'Condition' or 'Replicate'
            color = colormap(unique_vals.tolist().index(key) % colormap.N)  # consistent mapping
            group_df['Color'] = mcolors.to_hex(color)
            
        elif colormap is not None:
            # This is for metric-based color mapping (quantitative)
            norm = plt.Normalize(metric_min, metric_max)
            color = colormap(norm(track_row[lut_scaling_metric]))
            group_df['Color'] = mcolors.to_hex(color)

        elif c_mode in ['random colors', 'random greys']:
            group_df['Color'] = track_row['Color']

        elif c_mode == 'only-one-color':
            group_df['Color'] = only_one_color


        if (type(smoothing_index) is int or type(smoothing_index) is float) and smoothing_index > 1:
            group_df['X coordinate'] = group_df['X coordinate'].rolling(window=smoothing_index, min_periods=1).mean()
            group_df['Y coordinate'] = group_df['Y coordinate'].rolling(window=smoothing_index, min_periods=1).mean()
        else:
            group_df['X coordinate'] = group_df['X coordinate']
            group_df['Y coordinate'] = group_df['Y coordinate']

        ax.plot(group_df['X coordinate'], group_df['Y coordinate'], color=group_df['Color'].iloc[0], linewidth=lw)


        if len(group_df['X coordinate']) > 1 & arrows:
            # Use trigonometry to calculate the direction (dx, dy) from the angle
            dx = np.cos(track_row['Direction mean (rad)'])  # Change in x based on angle
            dy = np.sin(track_row['Direction mean (rad)'])  # Change in y based on angle

            # Create an arrow to indicate direction
            arrow = FancyArrowPatch(
                posA=(group_df['X coordinate'].iloc[-2], group_df['Y coordinate'].iloc[-2]),  # Start position (second-to-last point)
                posB=(group_df['X coordinate'].iloc[-2] + dx, group_df['Y coordinate'].iloc[-2] + dy),  # End position based on direction
                arrowstyle='-|>',  # Style of the arrow (you can adjust the style as needed)
                color=group_df['Color'].iloc[0],  # Set the color of the arrow
                mutation_scale=arrowsize,  # Scale the size of the arrow head (adjust this based on the plot scale)
                linewidth=1.2,  # Line width for the arrow
                zorder=30  # Ensure the arrow is drawn on top of the line
            )

            # Add the arrow to your plot (if you're using a `matplotlib` figure/axes)
            plt.gca().add_patch(arrow)


    Tracks_df.reset_index(drop=False, inplace=True)

    return plt.gcf()


def Visualize_normalized_tracks_plotly(Spots_df:pd.DataFrame, Tracks_df:pd.DataFrame, condition:None, replicate:None, c_mode:str, only_one_color:str, lut_scaling_metric:str, smoothing_index:float, lw:float, show_tracks:bool, let_me_look_at_these:tuple, I_just_wanna_be_normal:bool, metric_dictionary:dict, end_track_markers:bool, marker_size:float, markers:None):

    if show_tracks:
        pass
    else:
        lw=0

    
    if condition == None or replicate == None:
        pass
    else:
        try:
            condition = int(condition)
        except (ValueError, TypeError):
            pass
        try:
            replicate = int(replicate)
        except (ValueError, TypeError):
            pass
    

    let_me_look_at_these = list(let_me_look_at_these)
    if 'level_0' in Tracks_df.columns:
        Tracks_df.drop(columns=['level_0'], inplace=True)
    
    Tracks_df.reset_index(drop=False, inplace=True)


    sort_cols = ['Condition', 'Replicate', 'Track ID', 'Time point']
    missing_cols = [col for col in sort_cols if col not in Spots_df.columns]
    if not missing_cols:
        if condition == 'all':
            Spots_df = Spots_df.sort_values(by=sort_cols)
        elif condition != 'all' and replicate == 'all':
            Spots_df = Spots_df[Spots_df['Condition'] == condition].sort_values(by=sort_cols)
        elif condition != 'all' and replicate != 'all':
            Spots_df = Spots_df[(Spots_df['Condition'] == condition) & (Spots_df['Replicate'] == replicate)].sort_values(by=sort_cols)
    # else: skip sorting if columns are missing

    np.random.seed(42)  # For reproducibility

    # Set colors based on chosen mode
    if c_mode in ['random colors', 'random greys', 'only-one-color']:
        colormap = None
        if c_mode == 'random colors':
            track_colors = [_generate_random_color() for _ in range(len(Tracks_df))]
        elif c_mode == 'random greys':
            track_colors = [_generate_random_grey() for _ in range(len(Tracks_df))]
        else:
            track_colors = [only_one_color for _ in range(len(Tracks_df))]
        
        color_map_direct = dict(zip(Tracks_df['Track ID'], track_colors))
        Tracks_df['Color'] = Tracks_df['Track ID'].map(color_map_direct)

    elif c_mode in ['differentiate conditions', 'differentiate replicates']:
        if c_mode == 'differentiate conditions':
            colormap = plt.get_cmap('Set1')  # Use qualitative colormap
            unique_vals = Spots_df['Condition'].unique()
            val_column = 'Condition'
        else:
            colormap = plt.get_cmap('Set1')
            unique_vals = Spots_df['Replicate'].unique()
            val_column = 'Replicate'

        # Assign colors to each unique category
        val_to_color = {
            val: colormap(i % colormap.N)  # Wrap around if more values than colors
            for i, val in enumerate(sorted(unique_vals))
        }
        # Map those colors to the tracks
        Tracks_df['Color'] = Tracks_df[val_column].map(val_to_color)

    else:
        colormap = _get_cmap(c_mode)
        metric_min = Spots_df[lut_scaling_metric].min()
        metric_max = Spots_df[lut_scaling_metric].max()

    min_track_length = Tracks_df['Track length'].min()
    max_track_length = Tracks_df['Track length'].max()

    Spots_grouped = Spots_df.groupby(['Condition', 'Replicate', 'Track ID'])
    Tracks_df.set_index(['Condition', 'Replicate', 'Track ID'], inplace=True)

    processed_groups = []

    # Normalize each track's positions to start at (0, 0)
    for (cond, repl, track), group_df in Spots_grouped:
        # Apply smoothing if required
        if isinstance(smoothing_index, (int, float)) and smoothing_index > 1:
            group_df['X coordinate'] = group_df['X coordinate'].rolling(window=int(smoothing_index), min_periods=1).mean()
            group_df['Y coordinate'] = group_df['Y coordinate'].rolling(window=int(smoothing_index), min_periods=1).mean()

        # Normalize positions to start at (0, 0)
        start_x = group_df['X coordinate'].iloc[0]
        start_y = group_df['Y coordinate'].iloc[0]

        group_df['X coordinate'] -= start_x
        group_df['Y coordinate'] -= start_y

        processed_groups.append(group_df)

    # Concatenate everything back into one DataFrame
    Spots_df = pd.concat(processed_groups)


    # Convert to polar coordinates (theta in radians & degrees)
    Spots_df['r'] = np.sqrt(Spots_df['X coordinate']**2 + Spots_df['Y coordinate']**2)
    Spots_df['theta'] = np.arctan2(Spots_df['Y coordinate'], Spots_df['X coordinate'])
    Spots_df['theta_deg'] = np.degrees(Spots_df['theta'])
    
    # Determine maximum radius for layout
    y_max = Spots_df['r'].max()
    y_max_r = y_max * 1.12
    y_max_a = y_max * 1.1

    # Create Plotly polar figure
    fig = go.Figure()

    Spots_grouped = Spots_df.groupby(['Condition', 'Replicate', 'Track ID'])

    # Loop over tracks and add a polar trace for each
    for (cond, repl, track), group_df in Spots_grouped:

        track_row = Tracks_df.loc[(cond, repl, track)]
        track_row['Condition'] = cond
        track_row['Replicate'] = repl
        track_row['Track ID'] = track

        if colormap is not None and c_mode in ['differentiate conditions', 'differentiate replicates']:
            key = track_row[val_column]  # val_column is either 'Condition' or 'Replicate'
            color = colormap(unique_vals.tolist().index(key) % colormap.N)  # consistent mapping
            group_df['Color'] = mcolors.to_hex(color)
            
        elif colormap is not None:
            # This is for metric-based color mapping (quantitative)
            norm = plt.Normalize(metric_min, metric_max)
            color = colormap(norm(track_row[lut_scaling_metric]))
            group_df['Color'] = mcolors.to_hex(color)
            
        elif c_mode in ['random colors', 'random greys']:
            group_df['Color'] = track_row['Color']

        elif c_mode == 'only-one-color':
            group_df['Color'] = only_one_color


        # Create hover text if extra metrics are provided
        hover_text = ""
        if len(let_me_look_at_these) > 0:
            hover_dict = {}
            for metric in let_me_look_at_these:
                hover_dict[metric_dictionary[metric]] = track_row[metric]
            hover_text = "<br>".join(f"{key}: {value}" for key, value in hover_dict.items())

        # Add the polar trace
        fig.add_trace(go.Scatterpolar(
            r=group_df['r'],
            theta=group_df['theta_deg'],
            mode='lines',
            line=dict(color=group_df['Color'].iloc[0], width=lw),
            name=f"Track {track}",
            hovertemplate=hover_text + "<extra></extra>"
        ))
        
        # Optionally add markers for the end of each track within the polar coordinate system
        if end_track_markers:
            if I_just_wanna_be_normal:
                fig.add_trace(go.Scatterpolar(
                    r=[group_df['r'].iloc[-1]],
                    theta=[group_df['theta_deg'].iloc[-1]],
                    mode='markers',
                    marker=dict(
                        symbol=markers,
                        size=marker_size,
                        color=group_df['Color'].iloc[0],
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            else:
                if markers == 'cell':
                    marker = _cell  # Ensure you define _cell if needed
                elif markers in ['scaled', 'trains']:
                    markers_ = _get_markers(markers)
                    percentile_value = ((track_row['Track length'] - min_track_length) / (max_track_length - min_track_length)) * 100
                    marker = _assign_marker(percentile_value, markers_)
                elif markers in ['random','farm','safari','insects','birds','forest','aquarium']:
                    markers_ = _get_markers(markers)
                    marker = np.random.choice(markers_)
                else:
                    marker = ''

                fig.add_trace(go.Scatterpolar(
                    r=[group_df['r'].iloc[-1]],
                    theta=[group_df['theta_deg'].iloc[-1]],
                    mode='text',
                    text=marker,
                    textposition='middle center',
                    textfont=dict(size=marker_size),
                    showlegend=False,
                    hoverinfo='skip'
                ))

    # Update layout for a polar plot
    fig.update_layout(
        title="Normalized Tracks",
        title_x=0.5,
        polar=dict(
            bgcolor='white',
            radialaxis=dict(
                range=[0, y_max_r],
                gridcolor='lightgrey',
                showticklabels=False,
                ticks=''
            ),
            angularaxis=dict(
                gridcolor='grey',
                showticklabels=False,
                ticks=''
            ),
        ),
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # Add annotation for maximum radius if desired
    fig.add_annotation(
        text=f'{int(round(y_max_a, -1))} µm',
        x=0.74, y=0.5, xref='paper', yref='paper',
        showarrow=False,
        font=dict(size=12, color="black")
    )

    return fig

def Visualize_normalized_tracks_matplotlib(Spots_df:pd.DataFrame, Tracks_df:pd.DataFrame, condition:None, replicate:None, c_mode:str, only_one_color:str, lut_scaling_metric:str, smoothing_index:float, lw:float, show_tracks:bool, grid:bool, arrows:bool, arrowsize:int):

    arrow_length = 1  # Length of the arrow in data units

    if show_tracks:
        pass
    else:
        lw=0

    
    if condition == None or replicate == None:
        pass
    else:
        try:
            condition = int(condition)
        except (ValueError, TypeError):
            pass
        try:
            replicate = int(replicate)
        except (ValueError, TypeError):
            pass
    

    if 'level_0' in Tracks_df.columns:
        Tracks_df.drop(columns=['level_0'], inplace=True)
    
    Tracks_df.reset_index(drop=False, inplace=True)


    sort_cols = ['Condition', 'Replicate', 'Track ID', 'Time point']
    missing_cols = [col for col in sort_cols if col not in Spots_df.columns]
    if not missing_cols:
        if condition == 'all':
            Spots_df = Spots_df.sort_values(by=sort_cols)
        elif condition != 'all' and replicate == 'all':
            Spots_df = Spots_df[Spots_df['Condition'] == condition].sort_values(by=sort_cols)
        elif condition != 'all' and replicate != 'all':
            Spots_df = Spots_df[(Spots_df['Condition'] == condition) & (Spots_df['Replicate'] == replicate)].sort_values(by=sort_cols)
    # else: skip sorting if columns are missing
    
    np.random.seed(42)  # For reproducibility
    
    # Set colors based on chosen mode
    if c_mode in ['random colors', 'random greys', 'only-one-color']:
        colormap = None
        if c_mode == 'random colors':
            track_colors = [_generate_random_color() for _ in range(len(Tracks_df))]
        elif c_mode == 'random greys':
            track_colors = [_generate_random_grey() for _ in range(len(Tracks_df))]
        else:
            track_colors = [only_one_color for _ in range(len(Tracks_df))]
        
        color_map_direct = dict(zip(Tracks_df['Track ID'], track_colors))
        Tracks_df['Color'] = Tracks_df['Track ID'].map(color_map_direct)

    elif c_mode in ['differentiate conditions', 'differentiate replicates']:
        if c_mode == 'differentiate conditions':
            colormap = plt.get_cmap('Set1')  # Use qualitative colormap
            unique_vals = Spots_df['Condition'].unique()
            val_column = 'Condition'
        else:
            colormap = plt.get_cmap('Set1')
            unique_vals = Spots_df['Replicate'].unique()
            val_column = 'Replicate'

        # Assign colors to each unique category
        val_to_color = {
            val: colormap(i % colormap.N)  # Wrap around if more values than colors
            for i, val in enumerate(sorted(unique_vals))
        }
        # Map those colors to the tracks
        Tracks_df['Color'] = Tracks_df[val_column].map(val_to_color)

    else:
        colormap = _get_cmap(c_mode)
        metric_min = Spots_df[lut_scaling_metric].min()
        metric_max = Spots_df[lut_scaling_metric].max()


    Spots_grouped = Spots_df.groupby(['Condition', 'Replicate', 'Track ID'])
    Tracks_df.set_index(['Condition', 'Replicate', 'Track ID'], inplace=True)
    
    processed_groups = []

    # Normalize each track's positions to start at (0, 0)
    for (cond, repl, track), group_df in Spots_grouped:
        # Apply smoothing if required
        if isinstance(smoothing_index, (int, float)) and smoothing_index > 1:
            group_df['X coordinate'] = group_df['X coordinate'].rolling(window=int(smoothing_index), min_periods=1).mean()
            group_df['Y coordinate'] = group_df['Y coordinate'].rolling(window=int(smoothing_index), min_periods=1).mean()

        # Normalize positions to start at (0, 0)
        start_x = group_df['X coordinate'].iloc[0]
        start_y = group_df['Y coordinate'].iloc[0]

        group_df['X coordinate'] -= start_x
        group_df['Y coordinate'] -= start_y

        processed_groups.append(group_df)

    # Concatenate everything back into one DataFrame
    Spots_df = pd.concat(processed_groups)

    # Convert to polar coordinates.
    Spots_df['r'] = np.sqrt(Spots_df['X coordinate']**2 + Spots_df['Y coordinate']**2)
    Spots_df['theta'] = np.arctan2(Spots_df['Y coordinate'], Spots_df['X coordinate'])

    Spots_grouped = Spots_df.groupby(['Condition', 'Replicate', 'Track ID'])
    
    fig, ax = plt.subplots(figsize=(12.5, 9.5), subplot_kw={'projection': 'polar'})
    y_max = Spots_df['r'].max() * 1.1

    x_grid_color = 'grey'
    y_grid_color = 'lightgrey'
    ax.set_facecolor('white')

    ax.set_title('Normalized Tracks')
    ax.set_ylim(0, y_max)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines['polar'].set_visible(False)
    ax.grid(grid)
    

    # Plot all tracks along with an arrow at the track's end pointing toward the mean direction
    for (cond, repl, track), group_df in Spots_grouped:

        track_row = Tracks_df.loc[(cond, repl, track)]
        track_row['Condition'] = cond
        track_row['Replicate'] = repl
        track_row['Track ID'] = track

        if colormap is not None and c_mode in ['differentiate conditions', 'differentiate replicates']:
            key = track_row[val_column]  # val_column is either 'Condition' or 'Replicate'
            color = colormap(unique_vals.tolist().index(key) % colormap.N)  # consistent mapping
            group_df['Color'] = mcolors.to_hex(color)
            
        elif colormap is not None:
            # This is for metric-based color mapping (quantitative)
            norm = plt.Normalize(metric_min, metric_max)
            color = colormap(norm(track_row[lut_scaling_metric]))
            group_df['Color'] = mcolors.to_hex(color)
            
        elif c_mode in ['random colors', 'random greys']:
            group_df['Color'] = track_row['Color']

        elif c_mode == 'only-one-color':
            group_df['Color'] = only_one_color


        # Plot the track using computed color.
        ax.plot(group_df['theta'], group_df['r'], lw=lw, color=group_df['Color'].iloc[0])

        # If arrows flag is True, add an arrow at the end of the track.
        if arrows:
            # Get the last point of the current track.
            last_point = group_df.iloc[-1]
            r_end = last_point['r']
            theta_end = last_point['theta']

            # Get the mean direction from the track metadata (in radians)
            mean_dir = track_row['Direction mean (rad)']

            # Convert last point from polar to Cartesian coordinates.
            x_end = r_end * np.cos(theta_end)
            y_end = r_end * np.sin(theta_end)
            
            # Calculate the arrow tip in Cartesian coordinates using the mean direction.
            x_tip = x_end + arrow_length * np.cos(mean_dir)
            y_tip = y_end + arrow_length * np.sin(mean_dir)
            
            # Convert the computed Cartesian arrow tip back to polar coordinates.
            r_tip = np.sqrt(x_tip**2 + y_tip**2)
            theta_tip = np.arctan2(y_tip, x_tip)

            ax.annotate(
                '',
                xy=(theta_tip, r_tip),
                xytext=(theta_end, r_end),
                arrowprops=dict(arrowstyle='-|>', 
                color=group_df['Color'].iloc[0], 
                lw=lw, 
                mutation_scale=arrowsize),
                annotation_clip=False
            )

    Tracks_df.reset_index(drop=False, inplace=True)

    # Style the polar grid.
    for i, line in enumerate(ax.get_xgridlines()):
        if i % 2 == 0:
            line.set_linestyle('--')
            line.set_color(x_grid_color)
            line.set_linewidth(0.5)

    for line in ax.get_ygridlines():
        line.set_linestyle('-.')
        line.set_color(y_grid_color)
        line.set_linewidth(0.5)

    return plt.gcf()


def Lut_map(Tracks_df:pd.DataFrame, c_mode:str, lut_scaling_metric:str, metrics_dict:dict):

    lut_norm_df = Tracks_df[['Track ID', lut_scaling_metric]].drop_duplicates()

    # Normalize the Net distance to a 0-1 range
    lut_min = lut_norm_df[lut_scaling_metric].min()
    lut_max = lut_norm_df[lut_scaling_metric].max()
    norm = plt.Normalize(vmin=lut_min, vmax=lut_max)

    if c_mode not in ['random colors', 'random greys', 'only-one-color']:
        # Get the colormap based on the selected mode
        colormap = _get_cmap(c_mode)
    
        # Add a colorbar to show the LUT map
        sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        # Create a separate figure for the LUT map (colorbar)
        fig_lut, ax_lut = plt.subplots(figsize=(2, 6))
        ax_lut.axis('off')
        cbar = fig_lut.colorbar(sm, ax=ax_lut, orientation='vertical', extend='both', shrink=0.85)
        cbar.set_label(metrics_dict[lut_scaling_metric], fontsize=10)

        plt.gcf()
    
    else:
        pass



# =========================================================================================================================================================================================================================================================================================================================================================================================================
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Time series plots


def Scatter_poly_fit_chart_altair(Time_df:pd.DataFrame, condition:None, replicate:None, replicates_separately:bool, metric:str, Metric:str, cmap:str, degree:list, point_fill:bool, point_size:int, point_outline:bool, point_outline_width:float, opacity:float):

    if point_outline:
        outline_color = 'black'
    else:
        outline_color = ''
    
    if condition == None or replicate == None:
        pass
    else:
        try:
            condition = int(condition)
        except ValueError or TypeError:
            pass
        try:
            replicate = int(replicate)
        except ValueError or TypeError:
            pass

    Time_df = Time_df.sort_values(by=['Condition', 'Replicate', 'Time point'])

    if condition == 'all':
        Time_df = Time_df.groupby(['Condition', 'Time point']).agg({metric: 'mean'}).reset_index()
        element = 'Condition'
        shorthand = 'Condition:N'
        what = ''
    elif condition != 'all' and replicate == 'all':
        if replicates_separately:
            Time_df = Time_df[Time_df['Condition'] == condition]
            shorthand = 'Replicate:N'
            element = 'Replicate'
            what = f'for condition {condition}'
        else:
            Time_df = Time_df[Time_df['Condition'] == condition].groupby(['Condition', 'Time point']).agg({metric: 'mean'}).reset_index()
            shorthand = 'Condition:N'
            element = 'Condition'
            what = f'for condition {condition}'
    elif condition != 'all' and replicate != 'all':
        Time_df = Time_df[(Time_df['Condition'] == condition) & (Time_df['Replicate'] == replicate)].sort_values(by=['Condition', 'Replicate', 'Time point'])
        shorthand = 'Replicate:N'
        element = 'Replicate'
        what = f'for condition {condition} and replicate {replicate}'   
    

    # Retrieve unique conditions and assign colors from the selected qualitative colormap.
    elements = Time_df[element].unique().tolist()
    colors = _make_cmap(elements, cmap)

    highlight = alt.selection_point(
        on="pointerover", fields=[element], nearest=True
        )
    
    # Create a base chart with the common encodings.
    base = alt.Chart(Time_df).encode(
        x=alt.X('Time point', title='Time position'),
        y=alt.Y(metric, title=Metric),
        color=alt.Color(shorthand, title='Condition', scale=alt.Scale(domain=elements, range=colors))
        )
    
    ignore = 0.1 

    if point_fill:
        scatter = base.mark_circle(
            size=point_size,
            stroke=outline_color,
            strokeWidth=point_outline_width,
        ).encode(
            opacity=alt.when(~highlight).then(alt.value(ignore)).otherwise(alt.value(opacity)),
            tooltip=alt.value(None),
        ).add_params(
            highlight
        )
    else:
        # Scatter layer: displays the actual data points.
        scatter = base.mark_point(
            size=point_size, 
            filled=point_fill, 
            strokeWidth=point_outline_width,
        ).encode(
            opacity=alt.when(~highlight).then(alt.value(ignore)).otherwise(alt.value(opacity)),
            # tooltip=alt.Tooltip(metric, title=f'Cond: {condition}, Repl: {condition}'),
            tooltip=['Time point', metric, shorthand],
        ).add_params(
            highlight
        )
    
    if degree[0] == 0:
        chart = alt.layer(scatter).properties(
            width=1100,
            height=350,
            title=f'{Metric} with Polynomial Fits'
        ).configure_view(
            strokeWidth=1
        )
    else:
        # Build a list of polynomial fit layers, one for each specified degree.
        polynomial_fit = [
            base.transform_regression(
                "Time point", metric,
                method="poly",
                order=order,
                groupby=[element],
                as_=["Time point", str(order)]
            ).mark_line(
            ).transform_fold(
                [str(order)], as_=["degree", metric]
            ).encode(
                x=alt.X('Time point', title='Time position'),
                y=alt.Y(metric, title=Metric),
                color=alt.Color(shorthand, title='Condition', scale=alt.Scale(domain=elements, range=colors)),
                size=alt.when(~highlight).then(alt.value(1.25)).otherwise(alt.value(3))
            )
            for order in degree
            ]
        
        # Layer the scatter points with all polynomial fits.
        chart = alt.layer(scatter, *polynomial_fit).properties(
            width=1100,
            height=350,
            title=f'{Metric} with Polynomial Fits for {what}'
        ).configure_view(
            strokeWidth=1
        ).interactive()

    return chart


def Line_chart_altair(Time_df:pd.DataFrame, condition:None, replicate:None, replicates_separately:bool, metric:str, Metric:str, cmap:str, interpolation:str, show_median:bool):

    if interpolation == 'None':
        interpolation = None
        
    metric_mean = metric
    metric_median = metric.replace('MEAN', 'MEDIAN')

    if condition == None or replicate == None:
        pass
    else:
        try:
            condition = int(condition)
        except ValueError or TypeError:
            pass
        try:
            replicate = int(replicate)
        except ValueError or TypeError:
            pass

    Time_df.sort_values(by=['Condition', 'Replicate', 'Time point'])

    if condition == 'all':
        Time_df = Time_df.groupby(['Condition','Time point']).agg({metric_mean: 'mean', metric_median: 'median'}).reset_index()
        shorthand = 'Condition:N'
        element = 'Condition'
        what = None
    elif condition != 'all' and replicate == 'all':
        if replicates_separately:
            Time_df = Time_df[Time_df['Condition'] == condition]
            shorthand = 'Replicate:N'
            element = 'Replicate'
            what = f'for condition {condition}'
        else:
            Time_df = Time_df[Time_df['Condition'] == condition].groupby(['Condition', 'Time point']).agg({metric_mean: 'mean', metric_median: 'median'}).reset_index()
            shorthand = 'Condition:N'
            element = 'Condition'
            what = f'for condition {condition}'
    elif condition != 'all' and replicate != 'all':
        Time_df = Time_df[(Time_df['Condition'] == condition) & (Time_df['Replicate'] == replicate)].sort_values(by=['Condition', 'Replicate', 'Time point'])
        shorthand = 'Replicate:N'
        element = 'Replicate'
        what = f'for condition {condition} and replicate {replicate}' 

    if show_median:
        median_opacity = 0.85
        median_mark_opacity = 1
        Metric = Metric + ' and median'
    else:
        median_opacity = 0
        median_mark_opacity = 0
            
    # Retrieve unique conditions and assign colors from the selected qualitative colormap.
    elements = Time_df[element].unique().tolist()
    colors = _make_cmap(elements, cmap)
    

    nearest = alt.selection_point(nearest=True, on="pointerover",
                              fields=['Time point'], empty=False)    
    
    color_scale = alt.Scale(domain=elements, range=colors)

    mean_base = alt.Chart(Time_df).encode(
        x=alt.X('Time point', title='Time position'),
        y=alt.Y(metric_mean, title=None),
        color=alt.Color(shorthand, title='Condition', scale=color_scale),
        )
    
    median_base = alt.Chart(Time_df).encode(
        x=alt.X('Time point', title='Time position'),
        y=alt.Y(metric_median, title='Median ' + Metric),
        color=alt.Color(shorthand, title='Condition', scale=color_scale),
        )

    if interpolation is not None:
        mean_line = mean_base.mark_line(interpolate=interpolation)
        median_line = median_base.mark_line(interpolate=interpolation, strokeDash=[4, 3]).encode(
            opacity=alt.value(median_opacity)
            )
        text_median = 0
    else:
        mean_line = mean_base.mark_line()
        median_line = median_base.mark_line(strokeDash=[4, 3]).encode(
            opacity=alt.value(median_opacity)
            )


    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    mean_selectors = mean_base.mark_point().encode(
        opacity=alt.value(0),
        ).add_params(
        nearest
        )
    median_selectors = median_base.mark_point().encode(
        opacity=alt.value(0),
        ).add_params(
        nearest
        )
    
    when_near = alt.when(nearest)

    # Draw points on the line, and highlight based on selection
    mean_points = mean_base.mark_point().encode(
        opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
        )
    median_points = median_base.mark_point().encode(
        opacity=when_near.then(alt.value(median_mark_opacity)).otherwise(alt.value(0))
        )

    # Draw text labels near the points, and highlight based on selection
    text_mean = mean_line.mark_text(align="left", dx=6, dy=-6).encode(
        color=alt.value('black'),	
        text=when_near.then(metric_mean).otherwise(alt.value(" "))
        )
    text_median = median_line.mark_text(align="left", dx=5, dy=-5).encode(
        color=alt.value('grey'),
        text=when_near.then(metric_median).otherwise(alt.value(" "))
        )

    # Draw a rule at the location of the selection
    rules = alt.Chart(Time_df).mark_rule(color="gray").encode(
        x='Time point',
        ).transform_filter(
        nearest
        )


    chart = alt.layer(
        mean_line, mean_points, text_mean, mean_selectors, rules, median_line, median_points, text_median, median_selectors
        ).properties(
        width=1100,
        height=350,
        title=f'{Metric} across time {what}'
        ).configure_view(
        strokeWidth=1
        ).interactive()
    
        
    return chart


def Errorband_chart_altair(Time_df:pd.DataFrame, condition:None, replicate:None, replicates_separately:bool, metric:str, Metric:str, cmap:str, interpolation:str, extent:list, show_mean:bool):

    if interpolation == 'None':
        interpolation = None

    metric_mean = metric
    metric_std = metric.replace('MEAN', 'STD')
    metric_min = metric.replace('MEAN', 'MIN')
    metric_max = metric.replace('MEAN', 'MAX')

    if condition == None or replicate == None:
        pass
    else:
        try:
            condition = int(condition)
        except ValueError or TypeError:
            pass
        try:
            replicate = int(replicate)
        except ValueError or TypeError:
            pass

    Time_df.sort_values(by=['Condition', 'Replicate', 'Time point'])

    if condition == 'all':
        Time_df = Time_df.groupby(['Condition','Time point']).agg({metric_mean: 'mean', metric_std: 'std', metric_min: 'min', metric_max: 'max'}).reset_index()
        shorthand = 'Condition:N'
        element = 'Condition'
        element_ = 'Condition'
        what = None
    elif condition != 'all' and replicate == 'all':
        if replicates_separately:
            Time_df = Time_df[Time_df['Condition'] == condition]
            shorthand = 'Replicate:N'
            element = 'Replicate'
            element_ = 'Replicate'
            what = f'for condition {condition}'
        else:
            Time_df = Time_df[Time_df['Condition'] == condition].groupby(['Condition', 'Time point']).agg({metric_mean: 'mean', metric_std: 'std', metric_min: 'min', metric_max: 'max'}).reset_index()
            shorthand = 'Condition:N'
            element = 'Condition'
            element_ = 'Condition'
            what = f'for condition {condition}'
    elif condition != 'all' and replicate != 'all':
        Time_df = Time_df[(Time_df['Condition'] == condition) & (Time_df['Replicate'] == replicate)].sort_values(by=['Condition', 'Replicate', 'Time point'])
        shorthand = 'Replicate:N'
        element = 'Replicate'
        element_ = 'Replicate'
        what = f'for condition {condition} and replicate {replicate}'

    if show_mean:
        mean_opacity = 1
    else:
        mean_opacity = 0

    if condition == 'all' or (replicate == 'all' and replicates_separately):
        text_opacity = 0
        rule_opacity = 0
        marks_opacity = 0
    else:
        text_opacity = 1
        rule_opacity = 1
        marks_opacity = 1

    l = -75

    Time_df['lower'] = Time_df[metric_mean] - Time_df[metric_std]
    Time_df['upper'] = Time_df[metric_mean] + Time_df[metric_std]

    # Retrieve unique conditions and assign colors from the selected qualitative colormap.
    elements = Time_df[element].unique().tolist()
    colors = _make_cmap(elements, cmap)
    color_scale=alt.Scale(domain=elements, range=colors)
    
    if extent == 'min-max':
        band = alt.Chart(Time_df).encode(
            x=alt.X('Time point', title='Time position'),
            y=alt.Y(metric_min),
            y2=alt.Y2(metric_max),
            color=alt.Color(shorthand, title=element_, scale=color_scale),
            tooltip=alt.value(None)
            )
        if interpolation is not None:
            band = band.mark_errorband(interpolate=interpolation)
        else:
            band = band.mark_errorband()

    elif extent == 'std':
        band = alt.Chart(Time_df).encode(
            x=alt.X('Time point', title='Time position'),
            y=alt.Y('upper'),
            y2=alt.Y2('lower'),
            color=alt.Color(shorthand, title=element_, scale=color_scale),   
            opacity=alt.value(0.25),
            tooltip=alt.value(None)
        )
        if interpolation is not None:
            band = band.mark_errorband(interpolate=interpolation)
        else:
            band = band.mark_errorband()

    else:
        band = alt.Chart(Time_df).mark_errorband(extent=extent).encode(
            x=alt.X('Time point', title='Time position'),
            y=alt.Y('upper'),
            y2=alt.Y2('lower'),
            color=alt.Color(shorthand, title=element_, scale=color_scale),
            opacity=alt.value(0.25),
            tooltip=alt.value(None)
        )
        if interpolation is not None:
            band = band.mark_errorband(interpolate=interpolation)  
        else:
            band = band.mark_errorband()

    mean_base = alt.Chart(Time_df).encode(
        x=alt.X('Time point', title='Time position'),
        y=alt.Y(metric_mean, title=None),
        color=alt.Color(shorthand, title=element_, scale=color_scale),
        opacity=alt.value(mean_opacity),
        strokeWidth=alt.value(3),
        tooltip=alt.value(None)
        )

    if interpolation is not None:
        mean_line = mean_base.mark_line(interpolate=interpolation)
    else:
        mean_line = mean_base.mark_line()

    nearest = alt.selection_point(nearest=True, on="pointerover",
                              fields=['Time point'], empty=False)    
    
    color_scale = alt.Scale(domain=elements, range=colors)

    # Transparent selectors across the chart. This is what tells us
    mean_selectors = mean_base.mark_point().encode(
        opacity=alt.value(0),
        ).add_params(
        nearest
        )
    
    when_near = alt.when(nearest)

    mean_points = mean_base.mark_point().encode(
        opacity=when_near.then(alt.value(marks_opacity)).otherwise(alt.value(0)),
        )    

    # Draw a rule at the location of the selection
    rules = alt.Chart(Time_df).mark_rule(color="gray", opacity=rule_opacity).encode(
        x='Time point',
        ).transform_filter(
        nearest
        )


    # Calculate separate fields for each line of the tooltip.
    tooltip_data = alt.Chart(Time_df).transform_calculate(
        tooltip_line1=f'"{element_}: " + datum.Condition',
        tooltip_line2=f'"Time point: " + datum.Time point',
        tooltip_line3=f'"Mean: " + datum["{metric_mean}"]',
        tooltip_line4=f'"Std: " + datum["{metric_std}"]',
        tooltip_line5=f'"Min: " + datum["{metric_min}"]',
        tooltip_line6=f'"Max: " + datum["{metric_max}"]',
        )

    # Create text marks for each line.
    tooltip_line1_mark = tooltip_data.mark_text(
        align='left',
        dx=6,
        dy=-30-l,  # adjust vertical offset as needed
        fontSize=12,
        fontWeight='bold'
        ).encode(
        x='Time point',
        text=alt.condition(nearest, 'tooltip_line1:N', alt.value('')),
        opacity=alt.value(text_opacity)
        ).transform_filter(nearest)

    tooltip_line2_mark = tooltip_data.mark_text(
        align='left',
        dx=6,
        dy=-15-l,  # adjust vertical offset for the second line
        fontSize=12,
        fontWeight='bold'
        ).encode(
        x='Time point',
        text=alt.condition(nearest, 'tooltip_line2:N', alt.value('')),
        opacity=alt.value(text_opacity)
        ).transform_filter(nearest)

    tooltip_line3_mark = tooltip_data.mark_text(
        align='left',
        dx=6,
        dy=0-l,  # adjust vertical offset for the third line
        fontSize=12,
        fontWeight='bold'
        ).encode(
        x='Time point',
        text=alt.condition(nearest, 'tooltip_line3:N', alt.value('')),
        opacity=alt.value(text_opacity)
        ).transform_filter(nearest)
    
    tooltip_line4_mark = tooltip_data.mark_text(
        align='left',
        dx=6,
        dy=15-l,  # adjust vertical offset for the fourth line
        fontSize=12,
        fontWeight='bold'
        ).encode(
        x='Time point',
        text=alt.condition(nearest, 'tooltip_line4:N', alt.value('')),
        opacity=alt.value(text_opacity)
        ).transform_filter(nearest)
    
    tooltip_line5_mark = tooltip_data.mark_text(
        align='left',
        dx=6,
        dy=30-l,  # adjust vertical offset for the fifth line
        fontSize=12,
        fontWeight='bold'
        ).encode(
        x='Time point',
        text=alt.condition(nearest, 'tooltip_line5:N', alt.value('')),
        opacity=alt.value(text_opacity)
        ).transform_filter(nearest)
    
    tooltip_line6_mark = tooltip_data.mark_text(
        align='left',
        dx=6,
        dy=45-l,  # adjust vertical offset for the sixth line
        fontSize=12,
        fontWeight='bold'
        ).encode(
        x='Time point',
        text=alt.condition(nearest, 'tooltip_line6:N', alt.value('')),
        opacity=alt.value(text_opacity)
        ).transform_filter(nearest)

    # Layer the tooltip text marks with your other layers.
    chart = alt.layer(
        band, mean_line, mean_points, mean_selectors, rules,
        tooltip_line1_mark, tooltip_line2_mark, tooltip_line3_mark, tooltip_line4_mark, tooltip_line5_mark, tooltip_line6_mark
        ).properties(
        width=1100,
        height=350,
        title=f'{Metric} with its standard deviation across time {what}'
        ).interactive()

    return chart


# ==========================================================================================================================================================================================================================================================================================================
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





def Superplot_seaborn(
        df:pd.DataFrame,                                     
        metric:str, 
        Metric:str, 
        palette:str,

        show_violin:bool, 
        violin_fill_color:str, 
        violin_edge_color:str, 
        violin_alpha:float,
        violin_outline_width:float,

        show_mean:bool, 
        mean_span:float,
        mean_color:str,
        show_median:bool,
        median_span:float, 
        median_color:str,
        line_width:float,
        set_main_line:str,


        show_error_bars:bool,
        errorbar_capsize:int,
        errorbar_color:str,
        errorbar_lw:int,
        errorbar_alpha:float,  
        
        show_swarm:bool, 
        swarm_size:int, 
        swarm_outline_color:str,
        swarm_alpha:float,

        show_balls:bool,
        ball_size:int,
        ball_outline_color:str, 
        ball_outline_width:int, 
        ball_alpha:int,
        
        show_kde:bool,
        kde_inset_width:float,
        kde_outline:float,
        kde_alpha:float,
        kde_legend:bool,
        kde_fill:bool,

        p_test:bool, 

        show_legend:bool, 
        show_grid:bool,
        open_spine:bool,
        
        plot_width:int,
        plot_height:int,
        ):
    

    """
    MEGALOMANIC - DIRTY - EDGY   
    seaborn plotting function.

    1. Swarm plot
    2. Violin plot
    3. Scatter plot of replicate means
    4. Mean and median lines as well as errorbars
    5. KDE inset subplots
    6. P-testing using combinations of real conditions (compare every pair)

    """
        

    plt.figure(figsize=(plot_width, plot_height))
    

    df['Condition'] = df['Condition'].astype(str)
    conditions = df['Condition'].unique()


    if df['Replicate'].nunique() == 1:
        hue = 'Condition'
    else:
        hue = 'Replicate'
    

    # ======================= KDE INSET =========================
    # If True, ensures spacing inbetween conditions for KDE plots
    if show_kde:


        # ----------- Create artificial and dirty x-axis positions for the KDE plots ------------

        spaced_conditions = ['spacer_0']

        for i, condition in enumerate(conditions):
            spaced_conditions.append(condition)

            if i < len(conditions) - 1:
                spaced_conditions.append(f"spacer_{i+1}")
 
        df['Condition'] = pd.Categorical(df['Condition'], categories=spaced_conditions, ordered=True)
        
        
        # ----------------------- Swarm plot --------------------------

        if show_swarm:
            sns.swarmplot(
                data=df, 
                x='Condition', 
                y=metric,
                hue=hue, 
                palette=palette, 
                size=swarm_size, 
                dodge=False, 
                alpha=swarm_alpha,
                legend=False, 
                zorder=3, 
                edgecolor=swarm_outline_color, 
                order=spaced_conditions
                )
        

        # ------------------------ Violinplot -------------------------
        if show_violin:
            sns.violinplot(
                data=df, 
                x='Condition', 
                y=metric, 
                color=violin_fill_color, 
                edgecolor=violin_edge_color, 
                width=violin_outline_width, 
                inner=None, 
                alpha=violin_alpha, 
                zorder=2, 
                order=spaced_conditions
                )
        

        # ------------------------ Scatterplot of replicate means ------------------------------

        replicate_means = df.groupby(['Condition', 'Replicate'])[metric].mean().reset_index()
        if show_balls:
            sns.scatterplot(
                data=replicate_means, 
                x='Condition', 
                y=metric, 
                hue=hue, 
                palette=palette, 
                edgecolor=ball_outline_color, 
                s=ball_size, 
                legend=False, 
                alpha=ball_alpha, 
                linewidth=ball_outline_width, 
                zorder=4
                )
        

        # ---------------------------- Mean, Meadian and Error bars --------------------------------
 
        condition_stats = df.groupby('Condition')[metric].agg(['mean', 'median', 'std']).reset_index()

        cond_num_list = list(range(len(conditions)*2)) 
        for cond in cond_num_list:

            x_center = cond_num_list[cond]  # Get the x position for the condition

            if show_mean:
                sns.lineplot(
                    x=[x_center - mean_span, x_center + mean_span],
                    y=[condition_stats['mean'].iloc[cond], condition_stats['mean'].iloc[cond]],
                    color='black', 
                    linestyle='-', 
                    linewidth=line_width,
                    label='Mean' if cond == 0 else "", zorder=5
                    )
                
            if show_median:
                sns.lineplot(
                    x=[x_center - median_span, x_center + median_span],
                    y=[condition_stats['median'].iloc[cond], condition_stats['median'].iloc[cond]],
                    color='black', 
                    linestyle='--', 
                    linewidth=line_width,
                    label='Median' if cond == 0 else "", zorder=5
                    )
                
            if show_error_bars:
                plt.errorbar(
                    x_center, 
                    condition_stats['mean'].iloc[cond], 
                    yerr=condition_stats['std'].iloc[cond], 
                    fmt='None',
                    color='black', 
                    alpha=errorbar_alpha,
                    linewidth=errorbar_lw, 
                    capsize=errorbar_capsize, 
                    zorder=5, 
                    label='Mean ± SD' if cond == 0 else "",
                    )
                

        # -------------------------------- P-tests -------------------------------------

        if p_test:

            real_conditions = [cond for cond in spaced_conditions if not cond.startswith('spacer')]
            pos_mapping = {cat: idx for idx, cat in enumerate(spaced_conditions)}
        
            for i, (cond1, cond2) in enumerate(combinations(real_conditions, 2)):
                data1 = df[df['Condition'] == cond1][metric]
                data2 = df[df['Condition'] == cond2][metric]
                stat, p_value = mannwhitneyu(data1, data2)
                x1, x2 = pos_mapping[cond1], pos_mapping[cond2]
                y_max = df[metric].max()
                y_offset = y_max * 0.1
                y = y_max + y_offset * (i + 1)
                plt.plot([x1, x1, x2, x2],
                        [y+4.5, y + y_offset / 2.5, y + y_offset / 2.5, y+1.5],
                        lw=1, color='black')
                plt.text((x1 + x2) / 2, y + y_offset / 2,
                        f'p = {round(p_value, 3):.3f}', ha='center', va='bottom', 
                        fontsize=10, color='black')
        


        # ------------------------ Dirty B.     ..ars ----------------------------
        # A dirty way to shift the x-axis positions and make room for the KDE plots

        dirty_b = list(range(-1, len(conditions)*2))
        for i in dirty_b:
            x_val = dirty_b[i]
            sns.lineplot(
                x=x_val-0.5,
                y=[condition_stats['median'].iloc[i]],
                color='none', 
                linewidth=0,
                label="", 
                zorder=0
                )


        # ------------------------ KDE inset plots ----------------------------

        ax = plt.gca()
        y_ax_min, y_ax_max = ax.get_ylim()
        
        for cond in cond_num_list[::2]:
            group_df = df[df['Condition'] == conditions[cond // 2]]   # DataFrame group for a given condition

            y_max = group_df[metric].max()
            inset_height = y_ax_max * (y_max/y_ax_max) + abs(y_ax_min)   # height of the inset plot
            inset_y = y_ax_min   # y inset position

            x_val = cond_num_list[cond]  
            offset_x = 0

            inset_ax = ax.inset_axes([x_val - offset_x, inset_y, kde_inset_width, inset_height], transform=ax.transData, zorder=0, clip_on=True)
            
            sns.kdeplot(
                data=group_df,
                y=metric,
                hue=hue,
                fill=kde_fill,
                alpha=kde_alpha,
                lw=kde_outline,
                palette=palette,
                ax=inset_ax,
                legend=kde_legend,
                zorder=0,
                clip=(y_ax_min, y_ax_max)
                )
            
            inset_ax.invert_xaxis()
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_xlabel('')
            inset_ax.set_ylabel('')

            sns.despine(ax=inset_ax, left=True, bottom=False, top=True, right=True)



        # ------------------------ X axis clean-up ----------------------------
        # Another dirty trick - removing the spacer labels from the x-axis

        plt.xticks(
            ticks=range(len(spaced_conditions)),
            labels=[lbl if not lbl.startswith("spacer") else "" for lbl in spaced_conditions]
            )
        plt.yticks(ticks=np.arange(0, y_ax_max, step=25))




    # ======================= IF FALSE KDE INSET =========================
    
    elif show_kde == False:

        # ------------------------------------------ Swarm plot -----------------------------------------------------------

        if show_swarm:
            sns.swarmplot(
                data=df, 
                x='Condition', 
                y=metric, 
                hue=hue, 
                palette=palette, 
                size=swarm_size, 
                edgecolor=swarm_outline_color, 
                dodge=False, 
                alpha=swarm_alpha, 
                legend=False, 
                zorder=2
                )
        

        # ----------------------------------- Scatterplot of replicate means ------------------------------------------------------

        replicate_means = df.groupby(['Condition', 'Replicate'])[metric].mean().reset_index()
        if show_balls:
            sns.scatterplot(
                data=replicate_means,
                x='Condition', 
                y=metric, 
                hue=hue, 
                palette=palette, 
                edgecolor=ball_outline_color, 
                s=ball_size, 
                legend=False, 
                alpha=ball_alpha, 
                linewidth=ball_outline_width, 
                zorder=3
                )


        # -------------------------------------------- Violin plot ---------------------------------------------------------

        if show_violin:
            sns.violinplot(
                data=df, 
                x='Condition', 
                y=metric, 
                color=violin_fill_color, 
                edgecolor=violin_edge_color, 
                width=violin_outline_width, 
                inner=None, 
                alpha=violin_alpha, 
                zorder=1
                )
        

        #  ------------------------------------ Mean, median and errorbar lines -------------------------------------------

        condition_stats = df.groupby('Condition')[metric].agg(['mean', 'median', 'std']).reset_index()
        for i, row in condition_stats.iterrows():
            x_center = i   # x coordinate
            if show_mean:
                sns.lineplot(
                    x=[x_center - mean_span, x_center + mean_span], 
                    y=[row['mean'], row['mean']], 
                    color='black', 
                    linestyle='-', 
                    linewidth=line_width, 
                    label='Mean' if i == 0 else "", 
                    zorder=4
                    )
            
            if show_median:
                sns.lineplot(
                    x=[x_center - median_span, x_center + median_span], 
                    y=[row['median'], row['median']], 
                    color='black', 
                    linestyle='--', 
                    linewidth=line_width, 
                    label='Median' if i == 0 else "", 
                    zorder=4
                    )
            
            if show_error_bars:
                plt.errorbar(
                    x=x_center, 
                    y=row['mean'], 
                    yerr=row['std'], 
                    fmt='None',
                    color='black', 
                    alpha=errorbar_alpha, 
                    linewidth=errorbar_lw, 
                    capsize=errorbar_capsize, 
                    zorder=5, 
                    label='Mean ± SD' if i == 0 else ""
                    )
            
        
        # ---------------------------------------- P-tests ------------------------------------------------------------

        if p_test:
            conditions = df['Condition'].unique()
            pairs = list(combinations(conditions, 2))
            y_max = df[metric].max()
            y_offset = (y_max * 0.1)  # Offset for p-value annotations
            for i, (cond1, cond2) in enumerate(pairs):
                data1 = df[df['Condition'] == cond1][metric]
                data2 = df[df['Condition'] == cond2][metric]
                stat, p_value = mannwhitneyu(data1, data2)
                
                # Annotate the plot with the p-value
                x1, x2 = conditions.tolist().index(cond1), conditions.tolist().index(cond2)
                y = y_max + y_offset * (i + 1)
                plt.plot([x1, x1, x2, x2], [y+4.5, y + y_offset / 2.5, y + y_offset / 2.5, y+1.5], lw=1, color='black')
                plt.text((x1 + x2) / 2, y + y_offset / 2, f'p = {round(p_value, 3):.3f}', ha='center', va='bottom', fontsize=10, color='black')

        

    # ----------------------- Title settings ----------------------------

    if show_kde:
        if show_swarm & show_mean & show_median:
            title = f"Swarm Plot with Mean, Median and KDE for {Metric}"
        elif show_swarm & show_mean & show_median == False:
            title = f"Swarm Plot with Mean and KDE for {Metric}"
        elif show_swarm & show_mean == False & show_median:
            title = f"Swarm Plot with Median and KDE for {Metric}"
        elif show_swarm == False:
            if show_violin & show_mean & show_median:
                title = f"Violin Plot with Mean, Median and KDE for {Metric}"
            elif show_violin & show_mean & show_median == False:
                title = f"Violin Plot with Mean and KDE for {Metric}"
            elif show_violin & show_mean == False & show_median:
                title = f"Violin Plot with Median and KDE for {Metric}"
    else:
        if show_swarm & show_mean & show_median:
            title = f"Swarm Plot with Mean and Median for {Metric}"
        elif show_swarm & show_mean & show_median == False:
            title = f"Swarm Plot with Mean for {Metric}"
        elif show_swarm & show_mean == False & show_median:
            title = f"Swarm Plot with Median for {Metric}"
        elif show_swarm == False:
            if show_violin & show_mean & show_median:
                title = f"Violin Plot with Mean and Median for {Metric}"
            elif show_violin & show_mean & show_median == False:
                title = f"Violin Plot with Mean for {Metric}"
            elif show_violin & show_mean == False & show_median:
                title = f"Violin Plot with Median for {Metric}"
    
    
    plt.title(title)
    plt.xlabel("Condition")
    plt.ylabel(Metric)

    # Add a legend
    replicate_handle = mlines.Line2D([], [], marker='o', color='w', markerfacecolor=sns.color_palette('tab10')[0], markeredgecolor='black', markersize=10, label='Replicates')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.insert(0, replicate_handle)
    labels.insert(0, 'Replicates')
    
    if show_legend:
        plt.legend(handles=handles, labels=labels, title='Legend', title_fontsize='12', fontsize='10', loc='upper right', bbox_to_anchor=(1.15, 1), frameon=True)
    else:
        plt.legend().remove()
    
    sns.despine(top=open_spine, right=open_spine, bottom=False, left=False)
    plt.tick_params(axis='y', which='major', length=7, width=1.5, direction='out', color='black')
    plt.tick_params(axis='x', which='major', length=0)
    plt.grid(show_grid, axis='y', color='lightgrey', linewidth=1.5, alpha=0.2)

    plt.tight_layout()

    return plt.gcf()



def interactive_stripplot(
    df: pd.DataFrame, 
    metric:str, 
    Metrics:dict, 
    let_me_look_at_these:list, 
    palette:str, 
    width:int, 
    height:int, 
    jitter_outline_width:float,
    violin_edge_color:str,
    lowband:float,
    highband:float,
    see_outliars:bool
    ):
    
    """
    Create an interactive strip plot with aggregated violin plots using Plotly.
    
    For each Condition:
      - A grey aggregated violin plot is drawn with custom hover displaying 
        min, median, mean and max values.
      - Scatter (jitter) points are drawn per Replicate. By default these points 
        are colored by replicate; however, if see_outliars is True, non‐outlying 
        points are painted grey and only outliers retain their replicate colors.
      - A solid horizontal line is added at the condition mean, along with a dashed line for the median.
      - An error bar with cap is added at the condition mean representing one standard deviation.
      - A diamond marker is added at each (Condition, Replicate) group representing that group’s mean.
    
    Parameters:
        df (pd.DataFrame): Input data frame.
        metric (str): Column name for the y-axis metric.
        Metric (str): Display name for the y-axis metric.
        let_me_look_at_these (list): Additional columns to include in scatter hover text.
        palette (str): Seaborn color palette for replicates.
        width (int): Plot width.
        height (int): Plot height.
        slope (bool): Whether to adjust jitter based on density and metric values.
        jitter_outline_width (float): Outline width for jittered markers.
        violin_edge_color (str): Color for violin borders.
        see_outliars (bool): When True, paints non-outlying jitter points grey and outliers with their replicate colors.
    
    Returns:
        go.Figure: The resulting Plotly figure.
    """
    # Map each condition to a unique numeric x value
    conditions = df['Condition'].unique()
    condition_map = {cond: i for i, cond in enumerate(conditions)}
    
    np.random.seed(42)

    # Create jittered x values based on the condition mapping
    base_x_jitter = df['Condition'].map(condition_map)
    jitter = np.random.uniform(-0.275, 0.275, size=len(df))
    df['x_jitter'] = base_x_jitter + jitter

    # Compute outlier flag per condition using the standard IQR method
    # A point is an outlier if it falls below Q1 - 1.5*IQR or above Q3 + 1.5*IQR for its condition.
    df['is_outlier'] = df[metric].transform(
        lambda x: (x < (x.quantile(lowband) - 1.5*(x.quantile(highband)-x.quantile(lowband)))) | 
                  (x > (x.quantile(highband) + 1.5*(x.quantile(highband)-x.quantile(lowband))))
        )

    # Group data for scatter points by Condition and Replicate
    grouped = df.groupby(['Condition', 'Replicate'])
    
    # Assign a distinct color to each Replicate using the specified palette
    unique_replicates = df['Replicate'].unique()
    replicate_colors = dict(zip(unique_replicates, sns.color_palette(palette, len(unique_replicates)).as_hex()))
    
    fig = go.Figure()
    
    # Plot scatter (jitter) points for each replicate group
    for (cond, repl), group_df in grouped:
        # Prepare hover info for each point
        hover_text = []
        for _, row in group_df.iterrows():
            hover_info = f"{Metrics[metric]}: {row[metric]}"
            for col in let_me_look_at_these:
                hover_info += f"<br>{Metrics[col]}: {row[col]}"
            hover_text.append(hover_info)
            
        # When see_outliars is enabled, split into outliers and non-outliers
        if see_outliars:
            df_norm = group_df[~group_df['is_outlier']]
            df_out = group_df[group_df['is_outlier']]
            # Plot non-outlier points in grey
            if not df_norm.empty:
                fig.add_trace(go.Scatter(
                    x=df_norm['x_jitter'],
                    y=df_norm[metric],
                    mode='markers',
                    hoverinfo='text',
                    hovertext=hover_text,  # Note: hovertext corresponds to the entire group.
                    marker=dict(
                        color='grey',
                        size=8,
                        opacity=0.55,
                        line=dict(width=jitter_outline_width, color='black')
                    ),
                    showlegend=False
                ))
            # Plot outlier points in the replicate's assigned color
            if not df_out.empty:
                fig.add_trace(go.Scatter(
                    x=df_out['x_jitter'],
                    y=df_out[metric],
                    mode='markers',
                    hoverinfo='text',
                    hovertext=hover_text,
                    marker=dict(
                        color=replicate_colors[repl],
                        size=8,
                        opacity=0.55,
                        line=dict(width=jitter_outline_width, color='black')
                    ),
                    showlegend=False
                ))
        else:
            # Default behavior: all points colored by replicate
            fig.add_trace(go.Scatter(
                x=group_df['x_jitter'],
                y=group_df[metric],
                mode='markers',
                hoverinfo='text',
                hovertext=hover_text,
                marker=dict(
                    color=replicate_colors[repl],
                    size=8,
                    opacity=0.55,
                    line=dict(width=jitter_outline_width, color='black')
                ),
                showlegend=False
            ))
    
    # Add diamond markers representing the overall mean for each replicate in each condition.
    for (cond, repl), group_df in grouped:
        mean_val = group_df[metric].mean()
        x_center = condition_map[cond]
        fig.add_trace(go.Scatter(
            x=[x_center],
            y=[mean_val],
            mode='markers',
            marker=dict(
                symbol='diamond',
                size=12,
                color=replicate_colors[repl],
                line=dict(width=1, color='black')
            ),
            showlegend=False,
            hoverinfo='skip',
            zorder=20
        ))
    
    # Plot one aggregated violin for each condition (not separated by replicate)
    grouped_conditions = df.groupby('Condition')
    for cond, group_df in grouped_conditions:
        # Compute aggregate statistics for the condition
        min_val = group_df[metric].min()
        max_val = group_df[metric].max()
        mean_val = group_df[metric].mean()
        median_val = group_df[metric].median()
        std_val = group_df[metric].std()

        x_vals = np.full(len(group_df), condition_map[cond])
        hovertemplate = (
            f"Condition: {cond}<br>"
            "Value: %{y:.2f}<br>"
            f"Min: {min_val:.2f}<br>"
            f"Median: {median_val:.2f}<br>"
            f"Mean: {mean_val:.2f}<br>"
            f"Max: {max_val:.2f}<extra></extra>"
        )

        fig.add_trace(go.Violin(
            x=x_vals,
            y=group_df[metric],
            name=cond,
            opacity=0.8,
            fillcolor='grey',
            line_color=violin_edge_color,
            width=0.8,
            showlegend=False,
            hoverinfo='text',
            hovertemplate=hovertemplate,
            points=False
        ))
        
        # Define x span for horizontal lines (centered on condition_map[cond])
        x_center = condition_map[cond]
        x_offset = 0.35  # Adjust as desired
        
        # Add a solid horizontal line at the condition mean
        fig.add_shape(
            type="line",
            xref="x", yref="y",
            x0=x_center - x_offset,
            x1=x_center + x_offset,
            y0=mean_val,
            y1=mean_val,
            line=dict(color="black", width=2, dash="solid"),
            # layer="above"
        )
        
        # Add a dashed horizontal line at the condition median
        fig.add_shape(
            type="line",
            xref="x", yref="y",
            x0=x_center - x_offset,
            x1=x_center + x_offset,
            y0=median_val,
            y1=median_val,
            line=dict(color="black", width=2, dash="dash"),
            # layer="above"
        )
        
        # Add an invisible scatter marker at the mean with error bars for standard deviation
        fig.add_trace(go.Scatter(
            x=[x_center],
            y=[mean_val],
            mode="markers",
            marker=dict(color="black", size=0),
            showlegend=False,
            error_y=dict(
                type="data",
                array=[std_val],
                visible=True,
                color="black",
                thickness=2,
                width=4
            ),
            zorder=10
        ))
        
    fig.update_layout(
        xaxis=dict(
            tickvals=list(condition_map.values()),
            ticktext=list(condition_map.keys()),
            title='Condition'
        ),
        yaxis_title=Metrics[metric],
        template='plotly_white',
        showlegend=False,
        width=width,
        height=height
    )
    
    fig.add_annotation(
        text="Stripplot with Aggregated Violin Plots and Replicate Means",
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.15,
        showarrow=False,
        font=dict(size=16)
    )
    
    return fig