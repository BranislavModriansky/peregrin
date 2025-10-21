import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class Colors:

    @staticmethod
    def BuildRepPalette(df: pd.DataFrame, palette_fallback: str) -> dict:
        reps = df['Replicate'].unique().tolist()
        mp = {}
        if 'Replicate color' in df.columns:
            mp = (df[['Replicate', 'Replicate color']]
                    .dropna()
                    .drop_duplicates('Replicate')
            )
            mp = mp.set_index('Replicate')['Replicate color'].to_dict()

        missing = [r for r in reps if r not in mp]
        if missing:
            cyc = sns.color_palette(palette_fallback, n_colors=len(missing))
            mp.update({r: cyc[i] for i, r in enumerate(missing)})

        print(f"Replicate colors: {mp}")

        return mp
    

    @staticmethod
    def GenerateRandomColor() -> str:
        """
        Generate a random color in hexadecimal format.
        """

        r = np.random.randint(0, 255)   # Red LED intensity
        g = np.random.randint(0, 255)   # Green LED intensity
        b = np.random.randint(0, 255)   # Blue LED intensity

        return '#{:02x}{:02x}{:02x}'.format(r, g, b)


    @staticmethod
    def GenerateRandomGrey() -> str:
        """
        Generate a random grey color in hexadecimal format.
        """

        n = np.random.randint(0, 240)  # All LED intensities

        return '#{:02x}{:02x}{:02x}'.format(n, n, n)


    @staticmethod
    def MakeCmap(elements: list, cmap: str) -> list:
        """
        Generates a qualitative colormap for a given list of elements.
        """

        n = len(elements)   # Number of elements in the dictionary
        if n == 0:          # Return an empty list if there are no elements
            return []       
        
        cmap = plt.get_cmap(cmap)                                   # Get the colormap
        colors = [mcolors.to_hex(cmap(i / n)) for i in range(n)]    # Generate a color for each element

        return colors
    

    @staticmethod
    def GetCmap(c_mode: str) -> mcolors.Colormap:
        """
        Get a colormap according to the selected color mode.

        """

        if c_mode == 'greyscale LUT':
            return plt.cm.gist_gray
        elif c_mode == 'reverse grayscale LUT':
            return plt.cm.gist_yarg
        elif c_mode == 'jet LUT':
            return plt.cm.jet
        elif c_mode == 'brg LUT':
            return plt.cm.brg
        elif c_mode == 'cool LUT':
            return plt.cm.cool
        elif c_mode == 'hot LUT':
            return plt.cm.hot
        elif c_mode == 'inferno LUT':
            return plt.cm.inferno
        elif c_mode == 'plasma LUT':
            return plt.cm.plasma
        elif c_mode == 'CMR-map LUT':
            return plt.cm.CMRmap
        elif c_mode == 'gist-stern LUT':
            return plt.cm.gist_stern
        elif c_mode == 'gnuplot LUT':
            return plt.cm.gnuplot
        elif c_mode == 'viridis LUT':
            return plt.cm.viridis
        elif c_mode == 'cividis LUT':
            return plt.cm.cividis
        elif c_mode == 'rainbow LUT':
            return plt.cm.rainbow
        elif c_mode == 'turbo LUT':
            return plt.cm.turbo
        elif c_mode == 'nipy-spectral LUT':
            return plt.cm.nipy_spectral
        elif c_mode == 'gist-ncar LUT':
            return plt.cm.gist_ncar
        elif c_mode == 'twilight LUT':
            return plt.cm.twilight
        elif c_mode == 'seismic LUT':
            return plt.cm.seismic
        else:
            return None


    # @staticmethod
    # def AssignMarker(value, markers):
    #     """
    #     Qualitatively map a metric's percentile value to a symbol.
    #     """


    #     lut = []    # Initialize a list to store the ranges and corresponding symbols

    #     for key, val in markers.items():                # Iterate through the markers dictionary
    #         low, high = map(float, key.split('-'))      # Split the key into low and high values
    #         lut.append((low, high, val))                # Append the range and symbol to the list

    #     for low, high, symbol in lut:               # Return the symbol for the range that contains the given value
    #         if low <= value < high:                  # Check if the value falls within the range
    #             return symbol
        
    #     return list(markers.items())[-1][-1]            # Return the last symbol for thr 100th percentile (which is not included in the ranges)

