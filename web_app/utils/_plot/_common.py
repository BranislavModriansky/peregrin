from os import path
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .._handlers._reports import Level

# Use the singleton instance directly, don't create a new one
# buffer = _message_buffer.MessageBuffer()  # REMOVE THIS LINE

class Colors:
    
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
            return plt.cm.jet
        
    @staticmethod
    def BuildRepPalette(df: pd.DataFrame, tag: str = 'Replicate', **kwargs) -> dict:
        queue = kwargs.get('queue', None) if 'queue' in kwargs else None

        tags = df[tag].unique().tolist()
        mp = {}
        if f'{tag} color' in df.columns:
            mp = (df[[tag, f'{tag} color']]
                    .dropna()
                    .drop_duplicates(tag)
            )
            mp = mp.set_index(tag)[f'{tag} color'].to_dict()
        
        missing = [t for t in tags if t not in mp]

        if missing:
            # Use the singleton instance
            queue.Report(Level.warning, f"Missing colors in {tag} values. Generating random colors instead.")
            
            for t in missing:
                mp[t] = Colors.GenerateRandomColor()

        return mp

class Values:

    @staticmethod
    def Clamp01(value: float) -> float:
        """
        Clamp a value between 0 and 1.
        """

        if not (0.0 <= value <= 1.0):

            

            if value < 0.0:
                return 0.0
            else:
                return 1.0
            
        return value