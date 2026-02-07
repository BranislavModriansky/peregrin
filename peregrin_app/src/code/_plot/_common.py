import math
from os import path
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Any, List, Tuple

from .._handlers._reports import Level


class Colors:

    @staticmethod
    def GenerateRandomColor() -> str:
        """
        Generate a random color in hexadecimal format.
        """

        r = np.random.randint(0, 255)   # Red intensity
        g = np.random.randint(0, 255)   # Green intensity
        b = np.random.randint(0, 255)   # Blue intensity

        return '#{:02x}{:02x}{:02x}'.format(r, g, b)

    @staticmethod
    def GenerateRandomGrey() -> str:
        """
        Generate a random grey color in hexadecimal format.
        """

        n = np.random.randint(0, 240)  # All intensities

        return '#{:02x}{:02x}{:02x}'.format(n, n, n)

    @staticmethod
    def StockQualPalette(elements: list, cmap: str, **kwargs) -> list:
        """
        Generates a qualitative colormap for a given list of elements.
        """
        noticequeue = kwargs.get('noticequeue', None) if 'noticequeue' in kwargs else None

        try:
            n = len(elements)
            if n == 0:
                noticequeue.Report(Level.error, "No elements provided for colormap generation.")
                return []
            
            else:
                cmap = plt.get_cmap(cmap)
            colors = [mcolors.to_hex(cmap(i / n)) for i in range(n)]

            return colors
        
        except Exception as e:
            noticequeue.Report(Level.error, f"Error generating qualitative colormap: {str(e)}")
            return None
    
    @staticmethod
    def GetCmap(c_mode: str, **kwargs) -> mcolors.Colormap:
        """
        Get a colormap according to the selected color mode.
        """
        noticequeue = kwargs.get('noticequeue', None) if 'noticequeue' in kwargs else None

        match c_mode:
            case 'greyscale LUT':
                return plt.cm.gist_gray
            case 'reverse grayscale LUT':
                return plt.cm.gist_yarg
            case 'jet LUT':
                return plt.cm.jet
            case 'brg LUT':
                return plt.cm.brg
            case 'cool LUT':
                return plt.cm.cool
            case 'hot LUT':
                return plt.cm.hot
            case 'inferno LUT':
                return plt.cm.inferno
            case 'plasma LUT':
                return plt.cm.plasma
            case 'magma LUT':
                return plt.cm.magma
            case 'RdYlGn_r LUT':
                return plt.cm.RdYlGn_r
            case 'CMR-map LUT':
                return plt.cm.CMRmap
            case 'gist-stern LUT':
                return plt.cm.gist_stern
            case 'gist-heat LUT':
                return plt.cm.gist_heat
            case 'gnuplot LUT':
                return plt.cm.gnuplot
            case 'gnuplot2 LUT':
                return plt.cm.gnuplot2
            case 'viridis LUT':
                return plt.cm.viridis
            case 'cividis LUT':
                return plt.cm.cividis
            case 'copper LUT':
                return plt.cm.copper
            case 'rainbow LUT':
                return plt.cm.rainbow
            case 'turbo LUT':
                return plt.cm.turbo
            case 'HSV LUT':
                return plt.cm.hsv
            case 'nipy-spectral LUT':
                return plt.cm.nipy_spectral
            case 'gist-ncar LUT':
                return plt.cm.gist_ncar
            case 'cubehelix LUT':
                return plt.cm.cubehelix
            case 'winter LUT':
                return plt.cm.winter
            case 'spring LUT':
                return plt.cm.spring
            case 'summer LUT':
                return plt.cm.summer
            case 'autumn LUT':
                return plt.cm.autumn
            case 'ocean LUT':
                return plt.cm.ocean
            case 'pink LUT':
                return plt.cm.pink
            case 'twilight LUT':
                return plt.cm.twilight
            case 'twilight-shifted LUT':
                return plt.cm.twilight_shifted
            case 'seismic LUT':
                return plt.cm.seismic
            case 'BrBG LUT':
                return plt.cm.BrBG
            case 'PRGn LUT':
                return plt.cm.PRGn
            case 'PiYG LUT':
                return plt.cm.PiYG

            case None:
                if noticequeue:
                    noticequeue.Report(Level.warning, "No color mode specified. Using 'jet LUT' instead.")
                return plt.cm.jet

            case _:
                if noticequeue:
                    noticequeue.Report(Level.warning, f"Unsupported color mode: {c_mode}. Using 'jet LUT' instead.")
                return plt.cm.jet
        
    @staticmethod
    def BuildQualPalette(data: pd.DataFrame, tag: str = 'Replicate', *args, which: list = [], **kwargs) -> dict:
        noticequeue = kwargs.get('noticequeue', None) if 'noticequeue' in kwargs else None

        # data.reset_index(drop=False, inplace=True)
        tags = data[tag].unique().tolist() if not which else which

        mp = {}
        if f'{tag} color' in data.columns:
            mp = (data[[tag, f'{tag} color']]
                    .dropna()
                    .drop_duplicates(tag)
            )
            
            mp = mp.set_index(tag)[f'{tag} color'].to_dict()
        
        missing = [t for t in tags if t not in mp]

        if missing:
            noticequeue.Report(Level.warning, f"Missing colors for {tag} values. Generating random colors instead.", f"Assign colors for missing {tag} values: {', '.join(missing)} or use a stock color palette instead.")
            
            for t in missing:
                mp[t] = Colors.GenerateRandomColor()

        return mp
    
    


    
    


class Categorizer:

    def __init__(
        self,
        data: pd.DataFrame,
        conditions: list,
        *,
        replicates: list = [],
        aggby: list = [],
        aggdict: dict = {},
        **kwargs
    ):
        """
        Initialize with a DataFrame containing 'Condition' and 'Replicate' columns.
        """
        self.data = data
        self.conditions = conditions
        self.replicates = replicates
        self.aggby = aggby
        self.aggdict = aggdict
        self.noticequeue = kwargs.get('noticequeue', None)

    def _checkerrors(self) -> bool:
        """
        Check for errors in the provided categories and replicates.
        """
        if self.conditions == []:
            self.noticequeue.Report(Level.error, "Missing conditions.", "At least one condition (category) must be specified.")
            return True
        if self.replicates == []:
            self.noticequeue.Report(Level.error, "Missing replicates.", "At least one replicate (category) must be specified.")
            return True
        
        conds_not_found = [cond for cond in self.conditions if cond not in self.data['Condition'].values]
        reps_not_found = [rep for rep in self.replicates if rep not in self.data['Replicate'].values]
        if conds_not_found:
            self.noticequeue.Report(Level.error, "Conditions not found.", f"Conditions {', '.join(conds_not_found)} were not found in the data.")
            return True
        if reps_not_found:
            self.noticequeue.Report(Level.error, "Replicates not found.", f"Replicates {', '.join(reps_not_found)} were not found in the data.")
            return True
        
        return False

    def _filter(self) -> pd.DataFrame:
        """
        Filter DataFrame categories.
        """
        if self.replicates:
            filtered = self.data[
                (self.data['Condition'].isin(self.conditions)) &
                (self.data['Replicate'].isin(self.replicates))
            ]
        else:
            filtered = self.data[self.data['Condition'].isin(self.conditions)]
        
        return filtered

    def _aggregate(self) -> pd.DataFrame:
        """
        Aggregate the filtered DataFrame.
        """
        return self._filter().groupby(self.aggby).agg(self.aggdict).reset_index()

    def __call__(self) -> pd.DataFrame:
        """
        Making the instance callable.
        """
        if self._checkerrors():
            return pd.DataFrame()
        if self.aggdict and self.aggby:
            return self._aggregate()
        return self._filter()
    


class LutScale:

    def __init__(self, min_val: float, max_val: float, cmap: str, **kwargs):
        """
        Initialize the LUT scale with min and max values and a colormap.
        """
        self.min_val = min_val
        self.max_val = max_val
        self.cmap = cmap
        self.noticequeue = kwargs.get('noticequeue', None)

