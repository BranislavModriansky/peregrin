from __future__ import annotations

import math
from os import path
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Any, List, Tuple

from .._handlers._reports import Level
from .._infra._selections import Dyes


class Painter:

    def __init__(self, **kwargs):
        self.noticequeue = kwargs.get('noticequeue', None) if 'noticequeue' in kwargs else None
        

    def GenerateRandomColor(self) -> str:
        """
        Generate a random color in hexadecimal format.
        """

        r = np.random.randint(0, 255)   # Red intensity
        g = np.random.randint(0, 255)   # Green intensity
        b = np.random.randint(0, 255)   # Blue intensity

        return '#{:02x}{:02x}{:02x}'.format(r, g, b)


    def GenerateRandomGrey(self) -> str:
        """
        Generate a random grey color in hexadecimal format.
        """

        n = np.random.randint(0, 240)  # All intensities

        return '#{:02x}{:02x}{:02x}'.format(n, n, n)


    def StockQualPalette(self, elements: list, cmap: str) -> list:
        """
        Generates a qualitative colormap for a given list of elements.
        """

        try:
            n = len(elements)
            if n == 0:
                self.noticequeue.Report(Level.error, "No elements provided for colormap generation.")
                return []
            
            else:
                cmap = plt.get_cmap(cmap)
            colors = [mcolors.to_hex(cmap(i / n)) for i in range(n)]

            return colors
        
        except Exception as e:
            self.noticequeue.Report(Level.error, f"Error generating qualitative colormap: {str(e)}")
            return None
    
    def GetCmap(self, c_mode: str) -> mcolors.Colormap:
        """
        Get a colormap according to the selected color mode.
        """

        try:
            if c_mode.endswith('LUT'):
                c_mode = c_mode[:-4]
                print(f"Retrieving colormap for '{c_mode}'")
                return plt.cm.get_cmap(c_mode)
            else:
                print(f"Color mode '{c_mode}' does not end with ' LUT'. Defaulting to 'jet LUT'.")
                return plt.cm.jet

        except Exception as e:
            self.noticequeue.Report(Level.error, f"Error retrieving colormap '{c_mode}': {str(e)}")
            return plt.cm.jet
        

    def BuildQualPalette(self, data: pd.DataFrame, tag: str = 'Replicate', *, which: list = [], **kwargs) -> dict:

        tags = data[tag].unique().tolist() if which == [] else which

        mp = {}
        if f'{tag} color' in data.columns:
            mp = (data[[tag, f'{tag} color']]
                  .dropna()
                  .drop_duplicates(tag)
            )
            
            mp = mp.set_index(tag)[f'{tag} color'].to_dict()
        
        missing = [t for t in tags if t not in mp]

        if missing:
            self.noticequeue.Report(Level.warning, f"Missing colors for {tag} values. Generating random colors instead.", f"Assign colors for missing {tag} values: {', '.join(missing)} or use a stock color palette instead.")
            
            for t in missing:
                mp[t] = self.GenerateRandomColor()

        return mp
    

    def ShowcaseGradients(self, *, cmaps: list[str] = Dyes.QuantitativeCModes, **kwargs) -> plt.Figure:
        """
        ### *Showcase qualitative colormaps.*
        
        Parameters
        ----------
        cmaps : list[str], optional
            List of colormaps to showcase.

        Returns
        -------
        plt.Figure
            A figure showcasing the gradients of qualitative colormaps.
        """

        text_color = kwargs.get('text_color', 'black')
        strip_background = kwargs.get('strip_background', False)

        n = len(cmaps)

        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        # Calculate figure height based on the number of colormaps to display
        height = 0.35 + 0.15 + (n + (n - 1) * 0.1) * 0.22
        fig, axs = plt.subplots(nrows=n + 1, figsize=(6.4, height))

        # Adjust subplot parameters to create space for labels
        fig.subplots_adjust(
            top=1 - 0.35 / height, 
            bottom=0.15 / height,
            left=0.2, right=0.99
        )
        
        # Display the gradient for each colormap with its name as a label
        for ax, name in zip(axs, cmaps):
            ax.imshow(gradient, aspect='auto', cmap=self.GetCmap(name))
            ax.text(
                -0.02, 0.5, name[:-4] if name.endswith(' LUT') else name, 
                va='center', ha='right', 
                fontsize=10, color='black',
                fontfamily='monospace',
                transform=ax.transAxes,
            )

        # Turn off all axes and spines for a clean look
        for ax in axs:
            ax.set_axis_off()
            
        if strip_background:
            fig.set_facecolor('none')
                
        return plt.gcf()
    


    

    
    


    
    


class Categorizer:
    """
    #### *Categorize and aggregate data.*

    Attributes
    ----------
    data : pd.DataFrame
        *The input DataFrame to be categorized and aggregated*

    conditions : list, optional
        *A list of conditions to be included in the categorized DataFrame.*

    replicates : list, optional
        *A list of replicates to be included in the categorized DataFrame.*

    aggby : list, optional
        *A list of columns to group by for aggregation. Default is an empty list.*
    
    aggdict : dict, optional
        *A dictionary specifying the aggregation functions to apply to each column. Default is an empty dictionary.*

    noticequeue : NoticeQueue, optional
        *An optional NoticeQueue for reporting errors and warnings.*
    """
    def __init__(
        self,
        data: pd.DataFrame,
        conditions: list = [],
        replicates: list = [],
        *,
        aggby: list = [],
        aggdict: dict = {},
        **kwargs
    ):
        """
        Initialize the categorizer with data, conditions, replicates, and optional aggregation parameters.
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
            self.noticequeue.Report(Level.warning, "Unspecified conditions, returning input.", "Due to no conditions being specified, the original input DataFrame is returned.")
            self.conditions = self.data['Condition'].unique().tolist()
        if self.replicates == []:
            self.noticequeue.Report(Level.warning, "Unspecified replicates, returning input.", "Due to no replicates being specified, the original input DataFrame is returned.")
            self.replicates = self.data['Replicate'].unique().tolist()
        
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

