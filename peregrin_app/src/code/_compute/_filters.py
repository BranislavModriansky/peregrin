import time
import traceback
from dataclasses import dataclass, field
from unittest import case

import pandas as pd
import numpy as np
from typing import Any, Literal
from pandas.api.types import is_object_dtype
from math import floor, ceil

from .._handlers._reports import Level
from .._general import clock, is_empty
from ._stats import MainDataInventory, Frames, TimeIntervals



@dataclass
class Inventory2D:
    ...

    id: np.ndarray = np.array([1])
    idx_x: np.ndarray[np.ndarray] = np.array([])
    idx_y: np.ndarray[np.ndarray] = np.array([])
    property_x: np.ndarray[str] = np.array([])
    property_y: np.ndarray[str] = np.array([])
    series_x: np.ndarray[pd.Series] = np.array([])
    series_y: np.ndarray[pd.Series] = np.array([])
    threshold_x: np.ndarray[str] = np.array(["Literal"])
    threshold_y: np.ndarray[str] = np.array(["Literal"])



@dataclass
class Inventory1D:
    id_idx = np.array([0])
    property = np.array([None])

    filter = [None]
    selection = [None]
    mask = [None]
    series = [None]
    ambit = [None]

    spot_data = pd.DataFrame()
    track_data = pd.DataFrame()


class Filter1D:

    noticequeue: Any = None

    def __init__(self, eps: float = 1e-12):
        self.EPS = eps


    def Apply(self) -> pd.DataFrame:
        """
        Returns:
            tuple: (spotstats, trackstats, framestats, tintervalstats)
        """

        spotstats = MainDataInventory.Spots
        trackstats = MainDataInventory.Tracks
        framestats = MainDataInventory.Frames
        tintervalstats = MainDataInventory.TimeIntervals

        # Return empty dataframes if any input is empty
        if any(is_empty(df) for df in [spotstats, trackstats, framestats, tintervalstats]):
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Get the mask and ensure it's valid
        mask = Inventory1D.mask[-1]
        if mask is None or len(mask) == 0:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Filter using index intersection to avoid KeyError
        valid_spot_indices = spotstats.index.intersection(mask)
        valid_track_indices = trackstats.index.intersection(mask)
        
        spotstats = spotstats.loc[valid_spot_indices]
        trackstats = trackstats.loc[valid_track_indices]
        
        # Regenerate frame and time interval stats from filtered spots
        framestats = Frames(spotstats)
        tintervalstats = TimeIntervals(spotstats)()

        return spotstats, trackstats, framestats, tintervalstats

    def Initialize(self):
        """
        Initializes the filter inventory.
        """

        data = Inventory1D.track_data

        if is_empty(data):
            return

        Inventory1D.selection = [None]
        Inventory1D.ambit = [None]
        Inventory1D.mask = [None]
        Inventory1D.series = [None]

        self._stream(0, data)
        Inventory1D.selection[0] = Inventory1D.ambit[0][:2]


    def Downstream(self, start: int):
        """
        Passes data downstream.
        """
        for idx in Inventory1D.id_idx[:-1]:
            if idx < start:
                continue

            def missing_inventory():
                return (
                    len(Inventory1D.property) <= idx
                    or len(Inventory1D.filter) <= idx
                    or len(Inventory1D.selection) <= idx
                )

            if missing_inventory():
                return

            data = self._choose_data(idx)

            if is_empty(data):
                return
            
            self._stream(idx, data)
    

    def PopLast(self):
        """
        Removes the last threshold from the inventory.
        """

        Inventory1D.id_idx = Inventory1D.id_idx[:-1]
        Inventory1D.property = Inventory1D.property[:-1]
        Inventory1D.filter = Inventory1D.filter[:-1]
        Inventory1D.selection = Inventory1D.selection[:-1]
        Inventory1D.mask = Inventory1D.mask[:-1]
        Inventory1D.series = Inventory1D.series[:-1]
        Inventory1D.ambit = Inventory1D.ambit[:-1]

        if len(Inventory1D.id_idx) == 1:
            self._safe_end(0)
        else:
            self._safe_end(Inventory1D.id_idx[-2])
        


    def _stream(self, idx: int, data: pd.DataFrame):

        if idx == 0:
            Inventory1D.mask[idx] = data.index.to_numpy()
            Inventory1D.series[idx] = self._get_series(idx, data)

        else:
            Inventory1D.mask[idx] = self._get_mask(idx)
            Inventory1D.series[idx] = self._get_series(idx, data)
            
        Inventory1D.ambit[idx] = self._ambit(idx)

        if idx == Inventory1D.id_idx[-2]:
            self._safe_end(idx)

   

    def _safe_end(self, idx: int) -> None:
        """
        Sets the inventory at idx to None/empty.
        """
        new = idx + 1

        try:
            Inventory1D.property[new] = Inventory1D.property[idx]
            Inventory1D.filter[new] = Inventory1D.filter[idx]
            Inventory1D.selection[new] = Inventory1D.ambit[idx][:2]
            Inventory1D.mask[new] = self._get_mask(new)
            Inventory1D.series[new] = Inventory1D.series[idx]
            Inventory1D.ambit[new] = Inventory1D.ambit[idx]

        except Exception:
            Inventory1D.property.append(Inventory1D.property[idx])
            Inventory1D.filter.append(Inventory1D.filter[idx])
            Inventory1D.selection.append(Inventory1D.ambit[idx][:2])
            Inventory1D.mask.append(self._get_mask(new))
            Inventory1D.series.append(Inventory1D.series[idx])
            Inventory1D.ambit.append(Inventory1D.ambit[idx])


    def _choose_data(self, idx: int) -> pd.DataFrame:
        """
        Chooses between spot and track data for the given threshold index.
        """

        if Inventory1D.property[idx] in Inventory1D.track_data.columns.tolist():
            return Inventory1D.track_data
        elif Inventory1D.property[idx] in Inventory1D.spot_data.columns.tolist():
            return Inventory1D.spot_data
        else:
            return pd.DataFrame()


    def _get_mask(self, idx: int) -> np.ndarray:
        """
        Creates a mask from the previous threshold.
        """

        prev_mask = Inventory1D.mask[idx - 1]
        prev_series = Inventory1D.series[idx - 1]
        selected = Inventory1D.selection[idx - 1]

        if not isinstance(selected, (list, tuple)) or len(selected) != 2:
            return prev_mask

        if not isinstance(prev_series, pd.Series) or prev_series.empty:
            return prev_mask

        # Get valid indices that exist in the series
        valid_mask = prev_series.index.intersection(prev_mask)

        if valid_mask.empty:
            return np.array([], dtype=prev_mask.dtype)

        new_mask = prev_series.loc[valid_mask][
            (prev_series.loc[valid_mask] >= selected[0]) 
            & (prev_series.loc[valid_mask] <= selected[1])
        ].index.to_numpy()

        return new_mask


    def _get_series(self, idx: int, data: pd.DataFrame) -> pd.Series:
        """
        Extracts a pandas series for the given threshold based on its mask.
        """
    
        mask = Inventory1D.mask[idx]
        if mask is None:
            return pd.Series(dtype=float)

        # Use intersection to avoid KeyError if some labels are not present in this df
        idxs = data.index.intersection(mask)
        if idxs.empty or Inventory1D.property[idx] not in data.columns:
            return pd.Series(dtype=float)

        series = data.loc[idxs, Inventory1D.property[idx]].dropna()
        min, max = series.min(), series.max()

        match Inventory1D.filter[idx][0]:

            case "Normalized 0-1":

                if min == max:
                    series = pd.Series(0.0, index=series.index)
                else:
                    series = series.apply(lambda v: (v - min) / (max - min))

        return series


    def _ambit(self, idx: int) -> tuple[int | float, int | float, int | float]:
        """
        Returns min, max and step size.
        """
        
        if not isinstance(Inventory1D.property[idx], str):
            return (0, 100, 1)
        
        if not isinstance(Inventory1D.filter[idx], (list, tuple)):
            return (0, 100, 1)

        match Inventory1D.filter[idx][0]:

            case "Literal":
                min, max = self._get_range(idx)
                step = self._get_steps(max)
            
            case "Normalized 0-1":
                min, max, step = 0, 1, 0.01

            case "Percentile":
                min, max = 0, 100
                step = 100/float(Inventory1D.filter[idx][1])

            case "Relative to...":
                reference = Inventory1D.filter[idx][1]
                min, max = self._reference_delta(idx, reference)
                step = self._get_steps(max)

            case _:
                min, max = 0, 100
                step = 1

        return (min, max, step)
    

    def _get_range(self, idx: int) -> tuple[int, int]:
        """
        Gets the min and max values for a given data property.
        """
        
        if Inventory1D.series[idx] is not None and not Inventory1D.series[idx].empty:
            min, max = Inventory1D.series[idx].min(), Inventory1D.series[idx].max()
            if max > 1:
                min, max = self._clamp(min, max)
            else:
                min, max =round(min, 2), round(max, 2)
                
        else:
            min, max = 0, 100

        return (min, max)
    

    def _clamp(self, min: int | float, max: int | float) -> tuple[int, int]:
        """
        Clamps given min and max values to whole numbers.
        """

        if min is None or not isinstance(min, (int, float)):
            min = 0
        if max is None or not isinstance(max, (int, float)):
            max = 100
        if min > max:
            min, max = max, min

        min = floor(min)
        max = ceil(max)

        return (min, max)
    

    def _get_steps(self, max: int | float) -> int | float:
        """
        Finds the adequate step size based on the highest value of the range.
        """

        if max < 0.01:
            step = 0.0001
        elif 0.01 <= max < 0.1:
            step = 0.001
        elif 0.1 <= max < 1:
            step = 0.01
        elif 1 <= max < 10:
            step = 0.1
        elif 10 <= max < 1000:
            step = 1
        elif 1000 <= max < 100000:
            step = 10
        elif 100000 < max:
            step = 100
        else:
            step = 1
        return step


    def _reference_delta(self, idx: int, reference: str | int | float):
        """
        Returns (min_delta, max_delta) for the 'Relative to...' mode.
        max_delta is the farthest absolute distance from reference to any data point.
        """

        series = Inventory1D.series[idx]
        if series.empty:
            return 0.0, 1.0
        
        try:
            match reference:
                case "Mean":
                    ref = np.mean(series)
                case "Median":
                    ref = np.median(series)
                case _:
                    ref = reference if isinstance(reference, (int, float)) else np.mean(series)

            max_delta = ceil(np.max(np.abs(series - ref)))
            ref = round(ref, 2)
            
            Inventory1D.filter[idx] = [Inventory1D.filter[idx][0], ref]  # store computed reference value

            return 0.0, max_delta
        
        except Exception as e:
            self.noticequeue.Report(Level.error, f"Error computing reference and span: {e}", traceback.format_exc())

            return 0.0, 1.0



    @staticmethod
    def intersects_symmetric(i, bins, bottom, top, reference) -> bool:
        # interval A: [-b, -a], interval B: [a, b]
        left_hit  = (bins[i+1] >= reference - top) and (bins[i] <= reference - bottom) # far left to near left
        right_hit = (bins[i+1] >= reference + bottom) and (bins[i] <= reference + top) # near right to far right
        return left_hit or right_hit

    @staticmethod
    def bin_bounds(i, bins, bottom, top):
        return not(bins[i] < bottom or bins[i+1] > top)



class Filter2D:
    ...

