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

        print("_______________________________________________")
        print(f"spotstats index: {spotstats.index}")
        print(f"Inventory1D.mask[-1]: {Inventory1D.mask[-1]}")
        
        # Get the mask and ensure it's valid
        mask = Inventory1D.mask[-1]
        if mask is None or len(mask) == 0:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        print(f"Applying filter with mask of length: {len(mask)}")
        
        # Filter using index intersection to avoid KeyError
        valid_spot_indices = spotstats.index.intersection(mask)
        valid_track_indices = trackstats.index.intersection(mask)
        
        spotstats = spotstats.loc[valid_spot_indices]
        trackstats = trackstats.loc[valid_track_indices]
        
        # Regenerate frame and time interval stats from filtered spots
        framestats = Frames(spotstats)
        tintervalstats = TimeIntervals(spotstats)()

        print(f"Filtered spotstats index: {len(spotstats)}")

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

        print(Inventory1D.spot_data.index)
        print(Inventory1D.track_data.index)
        

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
        idx_ok = data.index.intersection(mask)
        if idx_ok.empty or Inventory1D.property[idx] not in data.columns:
            return pd.Series(dtype=float)

        series = data.loc[idx_ok, Inventory1D.property[idx]].dropna()
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

            case "N-tile":
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









#     def get_threshold_params(
#         self,
#         data: pd.DataFrame, 
#         property_name: str, 
#         threshold_type: str, 
#         quantile: int = None,
#         reference: str = None,
#         reference_value: float = None
#     ):
        
#         # instance method; use self and helpers
#         self = self  # no-op to emphasize instance usage
#         if threshold_type == "Literal":
#             if property_name in data.columns:
#                 min = data[property_name].min()
#                 max = data[property_name].max()
#             else:
#                 min, max = 0, 100
            
#             steps = self._get_steps(max)
#             min, max = floor(min), ceil(max)

#         elif threshold_type == "Normalized 0-1":
#             min, max = 0, 1
#             steps = 0.01

#         elif threshold_type == "N-tile":
#             min, max = 0, 100
#             steps = 100/float(quantile)

#         elif threshold_type == "Relative to...":
#             if property_name in data.columns:
#                 series = data[property_name]
#             else:
#                 series = pd.Series(dtype=float)

#             # Compute reference and span
#             reference_value, max_delta = self._compute_reference_and_span(series, reference, reference_value)

#             min = 0
#             max = ceil(max_delta) if np.isfinite(max_delta) else 0

#             min, max = self._format_numeric_pair((min, max))
#             steps = self._get_steps(max)

#         return min, max, steps, reference_value

#     def get_range(self, data: pd.DataFrame, property_name: str):
#         """
#         Get the global min and max for a given property across data.
#         """
        
#         if property_name not in data.columns:
#             return 0, 100
        
#         series = data[property_name].dropna()
#         if not series.empty:
#             min, max = series.min(), series.max()
#         else:
#             min, max = 0, 100

#         min, max = self._format_numeric_pair((min, max))

#         return min, max




#     def filter_data(self, df, threshold: tuple, property: str, threshold_type: str, reference: str = None, reference_value: float = None):
#         if df is None or df.empty:
#             return df
        
#         try:
#             working_df = df[property].dropna()
#         except Exception:
#             return df
        
#         _floor, _roof = threshold
#         if (
#             _floor is None or _roof is None
#             or not isinstance(_floor, (int, float)) or not isinstance(_roof, (int, float))
#         ):
#             return working_df

#         if threshold_type == "Literal":
#             return working_df[(working_df >= _floor) & (working_df <= _roof)]

#         elif threshold_type == "Normalized 0-1":
#             normalized = self.Normalize_01(df, property)
#             return normalized[(normalized >= _floor) & (normalized <= _roof)]

#         elif threshold_type == "N-tile":
            
#             q_floor, q_roof = _floor / 100, _roof / 100
#             if not 0 <= q_floor <= 1 or not 0 <= q_roof <= 1:
#                 q_floor, q_roof = 0, 1

#             lower_bound = np.quantile(working_df, q_floor)
#             upper_bound = np.quantile(working_df, q_roof)
#             return working_df[(working_df >= lower_bound) & (working_df <= upper_bound)]

#         elif threshold_type == "Relative to...":
#             # req(reference is not None)
#             if reference is None:
#                 reference = 0.0
#             ref, _ = self.compute_reference_and_span(working_df, reference, reference_value)

#             return working_df[
#                 (working_df >= (ref + _floor)) 
#                 & (working_df <= (ref + _roof))
#                 | (working_df <= (ref - _floor)) 
#                 & (working_df >= (ref - _roof))    
#             ]

#         return df
    




#     def nearly_equal_pair(self, a, b) -> bool:
#         try:
#             return (
#                 abs(float(a[0]) - float(b[0])) <= self.EPS and
#                 abs(float(a[1]) - float(b[1])) <= self.EPS
#             )
#         except Exception:
#             return False

#     def _is_whole_number(self, x) -> bool:
#         try:
#             fx = float(x)
#         except Exception:
#             return False
#         return abs(fx - round(fx)) < self.EPS

#     def _int_if_whole(self, x):
#         if x is None:
#             return None
#         try:
#             fx = float(x)
#         except Exception:
#             return x
#         if self._is_whole_number(fx):
#             return int(round(fx))
#         return fx
    

    
#     def _format_numeric_pair(self, values):
#         """
#         Normalize `values` into a (low, high) numeric pair.

#         Accepts:
#         - scalar numbers (including numpy.float64) -> returns (v, v)
#         - 1-length iterables -> returns (v, v)
#         - 2+-length iterables -> returns (first, second) but ensures low<=high where possible
#         - None or empty -> (None, None)
#         """

#         if values is None:
#             return None, None
#         if np.isscalar(values):
#             v = values.item() if hasattr(values, "item") else float(values)
#             v = self._int_if_whole(v)
#             return v, v
#         try:
#             seq = list(values)
#         except Exception:
#             try:
#                 v = float(values)
#                 v = self._int_if_whole(v)
#                 return v, v
#             except Exception:
#                 return None, None
#         if len(seq) == 0:
#             return None, None
#         if len(seq) == 1:
#             v = seq[0]
#             try:
#                 fv = float(v)
#                 fv = self._int_if_whole(fv)
#                 return fv, fv
#             except Exception:
#                 return v, v
#         a, b = seq[0], seq[1]
#         try:
#             fa = float(a)
#             fb = float(b)
#             if fa <= fb:
#                 return self._int_if_whole(fa), self._int_if_whole(fb)
#             else:
#                 return self._int_if_whole(fb), self._int_if_whole(fa)
#         except Exception:
#             return self._int_if_whole(a), self._int_if_whole(b)
        

    
#     def compute_reference_and_span(self, values_series: pd.Series, reference: str, my_value: float | None):
#         """
#         Returns (reference_value, max_delta) for the 'Relative to...' mode.
#         max_delta is the farthest absolute distance from reference to any data point.
#         """
#         vals = values_series.dropna()
#         if vals.empty:
#             return 0.0, 0.0

#         if reference == "Mean":
#             ref = float(vals.mean())
#         elif reference == "Median":
#             ref = float(vals.median())
#         elif reference == "My own value":
#             ref = float(my_value) if isinstance(my_value, (int, float)) else 0.0
#         else:
#             ref = float(vals.mean())

#         max_delta = float(np.max(np.abs(vals - ref)))
#         return ref, max_delta
    
    
    
    
  
#     def Normalize_01(self, df, col) -> pd.Series:
#         """
#         Normalize a column to the [0, 1] range.
#         """
#         # s = pd.to_numeric(df[col], errors='coerce')
#         try:
#             s = pd.Series(self._try_float(df[col]), dtype=float)
#             if self._has_strings(s):
#                 normalized = pd.Series(0.0, index=s.index, name=col)
#             lo, hi = s.min(), s.max()
#             if lo == hi:
#                 normalized = pd.Series(0.0, index=s.index, name=col)
#             else:
#                 normalized = pd.Series((s - lo) / (hi - lo), index=s.index, name=col)

#         except Exception:
#             normalized = pd.Series(0.0, index=df.index, name=col)

#         return normalized  # <-- keeps index

#     @staticmethod
#     def JoinByIndex(a: pd.Series, b: pd.Series) -> pd.DataFrame:
#         """
#         Join two Series of potentially different lengths into a DataFrame.
#         """

#         if b.index.is_unique and not a.index.is_unique:
#             df = a.rename(a.name).to_frame().set_index(a.index)
#             df[b.name] = b.reindex(df.index)
#         else:
#             df = b.rename(b.name).to_frame().set_index(b.index)
#             df[a.name] = a.reindex(df.index)

#         return df


#     def _has_strings(self, s: pd.Series) -> bool:
#         # pandas "string" dtype (pyarrow/python)
#         if isinstance(s.dtype, pd.StringDtype):
#             return s.notna().any()
#         # categorical of strings?
#         if isinstance(s.dtype, pd.CategoricalDtype):
#             return isinstance(s.dtype.categories.dtype, pd.StringDtype) and s.notna().any()
#         # numeric, datetime, bool, etc.
#         if not is_object_dtype(s.dtype):
#             return False
#         # Fallback for object-dtype (mixed types): min Python loop over NumPy array
#         arr = s.to_numpy(dtype=object, copy=False)
#         return any(isinstance(v, (str, np.str_)) for v in arr)


#     def _try_float(self, x: Any) -> Any:
#             """
#             Try to convert a string to an int or float, otherwise return the original value.
#             """
#             try:
#                 if isinstance(x, str):
#                     num = float(x.strip())
#                     if num.is_integer():
#                         return float(num)
#                     else:
#                         return num
#                 else:
#                     return x
#             except ValueError:
#                 return x


class Filter2D:
    ...

