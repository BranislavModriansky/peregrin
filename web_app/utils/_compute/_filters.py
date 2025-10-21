import pandas as pd
import numpy as np
from typing import Any
from pandas.api.types import is_object_dtype
from math import floor, ceil

from utils import Metrics



def _has_strings(s: pd.Series) -> bool:
    # pandas "string" dtype (pyarrow/python)
    if isinstance(s.dtype, pd.StringDtype):
        return s.notna().any()
    # categorical of strings?
    if isinstance(s.dtype, pd.CategoricalDtype):
        return isinstance(s.dtype.categories.dtype, pd.StringDtype) and s.notna().any()
    # numeric, datetime, bool, etc.
    if not is_object_dtype(s.dtype):
        return False
    # Fallback for object-dtype (mixed types): minimal Python loop over NumPy array
    arr = s.to_numpy(dtype=object, copy=False)
    return any(isinstance(v, (str, np.str_)) for v in arr)


def _try_float(x: Any) -> Any:
        """
        Try to convert a string to an int or float, otherwise return the original value.
        """
        try:
            if isinstance(x, str):
                num = float(x.strip())
                if num.is_integer():
                    return float(num)
                else:
                    return num
            else:
                return x
        except ValueError:
            return x


class Threshold:
    """
    Utility functions for data filtering and normalization.
    """

    def __init__(self, eps: float = 1e-12):
        self.EPS = eps


    def nearly_equal_pair(self, a, b) -> bool:
        try:
            return (
                abs(float(a[0]) - float(b[0])) <= self.EPS and
                abs(float(a[1]) - float(b[1])) <= self.EPS
            )
        except Exception:
            return False

    def is_whole_number(self, x) -> bool:
        try:
            fx = float(x)
        except Exception:
            return False
        return abs(fx - round(fx)) < self.EPS

    def int_if_whole(self, x):
        if x is None:
            return None
        try:
            fx = float(x)
        except Exception:
            return x
        if self.is_whole_number(fx):
            return int(round(fx))
        return fx

    def format_numeric_pair(self, values):
        """
        Normalize `values` into a (low, high) numeric pair.

        Accepts:
        - scalar numbers (including numpy.float64) -> returns (v, v)
        - 1-length iterables -> returns (v, v)
        - 2+-length iterables -> returns (first, second) but ensures low<=high where possible
        - None or empty -> (None, None)
        """

        if values is None:
            return None, None
        if np.isscalar(values):
            v = values.item() if hasattr(values, "item") else float(values)
            v = self.int_if_whole(v)
            return v, v
        try:
            seq = list(values)
        except Exception:
            try:
                v = float(values)
                v = self.int_if_whole(v)
                return v, v
            except Exception:
                return None, None
        if len(seq) == 0:
            return None, None
        if len(seq) == 1:
            v = seq[0]
            try:
                fv = float(v)
                fv = self.int_if_whole(fv)
                return fv, fv
            except Exception:
                return v, v
        a, b = seq[0], seq[1]
        try:
            fa = float(a)
            fb = float(b)
            if fa <= fb:
                return self.int_if_whole(fa), self.int_if_whole(fb)
            else:
                return self.int_if_whole(fb), self.int_if_whole(fa)
        except Exception:
            return self.int_if_whole(a), self.int_if_whole(b)
    
    def get_steps(self, highest):
        """
        Returns the step size for the slider based on the range.
        """
        if highest < 0.01:
            steps = 0.0001
        elif 0.01 <= highest < 0.1:
            steps = 0.001
        elif 0.1 <= highest < 1:
            steps = 0.01
        elif 1 <= highest < 10:
            steps = 0.1
        elif 10 <= highest < 1000:
            steps = 1
        elif 1000 <= highest < 100000:
            steps = 10
        elif 100000 < highest:
            steps = 100
        else:
            steps = 1
        return steps

    def compute_reference_and_span(self, values_series: pd.Series, reference: str, my_value: float | None):
        """
        Returns (reference_value, max_delta) for the 'Relative to...' mode.
        max_delta is the farthest absolute distance from reference to any data point.
        """
        vals = values_series.dropna()
        if vals.empty:
            return 0.0, 0.0

        if reference == "Mean":
            ref = float(vals.mean())
        elif reference == "Median":
            ref = float(vals.median())
        elif reference == "My own value":
            ref = float(my_value) if isinstance(my_value, (int, float)) else 0.0
        else:
            ref = float(vals.mean())

        max_delta = float(np.max(np.abs(vals - ref)))
        return ref, max_delta

    def get_threshold_value_params(
        self,
        spot_data: pd.DataFrame, 
        track_data: pd.DataFrame, 
        property_name: str, 
        threshold_type: str, 
        quantile: int = None,
        reference: str = None,
        reference_value: float = None
    ):
        
        # instance method; use self and helpers
        self = self  # no-op to emphasize instance usage
        if threshold_type == "Literal":
            if property_name in Metrics.Thresholding.SpotProperties:
                minimal = spot_data[property_name].min()
                maximal = spot_data[property_name].max()
            elif property_name in Metrics.Thresholding.TrackProperties:
                minimal = track_data[property_name].min()
                maximal = track_data[property_name].max()
            else:
                minimal, maximal = 0, 100

            steps = self.get_steps(maximal)
            minimal, maximal = floor(minimal), ceil(maximal)

        elif threshold_type == "Normalized 0-1":
            minimal, maximal = 0, 1
            steps = 0.01

        elif threshold_type == "Quantile":
            minimal, maximal = 0, 100
            steps = 100/float(quantile)

        elif threshold_type == "Relative to...":
            if property_name in Metrics.Thresholding.SpotProperties:
                series = spot_data[property_name]
            elif property_name in Metrics.Thresholding.TrackProperties:
                series = track_data[property_name]
            else:
                series = pd.Series(dtype=float)

            # Compute reference and span
            reference_value, max_delta = self.compute_reference_and_span(series, reference, reference_value)

            minimal = 0
            maximal = ceil(max_delta) if np.isfinite(max_delta) else 0
            steps = self.get_steps(maximal)

        return minimal, maximal, steps, reference_value

    def filter_data(self, df, threshold: tuple, property: str, threshold_type: str, reference: str = None, reference_value: float = None):
        if df is None or df.empty:
            return df
        
        try:
            working_df = df[property].dropna()
        except Exception:
            return df
        
        _floor, _roof = threshold
        if (
            _floor is None or _roof is None
            or not isinstance(_floor, (int, float)) or not isinstance(_roof, (int, float))
        ):
            return working_df

        if threshold_type == "Literal":
            return working_df[(working_df >= _floor) & (working_df <= _roof)]

        elif threshold_type == "Normalized 0-1":
            normalized = self.Normalize_01(df, property)
            return normalized[(normalized >= _floor) & (normalized <= _roof)]

        elif threshold_type == "Quantile":
            
            q_floor, q_roof = _floor / 100, _roof / 100
            if not 0 <= q_floor <= 1 or not 0 <= q_roof <= 1:
                q_floor, q_roof = 0, 1

            lower_bound = np.quantile(working_df, q_floor)
            upper_bound = np.quantile(working_df, q_roof)
            return working_df[(working_df >= lower_bound) & (working_df <= upper_bound)]

        elif threshold_type == "Relative to...":
            # req(reference is not None)
            if reference is None:
                reference = 0.0
            ref, _ = self.compute_reference_and_span(working_df, reference, reference_value)

            # print(f"Reference value: {ref}, Floor: {ref + _floor}, Roof: {ref + _roof}, -Floor: {ref - _floor}, -Roof: {ref - _roof}")

            return working_df[
                (working_df >= (ref + _floor)) 
                & (working_df <= (ref + _roof))
                | (working_df <= (ref - _floor)) 
                & (working_df >= (ref - _roof))    
            ]

        return df
    

    @staticmethod   
    def Normalize_01(df, col) -> pd.Series:
        """
        Normalize a column to the [0, 1] range.
        """
        # s = pd.to_numeric(df[col], errors='coerce')
        try:
            s = pd.Series(_try_float(df[col]), dtype=float)
            if _has_strings(s):
                normalized = pd.Series(0.0, index=s.index, name=col)
            lo, hi = s.min(), s.max()
            if lo == hi:
                normalized = pd.Series(0.0, index=s.index, name=col)
            else:
                normalized = pd.Series((s - lo) / (hi - lo), index=s.index, name=col)

        except Exception:
            normalized = pd.Series(0.0, index=df.index, name=col)

        return normalized  # <-- keeps index

    @staticmethod
    def JoinByIndex(a: pd.Series, b: pd.Series) -> pd.DataFrame:
        """
        Join two Series of potentially different lengths into a DataFrame.
        """

        if b.index.is_unique and not a.index.is_unique:
            df = a.rename(a.name).to_frame().set_index(a.index)
            df[b.name] = b.reindex(df.index)
        else:
            df = b.rename(b.name).to_frame().set_index(b.index)
            df[a.name] = a.reindex(df.index)

        return df


