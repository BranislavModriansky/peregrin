import pandas as pd
import numpy as np
from typing import Any
from pandas.api.types import is_object_dtype




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

