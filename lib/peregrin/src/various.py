import math
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import *
from itertools import zip_longest
from functools import wraps

from ._pckg_exceptions._pckg_warnings import *
from ._pckg_exceptions._pckg_errors import *








class Values:

    @staticmethod
    def Clamp01(value: float, **kwargs) -> float:
        """
        Clamp a value between 0 and 1.
        """
        noticequeue = kwargs.get('noticequeue', None) if 'noticequeue' in kwargs else None

        if not (0.0 <= value <= 1.0):    

            if value < 0.0:
                clamped = 0
            else:
                clamped = 1

            if noticequeue:
                noticequeue.Report(Level.warning, f"{value} out of 0-1 range. Clamping to {clamped}.")

            return clamped
        
        return value
    
    
    @staticmethod
    def cmap_lut(data: pd.Series, *args, min: float = None, max: float = None, **kwargs) -> Tuple[Any, Any]:

        try:
            if not isinstance(min, (int, float)):
                min = float(data.min())
            if not isinstance(max, (int, float)):
                max = float(data.max())

            if not (np.isfinite(max) or np.isfinite(min)):
                warnings.warn(message=f"Invalid LUT range. Max and min values are not finite. Using default range (0.0, 100.0).", 
                              category=LUTWarning, 
                              stacklevel=2)

                if not np.isfinite(min):
                    min = 0.0
                if not np.isfinite(max):
                    max = 100.0
                    
            if max <= min:
                warnings.warn(message=f"Invalid LUT range. Max value must be greater than min value. Using default range (0.0, 100.0).", 
                              category=LUTWarning, 
                              stacklevel=2)
                
                min = 0.0
                max = 100.0
            
            norm = plt.Normalize(min, max)
            vals = data.to_numpy()

            return norm, vals
        
        except Exception as e:
            raise LUTError(f"Error while computing LUT map: {str(e)}")
        


