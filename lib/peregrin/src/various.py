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




def clock(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        finish = time.time()
        print("")
        print(f"Clocked: '{f.__name__}' <- {finish - start:.4f} s")
        print("")

        return result
    
    return wrap



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
    def RoundSigFigs(x, sigfigs: int = 5, **kwargs) -> float:
        """
        Round a number to a given number of significant figures.

        Parameters
        ----------
        x : any
            The value to round.
        sigfigs : int
            Number of significant figures (default = 5).

        Returns
        -------
        int, float, or None
            Rounded value, or None if input is None.
        """

        noticequeue = kwargs.get('noticequeue', None) if 'noticequeue' in kwargs else None

        if x is None:
            return None

        try:
            x = float(x)

        except (TypeError, ValueError) as e:
            if noticequeue: noticequeue.Report(Level.Error, f"Cannot convert {type(x)}: {x} to float.", str(e))
            return None
        
        except Exception as e:
            if noticequeue: noticequeue.Report(Level.Error, f"Error converting {type(x)}: {x} to float.", str(e))
            return None

        if math.isnan(x) or math.isinf(x):
            return x

        if x == 0.0:
            return 0.0

        return round(x, sigfigs - int(math.floor(math.log10(abs(x)))) - 1)
    

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
        


