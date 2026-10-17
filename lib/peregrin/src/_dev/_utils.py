import time
from functools import wraps


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
