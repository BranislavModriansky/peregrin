import polars as pl

from warnings import warn



def ensure_polars(df) -> pl.DataFrame:
    """ Convert the :class:`InputMetadata` wrapper or a pandas DataFrame to a Polars DataFrame. """
    if isinstance(df, pl.DataFrame):
        return df
    if hasattr(df, 'df') and isinstance(getattr(df, 'df'), pl.DataFrame):
        return df.df  # loader's Input wrapper
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df)
    except ImportError:
        warn("pandas is not installed, cannot convert pandas DataFrame to polars DataFrame.")
    raise TypeError(f"Expected a polars DataFrame, got {type(df).__name__}.")