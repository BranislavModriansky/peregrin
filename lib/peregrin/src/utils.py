import polars as pl

from itertools import zip_longest
from warnings import warn


def ensure_polars(df) -> pl.DataFrame:
    """ Convert the :class:`InputMetadata` wrapper or a pandas DataFrame to a Polars DataFrame. """
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    if hasattr(df, 'df') and isinstance(getattr(df, 'df'), pl.DataFrame):
        return df.df  # loader's Input wrapper
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df)
    except ImportError:
        warn("pandas is not installed, cannot convert pandas DataFrame to polars DataFrame.")
    raise TypeError(f"Expected a polars DataFrame, got {type(df).__name__}.")


class CheckData:

    def is_empty(self, data: pl.DataFrame | pl.Series | list | dict | str | None, *, details: bool = False) -> bool:
        """
        Checks if a polars DataFrame/Series (or list, dict, str) is empty.
        """
        if data is None:
            return True

        if isinstance(data, (pl.DataFrame, pl.Series)):
            isempty = data.is_empty()
            if details and not isempty:
                self._get_details(data)
            return isempty

        if isinstance(data, (list, dict, str)):
            return len(data) == 0

        return False

    def _get_details(self, data: pl.DataFrame | pl.Series) -> None:
        """
        Print details of the DataFrame or Series.
        """
        if isinstance(data, pl.DataFrame):
            table = self._get_df_details(data)
        else:
            table = self._get_sr_details(data)

        self._print_table(table)

    @staticmethod
    def _print_table(table: dict) -> None:
        headers = list(table.keys())
        values = list(table.values())

        # Compute column widths
        col_widths = [
            max(len(headers[i]), max((len(v) for v in values[i]), default=0))
            for i in range(len(headers))
        ]

        header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        separator_line = "  ".join("-" * w for w in col_widths)

        print("")
        print(header_line)
        print(separator_line)

        # Values (shorter columns filled with empty strings)
        for row in zip_longest(*values, fillvalue=""):
            print("  ".join(cell.rjust(w) for cell, w in zip(row, col_widths)))

        print("")

    @staticmethod
    def _get_df_details(df: pl.DataFrame) -> dict:
        """
        Get a summary of the DataFrame's properties.
        """
        n_rows, n_cols = df.shape
        total_cells = n_rows * n_cols

        # Single pass for null counts across all columns
        total_nulls = df.null_count().sum_horizontal().item() if n_cols else 0
        missing_pct = (total_nulls / total_cells * 100) if total_cells else 0.0

        # Duplicated rows (is_duplicated marks all occurrences; count extras only)
        row_duplicates = (n_rows - df.n_unique()) if n_cols else 0

        return {
            "MemoryMB": [f"{df.estimated_size('mb'):.2f}"],
            "Rows": [f"{n_rows}"],
            "Columns": [f"{n_cols}"],
            "ColumnLabels": list(df.columns),
            "ColumnTypes": [str(dt) for dt in df.dtypes],
            "MissingValues%": [f"{missing_pct:.2f}"],
            "RowDuplicates": [f"{row_duplicates}"],
            # Polars enforces unique column names
            "ColumnDuplicates": ["0"],
        }

    @staticmethod
    def _get_sr_details(series: pl.Series) -> dict:
        """
        Get a summary of the Series' properties.
        """
        length = series.len()
        missing_pct = (series.null_count() / length * 100) if length else 0.0

        return {
            "MemoryMB": [f"{series.estimated_size('mb'):.2f}"],
            "Label": [series.name or "<unnamed>"],
            "Type": [str(series.dtype)],
            "Length": [f"{length}"],
            "MissingValues%": [f"{missing_pct:.2f}"],
            "Duplicates": [f"{length - series.n_unique()}"],
        }



class Kwargs:

    @staticmethod
    def get_kwarg(key: str, aliases: dict) -> str:
        """
        Get the canonical key for a given keyword argument.

        Parameters
        ----------
        key : str
            The keyword argument to check.
        aliases : dict
            A dictionary where keys are canonical names and values are lists of aliases.

        Returns
        -------
        str
            The canonical key if found, otherwise the original key.
        """
        for canonical, alias_list in aliases.items():
            if key in alias_list:
                return canonical
        return key

    @staticmethod
    def get_aliases(kwargs, aliases):
        """
        Resolves aliased kwargs to their canonical parameter names.

        Parameters
        ----------
        kwargs : dict
            The keyword arguments as passed by the user, e.g. {'colour': 'black', 'line_width': 1}
        aliases : dict
            Maps canonical name -> list of accepted alias names (including the
            canonical name itself), e.g. {'color': ['color', 'colour', 'c'], ...}

        Returns
        -------
        dict
            kwargs with all recognized aliases rewritten to their canonical key.
            Keys not found in `aliases` are passed through unchanged, with a warning.
        """
        # Build a reverse lookup: alias -> canonical name
        alias_to_canonical = {
            alias: canonical
            for canonical, alias_list in aliases.items()
            for alias in alias_list
        }

        resolved = {}

        for key, value in kwargs.items():
            canonical = alias_to_canonical.get(key, key)
            resolved[canonical] = value

        return resolved


get_aliases = Kwargs.get_aliases
is_empty = CheckData().is_empty