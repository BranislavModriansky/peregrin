import time
import pandas as pd
from itertools import zip_longest


class CheckData:

    def __init__(self):
        pass

    def is_empty(self, data: pd.DataFrame | pd.Series, *, details: bool = False) -> bool:
        """
        Checks if a pd.DataFrame or pd.Series is empty.
        """

        isempty = False
        
        if data is None or data.empty:
            isempty = True

        if details:
            self._get_details(isempty, data)
        
        return isempty


    def _get_details(self, isempty: bool, data: pd.DataFrame | pd.Series) -> None:
        """
        Print details of the DataFrame or Series.
        """

        if isempty:
            print("\nData structure is empty.\n")
            return

        if isinstance(data, pd.DataFrame):
            table = self._get_df_details(data)
        else:
            table = self._get_sr_details(data)

        self._print_table(table)


    def _print_table(self, table: dict) -> None:

        headers = list(table.keys())
        values = list(table.values())

        # Compute column widths
        col_widths = [
            max(len(headers[i]), max(len(v) for v in values[i]))
            for i in range(len(headers))
        ]

        # Headers
        header_line = "  ".join(
            headers[i].ljust(col_widths[i])
            for i in range(len(headers))
        )

        # Separators
        separator_line = "  ".join(
            "-" * col_widths[i]
            for i in range(len(col_widths))
        )

        print("")
        print(header_line)
        print(separator_line)

        # Values (shorter columns filled with empty strings)
        for row in zip_longest(*values, fillvalue=""):
            print(
                "  ".join(
                    row[i].rjust(col_widths[i])
                    for i in range(len(row))
                )
            )

        print("")

    def _get_df_details(self, df: pd.DataFrame) -> dict:
        """
        Get a summary of the DataFrame's properties.
        """

        df_shape = df.shape
        
        try:
            index_label = df.index.names if df.index.names is not None else "<unnamed>"
            index_type = df.index.dtypes
        except Exception:
            index_label = df.index.name if df.index.name is not None else "<unnamed>"
            index_type = df.index.dtype
            

        return {
            "MemoryMB": [f"{round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2)}"],
            "Rows": [f"{df_shape[0]}"],
            "Columns": [f"{df_shape[1]}"],
            "ColumnLabels": list(df.columns),
            "IndexLabel": [f"{index_label}"],
            "IndexType": [f"{index_type}"],
            "MissingValues%": [f"{(df.isna().sum().sum() / (df_shape[0] * df_shape[1]) * 100):.2f}"],
            "RowDuplicates": [f"{df.duplicated().sum()}"],
            "ColumnDuplicates": [f"{df.columns.duplicated().sum()}"],
        }
    
    def _get_sr_details(self, series: pd.Series) -> dict:
        """
        Get a summary of the Series' properties.
        """

        return {
            "MemoryMB": [f"{round(series.memory_usage(deep=True) / (1024 ** 2), 2)}"],
            "Label": [series.name],
            "Length": [f"{len(series)}"],
            "MissingValues%": [f"{(series.isna().sum() / len(series) * 100):.2f}"],
            "Duplicates": [f"{series.duplicated().sum()}"],
        }
    
is_empty = CheckData().is_empty


def clock(f):
    def wrap(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        finish = time.time()
        print("")
        print(f"Clocked: '{f.__name__}' <- {finish - start:.4f} s")
        print("")

        return result
    
    return wrap

