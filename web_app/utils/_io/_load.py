import re
import pandas as pd
import os.path as op
from typing import List


def _try_reading(path, encodings=("utf-8", "cp1252", "latin1", "iso8859_15")):
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue


class DataLoader:
    
    @staticmethod
    def GetDataFrame(filepath: str) -> pd.DataFrame:
        """
        Loads a DataFrame from a file based on its extension.
        Supported formats: CSV, Excel, Feather, Parquet, HDF5, JSON.
        """
        _, ext = op.splitext(filepath.lower())

        try:
            if ext == '.csv':
                return _try_reading(filepath)
            elif ext in ['.xls', '.xlsx']:
                return pd.read_excel(filepath)
            elif ext == '.feather':
                return pd.read_feather(filepath)
            elif ext == '.parquet':
                return pd.read_parquet(filepath)
            elif ext in ['.h5', '.hdf5']:
                return pd.read_hdf(filepath)
            elif ext == '.json':
                return pd.read_json(filepath)
        except ValueError as e:
            raise e(f"{ext} is not a supported file format.")
        except Exception as e:
            raise e(f"Failed to load file '{filepath}': {e}")
    

    @staticmethod
    def ExtractStripped(df: pd.DataFrame, id_col: str, t_col: str, x_col: str, y_col: str, mirror_y: bool = True) -> pd.DataFrame:
        # Keep only relevant columns and convert to numeric
        df = df[[id_col, t_col, x_col, y_col]].apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

        # Mirror Y if needed
        if mirror_y:
            """
            TrackMate may export y-coordinates mirrored.
            This does not affect statistics but leads to incorrect visualization.
            Here Y data is reflected across its midpoint for accurate directionality.
            """
            y_mid = (df[y_col].min() + df[y_col].max()) / 2
            df[y_col] = 2 * y_mid - df[y_col]

        # Standardize column names
        return df.rename(columns={id_col: 'Track ID', t_col: 'Time point', x_col: 'X coordinate', y_col: 'Y coordinate'})
    

    @staticmethod
    def ExtractFull(
        df: pd.DataFrame,
        id_col: str,
        t_col: str,
        x_col: str,
        y_col: str,
        mirror_y: bool = True
    ) -> pd.DataFrame:
        """
        Prepare tracking data:
        - Converts chosen coordinate columns to numeric.
        - Mirrors Y if requested.
        - Renames the 4 key columns to standard names.
        - Keeps all other columns intact.
        - Normalizes column labels (e.g. 'CONTRAST_CH' → 'Contrast ch').
        """
        df = df.copy()

        # convert only selected columns to numeric
        for c in [id_col, t_col, x_col, y_col]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # drop rows missing key coordinates
        df = df.dropna(subset=[id_col, t_col, x_col, y_col]).reset_index(drop=True)

        # mirror Y if needed
        if mirror_y:
            y_mid = (df[y_col].min() + df[y_col].max()) / 2
            df[y_col] = 2 * y_mid - df[y_col]

        # rename main columns
        rename_map = {
            id_col: "Track ID",
            t_col: "Time point",
            x_col: "X coordinate",
            y_col: "Y coordinate",
        }
        df = df.rename(columns=rename_map)

        # normalize column names for readability
        def clean_name(name: str) -> str:
            name = str(name)
            name = name.replace("_", " ")
            name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
            name = name.strip().capitalize()
            return name

        try:
            # df = df.rename(columns=lambda c: clean_name(c) if c != 'Track ID' else c)
            df.columns = [clean_name(c) if c != 'Track ID' else c for c in df.columns]
        except Exception as e:
            print(e)

        return df



    @staticmethod
    def GetColumns(path: str) -> List[str]:
        """
        Returns a list of column names from the DataFrame.
        """
        df = DataLoader.GetDataFrame(path)  # or pd.read_excel(path), depending on file type
        return df.columns.tolist()
    
    @staticmethod
    def FindMatchingColumn(columns: List[str], lookfor: List[str]) -> str:
        """
        Looks for matches with any of the provided strings.
        - First tries exact matches.
        - Then checks if the column starts with any of given terms.
        - Finally checks if any term is a substring of the column name.
        If no match is found, returns None.
        """

        # Normalize columns for matching
        normalized_columns = [
            (col, str(col).replace('_', ' ').strip().lower() if col is not None else '') for col in columns
        ]
        # Try exact matches first
        for col, norm_col in normalized_columns:
            for look in lookfor:
                if norm_col == look.lower():
                    return col
        # Then try startswith
        for col, norm_col in normalized_columns:
            for look in lookfor:
                if norm_col.startswith(look.lower()):
                    return col
        # Then try substring
        for col, norm_col in normalized_columns:
            for look in lookfor:
                if look.lower() in norm_col:
                    return col
        return None
        


