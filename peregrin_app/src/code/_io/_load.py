import re
import pandas as pd
import numpy as np
import os.path as op
from typing import List

from .._handlers._reports import Level


def _try_reading(path, encodings=("utf-8", "cp1252", "latin1", "iso8859_15"), **kwargs) -> pd.DataFrame:
    noticequeue = kwargs.get('noticequeue', None) if 'noticequeue' in kwargs else None

    try:
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except UnicodeDecodeError:
                continue
    except Exception as e:
        noticequeue.Report(Level.error, f"Failed to read files '{path}'", f"{str(e)}")


class DataLoader:

    def __init__(self):
        ...
    
    # @staticmethod
    def GetDataFrame(filepath: str, **kwargs) -> pd.DataFrame:
        """
        Loads a DataFrame from a file based on its extension.
        Supported formats: CSV, Excel.
        """
        noticequeue = kwargs.get('noticequeue', None) if 'noticequeue' in kwargs else None

        _, ext = op.splitext(filepath.lower())

        if ext not in ['.csv', '.xls', '.xlsx']:
            noticequeue.Report(Level.error, f"File format: {ext} is not supported.", f"Supported formats include: .csv, .xls, .xlsx.")
            return None

        if ext == '.csv':
            return _try_reading(filepath, noticequeue=noticequeue)
        elif ext in ['.xls', '.xlsx']:
            return pd.read_excel(filepath)
        elif ext == '.xml':
            return pd.read_xml(filepath)
        


    # @staticmethod
    def ExtractStripped(df: pd.DataFrame, id_col: str, t_col: str, x_col: str, y_col: str, *args, mirror_y: bool = True, **kwargs) -> pd.DataFrame:
        """
        Prepare tracking data:
        - Extract only the 4 key columns.
        - Converts them to numeric, dropping rows with missing values.
        - Mirrors Y if requested.
        """
        noticequeue = kwargs.get('noticequeue', None) if 'noticequeue' in kwargs else None

        try:
            if not all(col in df.columns for col in [id_col, t_col, x_col, y_col]):
                missing = [col for col in [id_col, t_col, x_col, y_col] if col not in df.columns]
                noticequeue.Report(Level.error, f"Specified columns were not found", f"Failed to find: {', '.join(missing)}")
                return pd.DataFrame()

            df = df[[id_col, t_col, x_col, y_col]].apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

        except Exception as e:
            noticequeue.Report(Level.error, f"Error in processing input DataFrame", f"{str(e)}")
            return pd.DataFrame()

        if mirror_y:
            """
            TrackMate may export y-coordinates mirrored.
            This does not affect statistics but leads to incorrect visualization.
            Here Y data is reflected across its midpoint for accurate directionality.
            """
            y_mid = (df[y_col].min() + df[y_col].max()) / 2
            df[y_col] = 2 * y_mid - df[y_col]

        return df.rename(columns={id_col: 'Track ID', t_col: 'Time point', x_col: 'X coordinate', y_col: 'Y coordinate'})
    


    

    @staticmethod
    def ExtractFull(df: pd.DataFrame, id_col: str, t_col: str, x_col: str, y_col: str, *args, mirror_y: bool = True, **kwargs) -> pd.DataFrame:
        """
        Prepare tracking data:
        - Converts chosen coordinate columns to numeric.
        - Mirrors Y if requested.
        - Renames the 4 key columns to standard names.
        - Keeps all other columns intact.
        - Normalizes column labels (e.g. 'CONTRAST_CH' → 'Contrast ch').
        """

        print("Extracting full data...")
        noticequeue = kwargs.get('noticequeue', None) if 'noticequeue' in kwargs else None

        try: 
            if not all(col in df.columns for col in [id_col, t_col, x_col, y_col]):
                missing = [col for col in [id_col, t_col, x_col, y_col] if col not in df.columns]
                noticequeue.Report(Level.error, f"Specified columns were not found.", f"Missing columns: {missing}")
                return pd.DataFrame()
            
            # convert only selected columns to numeric
            for c in [id_col, t_col, x_col, y_col]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            # drop rows missing key coordinates
            df = df.dropna(subset=[id_col, t_col, x_col, y_col]).reset_index(drop=True)
        except Exception as e:
            noticequeue.Report(Level.error, f"Error in processing input DataFrame", f"{str(e)}")
            return pd.DataFrame()

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

        # normalize other column names
        def clean_name(name: str) -> str:
            name = str(name)
            name = name.replace("_", " ")
            name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
            name = name.strip().capitalize()
            return name

        try:
            df.columns = [clean_name(c) if c != 'Track ID' else c for c in df.columns]
        except Exception as e:
            pass

        def _py_numeric_df(df: pd.DataFrame) -> None:
            for col in df.columns:
                try: 
                    df[col] = pd.to_numeric(df[col], errors='raise')
                except Exception as e: 
                    continue  # quietly move on if conversion fails

        _py_numeric_df(df)
            
        return df
    
    

    



    @staticmethod
    def GetColumns(path: str, **kwargs) -> List[str]:
        """
        Returns a list of column names from the DataFrame.
        """
        noticequeue = kwargs.get('noticequeue', None) if 'noticequeue' in kwargs else None

        df = DataLoader.GetDataFrame(path, noticequeue=noticequeue)  # or pd.read_excel(path), depending on file type
        return df.columns.tolist()
    

    @staticmethod
    def FindMatchingColumn(columns: List[str], lookfor: List[str], **kwargs) -> str:
        """
        Looks for matches with any of the provided strings.
        - First tries exact matches.
        - Then checks if the column starts with any of given terms.
        - Finally checks if any term is a substring of the column name.
        If no match is found, returns None.
        """
        noticequeue = kwargs.get('noticequeue', None) if 'noticequeue' in kwargs else None

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



