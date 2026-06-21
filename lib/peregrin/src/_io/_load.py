import re
import traceback
import pandas as pd
import numpy as np
import os.path as op
from typing import List

from .._handlers._reports import Level, Reporter
from .._compute._stats import Stats
from .._handlers._log import get_logger

_log = get_logger(__name__)



class DataLoader:

    def __init__(self, **kwargs):
        self.noticequeue = kwargs.get('noticequeue', None)
    

    def construct_data(self, files: List[List[str | dict]], strip_data: bool = True, cols: dict = {'id': None, 't': None, 'x': None, 'y': None},
                       *args, cond_labels: List[str | int] | None = None, auto_label: bool = False, **kwargs) -> pd.DataFrame:
        data_cache = []

        for group_idx, group_files in enumerate(files, start=1):
            cond_label = cond_labels[group_idx-1] if cond_labels is not None else None

            data_cache = self.load_data(
                group_files,
                strip_data=strip_data,
                cols=cols,
                cond_label=cond_label,
                auto_label=auto_label,
                cache=data_cache,
                iteration=group_idx,
                mirror_y=kwargs.get("mirror_y", True),
                mirror_x=kwargs.get("mirror_x", False),
            )

        if data_cache is not None:
            return pd.concat(data_cache, axis=0)
        else:
            raise ValueError("No valid data could be loaded from the provided files.")


    def load_data(self, files: List[str | dict], strip_data: bool = True, cols: dict = {'id': None, 't': None, 'x': None, 'y': None}, 
                  *args, cond_label: str | int | None = None, rep_labels: List[str | int] | None = None, auto_label: bool = False,
                  **kwargs) -> pd.DataFrame | List[pd.DataFrame]:

        cache = "cache" in kwargs
        data_cache = kwargs.get("cache", [])
        self.t_unit = kwargs.get('time_units', 's')
        self.time_conversion = kwargs.get('time_conversion', None)

        _rep_guard_list = []
        self.rep_multiplicates = False

        for file_idx, fileinfo in enumerate(files, start=1):
            try:
                try:
                    if isinstance(fileinfo, dict) and "datapath" in fileinfo:
                        df = self.GetDataFrame(fileinfo["datapath"])
                    else:
                        df = self.GetDataFrame(fileinfo)

                    if strip_data:
                        extracted = self.ExtractStripped(
                            df,
                            cols=cols,
                            mirror_y=kwargs.get("mirror_y", True),
                            mirror_x=kwargs.get("mirror_x", False),
                        )
                    else:
                        extracted = self.ExtractFull(
                            df,
                            cols=cols,
                            mirror_y=kwargs.get("mirror_y", True),
                            mirror_x=kwargs.get("mirror_x", False),
                        )

                except: continue

                if (auto_label 
                    and isinstance(fileinfo, dict) 
                    and fileinfo.get("name") 
                    and len(fileinfo.get("name").split("#")) >= 2):

                    file_info = fileinfo.get("name")

                    cond_label = file_info.split("#")[1]
                    rep_label = file_info.split("#")[2]

                    if file_idx > 1:
                        extracted = self._guard_replicates(rep_label, file_info, extracted, _rep_guard_list)

                elif (auto_label 
                    and isinstance(fileinfo, str)
                    and len(op.basename(fileinfo).split("#")) >= 2):

                    cond_label = op.basename(fileinfo).split("#")[1]
                    rep_label = op.basename(fileinfo).split("#")[2]

                    if file_idx > 1:
                        extracted = self._guard_replicates(rep_label, file_info, extracted, _rep_guard_list)

                else:
                    try:
                        cond_label = cond_label if cond_label not in (None, "") else kwargs.get("iteration", 1)
                        rep_label = rep_labels[file_idx-1] if rep_labels is not None else fileinfo.get("name") + f" (file {file_idx})" if isinstance(fileinfo, dict) else op.basename(fileinfo) + f" (file {file_idx})"
                    except Exception:
                        Reporter(Level.error, f"Error occurred while setting category labels -> defaulting to index int", trace=traceback.format_exc(), noticequeue=self.noticequeue)
                        cond_label = cond_label if cond_label not in (None, "") else kwargs.get("iteration", 1)
                        rep_label = rep_labels[file_idx-1] if rep_labels is not None else fileinfo.get("name", file_idx) + f" (file {file_idx})" if isinstance(fileinfo, dict) else op.basename(fileinfo) + f" (file {file_idx})"

                _rep_guard_list.append(rep_label)

                extracted["Condition"] = str(cond_label)
                extracted["Replicate"] = str(rep_label)

                data_cache.append(extracted)

                if self.rep_multiplicates:
                    Reporter(Level.info, f"Found multiple copies of the same replicate label -> Track IDs in these replicates have been prefixed to avoid conflicts.", noticequeue=self.noticequeue)

            except: continue

        if cache and data_cache is not None:
            return data_cache

        elif not cache and data_cache is not None:
            return pd.concat(data_cache, axis=0)
        
        else:
            _log.error(f"No valid data could be loaded from the provided files.\n{traceback.format_exc()}")


    def GetDataFrame(self, filepath: str) -> pd.DataFrame:
        """
        Loads a DataFrame from a file based on its extension.
        Supported formats: CSV, Excel.
        """
        _, ext = op.splitext(filepath.lower())

        if ext not in ['.csv', '.xls', '.xlsx', '.xml', '.json', '.parquet']:
            Reporter(Level.error, f"File format: {ext} is not supported.", details=f"Supported formats include: .csv, .xls, .xlsx, .xml., .json, .parquet", noticequeue=self.noticequeue)
            return None

        match ext:
            case '.csv':
                return self._try_reading(filepath)
            case '.xls' | '.xlsx':
                return pd.read_excel(filepath)
            case '.xml':
                return pd.read_xml(filepath)
            case '.parquet':
                return pd.read_parquet(filepath)
            case '.json':
                return pd.read_json(filepath)
            case _:
                Reporter(Level.error, f"File format: {ext} is not supported.", details=f"Supported formats include: .csv, .xls, .xlsx, .xml., .json, .parquet", noticequeue=self.noticequeue)
        

    def ExtractStripped(self, df: pd.DataFrame, cols: dict = {'id': None, 't': None, 'x': None, 'y': None}, 
                        *args, mirror_y: bool = True, mirror_x: bool = False) -> pd.DataFrame:
        """
        Prepare tracking data:
        - Extract only the 4 key columns.
        - Converts them to numeric, dropping rows with missing values.
        - Mirrors Y if requested.
        """

        try:
            id_col = cols['id']
            t_col = cols['t']
            x_col = cols['x']
            y_col = cols['y']

            if not all(col in df.columns for col in [id_col, t_col, x_col, y_col]):
                missing = [col for col in [id_col, t_col, x_col, y_col] if col not in df.columns]
                Reporter(Level.error, f"Specified columns were not found.", details=f"Missing columns: {missing}", noticequeue=self.noticequeue)
                return pd.DataFrame()

            df = df[[id_col, t_col, x_col, y_col]].apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

        except Exception as e:
            Reporter(Level.error, f"Error in processing input DataFrame", details=f"{str(e)}", noticequeue=self.noticequeue)
            return pd.DataFrame()

        if mirror_y:
            """
            TrackMate may export y-coordinates mirrored.
            This does not affect statistics but leads to incorrect visualization.
            Here Y data is reflected across its midpoint for accurate directionality.
            """
            y_mid = (df[y_col].min() + df[y_col].max()) / 2
            df[y_col] = 2 * y_mid - df[y_col]

        if mirror_x:
            x_mid = (df[x_col].min() + df[x_col].max()) / 2
            df[x_col] = 2 * x_mid - df[x_col]

        # rename main columns
        df = self._standname(df, id_col, t_col, x_col, y_col)

        df.sort_values('Time point', inplace=True)
        t_steps = np.diff(df['Time point'].unique())

        _log.info(f"[INFO] Parsing data -> stripped to key columns: Track ID = {id_col}, Time point = {t_col}, X coordinate = {x_col}, Y coordinate = {y_col}")
        _log.info(f"[INFO] Observed time steps:\n{t_steps}")
        _log.info(f"[INFO] Between unique time points: \n{df['Time point'].unique()}")

        try:
            if np.all(t_steps == t_steps[0]):
                Stats.t_step = float(t_steps[0])
            else:
                Stats.t_step = float(np.median(t_steps))
                Reporter(Level.warning, f"Time points are not uniformly spaced -> this will most probably lead to incorrect data computation.", details=f"Observed time steps:\n{t_steps}\nUsing: {Stats.t_step}", noticequeue=self.noticequeue)
        except Exception as e:
            Reporter(Level.error, f'{e} (spot stats)', trace=traceback.format_exc(), noticequeue=self.noticequeue)

        if self.time_conversion in ('s', 'min', 'h'):
            df = self._convert_time(df)

        return df
    

    def ExtractFull(self, df: pd.DataFrame, cols: dict = {'id': None, 't': None, 'x': None, 'y': None}, 
                    *args, mirror_y: bool = True, mirror_x: bool = False) -> pd.DataFrame:
        """
        Prepare tracking data:
        - Converts chosen coordinate columns to numeric.
        - Mirrors Y if requested.
        - Renames the 4 key columns to standard names.
        - Keeps all other columns intact.
        - Normalizes column labels (e.g. 'CONTRAST_CH' → 'Contrast ch').
        """

        id_col = cols['id']
        t_col = cols['t']
        x_col = cols['x']
        y_col = cols['y']

        try: 
            if not all(col in df.columns for col in [id_col, t_col, x_col, y_col]):
                missing = [col for col in [id_col, t_col, x_col, y_col] if col not in df.columns]
                Reporter(Level.error, f"Specified columns were not found.", details=f"Missing columns: {missing}", noticequeue=self.noticequeue)
                return pd.DataFrame()
            
            # convert only selected columns to numeric
            for c in [id_col, t_col, x_col, y_col]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            # drop rows missing key coordinates
            df = df.dropna(subset=[id_col, t_col, x_col, y_col]).reset_index(drop=True)
        except Exception as e:
            Reporter(Level.error, f"Error in processing input DataFrame", details=f"{str(e)}", noticequeue=self.noticequeue)
            return pd.DataFrame()

        # mirror Y if needed
        if mirror_y:
            y_mid = (df[y_col].min() + df[y_col].max()) / 2
            df[y_col] = 2 * y_mid - df[y_col]

        if mirror_x:
            x_mid = (df[x_col].min() + df[x_col].max()) / 2
            df[x_col] = 2 * x_mid - df[x_col]

        # rename main columns
        df = self._standname(df, id_col, t_col, x_col, y_col)

        df.sort_values('Time point', inplace=True)
        t_steps = np.diff(df['Time point'].unique())

        _log.info(f"[INFO] Parsing data -> preserving all columns")
        _log.info(f"[INFO] Observed time steps:\n{t_steps}")
        _log.info(f"[INFO] Between unique time points: \n{df['Time point'].unique()}")

        try:
            if np.all(t_steps == t_steps[0]):
                Stats.t_step = float(t_steps[0])
            else:
                Stats.t_step = float(np.median(t_steps))
                Reporter(Level.warning, f"Time points are not uniformly spaced -> this will most probably lead to incorrect data computation.", details=f"Observed time steps:\n{t_steps}\nUsing: {Stats.t_step}", noticequeue=self.noticequeue)
        except Exception as e:
            Reporter(Level.error, f'{e} on loading data', trace=traceback.format_exc(), noticequeue=self.noticequeue)

        # normalize other column names
        try:
            df.columns = [self._clean_name(c) if c != 'Track ID' else c for c in df.columns]
        except Exception:
            pass
        
        self._py_numeric_df(df)

        if self.time_conversion in ('s', 'min', 'h'):
            df = self._convert_time(df)
         
        return df
    

    def GetColumns(self, path: str) -> List[str]:
        """
        Returns a list of column names from the DataFrame.
        """

        df = self.GetDataFrame(path)  # or pd.read_excel(path), depending on file type
        return df.columns.tolist()
    

    def FindMatchingColumn(self, columns: List[str], lookfor: List[str]) -> str:
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
    

    def _guard_replicates(self, rep_label, file_info, data, _rep_guard_list) -> pd.DataFrame:

        if rep_label in _rep_guard_list:

            count = _rep_guard_list.count(rep_label)
            data['Track ID'] = data['Track ID'].apply(lambda x: f"{count}_{x}")

            _log.info(f"[INFO] Multiple ({count+1}) replicate labels: '{rep_label}' \n-> adding prefix: {count} to replicate label: {rep_label}\nFile info: {file_info}")
            self.rep_multiplicates = True
        
        return data



    def _try_reading(self, path, encodings=("utf-8", "cp1252", "latin1", "iso8859_15"), **kwargs) -> pd.DataFrame:

        try:
            for enc in encodings:
                try:
                    return pd.read_csv(path, encoding=enc, low_memory=False)
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            Reporter(Level.error, f"{str(e)} -> Failed to read file: {path}.", trace=traceback.format_exc(), noticequeue=self.noticequeue)
    
    
    def _py_numeric_df(self, df: pd.DataFrame) -> None:
        for col in df.columns:
            try: 
                df[col] = pd.to_numeric(df[col], errors='raise')
            except Exception as e: 
                continue  # quietly move on if conversion fails


    def _clean_name(self, name: str) -> str:
        name = str(name)
        name = name.replace("_", " ")
        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
        name = name.strip().capitalize()
        return name
    

    def _standname(self, df, id_col: str, t_col: str, x_col: str, y_col: str) -> pd.DataFrame:
        return df.rename(columns={id_col: 'Track ID', t_col: 'Time point', x_col: 'X coordinate', y_col: 'Y coordinate'})
    

    def _convert_time(self, df: pd.DataFrame) -> pd.DataFrame:
        divide   = False
        multiply = False
        match (self.t_unit, self.time_conversion):
            case ('s', 'min') | ('min', 'h'):
                divide = True
                factor = 60
            case ('s', 'h'):
                divide = True
                factor = 3600
            case ('s', 'day'):
                divide = True
                factor = 86400
            case ('min', 'day'):
                divide = True
                factor = 1440
            case ('h', 'day'):
                divide = True
                factor = 24


            case ('h', 'min') | ('min', 's'):
                multiply = True
                factor = 60
            case ('h', 's'):
                multiply = True
                factor = 3600
            case ('day', 's'):
                multiply = True
                factor = 86400
            case ('day', 'min'):
                multiply = True
                factor = 1440
            case ('day', 'h'):
                multiply = True
                factor = 24

        if divide:
            df['Time point'] = df['Time point'] / factor
            Stats.t_step = Stats.t_step / factor
            _log.info(f"[INFO] Time data converted from {self.t_unit} to {self.time_conversion} <- time point values and time step divided by {factor}.")
        elif multiply:
            df['Time point'] = df['Time point'] * factor
            Stats.t_step = Stats.t_step * factor
            _log.info(f"[INFO] Time data converted from {self.t_unit} to {self.time_conversion} <- time point values and time step multiplied by {factor}.")

        return df
            