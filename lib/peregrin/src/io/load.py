import re
import traceback
import pandas as pd
import numpy as np
import os.path as op
from typing import List

from ..compute.stats import stats


class DataLoader:

    def __init__(self, **kwargs):
        self.ntcq = kwargs.get('noticequeue', None)
    

    # def construct_data(self, files: List[List[str | dict]], strip_data: bool = True, cols: dict = {'id': None, 't': None, 'x': None, 'y': None},
    #                    *args, cond_labels: List[str | int] | None = None, auto_label: bool = False, **kwargs) -> pd.DataFrame:
    #     data_cache = []

    #     for group_idx, group_files in enumerate(files, start=1):
    #         cond_label = cond_labels[group_idx-1] if cond_labels is not None else None

    #         data_cache = self.load_data(
    #             group_files,
    #             strip_data=strip_data,
    #             cols=cols,
    #             cond_label=cond_label,
    #             auto_label=auto_label,
    #             cache=data_cache,
    #             iteration=group_idx,
    #             mirror_y=kwargs.get("mirror_y", True),
    #             mirror_x=kwargs.get("mirror_x", False),
    #         )

    #     if data_cache is not None:
    #         return pd.concat(data_cache, axis=0)
    #     else:
    #         raise ValueError("No valid data could be loaded from the provided files.")


    def load_data(
            self, 
            files: List[str | dict], 
            columns: dict = {
                'id': None, 
                't': None, 
                'x': None, 
                'y': None
            }, 
            cond_label: str | int | None = None, 
            rep_labels: List[str | int] | None = None, 
            *,
            auto_label: bool = False,
            devider_character: str = "#",
            strip_data: bool = True, 
            **kwargs
        ) -> pd.DataFrame | List[pd.DataFrame]:

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
                        df = self.get_data_frame(fileinfo["datapath"])
                    else:
                        df = self.get_data_frame(fileinfo)

                    if strip_data:
                        extracted = self.extract_stripped(
                            df,
                            columns=columns,
                            mirror_y=kwargs.get("mirror_y", True),
                            mirror_x=kwargs.get("mirror_x", False),
                        )
                    else:
                        extracted = self.extract_full(
                            df,
                            columns=columns,
                            mirror_y=kwargs.get("mirror_y", True),
                            mirror_x=kwargs.get("mirror_x", False),
                        )

                except: continue

                if (auto_label 
                    and isinstance(fileinfo, dict) 
                    and fileinfo.get("name") 
                    and len(fileinfo.get("name").split(devider_character)) >= 2):

                    file_info = fileinfo.get("name")

                    cond_label = file_info.split(devider_character)[1]
                    rep_label = file_info.split(devider_character)[2]

                    if file_idx > 1:
                        extracted = self._guard_replicates(rep_label, file_info, extracted, _rep_guard_list)

                elif (auto_label 
                    and isinstance(fileinfo, str)
                    and len(op.basename(fileinfo).split(devider_character)) >= 2):

                    cond_label = op.basename(fileinfo).split(devider_character)[1]
                    rep_label = op.basename(fileinfo).split(devider_character)[2]

                    if file_idx > 1:
                        extracted = self._guard_replicates(rep_label, file_info, extracted, _rep_guard_list)

                else:
                    try:
                        cond_label = cond_label if cond_label not in (None, "") else kwargs.get("iteration", 1)
                        rep_label = rep_labels[file_idx-1] if rep_labels is not None else fileinfo.get("name") + f" (file {file_idx})" if isinstance(fileinfo, dict) else op.basename(fileinfo) + f" (file {file_idx})"
                    except Exception:
                        if self.ntcq is not None:
                            Reporter(Level.error, f"Error occurred while setting category labels -> defaulting to index int", trace=traceback.format_exc(), ntcq=self.ntcq)
                        else:
                            raise ValueError(f"Error occurred while setting category labels -> defaulting to index int\n{traceback.format_exc()}")
                        cond_label = cond_label if cond_label not in (None, "") else kwargs.get("iteration", 1)
                        rep_label = rep_labels[file_idx-1] if rep_labels is not None else fileinfo.get("name", file_idx) + f" (file {file_idx})" if isinstance(fileinfo, dict) else op.basename(fileinfo) + f" (file {file_idx})"

                _rep_guard_list.append(rep_label)

                extracted["condition"] = str(cond_label)
                extracted["replicate"] = str(rep_label)

                data_cache.append(extracted)

                if self.rep_multiplicates:
                    Reporter(Level.info, f"Found multiple copies of the same replicate label -> Track IDs in these replicates have been prefixed to avoid conflicts.", ntcq=self.ntcq)

            except: continue

        if cache and data_cache != []:
            return data_cache

        elif not cache and data_cache != []:
            return pd.concat(data_cache, axis=0)
        
        else:
            if self.ntcq is not None:
                Reporter(Level.error, f"No valid data could be loaded from the provided files.", trace=traceback.format_exc(), ntcq=self.ntcq)
            else:
                raise ValueError(f"No valid data could be loaded from the provided files.\n{traceback.format_exc()}")
            _log.error(f"No valid data could be loaded from the provided files.\n{traceback.format_exc()}")


    def get_data_frame(self, filepath: str) -> pd.DataFrame:
        """
        Loads a DataFrame from a file based on its extension.
        Supported formats: CSV, Excel.
        """
        _, ext = op.splitext(filepath.lower())

        if ext not in ['.csv', '.xls', '.xlsx', '.xml', '.json', '.parquet']:
            if self.ntcq is not None:
                Reporter(Level.error, f"File format: {ext} is not supported.", details=f"Supported formats include: .csv, .xls, .xlsx, .xml., .json, .parquet", ntcq=self.ntcq)
            else:
                raise ValueError(f"File format: {ext} is not supported. Supported formats include: .csv, .xls, .xlsx, .xml., .json, .parquet")
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
                if self.ntcq is not None:
                    Reporter(Level.error, f"File format: {ext} is not supported.", details=f"Supported formats include: .csv, .xls, .xlsx, .xml., .json, .parquet", ntcq=self.ntcq)
                else:
                    raise ValueError(f"File format: {ext} is not supported. Supported formats include: .csv, .xls, .xlsx, .xml., .json, .parquet")
        

    def extract_stripped(self, df: pd.DataFrame, columns: dict = {'id': None, 't': None, 'x': None, 'y': None}, 
                        *args, mirror_y: bool = True, mirror_x: bool = False) -> pd.DataFrame:
        """
        Prepare tracking data:
        - Extract only the 4 key columns.
        - Converts them to numeric, dropping rows with missing values.
        - Mirrors Y if requested.
        """

        try:
            id_col = columns['id']
            t_col  = columns['t']
            x_col  = columns['x']
            y_col  = columns['y']

            if not all(col in df.columns for col in [id_col, t_col, x_col, y_col]):
                missing = [col for col in [id_col, t_col, x_col, y_col] if col not in df.columns]
                if self.ntcq is not None:
                    Reporter(Level.error, f"Specified columns were not found.", details=f"Missing columns: {missing}", ntcq=self.ntcq)
                else:
                    raise ValueError(f"Specified columns were not found. Missing columns: {missing}")
                return pd.DataFrame()

            df = df[[id_col, t_col, x_col, y_col]].apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

        except Exception as e:
            if self.ntcq is not None:
                Reporter(Level.error, f"Error in processing input DataFrame", details=f"{str(e)}", ntcq=self.ntcq)
            else:
                raise ValueError(f"Error in processing input DataFrame: {str(e)}")
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

        df.sort_values('time_point', inplace=True)
        t_steps = np.diff(df['time_point'].unique())

        _log.info(f"[INFO] Parsing data -> stripped to key columns: track_id = {id_col}, time_point = {t_col}, x_coordinate = {x_col}, y_coordinate = {y_col}")
        _log.info(f"[INFO] Observed time steps:\n{t_steps}")
        _log.info(f"[INFO] Between unique time points: \n{df['time_point'].unique()}")

        try:
            if np.all(t_steps == t_steps[0]):
                Stats.t_step = float(t_steps[0])
            else:
                Stats.t_step = float(np.median(t_steps))
                Reporter(Level.warning, f"Time points are not uniformly spaced -> this will most probably lead to incorrect data computation.", details=f"Observed time steps:\n{t_steps}\nUsing: {Stats.t_step}", ntcq=self.ntcq)
        except Exception as e:
            Reporter(Level.error, f'{e} (spot stats)', trace=traceback.format_exc(), ntcq=self.ntcq)

        # run time conversion for any requested target unit handled by _convert_time
        if self.time_conversion not in (None, ""):
            df = self._convert_time(df)

        return df
    

    def extract_full(self, df: pd.DataFrame, columns: dict = {'id': None, 't': None, 'x': None, 'y': None}, 
                    *args, mirror_y: bool = True, mirror_x: bool = False) -> pd.DataFrame:
        """
        Prepare tracking data:
        - Converts chosen coordinate columns to numeric.
        - Mirrors Y if requested.
        - Renames the 4 key columns to standard names.
        - Keeps all other columns intact.
        - Normalizes column labels (e.g. 'CONTRAST_CH' → 'Contrast ch').
        """

        id_col = columns['id']
        t_col = columns['t']
        x_col = columns['x']
        y_col = columns['y']

        try: 
            if not all(col in df.columns for col in [id_col, t_col, x_col, y_col]):
                missing = [col for col in [id_col, t_col, x_col, y_col] if col not in df.columns]
                if self.ntcq is not None:
                    Reporter(Level.error, f"Specified columns were not found.", details=f"Missing columns: {missing}", ntcq=self.ntcq)
                else:
                    raise ValueError(f"Specified columns were not found. Missing columns: {missing}")
                return pd.DataFrame()
            
            # convert only selected columns to numeric
            for c in [id_col, t_col, x_col, y_col]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            # drop rows missing key coordinates
            df = df.dropna(subset=[id_col, t_col, x_col, y_col]).reset_index(drop=True)
        except Exception as e:
            if self.ntcq is not None:
                Reporter(Level.error, f"Error in processing input DataFrame", details=f"{str(e)}", ntcq=self.ntcq)
            else:
                raise ValueError(f"Error in processing input DataFrame: {str(e)}")
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

        df.sort_values('time_point', inplace=True)
        t_steps = np.diff(df['time_point'].unique())

        _log.info(f"[INFO] Parsing data -> preserving all columns")
        _log.info(f"[INFO] Observed time steps:\n{t_steps}")
        _log.info(f"[INFO] Between unique time points: \n{df['time_point'].unique()}")

        try:
            if np.all(t_steps == t_steps[0]):
                Stats.t_step = float(t_steps[0])
            else:
                Stats.t_step = float(np.median(t_steps))
                Reporter(Level.warning, f"Time points are not uniformly spaced -> this will most probably lead to incorrect data computation.", details=f"Observed time steps:\n{t_steps}\nUsing: {Stats.t_step}", ntcq=self.ntcq)
        except Exception as e:
            if self.ntcq is not None:
                Reporter(Level.error, f'{e} on loading data', trace=traceback.format_exc(), ntcq=self.ntcq)
            else:
                raise ValueError(f'{e} on loading data')

        # normalize other column names
        try:
            df.columns = [self._clean_name(c) if c != 'track_id' else c for c in df.columns]
        except Exception:
            pass
        
        self._py_numeric_df(df)

        # run time conversion for any requested target unit handled by _convert_time
        if self.time_conversion not in (None, ""):
            df = self._convert_time(df)
         
        return df
    

    def get_columns(self, path: str) -> List[str]:
        """
        Returns a list of column names from the DataFrame.
        """

        df = self.get_data_frame(path)  # or pd.read_excel(path), depending on file type
        return df.columns.tolist()
    

    def find_matching_column(self, columns: List[str], lookfor: List[str]) -> str:
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
            data['track_id'] = data['track_id'].apply(lambda x: f"{count}_{x}")

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
            if self.ntcq is not None:
                Reporter(Level.error, f"{str(e)} -> Failed to read file: {path}.", trace=traceback.format_exc(), ntcq=self.ntcq)
            else:
                raise ValueError(f"{str(e)} -> Failed to read file: {path}.")
    
    
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
        return df.rename(columns={id_col: 'track_id', t_col: 'time_point', x_col: 'x_coordinate', y_col: 'y_coordinate'})
    

    def _convert_time(self, df: pd.DataFrame) -> pd.DataFrame:
        unit_aliases = {
            "ms": "ms",
            "millisecond": "ms",
            "milliseconds": "ms",
            "milisecond": "ms",   # misspelling
            "miliseconds": "ms",  # misspelling
            "s": "s",
            "sec": "s",
            "second": "s",
            "seconds": "s",
            "min": "min",
            "minute": "min",
            "minutes": "min",
            "h": "h",
            "hr": "h",
            "hour": "h",
            "hours": "h",
            "day": "day",
            "days": "day",
            "d": "day",
        }

        unit_to_seconds = {
            "ms": 1e-3,
            "s": 1.0,
            "min": 60.0,
            "h": 3600.0,
            "day": 86400.0,
        }

        from_unit = unit_aliases.get(str(self.t_unit).strip().lower())
        to_unit = unit_aliases.get(str(self.time_conversion).strip().lower())

        if from_unit is None or to_unit is None:
            Reporter(
                Level.warning,
                f"Unsupported time conversion: {self.t_unit} -> {self.time_conversion}",
                details="Supported units: ms, s, min, h, day",
                ntcq=self.ntcq
            )
            return df

        if from_unit == to_unit:
            return df

        factor = unit_to_seconds[from_unit] / unit_to_seconds[to_unit]

        df["time_point"] = df["time_point"] * factor
        try:
            Stats.t_step = float(Stats.t_step) * factor
        except Exception:
            pass

        _log.info(
            f"[INFO] Time data converted from {from_unit} to {to_unit} "
            f"(multiplied by factor {factor})."
        )

        return df


loader = DataLoader()
load_data = loader.load_data
extract_stripped = loader.extract_stripped
extract_full = loader.extract_full
