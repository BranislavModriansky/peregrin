import re
import traceback
import warnings
import pandas as pd
import numpy as np
import os.path as op
from typing import Dict, List

from .._pckg_exceptions._pckg_errors import *
from .._pckg_exceptions._pckg_warnings import *
from ..compute.stats import stats


class DataLoader:

    ALIASES = {
        "noticequeue": "ntcq",
        "notice_queue": "ntcq",
        "ntcq": "ntcq",
        "devider": "devider_character",
        "devider_character": "devider_character",
        "devider_char": "devider_character",
        "split": "devider_character",
        "split_character": "devider_character",
        "split_char": "devider_character",
    }

    def __init__(self, **kwargs):

        kwargs = {self.ALIASES.get(k, k): v for k, v in kwargs.items()}

        self.ntcq = kwargs.get("ntcq")

    def load_data(
        self, 
        files: str | dict | List[str | dict] | List[List[str | dict]], 
        colnames: dict = {
            'id': None, 
            't': None, 
            'x': None, 
            'y': None
        }, 
        t_unit: str = 's',
        *,
        cond_lbls: str | int | List[str | int] | None = None, 
        rep_lbls: str | int | List[str | int] | List[List[str | int]] | None = None, 
        auto_label: bool = False,
        strip_data: bool = True,
        **kwargs
    ) -> pd.DataFrame | Dict[str, Dict[str, pd.DataFrame]]:

        """
        Load tracking data from any number of files into a single DataFrame, 
        while assigning condition and replicate labels. This method is used
        to prepare data for further handling in the computations, using the 
        `peregrin` library.
        
        Parameters
        ----------
        files : str | dict | List[str | dict] | List[List[str | dict]]
            Either a single file path or dictionary (single file - single condition, single replicate), 
            a list of file paths or dictionaries (single set of files - single condition, multiple replicates), 
            or a list of lists of file paths or dictionaries (multiple sets of files - multiple conditions, multiple replicates). 
            Each file must contain tracking data with columns for track identifiers, time points, and x/y coordinates.

        colnames : dict
            A dictionary specifying the column names for track identifiers, time points, and x/y coordinates. 
            With the default {'id': None, 't': None, 'x': None, 'y': None}, the method will NOT attempt to 
            automatically detect these columns.

        t_unit : str, optional, default 's'
            The unit of time used in the input data. Supported units include: 'ms', 's', 'min', 'h', 'day'.

        cond_lbls : str | int | List[str | int], optional
            Condition labels to be assigned to the loaded data. The number of labels must match the number of conditions 
            (sets of files) being loaded. If None, the method will assign default labels: 1, 2, 3, ... for each condition.

        rep_lbls : str | int | List[str | int] | List[List[str | int]], optional
            Replicate labels to be assigned to the loaded data. The number of labels must match the number of replicates 
            (files) being loaded per set. If None, the method will assign default labels - file names or indices - for each replicate.
            
        auto_label : bool, optional, default False
            If True, the method will attempt to automatically extract condition and replicate labels from the file names, 
            using a devider character (default: '#'). First part of the file name - between the first and second occurrence 
            of the devider character - will be treated as the condition label, while the second part - between the second 
            and third occurrence of the devider character - will be treated as the replicate label.
        
        strip_data : bool, optional, default True
            If True, the method will extract only the 4 key columns (id, t, x, y) from the loaded data, dropping all other columns.
            If False, the method will keep all columns intact, while still renaming the 4 key columns to standard names.

        time_conversion : str, optional, default None
            If provided, time will be converted from the input unit `t_unit` to the specified target unit. 
            Supported units include: 'ms', 's', 'min', 'h', 'day'.

        devider_character : str, optional, default '#'
            The character used to split file names for automatic label extraction when `auto_label` is True

        mirror_y : bool, optional, default False
            If True, the method will mirror the y-coordinates across their midpoint - useful for correcting mirrored y-coordinates in exported data.
        
        mirror_x : bool, optional, default False
            If True, the method will mirror the x-coordinates across their midpoint - useful for correcting mirrored x-coordinates in exported data.

        merge: bool, optional, default True
            If True, the method will merge all loaded data into a single DataFrame.
            If False, the method will return a dictionary of dictionaries with DataFrames, organized as {condition: {replicate: DataFrame}}.

        Returns
        -------
        pd.DataFrame
            A single DataFrame containing all loaded data, with condition and replicate labels assigned to each row.
        
        """

        # Wrap single file or label inputs into lists for uniform handling
        if isinstance(files, (str, dict)):
            files = [files]
        if isinstance(files, list) and all(isinstance(f, (str, dict)) for f in files):
            files = [files]

        if isinstance(cond_lbls, (str, int)):
            cond_lbls = [cond_lbls]

        if isinstance(rep_lbls, (str, int)):
            rep_lbls = [rep_lbls]
        elif isinstance(rep_lbls, list) and all(isinstance(r, (str, int)) for r in rep_lbls):
            rep_lbls = [rep_lbls] 

        # Handle aliases for keyword arguments
        kwargs = {self.ALIASES.get(k, k): v for k, v in kwargs.items()}
        
        self.t_unit = t_unit
        self.time_conversion = kwargs.get('time_conversion', None)
        devider_character = kwargs.get("devider_character", "#")

        _rep_guard_list = []
        self.rep_multiplicates = False

        if kwargs.get("merge", True): 
            cache = []
        else: 
            cache = {}
        
        for file_set_idx, file_set in enumerate(files):
            if isinstance(file_set, (str, dict)):
                file_set = [file_set]
        
            for file_idx, fileinfo in enumerate(file_set):
                if isinstance(fileinfo, dict) and "datapath" in fileinfo:
                    df = self._read_file(fileinfo["datapath"])
                else:
                    df = self._read_file(fileinfo)

                extracted = self._extract(
                    df,
                    colnames,
                    strip_data,
                    mirror_y=kwargs.get("mirror_y", False),
                    mirror_x=kwargs.get("mirror_x", False),
                )

                if (auto_label 
                    and isinstance(fileinfo, dict) 
                    and fileinfo.get("name") 
                    and len(fileinfo.get("name").split(devider_character)) >= 2):

                    if cond_lbls is not None:
                        warnings.warn(message="cond_lbls are provided while auto_label is True -> cond_lbls will be ignored.",
                                        category=LabelWarning,
                                        stacklevel=2)

                    file_info = fileinfo.get("name")

                    cond_lbl = file_info.split(devider_character)[1]
                    rep_lbl = file_info.split(devider_character)[2]

                    if file_idx > 1:
                        extracted = self._guard_replicates(rep_lbl, file_info, extracted, _rep_guard_list)

                elif (auto_label 
                      and isinstance(fileinfo, str)
                      and len(op.basename(fileinfo).split(devider_character)) >= 2):

                    if cond_lbls is not None:
                        warnings.warn(message="cond_lbls are provided while auto_label is True -> cond_lbls will be ignored.",
                                        category=LabelWarning,
                                        stacklevel=2)

                    cond_lbl = op.basename(fileinfo).split(devider_character)[1]
                    rep_lbl = op.basename(fileinfo).split(devider_character)[2]

                    if file_idx > 1:
                        extracted = self._guard_replicates(rep_lbl, file_info, extracted, _rep_guard_list)

                else:
                    if cond_lbls is not None:
                        cond_lbl = cond_lbls[file_set_idx]
                    else:
                        cond_lbl = file_set_idx + 1

                    if rep_lbls is not None:
                        rep_lbl = rep_lbls[file_set_idx][file_idx]
                    else:
                        if isinstance(fileinfo, dict):
                            rep_lbl = f"file_{file_idx}_" + fileinfo.get("name", f"file_{file_idx}")
                        else:
                            rep_lbl = f"file_{file_idx}_" + op.basename(fileinfo)

                _rep_guard_list.append(rep_lbl)

                extracted["condition"] = str(cond_lbl)
                extracted["replicate"] = str(rep_lbl)

                if kwargs.get("merge", True):
                    cache.append(extracted)
                else:
                    if str(cond_lbl) not in cache:
                        cache[str(cond_lbl)] = {}
                    cache[str(cond_lbl)][str(rep_lbl)] = extracted

                if self.rep_multiplicates:
                    warnings.warn(message=f"Multiplicates of the same replicate label under the same condition -> track ids in these replicates have been prefixed to avoid conflicts.",
                                  category=LabelWarning,
                                  stacklevel=2)

        if not kwargs.get("merge", True):
            return pd.concat(cache, axis=0)
        else:
            return cache
            

    def _read_file(self, filepath: str) -> pd.DataFrame:
        
        _, ext = op.splitext(filepath.lower())

        match ext:
            case '.csv':
                return self._read_csv(filepath)
            case '.xls' | '.xlsx':
                return pd.read_excel(filepath)
            case '.xml':
                return pd.read_xml(filepath)
            case '.parquet':
                return pd.read_parquet(filepath)
            case '.json':
                return pd.read_json(filepath)
            case _:
                raise FileFormatError(f"{ext} is not supported. Supported formats include: .csv, .xls, .xlsx, .xml., .json, .parquet")

    def _extract(
        self,
        df: pd.DataFrame, 
        colnames: dict,
        strip_data: bool = True,
        *, 
        mirror_y: bool = False, 
        mirror_x: bool = False
    ) -> pd.DataFrame:

        id_col = colnames['id']
        t_col  = colnames['t']
        x_col  = colnames['x']
        y_col  = colnames['y']

        if not all(col in df.columns for col in [id_col, t_col, x_col, y_col]):
            missing = [col for col in [id_col, t_col, x_col, y_col] if col not in df.columns]
            raise ColumnsNotFoundError(f"missing columns: {missing}")

        if strip_data:
            df = df[[id_col, t_col, x_col, y_col]].apply(pd.to_numeric, errors='coerce')
        else:
            for c in [id_col, t_col, x_col, y_col]:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        df = df.dropna(subset=[id_col, t_col, x_col, y_col]).reset_index(drop=True)
        
        if mirror_y:
            y_mid = (df[y_col].min() + df[y_col].max()) / 2
            df[y_col] = 2 * y_mid - df[y_col]
        if mirror_x:
            x_mid = (df[x_col].min() + df[x_col].max()) / 2
            df[x_col] = 2 * x_mid - df[x_col]

        # normalize column names
        df = self._standname(df, id_col, t_col, x_col, y_col)
        try:
            df.columns = [self._clean_name(c) if c != 'track_id' else c for c in df.columns]
        except Exception:
            pass
        
        df.sort_values('time_point', inplace=True)
        t_steps = np.diff(df['time_point'].unique())

        if np.all(t_steps == t_steps[0]):
            stats.t_step = float(t_steps[0])
        else:
            stats.t_step = float(np.median(t_steps))
            warnings.warn(message=f"Non-uniformly spaced time point data -> will probably lead to incorrect data computation.\nObserved time steps:\n{t_steps}\nUsing: {stats.t_step}",
                            category=LabelWarning,
                            stacklevel=2)
        
        self._py_numeric_df(df)

        # run time conversion for any requested target unit handled by _convert_time
        if self.time_conversion not in (None, ""):
            df = self._convert_time(df)
         
        return df


    def get_columns(self, path: str) -> List[str]:
        """
        Returns a list of column names from the DataFrame.
        """
        df = self._read_file(path)  # or pd.read_excel(path), depending on file type
        return df.columns.tolist()
    

    def match_columns(self, columns: List[str], lookfor: List[str]) -> str:
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
    

    def _guard_replicates(self, rep_lbl, file_info, data, _rep_guard_list) -> pd.DataFrame:

        if rep_lbl in _rep_guard_list:

            count = _rep_guard_list.count(rep_lbl)
            data['track_id'] = data['track_id'].apply(lambda x: f"{count}_{x}")

            warnings.warn(message=f"Multiple ({count+1}) replicate labels: '{rep_lbl}' \n-> adding prefix: {count} to replicate label: {rep_lbl}\nFile info: {file_info}",
                          category=LabelWarning,
                          stacklevel=2)
            
            self.rep_multiplicates = True
        
        return data


    def _read_csv(self, path, encodings=("utf-8", "cp1252", "latin1", "iso8859_15"), **kwargs) -> pd.DataFrame:
        try:
            for enc in encodings:
                try:
                    return pd.read_csv(path, encoding=enc, low_memory=False)
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            raise FileFormatError(f"{str(e)} -> Failed to read CSV file: {path}. Tried encodings: {encodings}.")
    
    
    def _py_numeric_df(self, df: pd.DataFrame) -> None:
        for col in df.columns:
            try: 
                df[col] = pd.to_numeric(df[col], errors='raise')
            except Exception as e: 
                continue  # quietly move on if conversion fails


    def _clean_name(self, name: str) -> str:
        name = str(name)
        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
        name = name.strip().lower()
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
            warnings.warn(
                message=f"Unsupported time conversion: {self.t_unit} -> {self.time_conversion}. Skipping conversion.",
                category=LabelWarning,
                stacklevel=2
            )
            return df

        if from_unit == to_unit:
            return df

        factor = unit_to_seconds[from_unit] / unit_to_seconds[to_unit]

        df["time_point"] = df["time_point"] * factor
        try:
            stats.t_step = float(stats.t_step) * factor
        except Exception:
            pass

        return df


loader = DataLoader()
load_data = loader.load_data
get_columns = loader.get_columns
match_columns = loader.match_columns