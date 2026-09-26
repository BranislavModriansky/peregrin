from __future__ import annotations

import polars as pl
from typing import Dict, List, Optional, Any

from warnings import warn
from .._pckg_exceptions._pckg_errors import *
from .._pckg_exceptions._pckg_warnings import *



class Input:
    """Wrapper around a polars DataFrame for carrying a `.metadata` (InputMetadata) attribute."""

    def __init__(self, df: pl.DataFrame, metadata: "InputMetadata" = None):
        self._df = df
        self.metadata = metadata

    # Property for accessing the underlying polars DataFrame
    @property
    def df(self) -> pl.DataFrame:
        return self._df

    # Delegate attribute access to the underlying polars DataFrame for all other attributes
    def __getattr__(self, name):
        return getattr(self._df, name)

    # Delegate item access to the underlying polars DataFrame for indexing operations
    def __getitem__(self, key):
        return self._df[key]

    # Delegate length access to the underlying polars DataFrame
    def __len__(self):
        return len(self._df)

    # Delegate representation to the underlying polars DataFrame
    def __repr__(self):
        return repr(self._df)


class InputMetadata:
    """
    Input metadata container:
    -------------------------

    Keeps a dictionary (1) of the metadata common across all files.

        {
            "spatial_units": str,
            "time_units": str, 
            "time_interval": float,
            "n_frames": int,
            "columns": list[str],
        }

    and a dictionary (2) mapping metadata to files:

        {
            "file1.csv": {
                "spatial_units": str,
                "time_units": str, 
                "time_interval": float,
                "n_frames": int,
                "columns": list[str],
            },
            "file2.csv": ...,
        } 
    """

    # Unit standardization: map aliases to canonical units
    UNIT_ALIASES = {
        'nm':  ['nm', 'nanometer', 'nanometers'],
        'μm':  ['μm', 'um', 'micron', 'microns', 'micrometer', 'micrometers'],
        's':   ['s', 'sec', 'second', 'seconds'],
        'min': ['m', 'min', 'minute', 'minutes'],
        'ms':  ['ms', 'millisecond', 'milliseconds'],
        'h':   ['h', 'hr', 'hour', 'hours'],
        'd':   ['d', 'day', 'days'],
        'px':  ['px', 'pixel', 'pixels'],
        'mm':  ['mm', 'millimeter', 'millimeters'],
        'cm':  ['cm', 'centimeter', 'centimeters'],
        'm':   ['m', 'meter', 'meters'],
    }

    def __init__(self):
        self.input_metadata_common = {}      # init dict for common metadata
        self.input_metadata_individual = {}  # init dict for per-file metadata

    # Delegate attribute access to the underlying polars DataFrame for all other attributes
    def __getitem__(self, key):
        return self.input_metadata_individual[key]

    # Delegate item assignment to the underlying polars DataFrame for indexing operations
    def __setitem__(self, key, value):
        self.input_metadata_individual[key] = value
        if key in ('spatialunits', 'timeunits', 'timestep', 'nframes', 'columns'):
            self.input_metadata_common[key] = value

    def get(self, metadata_key: str = None) -> Optional[Dict[str, str] | str]:
        """
        Get the common metadata or a specific metadata value by key.
        
        Parameters
        ----------
        metadata_key : str, optional
            The key of the metadata value to retrieve. If None, returns the entire common metadata dictionary.

        Returns
        -------
        dict[str, str] | str | None
            The common metadata dictionary if metadata_key is None, otherwise the value corresponding to the specified key. Returns None if the key does not exist.
        """
        if metadata_key is not None:
            return self.input_metadata_common.get(metadata_key)
        return self.input_metadata_common

    def get_each(self, file_name: Optional[str] = None, metadata_keys: Optional[List[str]] = None) -> Optional[Dict[str, Dict[str, str]] | Dict[str, str] | str]:
        """
        Get the metadata for individual files, optionally filtered by file name and/or metadata keys.

        Parameters
        ----------
        file_name : str, optional
            The name of the file for which to retrieve metadata. If None, returns metadata for all files.
        metadata_keys : list of str, optional
            The list of metadata keys to retrieve for the specified file. If None, returns all metadata for the file.
        Returns
        -------
        dict[str, dict[str, str]] | dict[str, str] | str | None
            The metadata dictionary for the specified file(s), filtered by the specified keys if provided. Returns None if the file does not exist.
        """
        if file_name is None:
            if metadata_keys is not None:
                return {file: {
                    key: metadata.get(key) for key in metadata_keys
                } for file, metadata in self.input_metadata_individual.items()}
            return self.input_metadata_individual
        else:
            if metadata_keys is not None:
                return {key: self.input_metadata_individual[file_name].get(key) for key in metadata_keys}
            return self.input_metadata_individual[file_name]

    def write(self, *, spatialunits = None, timeunits = None):
        """
        Manually update the common metadata values. This method allows the user to specify or override 
        the common metadata values for spatial units, time units, time interval, number of frames, and columns. 
        If a parameter is not provided (i.e., None), the existing value in the common metadata will be retained.

        Parameters
        ----------
        spatialunits : str, optional
            The spatial units to set (e.g., 'nm', 'μm', 'px'). If None, the existing value is retained.
        timeunits : str, optional
            The time units to set (e.g., 's', 'ms', 'min'). If None, the existing value is retained.
        """
        self.input_metadata_common["spatialunits"] = self._get_alias(spatialunits) if spatialunits is not None else self.input_metadata_common.get("spatialunits")
        self.input_metadata_common["timeunits"] = self._get_alias(timeunits) if timeunits is not None else self.input_metadata_common.get("timeunits")

    def _update(self, metadata: Dict[str, dict]):
        """Update the metadata for individual files."""

        for key, item in metadata.items():
            for sub_key, sub_item in item.items():

                if not isinstance(sub_item, str):
                    continue  # e.g. 'columns' list; or numeric timestep/nframes

                for unit, aliases in self.UNIT_ALIASES.items():
                    # if the sub_item str matches any of the aliases, replace it with the canonical unit
                    if sub_item in aliases:
                        metadata[key][sub_key] = unit
                        break

        # Update the individual metadata dictionary and check for consistency
        self.input_metadata_individual.update(metadata)
        self._check()

    
    def _check(self) -> None:
        all_time_units = set()
        all_spatial_units = set()
        all_n_frames = set()
        all_time_intervals = set()
        columns_set = set()

        for _, metadata in self.input_metadata_individual.items(): 
            all_time_units.add(metadata.get("timeunits"))
            all_spatial_units.add(metadata.get("spatialunits"))
            all_n_frames.add(metadata.get("nframes"))
            all_time_intervals.add(metadata.get("timestep"))
            columns_set.add(tuple(metadata.get("columns")) if metadata.get("columns") is not None else None)

        first_metadata = next(iter(self.input_metadata_individual.values()))
        self.input_metadata_common["timeunits"] = first_metadata.get("timeunits")
        self.input_metadata_common["spatialunits"] = first_metadata.get("spatialunits")
        self.input_metadata_common["nframes"] = first_metadata.get("nframes")
        self.input_metadata_common["timestep"] = first_metadata.get("timestep")
        self.input_metadata_common["columns"] = first_metadata.get("columns")

        if self.input_metadata_common.get("timeunits") is None or self.input_metadata_common.get("timeunits") == '':
            warn("No time units found in input files.\n Please specify the time units using <load_data result>.metadata.write(time_unit=\"<unit>\")", InputWarning, stacklevel=2)
        if self.input_metadata_common.get("spatialunits") is None or self.input_metadata_common.get("spatialunits") == '':
            warn("No spatial units found in input files.\n Please specify the spatial units using <load_data result>.metadata.write(spatial_unit=\"<unit>\")", InputWarning, stacklevel=2)
        
        if len(all_time_units) > 1:
            self.input_metadata_common["timeunits"] = ''
            warn(f"Inconsistent time units across input files -> found {all_time_units}.", InputWarning, stacklevel=2)
        if len(all_spatial_units) > 1:
            self.input_metadata_common["spatialunits"] = ''
            warn(f"Inconsistent spatial units across input files -> found {all_spatial_units}.", InputWarning, stacklevel=2)
        if len(all_n_frames) > 1:
            self.input_metadata_common["nframes"] = ''
            warn(f"Inconsistent number of frames across input files -> found {all_n_frames}.", InputWarning, stacklevel=2)
        if len(all_time_intervals) > 1:
            raise InputError(f"Inconsistent time intervals across input files -> found {all_time_intervals}. Please ensure that all input files have the same time interval.")

        if len(columns_set) > 1:
            self.input_metadata_common["columns"] = ''
            warn(f"Inconsistent columns across input files -> found {len(columns_set)} distinct schemas.", InputWarning, stacklevel=2)

    def _get_alias(self, unit: str) -> str:
        for alias, aliases in self.UNIT_ALIASES.items():
            if unit in aliases:
                return alias
        raise ValueError(f"Unit '{unit}' is not recognized. Please use one of the following units: {list(self.UNIT_ALIASES.keys())}")
