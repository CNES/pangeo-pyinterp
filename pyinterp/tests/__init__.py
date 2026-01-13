# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test datasets for pyinterp package."""

import json
import pathlib

import numpy as np
import xarray as xr


# Define the root path for datasets
ROOT = pathlib.Path(__file__).parent.joinpath("dataset").resolve()


def _mask_and_scale(array: xr.DataArray) -> np.ndarray:
    """Mask and scale data."""
    add_offset = array.attrs.pop("add_offset", 0)
    scale_factor = array.attrs.pop("scale_factor", 1)
    fill_value = array.attrs.pop("_FillValue", None)

    result = array.data.astype("float64")
    result = result * scale_factor + add_offset
    if fill_value is not None:
        result[array.data == fill_value] = np.nan
    return result


def load_grid2d() -> xr.Dataset:
    """Load the Grid 2D."""
    path = ROOT.joinpath("mss.json")
    with path.open("r") as stream:
        data = json.load(stream)
    ds = xr.Dataset.from_dict(data)
    ds.variables["mss"].values = _mask_and_scale(ds["mss"])
    return ds


def load_grid3d() -> xr.Dataset:
    """Load the Grid 3D."""

    def _decode_datetime64(array: np.ndarray) -> np.ndarray:
        """Decode datetime64 data."""
        array = array.astype("timedelta64[h]") + np.datetime64("1900-01-01")
        return array.astype("M8[ns]")

    path = ROOT.joinpath("tcw.json")
    with path.open("r") as stream:
        data = json.load(stream)
    ds = xr.Dataset.from_dict(data)
    ds.variables["tcw"].values = _mask_and_scale(ds["tcw"])
    ds["time"] = xr.DataArray(
        _decode_datetime64(ds["time"].values),
        dims=["time"],
        attrs={"long_name": "time"},
    )
    return ds


def load_grid4d() -> xr.Dataset:
    """Return path to the Grid 4D."""

    def _decode_datetime64(array: np.ndarray) -> np.ndarray:
        """Decode datetime64 data."""
        array = array.astype("timedelta64[s]") + np.datetime64("1970-01-01")
        return array.astype("M8[ns]")

    path = ROOT.joinpath("pres_temp_4d.json")

    with path.open("r") as stream:
        data = json.load(stream)
    ds = xr.Dataset.from_dict(data)
    ds["time"] = xr.DataArray(
        _decode_datetime64(ds["time"].values),
        dims=["time"],
        attrs={"long_name": "time"},
    )
    ds = ds.assign_coords(
        {
            "longitude": ds["longitude"].astype("float32"),
            "latitude": ds["latitude"].astype("float32"),
        }
    )

    ds["pressure"] = ds["pressure"].astype("float32")
    ds["temperature"] = ds["temperature"].astype("float32")
    return ds


def ephemeris_path() -> pathlib.Path:
    """Return the path to the ephemeris test file."""
    return ROOT / "ephemeris.txt"


def load_aoml() -> xr.Dataset:
    """Return path to the AOML dataset."""

    def _decode_datetime64(array: np.ndarray) -> np.ndarray:
        """Decode datetime64 data."""
        array = array.astype("timedelta64[h]") + np.datetime64(
            "2001-12-19T18:00:00"
        )
        return array.astype("M8[ns]")

    path = ROOT.joinpath("aoml_v2019.json")
    with path.open("r") as stream:
        data = json.load(stream)
    for item in ("ud", "vd"):
        data["data_vars"][item]["data"] = [
            x if x is not None else float("nan")
            for x in data["data_vars"][item]["data"]
        ]
    ds = xr.Dataset.from_dict(data)
    ds["time"] = xr.DataArray(
        _decode_datetime64(ds["time"].values),
        dims=["time"],
        attrs={"long_name": "time"},
    )
    return ds
