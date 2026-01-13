# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Module implementing geohash utilities."""

import numpy as np
import xarray as xr

from .core import Axis
from .core.geohash import (
    GeoHash,
    area,
    bounding_boxes,
    decode,
    encode,
    transform,
)
from .type_hints import NDArray1D, NDArray1DStr


__all__ = [
    "GeoHash",
    "area",
    "bounding_boxes",
    "decode",
    "encode",
    "to_xarray",
    "transform",
]


def to_xarray(hashes: NDArray1DStr, data: NDArray1D) -> xr.DataArray:
    """Get the XArray grid representing the GeoHash grid.

    Args:
        hashes: Geohash codes.
        data: The data associated with the codes provided.

    Returns:
        The XArray grid representing the GeoHash grid.

    """
    if hashes.shape != data.shape:
        raise ValueError(
            "hashes, data could not be broadcast together with shape "
            f"{hashes.shape}, f{data.shape}"
        )
    if hashes.dtype.kind != "S":
        raise TypeError("hashes must be a string array")
    lon, lat = decode(bounding_boxes(precision=hashes.dtype.itemsize))
    x_axis = Axis(
        np.unique(lon),  # type: ignore[arg-type]
        period=360,
    )
    y_axis = Axis(np.unique(lat))  # type: ignore[arg-type]

    dtype = data.dtype
    if np.issubdtype(dtype, np.dtype("object")):
        grid = np.empty((len(y_axis), len(x_axis)), dtype)
    else:
        grid = np.zeros((len(y_axis), len(x_axis)), dtype)

    lon, lat = decode(hashes)
    grid[y_axis.find_index(lat), x_axis.find_index(lon)] = data

    return xr.DataArray(
        grid,
        dims=("lat", "lon"),
        coords={
            "lon": xr.DataArray(
                x_axis, dims=("lon",), attrs={"units": "degrees_north"}
            ),
            "lat": xr.DataArray(
                y_axis, dims=("lat",), attrs={"units": "degrees_east"}
            ),
        },
    )
