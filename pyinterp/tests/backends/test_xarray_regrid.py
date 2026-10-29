# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests of the conservative regridding of the xarray accessors."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ... import regrid
from ...core.geometry.geographic import Spheroid


def _dataset() -> xr.Dataset:
    """Global 1° dataset stored as (time, lat, depth, lon), with CF bounds."""
    lon = np.arange(0, 360.0) + 0.5
    lat = np.arange(-90, 90.0) + 0.5
    time = np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[ns]")
    depth = np.array([0.0, 10.0, 50.0])
    x, y = np.meshgrid(np.radians(lon), np.radians(lat))
    field = np.cos(y) * np.sin(3 * x) + 0.5 * np.sin(2 * y)
    values = (
        field[None, :, None, :]
        * np.arange(1, 3)[:, None, None, None]
        * np.arange(1, 4)[None, None, :, None]
    )
    return xr.Dataset(
        {
            "sst": (
                ("time", "lat", "depth", "lon"),
                values,
                {"units": "K", "long_name": "temperature"},
            ),
            "mask": (("lat", "lon"), np.isfinite(field).astype("i1")),
            "lon_profile": (("lon",), lon),
            "level": (("depth",), depth * 2),
            "lon_bnds": (("lon", "nv"), np.stack((lon - 0.5, lon + 0.5), -1)),
            "lat_bnds": (("lat", "nv"), np.stack((lat - 0.5, lat + 0.5), -1)),
        },
        coords={
            "lon": (
                "lon",
                lon,
                {"units": "degrees_east", "bounds": "lon_bnds"},
            ),
            "lat": (
                "lat",
                lat,
                {"units": "degrees_north", "bounds": "lat_bnds"},
            ),
            "time": ("time", time),
            "depth": ("depth", depth, {"units": "m"}),
        },
        attrs={"title": "test"},
    )


def test_data_array() -> None:
    """Labels, name, attributes and the other dimensions are kept."""
    sst = _dataset().sst
    target_lon = np.arange(-180, 180, 5.0) + 2.5
    target_lat = np.arange(-90, 90, 5.0) + 2.5
    result = sst.pyinterp.regrid(lon=target_lon, lat=target_lat)
    assert result.dims == sst.dims
    assert result.shape == (2, 36, 3, 72)
    assert result.name == "sst"
    assert result.attrs == sst.attrs
    np.testing.assert_array_equal(result.lon, target_lon)
    np.testing.assert_array_equal(result.lat, target_lat)
    np.testing.assert_array_equal(result.time, sst.time)
    np.testing.assert_array_equal(result.depth, sst.depth)
    assert result.lon.attrs == {"units": "degrees_east"}
    assert result.depth.attrs == {"units": "m"}
    expected = regrid.conservative(
        sst.isel(time=1, depth=2).values,
        sst.lon.values,
        sst.lat.values,
        target_lon,
        target_lat,
        axes=(1, 0),
    )
    np.testing.assert_allclose(
        result.isel(time=1, depth=2).values, expected, rtol=1e-14
    )
    # The coordinates can be given as a mapping, and the options are passed
    mapping = sst.pyinterp.regrid(
        {"lon": target_lon, "lat": target_lat},
        spheroid=Spheroid(),
        num_threads=1,
    )
    # WGS-84 instead of a sphere: the fields reach 6 in absolute value
    assert 0 < np.abs(mapping - result).max() < 1e-4


def test_cells_as_bounds() -> None:
    """Target cells given by their bounds, source bounds given explicitly."""
    sst = _dataset().sst.isel(time=0, depth=0)
    box = sst.pyinterp.regrid(lon=[[-10.0, 10.0]], lat=[[40.0, 50.0]])
    assert box.shape == (1, 1)
    assert box.lon.item() == 0.0 and box.lat.item() == 45.0
    lat = sst.lat.values
    weights = np.diff(np.sin(np.radians(np.arange(40, 51.0))))
    lon_index = np.r_[350:360, 0:10]
    expected = np.sum(
        sst.values[130:140][:, lon_index].mean(axis=1) * weights
    ) / np.sum(weights)
    assert lat[130] == 40.5
    assert box.item() == pytest.approx(expected, rel=1e-12)
    # Source bounds narrower than the cells: half of the box is covered
    half = sst.pyinterp.regrid(
        lon=[[-10.0, 10.0]],
        lat=[[40.0, 50.0]],
        source_bounds={
            "lon": np.stack((sst.lon - 0.25, sst.lon + 0.25), axis=-1)
        },
        normalize=False,
    )
    assert half.item() == pytest.approx(expected / 2, rel=1e-12)
    # A data array holding the bounds, with the bounds dimension first
    bounds = xr.DataArray(np.array([[-10.0], [10.0]]), dims=("nv", "lon"))
    result = sst.pyinterp.regrid(lon=bounds, lat=[[40.0, 50.0]])
    assert result.item() == pytest.approx(expected, rel=1e-12)


def test_cartesian() -> None:
    """Dimensions that are not a longitude and a latitude."""
    x = np.arange(0.5, 10)
    y = np.arange(0.5, 6)
    da = xr.DataArray(
        np.add.outer(10 * y, x), dims=("y", "x"), coords={"x": x, "y": y}
    )
    result = da.pyinterp.regrid(x=np.arange(1, 10, 2.0), y=[1.5, 4.5])
    expected = da.values.reshape(2, 3, 5, 2).mean(axis=(1, 3))
    np.testing.assert_allclose(result.values, expected, rtol=1e-14)


def test_errors() -> None:
    """Invalid dimensions, methods, options and cells."""
    sst = _dataset().sst
    with pytest.raises(ValueError, match="not dimensions"):
        sst.pyinterp.regrid({"lon": [0.0, 1.0], "x": [1.0, 2.0]})
    with pytest.raises(TypeError, match="unexpected options"):
        sst.pyinterp.regrid(lon=[0.0, 1.0], lat=[0.0, 1.0], x=[1.0, 2.0])
    with pytest.raises(ValueError, match="two dimensions"):
        sst.pyinterp.regrid(lon=[0.0, 1.0])
    with pytest.raises(ValueError, match="mix a longitude"):
        sst.pyinterp.regrid(lon=[0.0, 1.0], depth=[0.0, 1.0])
    with pytest.raises(ValueError, match="unknown regridding method"):
        sst.pyinterp.regrid(lon=[0.0, 1.0], lat=[0.0, 1.0], method="patch")
    with pytest.raises(TypeError, match="unexpected options"):
        sst.pyinterp.regrid(lon=[0.0, 1.0], lat=[0.0, 1.0], skipna=True)
    with pytest.raises(ValueError, match="give its bounds instead"):
        sst.pyinterp.regrid(lon=0.0, lat=[0.0, 1.0])
    curvilinear = xr.DataArray(
        np.zeros((3, 4)),
        dims=("y", "x"),
        coords={
            "lon": (("y", "x"), np.zeros((3, 4)), {"units": "degrees_east"}),
            "lat": (("y", "x"), np.zeros((3, 4)), {"units": "degrees_north"}),
        },
    )
    with pytest.raises(ValueError, match="rectilinear grids only"):
        curvilinear.pyinterp.regrid(x=[0.0, 1.0], y=[0.0, 1.0])


def test_dataset() -> None:
    """Variables regridded, kept and dropped; CF bounds read and written."""
    ds = _dataset()
    target_lon = np.arange(-180, 180, 5.0) + 2.5
    target_lat = np.arange(-90, 90, 5.0) + 2.5
    result = ds.pyinterp.regrid(lon=target_lon, lat=target_lat)
    assert set(result.data_vars) == {
        "sst",
        "mask",
        "level",
        "lon_bnds",
        "lat_bnds",
    }
    assert result.attrs == ds.attrs
    assert result.sst.dims == ds.sst.dims
    np.testing.assert_allclose(
        result.sst.values,
        ds.sst.pyinterp.regrid(lon=target_lon, lat=target_lat),
    )
    np.testing.assert_array_equal(result.level, ds.level)
    assert result.lon.attrs == {"units": "degrees_east", "bounds": "lon_bnds"}
    assert result.lon_bnds.dims == ("lon", "nv")
    np.testing.assert_allclose(
        result.lon_bnds.values,
        np.stack((target_lon - 2.5, target_lon + 2.5), axis=-1),
    )
    np.testing.assert_allclose(result.lat_bnds[0], [-90.0, -85.0])

    # The CF bounds of the source are used: narrower cells covering half of
    # the longitudes.
    narrow = ds.assign(
        lon_bnds=(("lon", "nv"), np.stack((ds.lon - 0.25, ds.lon + 0.25), -1))
    )
    ones = narrow.assign(sst=xr.ones_like(narrow.sst))
    result = ones.pyinterp.regrid(
        lon=target_lon, lat=target_lat, normalize=False
    )
    np.testing.assert_allclose(result.sst.values, 0.5)

    # Without bounds variables, the target bounds are named after the
    # dimensions.
    plain = ds.drop_vars(["lon_bnds", "lat_bnds"])
    plain.lon.attrs.pop("bounds")
    plain.lat.attrs.pop("bounds")
    result = plain.pyinterp.regrid(lon=target_lon, lat=target_lat)
    assert result.lat.attrs["bounds"] == "lat_bnds"
    assert result.lat_bnds.dims == ("lat", "bnds")

    with pytest.raises(ValueError, match="no variable"):
        ds[["lon_profile", "lat_bnds"]].pyinterp.regrid(
            lon=target_lon, lat=target_lat
        )
