# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for the ``pyinterp`` accessors of xarray objects."""

import numpy as np
import pytest
import xarray as xr

from pyinterp.backends.xarray import RegularGridInterpolator


@pytest.fixture
def mss() -> xr.DataArray:
    """Create a 2D geodetic data array stored in the (lat, lon) order."""
    lon = np.arange(0.0, 360.0, 5.0)
    lat = np.arange(-80.0, 81.0, 5.0)
    lon_2d, lat_2d = np.meshgrid(lon, lat)
    data = np.cos(np.radians(lat_2d)) * np.sin(np.radians(lon_2d))
    return xr.DataArray(
        data,
        coords={
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
            "crs": 0,
        },
        dims=("lat", "lon"),
        name="mss",
        attrs={"units": "m"},
    )


@pytest.fixture
def sst() -> xr.DataArray:
    """Create a 3D geodetic data array with a temporal axis."""
    lon = np.arange(0.0, 360.0, 10.0)
    lat = np.arange(-80.0, 81.0, 10.0)
    time = np.arange(
        np.datetime64("2020-01-01"), np.datetime64("2020-01-05")
    ).astype("datetime64[ns]")
    rng = np.random.default_rng(42)
    return xr.DataArray(
        rng.random((len(lon), len(lat), len(time))),
        coords={
            "lon": ("lon", lon, {"units": "degrees_east"}),
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "time": time,
        },
        dims=("lon", "lat", "time"),
        name="sst",
    )


def test_grid(mss: xr.DataArray) -> None:
    """One-dimensional coordinates define a grid."""
    lon = np.arange(-180.0, 180.0, 7.5)
    lat = np.arange(-60.0, 60.0, 3.5)
    result = mss.pyinterp.interp(lon=lon, lat=lat, method="bicubic")

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("lat", "lon")
    np.testing.assert_array_equal(result.lon, lon)
    np.testing.assert_array_equal(result.lat, lat)
    assert result.name == "mss"
    assert result.attrs == {"units": "m"}
    assert result.lon.attrs == {"units": "degrees_east"}
    assert result.lat.attrs == {"units": "degrees_north"}
    assert int(result.crs) == 0

    lon_2d, lat_2d = np.meshgrid(lon, lat)
    expected = RegularGridInterpolator(mss)(
        {"lon": lon_2d, "lat": lat_2d}, method="bicubic"
    )
    np.testing.assert_allclose(result.values, expected)


def test_attributes_not_shared(mss: xr.DataArray) -> None:
    """The result does not share the attributes of the source."""
    result = mss.pyinterp.interp(lon=[10.0], lat=[20.0])
    result.attrs["units"] = "cm"
    result.lon.attrs["units"] = "degrees"
    assert mss.attrs["units"] == "m"
    assert mss.lon.attrs["units"] == "degrees_east"


def test_points(mss: xr.DataArray) -> None:
    """Data arrays sharing a dimension are interpolated point by point."""
    lon = xr.DataArray(
        [10.0, 20.0, 30.0], dims="obs", coords={"obs": [7, 8, 9]}
    )
    lat = xr.DataArray([-5.0, 0.0, 5.0], dims="obs")
    result = mss.pyinterp.interp(lon=lon, lat=lat)

    assert result.dims == ("obs",)
    np.testing.assert_array_equal(result.obs, [7, 8, 9])
    np.testing.assert_array_equal(result.lon, lon)
    np.testing.assert_array_equal(result.lat, lat)
    expected = RegularGridInterpolator(mss)(
        {"lon": lon.values, "lat": lat.values}
    )
    np.testing.assert_allclose(result.values, expected)


def test_scalar(mss: xr.DataArray) -> None:
    """A scalar gives a scalar coordinate."""
    result = mss.pyinterp.interp(lon=10.0, lat=[0.0, 5.0])
    assert result.dims == ("lat",)
    assert float(result.lon) == 10.0

    result = mss.pyinterp.interp(lon=10.0, lat=5.0)
    assert result.dims == ()
    assert float(result) == pytest.approx(
        RegularGridInterpolator(mss)({"lon": 10.0, "lat": 5.0})
    )


def test_dates_as_strings(sst: xr.DataArray) -> None:
    """Dates given as strings are converted to the type of the axis."""
    result = sst.pyinterp.interp(
        lon=[15.0, 25.0], lat=[5.0], time="2020-01-02T12:00"
    )
    assert result.dims == ("lon", "lat")
    assert result.time.dtype == sst.time.dtype
    assert result.time.values == np.datetime64("2020-01-02T12:00", "ns")
    expected = RegularGridInterpolator(sst)(
        {
            "lon": np.array([[15.0], [25.0]]),
            "lat": np.array([[5.0], [5.0]]),
            "time": np.datetime64("2020-01-02T12:00", "ns"),
        }
    )
    np.testing.assert_allclose(result.values, expected)


def test_options(mss: xr.DataArray) -> None:
    """Keyword arguments that are not dimensions are options."""
    with pytest.raises(ValueError, match="is out of bounds"):
        mss.pyinterp.interp(lon=[10.0], lat=[89.0], bounds_error=True)
    with pytest.raises(TypeError):
        mss.pyinterp.interp(lon=[10.0], lat=[20.0], not_an_option=True)

    result = mss.pyinterp.interp(
        lon=[10.0],
        lat=[20.0],
        method="bicubic",
        half_window_size_x=3,
        half_window_size_y=3,
    )
    assert result.shape == (1, 1)


def test_coords_mapping(mss: xr.DataArray) -> None:
    """Coordinates can be given as a mapping."""
    result = mss.pyinterp.interp({"lon": [10.0, 20.0]}, lat=[0.0])
    assert result.shape == (1, 2)
    with pytest.raises(ValueError, match="both in coords"):
        mss.pyinterp.interp({"lon": [10.0]}, lon=[20.0], lat=[0.0])


def test_errors(mss: xr.DataArray) -> None:
    """Invalid coordinates are reported."""
    with pytest.raises(ValueError, match=r"\('lat',\) have no coordinate"):
        mss.pyinterp.interp(lon=[10.0])
    with pytest.raises(ValueError, match="are not dimensions"):
        mss.pyinterp.interp({"lon": [10.0], "lat": [0.0], "depth": [1.0]})
    with pytest.raises(ValueError, match="give a multidimensional"):
        mss.pyinterp.interp(lon=np.zeros((2, 2)), lat=[0.0])


def test_dask(mss: xr.DataArray) -> None:
    """A variable backed by dask is loaded before interpolation."""
    pytest.importorskip("dask")
    expected = mss.pyinterp.interp(lon=[10.0, 20.0], lat=[0.0])
    result = mss.chunk({"lat": 10}).pyinterp.interp(
        lon=[10.0, 20.0], lat=[0.0]
    )
    xr.testing.assert_identical(result, expected)


def test_dataset(mss: xr.DataArray) -> None:
    """The variables of a dataset are interpolated, kept or dropped."""
    dataset = xr.Dataset(
        {
            "mss": mss,
            "error": mss * 0.1,
            "lon_bnds": (("lon", "nv"), np.zeros((mss.lon.size, 2))),
            "version": ((), 2),
        },
        attrs={"title": "test"},
    )
    result = dataset.pyinterp.interp(lon=[10.0, 20.0], lat=[0.0])

    assert set(result.data_vars) == {"mss", "error", "version"}
    assert result.attrs == {"title": "test"}
    assert int(result.version) == 2
    xr.testing.assert_identical(
        result.mss, mss.pyinterp.interp(lon=[10.0, 20.0], lat=[0.0])
    )

    with pytest.raises(ValueError, match="no variable"):
        dataset.pyinterp.interp(lat=[0.0], nv=[0.0])
