# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for the xarray backend module."""

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

from pyinterp.backends.xarray import (
    AxisIdentifier,
    Grid2D,
    Grid3D,
    Grid4D,
    RegularGridInterpolator,
    _coords,
    _get_canonical_dimensions,
)


if TYPE_CHECKING:
    from collections.abc import Hashable

    from ...type_hints import NDArray1D


@pytest.fixture
def geodetic_2d_data() -> xr.DataArray:
    """Create a 2D geodetic data array."""
    lon = np.linspace(0, 360, 10)
    lat = np.linspace(-90, 90, 8)
    rng = np.random.default_rng()
    data = rng.random((10, 8))

    return xr.DataArray(
        data,
        coords={
            "lon": (["lon"], lon, {"units": "degrees_east"}),
            "lat": (["lat"], lat, {"units": "degrees_north"}),
        },
        dims=["lon", "lat"],
    )


@pytest.fixture
def cartesian_2d_data() -> xr.DataArray:
    """Create a 2D Cartesian data array."""
    x = np.linspace(0, 100, 10)
    y = np.linspace(0, 50, 8)
    rng = np.random.default_rng()
    data = rng.random((10, 8))

    return xr.DataArray(
        data,
        coords={"x": (["x"], x), "y": (["y"], y)},
        dims=["x", "y"],
    )


@pytest.fixture
def geodetic_3d_data() -> xr.DataArray:
    """Create a 3D geodetic data array with depth."""
    lon = np.linspace(0, 360, 10)
    lat = np.linspace(-90, 90, 8)
    depth = np.array([0, 100, 500, 1000, 5000])
    rng = np.random.default_rng()
    data = rng.random((10, 8, 5))

    return xr.DataArray(
        data,
        coords={
            "lon": (["lon"], lon, {"units": "degrees_east"}),
            "lat": (["lat"], lat, {"units": "degrees_north"}),
            "depth": (["depth"], depth),
        },
        dims=["lon", "lat", "depth"],
    )


@pytest.fixture
def geodetic_3d_temporal_data() -> xr.DataArray:
    """Create a 3D geodetic data array with temporal axis."""
    lon = np.linspace(0, 360, 10)
    lat = np.linspace(-90, 90, 8)
    time = np.array(
        ["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]"
    )
    rng = np.random.default_rng()
    data = rng.random((10, 8, 3))

    return xr.DataArray(
        data,
        coords={
            "lon": (["lon"], lon, {"units": "degrees_east"}),
            "lat": (["lat"], lat, {"units": "degrees_north"}),
            "time": (["time"], time),
        },
        dims=["lon", "lat", "time"],
    )


@pytest.fixture
def geodetic_4d_data() -> xr.DataArray:
    """Create a 4D geodetic data array."""
    rng = np.random.default_rng()
    lon = np.linspace(0, 360, 5)
    lat = np.linspace(-90, 90, 4)
    depth = np.array([0, 100, 500])
    time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    data = rng.random((5, 4, 3, 2))

    return xr.DataArray(
        data,
        coords={
            "lon": (["lon"], lon, {"units": "degrees_east"}),
            "lat": (["lat"], lat, {"units": "degrees_north"}),
            "depth": (["depth"], depth),
            "time": (["time"], time),
        },
        dims=["lon", "lat", "depth", "time"],
    )


class TestAxisIdentifier:
    """Test AxisIdentifier class."""

    def test_longitude_detection(self, geodetic_2d_data: xr.DataArray) -> None:
        """Test longitude axis detection."""
        identifier = AxisIdentifier(geodetic_2d_data)
        assert identifier.longitude() == "lon"

    def test_latitude_detection(self, geodetic_2d_data: xr.DataArray) -> None:
        """Test latitude axis detection."""
        identifier = AxisIdentifier(geodetic_2d_data)
        assert identifier.latitude() == "lat"

    def test_no_longitude(self, cartesian_2d_data: xr.DataArray) -> None:
        """Test when no longitude axis is found."""
        identifier = AxisIdentifier(cartesian_2d_data)
        assert identifier.longitude() is None

    def test_no_latitude(self, cartesian_2d_data: xr.DataArray) -> None:
        """Test when no latitude axis is found."""
        identifier = AxisIdentifier(cartesian_2d_data)
        assert identifier.latitude() is None


class TestDimsFromDataArray:
    """Test _dims_from_data_array function."""

    def test_geodetic_2d(self, geodetic_2d_data: xr.DataArray) -> None:
        """Test geodetic 2D array dimension extraction."""
        dim_info = _get_canonical_dimensions(geodetic_2d_data, ndims=2)
        assert dim_info.dims == ("lon", "lat")
        assert dim_info.has_longitude is True
        assert dim_info.has_temporal is False

    def test_cartesian_2d(self, cartesian_2d_data: xr.DataArray) -> None:
        """Test Cartesian 2D array dimension extraction."""
        dim_info = _get_canonical_dimensions(cartesian_2d_data, ndims=2)
        assert dim_info.dims == ("x", "y")
        assert dim_info.has_longitude is False
        assert dim_info.has_temporal is False
        assert dim_info.should_be_transposed is False

    def test_geodetic_3d(self, geodetic_3d_data: xr.DataArray) -> None:
        """Test geodetic 3D array dimension extraction."""
        dim_info = _get_canonical_dimensions(geodetic_3d_data, ndims=3)
        assert dim_info.dims == ("lon", "lat", "depth")
        assert dim_info.has_longitude is True
        assert dim_info.has_temporal is False
        assert dim_info.should_be_transposed is False

    def test_geodetic_3d_temporal(
        self, geodetic_3d_temporal_data: xr.DataArray
    ) -> None:
        """Test geodetic 3D temporal array dimension extraction."""
        dim_info = _get_canonical_dimensions(
            geodetic_3d_temporal_data, ndims=3
        )
        assert dim_info.dims == ("lon", "lat", "time")
        assert dim_info.has_longitude is True
        assert dim_info.has_temporal is True
        assert dim_info.should_be_transposed is False

    def test_geodetic_4d(self, geodetic_4d_data: xr.DataArray) -> None:
        """Test geodetic 4D array dimension extraction."""
        dim_info = _get_canonical_dimensions(geodetic_4d_data, ndims=4)
        assert dim_info.dims == ("lon", "lat", "time", "depth")
        assert dim_info.has_longitude is True
        assert dim_info.has_temporal is True
        assert dim_info.should_be_transposed is True

    def test_dimension_mismatch(self, geodetic_2d_data: xr.DataArray) -> None:
        """Test error on dimension mismatch."""
        with pytest.raises(ValueError, match="number of dimensions"):
            _get_canonical_dimensions(geodetic_2d_data, ndims=3)


class TestCoords:
    """Test _coords function."""

    def test_coords_basic(self, geodetic_2d_data: xr.DataArray) -> None:
        """Test basic coordinate extraction."""
        coords_dict: dict[Hashable, NDArray1D] = {
            "lon": np.array([10.0, 20.0]),
            "lat": np.array([45.0, -30.0]),
        }
        result = _coords(coords_dict, ("lon", "lat"))
        assert len(result) == 2
        assert np.allclose(result[0], [10.0, 20.0])
        assert np.allclose(result[1], [45.0, -30.0])

    def test_coords_wrong_type(self) -> None:
        """Test error when coords is not a dict."""
        with pytest.raises(TypeError, match="coords must be"):
            _coords([1, 2], ("lon", "lat"))  # type: ignore[arg-type]

    def test_coords_length_mismatch(self) -> None:
        """Test error when number of coordinates doesn't match."""
        coords_dict: dict[Hashable, NDArray1D] = {
            "lon": np.array([10.0, 20.0])
        }
        with pytest.raises(IndexError):
            _coords(coords_dict, ("lon", "lat"))

    def test_coords_unknown_axes(self) -> None:
        """Test error for unknown axes."""
        coords_dict: dict[Hashable, NDArray1D] = {
            "lon": np.array([10.0, 20.0]),
            "lat": np.array([45.0, -30.0]),
            "unknown": np.array([1, 2]),
        }
        with pytest.raises(
            IndexError,
            match="doesn't match number of dimensions",
        ):
            _coords(coords_dict, ("lon", "lat"))


class TestGrid2D:
    """Test Grid2D class."""

    def test_init_geodetic(self, geodetic_2d_data: xr.DataArray) -> None:
        """Test Grid2D initialization with geodetic data."""
        grid = Grid2D(geodetic_2d_data)
        assert grid.ndim == 2

    def test_init_cartesian(self, cartesian_2d_data: xr.DataArray) -> None:
        """Test Grid2D initialization with Cartesian data."""
        grid = Grid2D(cartesian_2d_data)
        assert grid.ndim == 2

    def test_init_dimension_error(
        self, geodetic_3d_data: xr.DataArray
    ) -> None:
        """Test Grid2D initialization fails with 3D data."""
        with pytest.raises(ValueError, match="number of dimensions"):
            Grid2D(geodetic_3d_data)

    def test_bivariate_interpolation(
        self, geodetic_2d_data: xr.DataArray
    ) -> None:
        """Test bivariate interpolation."""
        grid = Grid2D(geodetic_2d_data)
        coords: dict[Hashable, NDArray1D] = {
            "lon": np.array([10.0]),
            "lat": np.array([45.0]),
        }
        result = grid.bivariate(coords, method="bilinear")
        assert result.shape == (1,)


class TestGrid3D:
    """Test Grid3D class."""

    def test_init_with_depth(self, geodetic_3d_data: xr.DataArray) -> None:
        """Test Grid3D initialization with depth axis."""
        grid = Grid3D(geodetic_3d_data)
        assert grid.ndim == 3

    def test_init_with_temporal(
        self, geodetic_3d_temporal_data: xr.DataArray
    ) -> None:
        """Test Grid3D initialization with temporal axis."""
        grid = Grid3D(geodetic_3d_temporal_data)
        assert grid.ndim == 3
        assert grid._datetime64 is not None

    def test_init_dimension_error(
        self, geodetic_2d_data: xr.DataArray
    ) -> None:
        """Test Grid3D initialization fails with 2D data."""
        with pytest.raises(ValueError, match="number of dimensions"):
            Grid3D(geodetic_2d_data)

    def test_trivariate_interpolation(
        self, geodetic_3d_data: xr.DataArray
    ) -> None:
        """Test trivariate interpolation."""
        grid = Grid3D(geodetic_3d_data)
        coords: dict[Hashable, NDArray1D] = {
            "lon": np.array([10.0]),
            "lat": np.array([45.0]),
            "depth": np.array([100.0]),
        }
        result = grid.trivariate(coords, method="bilinear")
        assert result.shape == (1,)


class TestGrid4D:
    """Test Grid4D class."""

    def test_init_4d(self, geodetic_4d_data: xr.DataArray) -> None:
        """Test Grid4D initialization."""
        grid = Grid4D(geodetic_4d_data)
        assert grid.ndim == 4

    def test_init_dimension_error(
        self, geodetic_2d_data: xr.DataArray
    ) -> None:
        """Test Grid4D initialization fails with 2D data."""
        with pytest.raises(ValueError, match="number of dimensions"):
            Grid4D(geodetic_2d_data)


class TestRegularGridInterpolator:
    """Test RegularGridInterpolator class."""

    def test_init_2d(self, geodetic_2d_data: xr.DataArray) -> None:
        """Test RegularGridInterpolator with 2D data."""
        interp = RegularGridInterpolator(geodetic_2d_data)
        assert interp.ndim == 2
        assert isinstance(interp.grid, Grid2D)

    def test_init_3d(self, geodetic_3d_data: xr.DataArray) -> None:
        """Test RegularGridInterpolator with 3D data."""
        interp = RegularGridInterpolator(geodetic_3d_data)
        assert interp.ndim == 3
        assert isinstance(interp.grid, Grid3D)

    def test_init_temporal(
        self, geodetic_3d_temporal_data: xr.DataArray
    ) -> None:
        """Test RegularGridInterpolator with temporal data."""
        interp = RegularGridInterpolator(geodetic_3d_temporal_data)
        assert interp.ndim == 3

    def test_call_2d(self, geodetic_2d_data: xr.DataArray) -> None:
        """Test calling RegularGridInterpolator for 2D."""
        interp = RegularGridInterpolator(geodetic_2d_data)
        coords = {"lon": np.array([10.0]), "lat": np.array([45.0])}
        result = interp(coords, method="bilinear")
        assert result.shape == (1,)

    def test_call_3d(self, geodetic_3d_data: xr.DataArray) -> None:
        """Test calling RegularGridInterpolator for 3D."""
        interp = RegularGridInterpolator(geodetic_3d_data)
        coords = {
            "lon": np.array([10.0]),
            "lat": np.array([45.0]),
            "depth": np.array([100.0]),
        }
        result = interp(coords, method="bilinear")
        assert result.shape == (1,)

    def test_invalid_dimensions(self) -> None:
        """Test error with invalid number of dimensions."""
        rng = np.random.default_rng()
        data = xr.DataArray(
            rng.random(5), dims=["x"], coords={"x": np.arange(5)}
        )
        with pytest.raises(NotImplementedError):
            RegularGridInterpolator(data)
