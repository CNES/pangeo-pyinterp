# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Interpolation on the grid defined by 1D coordinates.

The ``*_outer`` functions must give the same results as the point-by-point
interpolation of the flattened grid of points.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from .... import core
from ....core.config import geometric


#: Geometric interpolation methods
METHODS = ["bilinear", "idw", "nearest"]

#: Longitudes of the grids: periodic, from 0 to 357.5
LON = np.arange(0.0, 360.0, 2.5)

#: Longitudes to interpolate at: in [-180, 180), in the last cell of the
#: axis (between 357.5 and 360), on the nodes, beyond 360 and undefined
LON_QUERY = np.concatenate(
    [
        np.arange(-180.0, 180.0, 7.3),
        np.linspace(357.5, 360.0, 7),
        [0.0, 2.5, 180.0, 365.1, -0.4, np.nan],
    ]
)

#: Latitudes to interpolate at, some out of the grids, which stop at 80
LAT_QUERY = np.concatenate(
    [np.arange(-85.0, 86.0, 3.1), [-80.0, 80.0, 0.0, np.nan]]
)


def _data(shape: tuple[int, ...], dtype: Any) -> np.ndarray:  # noqa: ANN401
    """Random values with undefined cells."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(shape)
    data[10:13, 5:7, ...] = np.nan
    data[-1, 20, ...] = np.nan
    return np.ascontiguousarray(data.astype(dtype))


def _points(*axes: np.ndarray) -> list[np.ndarray]:
    """Flatten the grid of points with the first axis varying fastest."""
    return [
        item.ravel(order="F") for item in np.meshgrid(*axes, indexing="ij")
    ]


def _points2d(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten the 2D grid of points with x varying fastest."""
    x_points, y_points = _points(x, y)
    return x_points, y_points


def _check(expected: np.ndarray, actual: np.ndarray) -> None:
    """Check that the two interpolations agree."""
    assert expected.dtype == actual.dtype
    assert expected.shape == actual.shape
    np.testing.assert_array_equal(np.isnan(expected), np.isnan(actual))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("num_threads", [1, 0])
def test_bivariate(
    method: str,
    dtype: Any,  # noqa: ANN401
    descending: bool,
    num_threads: int,
) -> None:
    """The grid of points is interpolated as its flattened points."""
    lat = np.arange(-80.0, 80.1, 2.0)
    if descending:
        lat = lat[::-1]
    grid = core.Grid(
        core.Axis(LON, period=360.0),
        core.Axis(lat),
        _data((LON.size, lat.size), dtype),
    )
    config = getattr(geometric.Bivariate, method)().with_num_threads(
        num_threads
    )
    x, y = _points(LON_QUERY, LAT_QUERY)
    _check(
        core.bivariate(grid, x, y, config),
        core.bivariate_outer(grid, LON_QUERY, LAT_QUERY, config),
    )


def test_bivariate_seam() -> None:
    """The last cell of a periodic axis is interpolated across the seam."""
    lon = np.arange(0.0, 360.0, 0.25)
    lat = np.array([-1.0, 1.0])
    data = np.repeat(np.cos(np.radians(lon))[:, np.newaxis], 2, axis=1)
    grid = core.Grid(core.Axis(lon, period=360.0), core.Axis(lat), data)
    x = np.array([359.75, 359.8, 359.9, 360.0, -0.1, 0.0])
    result = core.bivariate_outer(
        grid, x, np.array([0.0]), geometric.Bivariate.bilinear()
    )
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, np.cos(np.radians(x)), atol=1e-5)


def test_bounds_error() -> None:
    """A coordinate out of the grid raises an error if requested."""
    grid = core.Grid(
        core.Axis(LON, period=360.0),
        core.Axis(np.arange(-80.0, 80.1, 2.0)),
        _data((LON.size, 81), np.float64),
    )
    config = geometric.Bivariate.bilinear().with_bounds_error(True)
    x = np.array([10.0, 370.0])
    with pytest.raises(ValueError, match="out of bounds"):
        core.bivariate_outer(grid, x, np.array([0.0, 85.0]), config)
    with pytest.raises(ValueError, match="out of bounds"):
        core.bivariate(grid, *_points2d(x, np.array([0.0, 85.0])), config)
    _check(
        core.bivariate(grid, *_points2d(x, np.array([0.0, 5.0])), config),
        core.bivariate_outer(grid, x, np.array([0.0, 5.0]), config),
    )


def test_empty() -> None:
    """An empty coordinate gives an empty result."""
    grid = core.Grid(
        core.Axis(LON, period=360.0),
        core.Axis(np.arange(-80.0, 80.1, 2.0)),
        _data((LON.size, 81), np.float64),
    )
    config = geometric.Bivariate.bilinear().with_bounds_error(True)
    result = core.bivariate_outer(grid, np.array([400.0]), np.zeros(0), config)
    assert result.shape == (0,)


def _third_axis(
    temporal: bool,
) -> tuple[core.Axis | core.TemporalAxis, np.ndarray]:
    """Temporal or spatial third axis, and the coordinates to query."""
    if temporal:
        dates = np.arange(
            np.datetime64("2020-01-01"), np.datetime64("2020-01-06")
        ).astype("datetime64[h]")
        query = np.array(
            ["2019-12-31", "2020-01-01", "2020-01-02T07", "2020-01-05T00"],
            dtype="datetime64[h]",
        )
        return core.TemporalAxis(dates), query
    return (
        core.Axis(np.array([0.0, 10.0, 50.0, 100.0, 500.0])),
        np.array([-1.0, 0.0, 33.3, 500.0, 120.0]),
    )


#: Latitudes of the 3D and 4D grids
LAT_3D = np.arange(-80.0, 80.1, 4.0)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("temporal", [False, True])
def test_trivariate(method: str, dtype: Any, temporal: bool) -> None:  # noqa: ANN401
    """The grid of points is interpolated as its flattened points."""
    z_axis, z = _third_axis(temporal)
    grid = core.Grid(
        core.Axis(LON, period=360.0),
        core.Axis(LAT_3D),
        z_axis,
        _data((LON.size, LAT_3D.size, len(z_axis)), dtype),
    )
    config = getattr(geometric.Trivariate, method)()
    x, y, z_points = _points(LON_QUERY, LAT_QUERY, z)
    _check(
        core.trivariate(grid, x, y, z_points, config),
        core.trivariate_outer(grid, LON_QUERY, LAT_QUERY, z, config),
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("temporal", [False, True])
def test_quadrivariate(method: str, temporal: bool) -> None:
    """The grid of points is interpolated as its flattened points."""
    z_axis, z = _third_axis(temporal)
    u = np.array([1.0, 2.0, 4.0])
    grid = core.Grid(
        core.Axis(LON, period=360.0),
        core.Axis(LAT_3D),
        z_axis,
        core.Axis(u),
        _data((LON.size, LAT_3D.size, len(z_axis), u.size), np.float64),
    )
    u_query = np.array([0.5, 1.0, 1.7, 4.0])
    config = getattr(geometric.Quadrivariate, method)()
    x, y, z_points, u_points = _points(LON_QUERY, LAT_QUERY, z, u_query)
    _check(
        core.quadrivariate(grid, x, y, z_points, u_points, config),
        core.quadrivariate_outer(
            grid, LON_QUERY, LAT_QUERY, z, u_query, config
        ),
    )
