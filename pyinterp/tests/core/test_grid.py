# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for Grid."""

import numpy as np
import pytest

from ... import core
from .. import load_grid2d, load_grid3d, load_grid4d


DType = type[np.int8] | type[np.float32] | type[np.float64]


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.float32,
        np.float64,
    ],
)
def test_grid2d_load(dtype: DType) -> None:
    """Test loading a 2D grid."""
    grid_data = load_grid2d()
    x_axis = core.Axis(grid_data.lon.values, period=360.0)
    y_axis = core.Axis(grid_data.lat.values)
    matrix = np.ascontiguousarray(grid_data.mss.values.transpose())
    if np.issubdtype(dtype, np.integer):
        matrix[np.isnan(matrix)] = np.iinfo(dtype).min  # type: ignore[type-var]
    matrix = matrix.astype(dtype)

    grid = core.Grid(x_axis, y_axis, matrix)
    assert len(grid.x) == matrix.shape[0]
    assert len(grid.y) == matrix.shape[1]
    assert np.shares_memory(grid.array, matrix)
    assert "Grid" in repr(grid)
    np.testing.assert_array_equal(grid.array, matrix)


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.float32,
        np.float64,
    ],
)
@pytest.mark.parametrize("temporal_axis", [True, False])
def test_grid3d_load(dtype: DType, temporal_axis: bool) -> None:
    """Test loading a 3D grid."""
    grid_data = load_grid3d()
    x_axis = core.Axis(grid_data.longitude.values, period=360.0)
    y_axis = core.Axis(grid_data.latitude.values)
    z_axis = (
        core.TemporalAxis(grid_data.time.values)
        if temporal_axis
        else core.Axis(grid_data.time.values.astype("float64"))
    )
    matrix = np.ascontiguousarray(grid_data.tcw.values.transpose())
    if np.issubdtype(dtype, np.integer):
        matrix[np.isnan(matrix)] = np.iinfo(dtype).min  # type: ignore[type-var]
    matrix = matrix.astype(dtype)

    grid = core.Grid(x_axis, y_axis, z_axis, matrix)
    assert len(grid.x) == matrix.shape[0]
    assert len(grid.y) == matrix.shape[1]
    assert len(grid.z) == matrix.shape[2]
    assert np.shares_memory(grid.array, matrix)
    assert "Grid" in repr(grid)
    np.testing.assert_array_equal(grid.array, matrix)


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.float32,
        np.float64,
    ],
)
@pytest.mark.parametrize("temporal_axis", [True, False])
def test_grid4d_load(dtype: DType, temporal_axis: bool) -> None:
    """Test loading a 4D grid."""
    grid_data = load_grid4d()
    x_axis = core.Axis(grid_data.longitude.values, period=360.0)
    y_axis = core.Axis(grid_data.latitude.values)
    z_axis = (
        core.TemporalAxis(grid_data.time.values)
        if temporal_axis
        else core.Axis(grid_data.time.values.astype("float64"))
    )
    u_axis = core.Axis(grid_data.level.values)
    matrix = np.ascontiguousarray(grid_data.temperature.values.transpose())
    if np.issubdtype(dtype, np.integer):
        matrix[np.isnan(matrix)] = np.iinfo(dtype).min  # type: ignore[type-var]
    matrix = matrix.astype(dtype)

    grid = core.Grid(x_axis, y_axis, z_axis, u_axis, matrix)
    assert len(grid.x) == matrix.shape[0]
    assert len(grid.y) == matrix.shape[1]
    assert len(grid.z) == matrix.shape[2]
    assert len(grid.u) == matrix.shape[3]
    assert np.shares_memory(grid.array, matrix)
    assert "Grid" in repr(grid)
    np.testing.assert_array_equal(grid.array, matrix)
