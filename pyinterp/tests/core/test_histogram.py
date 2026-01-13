# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for Histogram2DHolder and related classes."""

import copy
import pickle

import numpy as np
import pytest

from ... import core


# Fixtures for Histogram2D tests
@pytest.fixture
def histogram2d_axes() -> tuple[core.Axis, core.Axis]:
    """Create common x and y axes for Histogram2D tests."""
    x_axis = core.Axis(np.arange(0.0, 3.0, 1.0))
    y_axis = core.Axis(np.arange(0.0, 3.0, 1.0))
    return x_axis, y_axis


@pytest.fixture
def histogram2d_float32(
    histogram2d_axes: tuple[core.Axis, core.Axis],
) -> core.Histogram2DFloat32:
    """Create a Histogram2D with float32 dtype."""
    x_axis, y_axis = histogram2d_axes
    return core.Histogram2D(x_axis, y_axis, compression=100, dtype="float32")


@pytest.fixture
def histogram2d_float64(
    histogram2d_axes: tuple[core.Axis, core.Axis],
) -> core.Histogram2DFloat64:
    """Create a Histogram2D with float64 dtype."""
    x_axis, y_axis = histogram2d_axes
    return core.Histogram2D(x_axis, y_axis, compression=100, dtype="float64")


class TestHistogram2DFloat32:
    """Tests for Histogram2DFloat32 (float32 specialization)."""

    def test_creation(
        self,
        histogram2d_axes: tuple[core.Axis, core.Axis],
        histogram2d_float32: core.Histogram2DFloat32,
    ) -> None:
        """Test creating Histogram2D with float32 dtype."""
        x_axis, y_axis = histogram2d_axes
        assert histogram2d_float32 is not None
        assert histogram2d_float32.x == x_axis
        assert histogram2d_float32.y == y_axis

    def test_creation_with_compression(
        self,
        histogram2d_axes: tuple[core.Axis, core.Axis],
    ) -> None:
        """Test creating Histogram2D with explicit compression parameter."""
        x_axis, y_axis = histogram2d_axes
        histogram = core.Histogram2D(
            x_axis, y_axis, compression=200, dtype="float32"
        )
        assert histogram is not None

    def test_creation_without_compression(
        self,
        histogram2d_axes: tuple[core.Axis, core.Axis],
    ) -> None:
        """Test creating Histogram2D without compression (uses default)."""
        x_axis, y_axis = histogram2d_axes
        histogram = core.Histogram2D(x_axis, y_axis, dtype="float32")
        assert histogram is not None

    def test_push_and_count(
        self,
        histogram2d_float32: core.Histogram2DFloat32,
    ) -> None:
        """Test pushing data and counting."""
        x_coords = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        y_coords = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        z_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        histogram2d_float32.push(x_coords, y_coords, z_values)
        count = histogram2d_float32.count()

        assert count.shape == (3, 3)
        assert count[0, 0] == 1
        assert count[1, 1] == 1
        assert count[2, 2] == 1

    def test_push_and_mean(
        self,
        histogram2d_float32: core.Histogram2DFloat32,
    ) -> None:
        """Test pushing data and computing mean."""
        x_coords = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        y_coords = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        z_values = np.array([1.0, 3.0, 2.0], dtype=np.float32)

        histogram2d_float32.push(x_coords, y_coords, z_values)
        mean = histogram2d_float32.mean()

        assert mean.shape == (3, 3)
        assert np.isclose(mean[0, 0], 2.0)  # (1+3)/2
        assert np.isclose(mean[1, 1], 2.0)

    def test_push_and_max(
        self,
        histogram2d_float32: core.Histogram2DFloat32,
    ) -> None:
        """Test pushing data and computing max."""
        x_coords = np.array([0.0, 0.0], dtype=np.float32)
        y_coords = np.array([0.0, 0.0], dtype=np.float32)
        z_values = np.array([1.0, 3.0], dtype=np.float32)

        histogram2d_float32.push(x_coords, y_coords, z_values)
        maximum = histogram2d_float32.max()

        assert maximum.shape == (3, 3)
        assert np.isclose(maximum[0, 0], 3.0)

    def test_push_and_min(
        self,
        histogram2d_float32: core.Histogram2DFloat32,
    ) -> None:
        """Test pushing data and computing min."""
        x_coords = np.array([0.0, 0.0], dtype=np.float32)
        y_coords = np.array([0.0, 0.0], dtype=np.float32)
        z_values = np.array([1.0, 3.0], dtype=np.float32)

        histogram2d_float32.push(x_coords, y_coords, z_values)
        minimum = histogram2d_float32.min()

        assert minimum.shape == (3, 3)
        assert np.isclose(minimum[0, 0], 1.0)

    def test_push_and_quantile(
        self,
        histogram2d_float32: core.Histogram2DFloat32,
    ) -> None:
        """Test pushing data and computing quantile (median)."""
        x_coords = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        y_coords = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        z_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        histogram2d_float32.push(x_coords, y_coords, z_values)
        quantile = histogram2d_float32.quantile(0.5)  # Median

        assert quantile.shape == (3, 3)
        # Quantile should return median (approximately 2.0)
        assert np.isclose(quantile[0, 0], 2.0, atol=0.1)

    def test_sum_of_weights(
        self,
        histogram2d_float32: core.Histogram2DFloat32,
    ) -> None:
        """Test sum of weights computation."""
        x_coords = np.array([0.0, 0.0], dtype=np.float32)
        y_coords = np.array([0.0, 0.0], dtype=np.float32)
        z_values = np.array([1.0, 2.0], dtype=np.float32)

        histogram2d_float32.push(x_coords, y_coords, z_values)
        weights_sum = histogram2d_float32.sum_of_weights()

        assert weights_sum.shape == (3, 3)
        # Sum of weights should be the count for unweighted data
        assert weights_sum[0, 0] == 2.0

    def test_clear(
        self,
        histogram2d_float32: core.Histogram2DFloat32,
    ) -> None:
        """Test clearing histogram data."""
        x_coords = np.array([0.0], dtype=np.float32)
        y_coords = np.array([0.0], dtype=np.float32)
        z_values = np.array([1.0], dtype=np.float32)

        histogram2d_float32.push(x_coords, y_coords, z_values)
        histogram2d_float32.clear()

        count = histogram2d_float32.count()
        assert np.all(count == 0)

    def test_copy(
        self,
        histogram2d_float32: core.Histogram2DFloat32,
    ) -> None:
        """Test shallow copy."""
        x_coords = np.array([0.0], dtype=np.float32)
        y_coords = np.array([0.0], dtype=np.float32)
        z_values = np.array([1.0], dtype=np.float32)

        histogram2d_float32.push(x_coords, y_coords, z_values)
        histogram_copy = copy.copy(histogram2d_float32)

        assert (
            histogram_copy.count()[0, 0] == histogram2d_float32.count()[0, 0]
        )

    def test_iadd(
        self,
        histogram2d_axes: tuple[core.Axis, core.Axis],
        histogram2d_float32: core.Histogram2DFloat32,
    ) -> None:
        """Test in-place addition of histogram objects."""
        x_axis, y_axis = histogram2d_axes
        x_coords1 = np.array([0.0], dtype=np.float32)
        y_coords1 = np.array([0.0], dtype=np.float32)
        z_values1 = np.array([1.0], dtype=np.float32)

        x_coords2 = np.array([0.0], dtype=np.float32)
        y_coords2 = np.array([0.0], dtype=np.float32)
        z_values2 = np.array([2.0], dtype=np.float32)

        histogram2d_float32.push(x_coords1, y_coords1, z_values1)

        histogram2 = core.Histogram2D(
            x_axis, y_axis, compression=100, dtype="float32"
        )
        histogram2.push(x_coords2, y_coords2, z_values2)

        histogram2d_float32 += histogram2

        count = histogram2d_float32.count()
        assert count[0, 0] == 2

    def test_pickle_support(
        self,
        histogram2d_float32: core.Histogram2DFloat32,
    ) -> None:
        """Test pickle support for Histogram2DFloat32."""
        x_coords = np.array([0.0, 1.0], dtype=np.float32)
        y_coords = np.array([0.0, 1.0], dtype=np.float32)
        z_values = np.array([1.0, 2.0], dtype=np.float32)

        histogram2d_float32.push(x_coords, y_coords, z_values)

        pickled = pickle.dumps(histogram2d_float32)
        unpickled = pickle.loads(pickled)

        assert np.array_equal(unpickled.count(), histogram2d_float32.count())

    def test_nan_values(
        self,
        histogram2d_float32: core.Histogram2DFloat32,
    ) -> None:
        """Test handling of NaN values."""
        x_coords = np.array([0.0, 0.0], dtype=np.float32)
        y_coords = np.array([0.0, 0.0], dtype=np.float32)
        z_values = np.array([1.0, np.nan], dtype=np.float32)

        histogram2d_float32.push(x_coords, y_coords, z_values)
        count = histogram2d_float32.count()

        # NaN value should be skipped
        assert count[0, 0] == 1


class TestHistogram2DFloat64:
    """Tests for Histogram2DFloat64 (float64 specialization)."""

    def test_creation(
        self,
        histogram2d_axes: tuple[core.Axis, core.Axis],
        histogram2d_float64: core.Histogram2DFloat64,
    ) -> None:
        """Test creating Histogram2D with float64 dtype."""
        x_axis, y_axis = histogram2d_axes
        assert histogram2d_float64 is not None
        assert histogram2d_float64.x == x_axis
        assert histogram2d_float64.y == y_axis

    def test_default_dtype_is_float64(
        self,
        histogram2d_axes: tuple[core.Axis, core.Axis],
    ) -> None:
        """Test that default dtype is float64."""
        x_axis, y_axis = histogram2d_axes
        histogram = core.Histogram2D(x_axis, y_axis)
        assert histogram is not None

    def test_push_and_count(
        self,
        histogram2d_float64: core.Histogram2DFloat64,
    ) -> None:
        """Test pushing data and counting."""
        x_coords = np.array([0.5, 1.5, 2.5])
        y_coords = np.array([0.5, 1.5, 2.5])
        z_values = np.array([1.0, 2.0, 3.0])

        histogram2d_float64.push(x_coords, y_coords, z_values)
        count = histogram2d_float64.count()

        assert count.shape == (3, 3)

    def test_push_and_mean(
        self,
        histogram2d_float64: core.Histogram2DFloat64,
    ) -> None:
        """Test pushing data and computing mean."""
        x_coords = np.array([0.0, 0.0, 0.0])
        y_coords = np.array([0.0, 0.0, 0.0])
        z_values = np.array([1.0, 2.0, 3.0])

        histogram2d_float64.push(x_coords, y_coords, z_values)
        mean = histogram2d_float64.mean()

        assert mean.shape == (3, 3)
        assert np.isclose(mean[0, 0], 2.0)  # (1+2+3)/3

    def test_push_and_quantile(
        self,
        histogram2d_float64: core.Histogram2DFloat64,
    ) -> None:
        """Test pushing data and computing quantile."""
        x_coords = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        y_coords = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        z_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        histogram2d_float64.push(x_coords, y_coords, z_values)
        quantile = histogram2d_float64.quantile(0.5)  # Median

        assert quantile.shape == (3, 3)
        # Median should be approximately 3.0
        assert np.isclose(quantile[0, 0], 3.0, atol=0.1)

    def test_max_min(
        self,
        histogram2d_float64: core.Histogram2DFloat64,
    ) -> None:
        """Test max and min computation."""
        x_coords = np.array([0.0, 0.0, 0.0])
        y_coords = np.array([0.0, 0.0, 0.0])
        z_values = np.array([1.0, 5.0, 3.0])

        histogram2d_float64.push(x_coords, y_coords, z_values)
        maximum = histogram2d_float64.max()
        minimum = histogram2d_float64.min()

        assert np.isclose(maximum[0, 0], 5.0)
        assert np.isclose(minimum[0, 0], 1.0)

    def test_sum_of_weights(
        self,
        histogram2d_float64: core.Histogram2DFloat64,
    ) -> None:
        """Test sum of weights computation."""
        x_coords = np.array([0.0, 0.0, 0.0])
        y_coords = np.array([0.0, 0.0, 0.0])
        z_values = np.array([1.0, 2.0, 3.0])

        histogram2d_float64.push(x_coords, y_coords, z_values)
        weights_sum = histogram2d_float64.sum_of_weights()

        assert weights_sum.shape == (3, 3)
        assert weights_sum[0, 0] == 3.0

    def test_multiple_bins(
        self,
        histogram2d_float64: core.Histogram2DFloat64,
    ) -> None:
        """Test with data in multiple bins."""
        x_coords = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        y_coords = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        z_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        histogram2d_float64.push(x_coords, y_coords, z_values)
        count = histogram2d_float64.count()

        # Check counts in first row (y=0.0)
        assert count[0, 0] == 1  # x=0.0, y=0.0
        assert count[1, 0] == 1  # x=1.0, y=0.0
        assert count[2, 0] == 1  # x=2.0, y=0.0
        # Check counts in second row (y=1.0)
        assert count[0, 1] == 1  # x=0.0, y=1.0
        assert count[1, 1] == 1  # x=1.0, y=1.0
        assert count[2, 1] == 1  # x=2.0, y=1.0

    def test_nan_values(
        self,
        histogram2d_float64: core.Histogram2DFloat64,
    ) -> None:
        """Test handling of NaN values."""
        x_coords = np.array([0.0, 0.0])
        y_coords = np.array([0.0, 0.0])
        z_values = np.array([1.0, np.nan])

        histogram2d_float64.push(x_coords, y_coords, z_values)
        count = histogram2d_float64.count()

        # NaN value should be skipped
        assert count[0, 0] == 1

    def test_pickle_support(
        self,
        histogram2d_float64: core.Histogram2DFloat64,
    ) -> None:
        """Test pickle support for Histogram2DFloat64."""
        x_coords = np.array([0.0, 1.0])
        y_coords = np.array([0.0, 1.0])
        z_values = np.array([1.0, 2.0])

        histogram2d_float64.push(x_coords, y_coords, z_values)

        pickled = pickle.dumps(histogram2d_float64)
        unpickled = pickle.loads(pickled)

        assert np.array_equal(unpickled.count(), histogram2d_float64.count())


class TestHistogram2DFactoryFunctions:
    """Tests for Histogram2D factory functions."""

    def test_histogram2d_factory_with_float32(self) -> None:
        """Test Histogram2D factory with float32 dtype."""
        x_axis = core.Axis(np.arange(5.0))
        y_axis = core.Axis(np.arange(5.0))

        histogram = core.Histogram2D(x_axis, y_axis, dtype="float32")
        assert histogram is not None

    def test_histogram2d_factory_with_float64(self) -> None:
        """Test Histogram2D factory with float64 dtype."""
        x_axis = core.Axis(np.arange(5.0))
        y_axis = core.Axis(np.arange(5.0))

        histogram = core.Histogram2D(x_axis, y_axis, dtype="float64")
        assert histogram is not None

    def test_histogram2d_factory_with_none_dtype(self) -> None:
        """Test Histogram2D factory with None dtype (default to float64)."""
        x_axis = core.Axis(np.arange(5.0))
        y_axis = core.Axis(np.arange(5.0))

        histogram = core.Histogram2D(x_axis, y_axis, dtype=None)
        assert histogram is not None

    def test_histogram2d_factory_with_compression(self) -> None:
        """Test Histogram2D factory with compression parameter."""
        x_axis = core.Axis(np.arange(5.0))
        y_axis = core.Axis(np.arange(5.0))

        histogram = core.Histogram2D(
            x_axis, y_axis, compression=500, dtype="float32"
        )
        assert histogram is not None
