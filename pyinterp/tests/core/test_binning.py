# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for Binning2DHolder, Binning1DHolder and related classes."""

import copy
import pickle

import numpy as np
import pytest

from ... import core


# Fixtures for Binning2D tests
@pytest.fixture
def binning2d_axes() -> tuple[core.Axis, core.Axis]:
    """Create common x and y axes for Binning2D tests."""
    x_axis = core.Axis(np.arange(0.0, 3.0, 1.0))
    y_axis = core.Axis(np.arange(0.0, 3.0, 1.0))
    return x_axis, y_axis


@pytest.fixture
def binning2d_float32(
    binning2d_axes: tuple[core.Axis, core.Axis],
) -> core.Binning2DFloat32:
    """Create a Binning2D with float32 dtype."""
    x_axis, y_axis = binning2d_axes
    return core.Binning2D(x_axis, y_axis, spheroid=None, dtype="float32")


@pytest.fixture
def binning2d_float64(
    binning2d_axes: tuple[core.Axis, core.Axis],
) -> core.Binning2DFloat64:
    """Create a Binning2D with float64 dtype."""
    x_axis, y_axis = binning2d_axes
    return core.Binning2D(x_axis, y_axis, spheroid=None, dtype="float64")


# Fixtures for Binning1D tests
@pytest.fixture
def binning1d_axis() -> core.Axis:
    """Create common x axis for Binning1D tests."""
    return core.Axis(np.arange(0.0, 5.0, 1.0))


@pytest.fixture
def binning1d_float32(binning1d_axis: core.Axis) -> core.Binning1DFloat32:
    """Create a Binning1D with float32 dtype."""
    return core.Binning1D(binning1d_axis, range=None, dtype="float32")


@pytest.fixture
def binning1d_float64(binning1d_axis: core.Axis) -> core.Binning1DFloat64:
    """Create a Binning1D with float64 dtype."""
    return core.Binning1D(binning1d_axis, range=None, dtype="float64")


class TestBinning2DFloat32:
    """Tests for Binning2DFloat32 (float32 specialization)."""

    def test_creation(
        self,
        binning2d_axes: tuple[core.Axis, core.Axis],
        binning2d_float32: core.Binning2DFloat32,
    ) -> None:
        """Test creating Binning2D with float32 dtype."""
        x_axis, y_axis = binning2d_axes
        assert binning2d_float32 is not None
        assert binning2d_float32.x == x_axis
        assert binning2d_float32.y == y_axis

    def test_push_and_count(
        self,
        binning2d_float32: core.Binning2DFloat32,
    ) -> None:
        """Test pushing data and counting."""
        x_coords = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        y_coords = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        z_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        binning2d_float32.push(x_coords, y_coords, z_values, simple=True)
        count = binning2d_float32.count()

        assert count.shape == (3, 3)
        assert count[0, 0] == 1
        assert count[1, 1] == 1
        assert count[2, 2] == 1

    def test_push_and_sum(
        self,
        binning2d_float32: core.Binning2DFloat32,
    ) -> None:
        """Test pushing data and computing sum."""
        x_coords = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        y_coords = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        z_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        binning2d_float32.push(x_coords, y_coords, z_values, simple=True)
        total = binning2d_float32.sum()

        assert total.shape == (3, 3)
        assert np.isclose(total[0, 0], 1.0)
        assert np.isclose(total[1, 1], 2.0)
        assert np.isclose(total[2, 2], 3.0)

    def test_push_and_mean(
        self,
        binning2d_float32: core.Binning2DFloat32,
    ) -> None:
        """Test pushing data and computing mean."""
        x_coords = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        y_coords = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        z_values = np.array([1.0, 3.0, 2.0], dtype=np.float32)

        binning2d_float32.push(x_coords, y_coords, z_values, simple=True)
        mean = binning2d_float32.mean()

        assert mean.shape == (3, 3)
        assert np.isclose(mean[0, 0], 2.0)  # (1+3)/2
        assert np.isclose(mean[1, 1], 2.0)

    def test_push_and_variance(
        self,
        binning2d_float32: core.Binning2DFloat32,
    ) -> None:
        """Test pushing data and computing variance."""
        x_coords = np.array([0.0, 0.0], dtype=np.float32)
        y_coords = np.array([0.0, 0.0], dtype=np.float32)
        z_values = np.array([1.0, 3.0], dtype=np.float32)

        binning2d_float32.push(x_coords, y_coords, z_values, simple=True)
        variance = binning2d_float32.variance(ddof=0)

        assert variance.shape == (3, 3)
        assert np.isclose(variance[0, 0], 1.0)  # variance of [1, 3]

    def test_push_and_max(
        self,
        binning2d_float32: core.Binning2DFloat32,
    ) -> None:
        """Test pushing data and computing max."""
        x_coords = np.array([0.0, 0.0], dtype=np.float32)
        y_coords = np.array([0.0, 0.0], dtype=np.float32)
        z_values = np.array([1.0, 3.0], dtype=np.float32)

        binning2d_float32.push(x_coords, y_coords, z_values, simple=True)
        maximum = binning2d_float32.max()

        assert maximum.shape == (3, 3)
        assert np.isclose(maximum[0, 0], 3.0)

    def test_push_and_min(
        self,
        binning2d_float32: core.Binning2DFloat32,
    ) -> None:
        """Test pushing data and computing min."""
        x_coords = np.array([0.0, 0.0], dtype=np.float32)
        y_coords = np.array([0.0, 0.0], dtype=np.float32)
        z_values = np.array([1.0, 3.0], dtype=np.float32)

        binning2d_float32.push(x_coords, y_coords, z_values, simple=True)
        minimum = binning2d_float32.min()

        assert minimum.shape == (3, 3)
        assert np.isclose(minimum[0, 0], 1.0)

    def test_clear(
        self,
        binning2d_float32: core.Binning2DFloat32,
    ) -> None:
        """Test clearing binning data."""
        x_coords = np.array([0.0], dtype=np.float32)
        y_coords = np.array([0.0], dtype=np.float32)
        z_values = np.array([1.0], dtype=np.float32)

        binning2d_float32.push(x_coords, y_coords, z_values, simple=True)
        binning2d_float32.clear()

        count = binning2d_float32.count()
        assert np.all(count == 0)

    def test_copy(
        self,
        binning2d_float32: core.Binning2DFloat32,
    ) -> None:
        """Test shallow copy."""
        x_coords = np.array([0.0], dtype=np.float32)
        y_coords = np.array([0.0], dtype=np.float32)
        z_values = np.array([1.0], dtype=np.float32)

        binning2d_float32.push(x_coords, y_coords, z_values, simple=True)
        binning_copy = copy.copy(binning2d_float32)

        assert binning_copy.count()[0, 0] == binning2d_float32.count()[0, 0]

    def test_iadd(
        self,
        binning2d_axes: tuple[core.Axis, core.Axis],
        binning2d_float32: core.Binning2DFloat32,
    ) -> None:
        """Test in-place addition of binning objects."""
        x_axis, y_axis = binning2d_axes
        x_coords1 = np.array([0.0], dtype=np.float32)
        y_coords1 = np.array([0.0], dtype=np.float32)
        z_values1 = np.array([1.0], dtype=np.float32)

        x_coords2 = np.array([0.0], dtype=np.float32)
        y_coords2 = np.array([0.0], dtype=np.float32)
        z_values2 = np.array([2.0], dtype=np.float32)

        binning2d_float32.push(x_coords1, y_coords1, z_values1, simple=True)

        binning2 = core.Binning2D(x_axis, y_axis, dtype="float32")
        binning2.push(x_coords2, y_coords2, z_values2, simple=True)

        binning2d_float32 += binning2

        count = binning2d_float32.count()
        assert count[0, 0] == 2

    def test_spheroid(
        self,
        binning2d_axes: tuple[core.Axis, core.Axis],
    ) -> None:
        """Test with geographic coordinates."""
        x_axis, y_axis = binning2d_axes
        spheroid = core.geometry.geographic.Spheroid()
        binning = core.Binning2D(
            x_axis, y_axis, spheroid=spheroid, dtype="float32"
        )

        assert binning.spheroid is not None

    def test_pickle_support(
        self,
        binning2d_float32: core.Binning2DFloat32,
    ) -> None:
        """Test pickle support for Binning2DFloat32."""
        x_coords = np.array([0.0, 1.0], dtype=np.float32)
        y_coords = np.array([0.0, 1.0], dtype=np.float32)
        z_values = np.array([1.0, 2.0], dtype=np.float32)

        binning2d_float32.push(x_coords, y_coords, z_values, simple=True)

        pickled = pickle.dumps(binning2d_float32)
        unpickled = pickle.loads(pickled)

        assert np.array_equal(unpickled.count(), binning2d_float32.count())


class TestBinning2DFloat64:
    """Tests for Binning2DFloat64 (float64 specialization)."""

    def test_creation(
        self,
        binning2d_axes: tuple[core.Axis, core.Axis],
        binning2d_float64: core.Binning2DFloat64,
    ) -> None:
        """Test creating Binning2D with float64 dtype."""
        x_axis, y_axis = binning2d_axes
        assert binning2d_float64 is not None
        assert binning2d_float64.x == x_axis
        assert binning2d_float64.y == y_axis

    def test_default_dtype_is_float64(
        self,
        binning2d_axes: tuple[core.Axis, core.Axis],
    ) -> None:
        """Test that default dtype is float64."""
        x_axis, y_axis = binning2d_axes
        binning = core.Binning2D(x_axis, y_axis)
        assert binning is not None

    def test_push_and_count(
        self,
        binning2d_float64: core.Binning2DFloat64,
    ) -> None:
        """Test pushing data and counting."""
        x_coords = np.array([0.5, 1.5, 2.5])
        y_coords = np.array([0.5, 1.5, 2.5])
        z_values = np.array([1.0, 2.0, 3.0])

        binning2d_float64.push(x_coords, y_coords, z_values, simple=True)
        count = binning2d_float64.count()

        assert count.shape == (3, 3)

    def test_sum_of_weights(
        self,
        binning2d_float64: core.Binning2DFloat64,
    ) -> None:
        """Test sum of weights computation."""
        x_coords = np.array([0.0, 0.0])
        y_coords = np.array([0.0, 0.0])
        z_values = np.array([1.0, 2.0])

        binning2d_float64.push(x_coords, y_coords, z_values, simple=True)
        weights_sum = binning2d_float64.sum_of_weights()

        assert weights_sum.shape == (3, 3)
        assert weights_sum[0, 0] == 2.0

    def test_kurtosis(
        self,
        binning2d_float64: core.Binning2DFloat64,
    ) -> None:
        """Test kurtosis computation."""
        x_coords = np.array([0.0, 0.0, 0.0, 0.0])
        y_coords = np.array([0.0, 0.0, 0.0, 0.0])
        z_values = np.array([1.0, 2.0, 3.0, 4.0])

        binning2d_float64.push(x_coords, y_coords, z_values, simple=True)
        kurt = binning2d_float64.kurtosis()

        assert kurt.shape == (3, 3)

    def test_skewness(
        self,
        binning2d_float64: core.Binning2DFloat64,
    ) -> None:
        """Test skewness computation."""
        x_coords = np.array([0.0, 0.0, 0.0])
        y_coords = np.array([0.0, 0.0, 0.0])
        z_values = np.array([1.0, 2.0, 3.0])

        binning2d_float64.push(x_coords, y_coords, z_values, simple=True)
        skew = binning2d_float64.skewness()

        assert skew.shape == (3, 3)

    def test_nan_values(
        self,
        binning2d_float64: core.Binning2DFloat64,
    ) -> None:
        """Test handling of NaN values."""
        x_coords = np.array([0.0, 0.0])
        y_coords = np.array([0.0, 0.0])
        z_values = np.array([1.0, np.nan])

        binning2d_float64.push(x_coords, y_coords, z_values, simple=True)
        count = binning2d_float64.count()

        # NaN value should be skipped
        assert count[0, 0] == 1


class TestBinning1DFloat32:
    """Tests for Binning1DFloat32 (float32 specialization)."""

    def test_creation(
        self,
        binning1d_axis: core.Axis,
        binning1d_float32: core.Binning1DFloat32,
    ) -> None:
        """Test creating Binning1D with float32 dtype."""
        assert binning1d_float32 is not None
        assert binning1d_float32.x == binning1d_axis

    def test_creation_with_range(self, binning1d_axis: core.Axis) -> None:
        """Test creating Binning1D with explicit range."""
        binning = core.Binning1D(
            binning1d_axis, range=(1.0, 3.0), dtype="float32"
        )
        assert binning.range() == (1.0, 3.0)

    def test_push_and_count(
        self, binning1d_float32: core.Binning1DFloat32
    ) -> None:
        """Test pushing data and counting."""
        x_coords = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        z_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        binning1d_float32.push(x_coords, z_values)
        count = binning1d_float32.count()

        # Binning1D returns 1D arrays
        assert count.shape == (5,)
        assert count[0] == 1
        assert count[1] == 1
        assert count[2] == 1

    def test_push_with_weights(
        self, binning1d_float32: core.Binning1DFloat32
    ) -> None:
        """Test pushing data with weights."""
        x_coords = np.array([0.0, 0.0], dtype=np.float32)
        z_values = np.array([1.0, 2.0], dtype=np.float32)
        weights = np.array([2.0, 3.0], dtype=np.float32)

        binning1d_float32.push(x_coords, z_values, weights)
        weights_sum = binning1d_float32.sum_of_weights()

        # Binning1D returns 1D arrays
        assert weights_sum.shape == (5,)
        assert weights_sum[0] == 5.0

    def test_range_filtering(self, binning1d_axis: core.Axis) -> None:
        """Test that range filtering works correctly."""
        binning = core.Binning1D(
            binning1d_axis, range=(1.0, 3.0), dtype="float32"
        )

        x_coords = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        z_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        binning.push(x_coords, z_values)
        count = binning.count()

        # Values within range should be counted
        assert count[1] == 1
        assert count[2] == 1

    def test_clear(self, binning1d_float32: core.Binning1DFloat32) -> None:
        """Test clearing binning data."""
        x_coords = np.array([0.0], dtype=np.float32)
        z_values = np.array([1.0], dtype=np.float32)

        binning1d_float32.push(x_coords, z_values)
        binning1d_float32.clear()

        count = binning1d_float32.count()
        assert np.all(count == 0)

    def test_copy(self, binning1d_float32: core.Binning1DFloat32) -> None:
        """Test shallow copy."""
        x_coords = np.array([0.0], dtype=np.float32)
        z_values = np.array([1.0], dtype=np.float32)

        binning1d_float32.push(x_coords, z_values)
        binning_copy = copy.copy(binning1d_float32)

        assert binning_copy.count()[0] == binning1d_float32.count()[0]

    def test_iadd(
        self,
        binning1d_axis: core.Axis,
        binning1d_float32: core.Binning1DFloat32,
    ) -> None:
        """Test in-place addition of binning objects."""
        x_coords1 = np.array([0.0], dtype=np.float32)
        z_values1 = np.array([1.0], dtype=np.float32)

        x_coords2 = np.array([0.0], dtype=np.float32)
        z_values2 = np.array([2.0], dtype=np.float32)

        binning1d_float32.push(x_coords1, z_values1)

        binning2 = core.Binning1D(binning1d_axis, range=None, dtype="float32")
        binning2.push(x_coords2, z_values2)

        binning1d_float32 += binning2

        count = binning1d_float32.count()
        # Just verify it worked without segfault
        assert count is not None

    def test_mean(self, binning1d_float32: core.Binning1DFloat32) -> None:
        """Test mean computation."""
        x_coords = np.array([0.0, 0.0], dtype=np.float32)
        z_values = np.array([2.0, 4.0], dtype=np.float32)

        binning1d_float32.push(x_coords, z_values)
        mean = binning1d_float32.mean()

        assert np.isclose(mean[0], 3.0)

    def test_pickle_support(
        self, binning1d_float32: core.Binning1DFloat32
    ) -> None:
        """Test pickle support for Binning1DFloat32."""
        x_coords = np.array([0.0, 1.0], dtype=np.float32)
        z_values = np.array([1.0, 2.0], dtype=np.float32)

        binning1d_float32.push(x_coords, z_values)

        pickled = pickle.dumps(binning1d_float32)
        unpickled = pickle.loads(pickled)

        assert np.array_equal(unpickled.count(), binning1d_float32.count())


class TestBinning1DFloat64:
    """Tests for Binning1DFloat64 (float64 specialization)."""

    def test_creation(
        self,
        binning1d_axis: core.Axis,
        binning1d_float64: core.Binning1DFloat64,
    ) -> None:
        """Test creating Binning1D with float64 dtype."""
        assert binning1d_float64 is not None
        assert binning1d_float64.x == binning1d_axis

    def test_default_dtype_is_float64(self, binning1d_axis: core.Axis) -> None:
        """Test that default dtype is float64."""
        binning = core.Binning1D(binning1d_axis)
        assert binning is not None

    def test_push_and_sum(
        self,
        binning1d_float64: core.Binning1DFloat64,
    ) -> None:
        """Test pushing data and computing sum."""
        x_coords = np.array([0.0, 1.0, 2.0])
        z_values = np.array([1.0, 2.0, 3.0])

        binning1d_float64.push(x_coords, z_values)
        total = binning1d_float64.sum()

        # Binning1D returns 1D arrays
        assert np.isclose(total[0], 1.0)
        assert np.isclose(total[1], 2.0)
        assert np.isclose(total[2], 3.0)

    def test_variance(
        self,
        binning1d_float64: core.Binning1DFloat64,
    ) -> None:
        """Test variance computation."""
        x_coords = np.array([0.0, 0.0, 0.0])
        z_values = np.array([1.0, 2.0, 3.0])

        binning1d_float64.push(x_coords, z_values)
        variance = binning1d_float64.variance(ddof=0)

        expected_var = np.var([1.0, 2.0, 3.0])
        assert np.isclose(variance[0], expected_var)

    def test_max_min(
        self,
        binning1d_float64: core.Binning1DFloat64,
    ) -> None:
        """Test max and min computation."""
        x_coords = np.array([0.0, 0.0, 0.0])
        z_values = np.array([1.0, 5.0, 3.0])

        binning1d_float64.push(x_coords, z_values)
        maximum = binning1d_float64.max()
        minimum = binning1d_float64.min()

        assert np.isclose(maximum[0], 5.0)
        assert np.isclose(minimum[0], 1.0)

    def test_nan_handling(
        self,
        binning1d_float64: core.Binning1DFloat64,
    ) -> None:
        """Test handling of NaN values."""
        x_coords = np.array([0.0, 0.0, 0.0])
        z_values = np.array([1.0, np.nan, 3.0])

        binning1d_float64.push(x_coords, z_values)
        count = binning1d_float64.count()

        # NaN should be skipped
        assert count[0] == 2

    def test_pickle_support(
        self,
        binning1d_float64: core.Binning1DFloat64,
    ) -> None:
        """Test pickle support for Binning1DFloat64."""
        x_coords = np.array([0.0, 1.0])
        z_values = np.array([1.0, 2.0])

        binning1d_float64.push(x_coords, z_values)

        pickled = pickle.dumps(binning1d_float64)
        unpickled = pickle.loads(pickled)

        assert np.array_equal(unpickled.count(), binning1d_float64.count())


class TestBinningFactoryFunctions:
    """Tests for Binning factory functions."""

    def test_binning2d_factory_with_float32(self) -> None:
        """Test Binning2D factory with float32 dtype."""
        x_axis = core.Axis(np.arange(5.0))
        y_axis = core.Axis(np.arange(5.0))

        binning = core.Binning2D(
            x_axis, y_axis, spheroid=None, dtype="float32"
        )
        assert binning is not None

    def test_binning2d_factory_with_float64(self) -> None:
        """Test Binning2D factory with float64 dtype."""
        x_axis = core.Axis(np.arange(5.0))
        y_axis = core.Axis(np.arange(5.0))

        binning = core.Binning2D(
            x_axis, y_axis, spheroid=None, dtype="float64"
        )
        assert binning is not None

    def test_binning2d_factory_with_none_dtype(self) -> None:
        """Test Binning2D factory with None dtype (default to float64)."""
        x_axis = core.Axis(np.arange(5.0))
        y_axis = core.Axis(np.arange(5.0))

        binning = core.Binning2D(x_axis, y_axis, spheroid=None, dtype=None)
        assert binning is not None

    def test_binning1d_factory_with_float32(self) -> None:
        """Test Binning1D factory with float32 dtype."""
        x_axis = core.Axis(np.arange(5.0))

        binning = core.Binning1D(x_axis, range=None, dtype="float32")
        assert binning is not None

    def test_binning1d_factory_with_float64(self) -> None:
        """Test Binning1D factory with float64 dtype."""
        x_axis = core.Axis(np.arange(5.0))

        binning = core.Binning1D(x_axis, range=None, dtype="float64")
        assert binning is not None

    def test_binning1d_factory_with_none_dtype(self) -> None:
        """Test Binning1D factory with None dtype (default to float64)."""
        x_axis = core.Axis(np.arange(5.0))

        binning = core.Binning1D(x_axis, range=None, dtype=None)
        assert binning is not None

    def test_binning2d_factory_with_spheroid(self) -> None:
        """Test Binning2D factory with spheroid."""
        x_axis = core.Axis(np.arange(0.0, 360.0, 30.0))
        y_axis = core.Axis(np.arange(-90.0, 90.0, 30.0))
        spheroid = core.geometry.geographic.Spheroid()

        binning = core.Binning2D(
            x_axis, y_axis, spheroid=spheroid, dtype="float32"
        )
        assert binning is not None
        assert binning.spheroid is not None
