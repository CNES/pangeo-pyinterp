# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for rtree interpolation wrapper functions.

This module provides comprehensive tests for the interpolation wrapper functions
that support both RTree3DHolder (Cartesian) and geographic.RTree (spherical)
coordinate systems.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pyinterp import RTree3D
from pyinterp.core.config import rtree
from pyinterp.core.geometry.geographic import RTree as GeographicRTree
from pyinterp.rtree import (
    inverse_distance_weighting,
    kriging,
    query,
    radial_basis_function,
    window_function,
)


if TYPE_CHECKING:
    from pyinterp.core import RTree3DHolder
    from pyinterp.type_hints import TwoDims


@pytest.fixture
def cartesian_rtree_2d() -> RTree3DHolder[np.float64]:
    """Create a 2D Cartesian RTree with sample data.

    Returns:
        Populated RTree3DHolder with 2D coordinates and values.

    """
    tree: RTree3DHolder[np.float64] = RTree3D(dtype="float64")

    # Create a 5x5 grid
    x = np.linspace(0.0, 4.0, 5)
    y = np.linspace(0.0, 4.0, 5)
    xx, yy = np.meshgrid(x, y)
    coordinates = np.column_stack([xx.ravel(), yy.ravel()])

    # Values as distance from origin
    values = np.sqrt(xx.ravel() ** 2 + yy.ravel() ** 2).astype(np.float64)

    tree.packing(coordinates, values)
    return tree


@pytest.fixture
def cartesian_rtree_3d() -> RTree3DHolder[np.float64]:
    """Create a 3D Cartesian RTree with sample data.

    Returns:
        Populated RTree3DHolder with 3D coordinates and values.

    """
    tree: RTree3DHolder[np.float64] = RTree3D(dtype="float64")

    # Create a 4x4x4 grid
    x = np.linspace(0.0, 3.0, 4)
    y = np.linspace(0.0, 3.0, 4)
    z = np.linspace(0.0, 3.0, 4)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    coordinates = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # Values as distance from origin
    values = np.sqrt(
        xx.ravel() ** 2 + yy.ravel() ** 2 + zz.ravel() ** 2
    ).astype(np.float64)

    tree.packing(coordinates, values)
    return tree


@pytest.fixture
def geographic_rtree() -> GeographicRTree:
    """Create a geographic RTree with sample data (N x 2 only).

    Returns:
        Populated geographic RTree with longitude/latitude coordinates.

    """
    tree = GeographicRTree()

    # Create points in European region
    longitudes = np.linspace(-5.0, 15.0, 5)
    latitudes = np.linspace(40.0, 50.0, 5)
    lons, lats = np.meshgrid(longitudes, latitudes)
    coordinates = np.column_stack([lons.ravel(), lats.ravel()])

    # Temperature-like values
    values = 10.0 + 0.5 * lons.ravel() + 0.3 * lats.ravel()
    values = values.astype(np.float64)

    tree.packing(coordinates, values)
    return tree


@pytest.fixture
def query_coords_2d() -> np.ndarray[TwoDims, np.dtype[np.float64]]:
    """Create query coordinates for 2D tests.

    Returns:
        Query points as (N, 2) array.

    """
    x = np.array([0.5, 1.5, 2.5, 3.5])
    y = np.array([0.5, 1.5, 2.5, 3.5])
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    return coords.astype(np.float64)


@pytest.fixture
def query_coords_3d() -> np.ndarray[TwoDims, np.dtype[np.float64]]:
    """Create query coordinates for 3D tests.

    Returns:
        Query points as (N, 3) array.

    """
    x = np.array([0.5, 1.5, 2.5])
    y = np.array([0.5, 1.5, 2.5])
    z = np.array([0.5, 1.5, 2.5])
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return coords.astype(np.float64)


@pytest.fixture
def query_coords_geographic() -> np.ndarray[TwoDims, np.dtype[np.float64]]:
    """Create query coordinates for geographic tests (N x 2 only).

    Returns:
        Query points as (N, 2) array representing (longitude, latitude).

    """
    longitudes = np.array([-2.0, 2.0, 8.0])
    latitudes = np.array([42.0, 45.0, 48.0])
    coords = np.column_stack([longitudes, latitudes])
    return coords.astype(np.float64)


class TestInverseDistanceWeighting:
    """Tests for inverse_distance_weighting interpolation function."""

    def test_idw_cartesian_2d_with_kwargs(
        self,
        cartesian_rtree_2d: RTree3DHolder[np.float64],
        query_coords_2d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test IDW interpolation on 2D Cartesian RTree using kwargs.

        Args:
            cartesian_rtree_2d: 2D Cartesian RTree fixture
            query_coords_2d: Query coordinates fixture

        """
        result, counts = inverse_distance_weighting(
            cartesian_rtree_2d,
            query_coords_2d,
            k=4,
            p=2,
        )

        assert isinstance(result, np.ndarray)
        assert isinstance(counts, np.ndarray)
        assert result.shape == (query_coords_2d.shape[0],)
        assert counts.shape == (query_coords_2d.shape[0],)
        assert np.all(counts > 0)
        assert np.all(np.isfinite(result))

    def test_idw_cartesian_3d_with_kwargs(
        self,
        cartesian_rtree_3d: RTree3DHolder[np.float64],
        query_coords_3d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test IDW interpolation on 3D Cartesian RTree using kwargs.

        Args:
            cartesian_rtree_3d: 3D Cartesian RTree fixture
            query_coords_3d: Query coordinates fixture

        """
        result, counts = inverse_distance_weighting(
            cartesian_rtree_3d,
            query_coords_3d,
            k=8,
            p=1,
        )

        assert result.shape == (query_coords_3d.shape[0],)
        assert counts.shape == (query_coords_3d.shape[0],)
        assert np.all(counts > 0)

    def test_idw_geographic_with_kwargs(
        self,
        geographic_rtree: GeographicRTree,
        query_coords_geographic: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test IDW interpolation on geographic RTree using kwargs.

        Args:
            geographic_rtree: Geographic RTree fixture
            query_coords_geographic: Query coordinates fixture

        """
        result, counts = inverse_distance_weighting(
            geographic_rtree,
            query_coords_geographic,
            k=5,
            p=2,
        )

        assert result.shape == (query_coords_geographic.shape[0],)
        assert np.all(counts > 0)

    def test_idw_with_config_object(
        self,
        cartesian_rtree_2d: RTree3DHolder[np.float64],
        query_coords_2d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test IDW with explicit config object.

        Args:
            cartesian_rtree_2d: 2D Cartesian RTree fixture
            query_coords_2d: Query coordinates fixture

        """
        config_obj = (
            rtree.InverseDistanceWeighting()
            .with_k(4)
            .with_p(2)
            .with_radius(10.0)
        )

        result, _counts = inverse_distance_weighting(
            cartesian_rtree_2d,
            query_coords_2d,
            config=config_obj,
        )

        assert result.shape == (query_coords_2d.shape[0],)
        assert np.all(np.isfinite(result))

    def test_idw_with_radius_and_boundary_check(
        self,
        cartesian_rtree_2d: RTree3DHolder[np.float64],
        query_coords_2d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test IDW with radius and boundary check constraints.

        Args:
            cartesian_rtree_2d: 2D Cartesian RTree fixture
            query_coords_2d: Query coordinates fixture

        """
        result, counts = inverse_distance_weighting(
            cartesian_rtree_2d,
            query_coords_2d,
            k=4,
            p=2,
            radius=5.0,
            boundary_check="convex_hull",
        )

        assert result.shape == (query_coords_2d.shape[0],)
        assert np.all(counts >= 0)

    def test_idw_with_num_threads(
        self,
        cartesian_rtree_2d: RTree3DHolder[np.float64],
        query_coords_2d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test IDW with explicit thread count.

        Args:
            cartesian_rtree_2d: 2D Cartesian RTree fixture
            query_coords_2d: Query coordinates fixture

        """
        result, _counts = inverse_distance_weighting(
            cartesian_rtree_2d,
            query_coords_2d,
            k=4,
            p=2,
            num_threads=2,
        )

        assert result.shape == (query_coords_2d.shape[0],)


class TestRadialBasisFunction:
    """Tests for radial_basis_function interpolation function."""

    def test_rbf_cartesian_2d_with_kwargs(
        self,
        cartesian_rtree_2d: RTree3DHolder[np.float64],
        query_coords_2d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test RBF interpolation on 2D Cartesian RTree using kwargs.

        Args:
            cartesian_rtree_2d: 2D Cartesian RTree fixture
            query_coords_2d: Query coordinates fixture

        """
        result, counts = radial_basis_function(
            cartesian_rtree_2d,
            query_coords_2d,
            k=10,
            rbf="thin_plate",
        )

        assert result.shape == (query_coords_2d.shape[0],)
        assert counts.shape == (query_coords_2d.shape[0],)
        assert np.all(np.isfinite(result))

    def test_rbf_cartesian_3d_with_kwargs(
        self,
        cartesian_rtree_3d: RTree3DHolder[np.float64],
        query_coords_3d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test RBF interpolation on 3D Cartesian RTree using kwargs.

        Args:
            cartesian_rtree_3d: 3D Cartesian RTree fixture
            query_coords_3d: Query coordinates fixture

        """
        result, _counts = radial_basis_function(
            cartesian_rtree_3d,
            query_coords_3d,
            k=15,
            rbf="gaussian",
        )

        assert result.shape == (query_coords_3d.shape[0],)

    def test_rbf_geographic_with_kwargs(
        self,
        geographic_rtree: GeographicRTree,
        query_coords_geographic: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test RBF interpolation on geographic RTree using kwargs.

        Args:
            geographic_rtree: Geographic RTree fixture
            query_coords_geographic: Query coordinates fixture

        """
        result, _counts = radial_basis_function(
            geographic_rtree,
            query_coords_geographic,
            k=8,
            rbf="multiquadric",
        )

        assert result.shape == (query_coords_geographic.shape[0],)

    def test_rbf_with_config_object(
        self,
        cartesian_rtree_2d: RTree3DHolder[np.float64],
        query_coords_2d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test RBF with explicit config object.

        Args:
            cartesian_rtree_2d: 2D Cartesian RTree fixture
            query_coords_2d: Query coordinates fixture

        """
        config_obj = (
            rtree.RadialBasisFunction()
            .with_k(10)
            .with_rbf(rtree.RBFKernel.THIN_PLATE)
            .with_smooth(0.1)
        )

        result, _counts = radial_basis_function(
            cartesian_rtree_2d,
            query_coords_2d,
            config=config_obj,
        )

        assert result.shape == (query_coords_2d.shape[0],)

    @pytest.mark.parametrize(
        "rbf_kernel",
        [
            "cubic",
            "gaussian",
            "inverse_multiquadric",
            "linear",
            "multiquadric",
            "thin_plate",
        ],
    )
    def test_rbf_all_kernels(
        self,
        cartesian_rtree_2d: RTree3DHolder[np.float64],
        query_coords_2d: np.ndarray[TwoDims, np.dtype[np.float64]],
        rbf_kernel: str,
    ) -> None:
        """Test RBF with all available kernels.

        Args:
            cartesian_rtree_2d: 2D Cartesian RTree fixture
            query_coords_2d: Query coordinates fixture
            rbf_kernel: Kernel type to test

        """
        result, _counts = radial_basis_function(
            cartesian_rtree_2d,
            query_coords_2d,
            k=8,
            rbf=rbf_kernel,  # type: ignore[arg-type]
        )

        assert result.shape == (query_coords_2d.shape[0],)
        assert np.all(np.isfinite(result))


class TestKriging:
    """Tests for kriging interpolation function."""

    def test_kriging_cartesian_2d_with_kwargs(
        self,
        cartesian_rtree_2d: RTree3DHolder[np.float64],
        query_coords_2d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test Kriging interpolation on 2D Cartesian RTree using kwargs.

        Args:
            cartesian_rtree_2d: 2D Cartesian RTree fixture
            query_coords_2d: Query coordinates fixture

        """
        result, _counts = kriging(
            cartesian_rtree_2d,
            query_coords_2d,
            k=10,
            covariance_model="gaussian",
        )

        assert result.shape == (query_coords_2d.shape[0],)
        assert np.all(np.isfinite(result))

    def test_kriging_cartesian_3d_with_kwargs(
        self,
        cartesian_rtree_3d: RTree3DHolder[np.float64],
        query_coords_3d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test Kriging interpolation on 3D Cartesian RTree using kwargs.

        Args:
            cartesian_rtree_3d: 3D Cartesian RTree fixture
            query_coords_3d: Query coordinates fixture

        """
        result, _counts = kriging(
            cartesian_rtree_3d,
            query_coords_3d,
            k=12,
            covariance_model="matern_32",
        )

        assert result.shape == (query_coords_3d.shape[0],)

    def test_kriging_geographic_with_kwargs(
        self,
        geographic_rtree: GeographicRTree,
        query_coords_geographic: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test Kriging interpolation on geographic RTree using kwargs.

        Args:
            geographic_rtree: Geographic RTree fixture
            query_coords_geographic: Query coordinates fixture

        """
        result, _counts = kriging(
            geographic_rtree,
            query_coords_geographic,
            k=8,
            covariance_model="spherical",
        )

        assert result.shape == (query_coords_geographic.shape[0],)

    def test_kriging_with_config_object(
        self,
        cartesian_rtree_2d: RTree3DHolder[np.float64],
        query_coords_2d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test Kriging with explicit config object.

        Args:
            cartesian_rtree_2d: 2D Cartesian RTree fixture
            query_coords_2d: Query coordinates fixture

        """
        config_obj = (
            rtree.Kriging()
            .with_k(10)
            .with_covariance_model(rtree.CovarianceFunction.GAUSSIAN)
            .with_sigma(1.0)
        )

        result, _counts = kriging(
            cartesian_rtree_2d,
            query_coords_2d,
            config=config_obj,
        )

        assert result.shape == (query_coords_2d.shape[0],)

    @pytest.mark.parametrize(
        "covariance",
        [
            "cauchy",
            "gaussian",
            "matern_12",
            "matern_32",
            "matern_52",
            "spherical",
            "wendland",
        ],
    )
    def test_kriging_all_covariance_models(
        self,
        cartesian_rtree_2d: RTree3DHolder[np.float64],
        query_coords_2d: np.ndarray[TwoDims, np.dtype[np.float64]],
        covariance: str,
    ) -> None:
        """Test Kriging with all available covariance models.

        Args:
            cartesian_rtree_2d: 2D Cartesian RTree fixture
            query_coords_2d: Query coordinates fixture
            covariance: Covariance model type to test

        """
        result, _counts = kriging(
            cartesian_rtree_2d,
            query_coords_2d,
            k=8,
            covariance_model=covariance,  # type: ignore[arg-type]
        )

        assert result.shape == (query_coords_2d.shape[0],)


class TestWindowFunction:
    """Tests for window_function interpolation function."""

    def test_window_function_cartesian_2d_with_kwargs(
        self,
        cartesian_rtree_2d: RTree3DHolder[np.float64],
        query_coords_2d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test window function interpolation on 2D Cartesian RTree.

        Args:
            cartesian_rtree_2d: 2D Cartesian RTree fixture
            query_coords_2d: Query coordinates fixture

        """
        result, _counts = window_function(
            cartesian_rtree_2d,
            query_coords_2d,
            k=10,
            wf="gaussian",
        )

        assert result.shape == (query_coords_2d.shape[0],)
        assert np.all(np.isfinite(result))

    def test_window_function_cartesian_3d_with_kwargs(
        self,
        cartesian_rtree_3d: RTree3DHolder[np.float64],
        query_coords_3d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test window function interpolation on 3D Cartesian RTree.

        Args:
            cartesian_rtree_3d: 3D Cartesian RTree fixture
            query_coords_3d: Query coordinates fixture

        """
        result, _counts = window_function(
            cartesian_rtree_3d,
            query_coords_3d,
            k=12,
            wf="boxcar",
        )

        assert result.shape == (query_coords_3d.shape[0],)

    def test_window_function_geographic_with_kwargs(
        self,
        geographic_rtree: GeographicRTree,
        query_coords_geographic: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test window function interpolation on geographic RTree.

        Args:
            geographic_rtree: Geographic RTree fixture
            query_coords_geographic: Query coordinates fixture

        """
        result, _counts = window_function(
            geographic_rtree,
            query_coords_geographic,
            k=8,
            wf="hamming",
        )

        assert result.shape == (query_coords_geographic.shape[0],)

    def test_window_function_with_config_object(
        self,
        cartesian_rtree_2d: RTree3DHolder[np.float64],
        query_coords_2d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test window function with explicit config object.

        Args:
            cartesian_rtree_2d: 2D Cartesian RTree fixture
            query_coords_2d: Query coordinates fixture

        """
        config_obj = (
            rtree.InterpolationWindow()
            .with_k(10)
            .with_wf(rtree.WindowKernel.GAUSSIAN)
            .with_arg(1.0)
        )

        result, _counts = window_function(
            cartesian_rtree_2d,
            query_coords_2d,
            config=config_obj,
        )

        assert result.shape == (query_coords_2d.shape[0],)

    @pytest.mark.parametrize(
        "window_kernel",
        [
            "blackman",
            "blackman_harris",
            "boxcar",
            "flat_top",
            "gaussian",
            "hamming",
            "lanczos",
            "nuttall",
            "parzen",
            "parzen_swot",
        ],
    )
    def test_window_function_all_kernels(
        self,
        cartesian_rtree_2d: RTree3DHolder[np.float64],
        query_coords_2d: np.ndarray[TwoDims, np.dtype[np.float64]],
        window_kernel: str,
    ) -> None:
        """Test window function with all available kernels.

        Args:
            cartesian_rtree_2d: 2D Cartesian RTree fixture
            query_coords_2d: Query coordinates fixture
            window_kernel: Window kernel type to test

        """
        result, _ = window_function(
            cartesian_rtree_2d,
            query_coords_2d,
            k=8,
            wf=window_kernel,  # type: ignore[arg-type]
        )

        assert result.shape == (query_coords_2d.shape[0],)
        assert np.all(np.isfinite(result))


class TestQuery:
    """Tests for query nearest neighbors function."""

    def test_query_cartesian_2d_with_kwargs(
        self,
        cartesian_rtree_2d: RTree3DHolder[np.float64],
        query_coords_2d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test query on 2D Cartesian RTree using kwargs.

        Args:
            cartesian_rtree_2d: 2D Cartesian RTree fixture
            query_coords_2d: Query coordinates fixture

        """
        distances, values = query(
            cartesian_rtree_2d,
            query_coords_2d,
            k=4,
        )

        assert isinstance(distances, np.ndarray)
        assert isinstance(values, np.ndarray)
        assert distances.shape == (query_coords_2d.shape[0], 4)
        assert values.shape == (query_coords_2d.shape[0], 4)
        assert np.all(distances >= 0.0)

    def test_query_cartesian_3d_with_kwargs(
        self,
        cartesian_rtree_3d: RTree3DHolder[np.float64],
        query_coords_3d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test query on 3D Cartesian RTree using kwargs.

        Args:
            cartesian_rtree_3d: 3D Cartesian RTree fixture
            query_coords_3d: Query coordinates fixture

        """
        distances, values = query(
            cartesian_rtree_3d,
            query_coords_3d,
            k=8,
        )

        assert distances.shape == (query_coords_3d.shape[0], 8)
        assert values.shape == (query_coords_3d.shape[0], 8)

    def test_query_geographic_with_kwargs(
        self,
        geographic_rtree: GeographicRTree,
        query_coords_geographic: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test query on geographic RTree using kwargs.

        Args:
            geographic_rtree: Geographic RTree fixture
            query_coords_geographic: Query coordinates fixture

        """
        distances, values = query(
            geographic_rtree,
            query_coords_geographic,
            k=5,
        )

        assert distances.shape == (query_coords_geographic.shape[0], 5)
        assert values.shape == (query_coords_geographic.shape[0], 5)

    def test_query_with_config_object(
        self,
        cartesian_rtree_2d: RTree3DHolder[np.float64],
        query_coords_2d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test query with explicit config object.

        Args:
            cartesian_rtree_2d: 2D Cartesian RTree fixture
            query_coords_2d: Query coordinates fixture

        """
        config_obj = rtree.Query().with_k(4).with_radius(5.0)

        distances, values = query(
            cartesian_rtree_2d,
            query_coords_2d,
            config=config_obj,
        )

        assert distances.shape == (query_coords_2d.shape[0], 4)
        assert values.shape == (query_coords_2d.shape[0], 4)

    def test_query_with_radius_constraint(
        self,
        cartesian_rtree_2d: RTree3DHolder[np.float64],
        query_coords_2d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test query with radius constraint.

        Args:
            cartesian_rtree_2d: 2D Cartesian RTree fixture
            query_coords_2d: Query coordinates fixture

        """
        distances, _values = query(
            cartesian_rtree_2d,
            query_coords_2d,
            k=8,
            radius=2.0,
        )

        # Some queries may return fewer than k neighbors due to radius
        assert distances.shape[0] == query_coords_2d.shape[0]
        assert np.all(distances >= 0.0)


class TestInterpolationConsistency:
    """Tests for consistency across different interpolation methods."""

    def test_all_methods_produce_valid_output(
        self,
        cartesian_rtree_2d: RTree3DHolder[np.float64],
        query_coords_2d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test that all interpolation methods produce valid output.

        Args:
            cartesian_rtree_2d: 2D Cartesian RTree fixture
            query_coords_2d: Query coordinates fixture

        """
        k = 8

        idw_result, _ = inverse_distance_weighting(
            cartesian_rtree_2d, query_coords_2d, k=k, p=2
        )
        rbf_result, _ = radial_basis_function(
            cartesian_rtree_2d, query_coords_2d, k=k, rbf="thin_plate"
        )
        kriging_result, _ = kriging(
            cartesian_rtree_2d,
            query_coords_2d,
            k=k,
            covariance_model="gaussian",
        )
        window_result, _ = window_function(
            cartesian_rtree_2d, query_coords_2d, k=k, wf="boxcar"
        )

        # All should have same shape
        assert (
            idw_result.shape
            == rbf_result.shape
            == kriging_result.shape
            == window_result.shape
        )

        # All should be finite
        assert np.all(np.isfinite(idw_result))
        assert np.all(np.isfinite(rbf_result))
        assert np.all(np.isfinite(kriging_result))
        assert np.all(np.isfinite(window_result))

    def test_config_and_kwargs_equivalence(
        self,
        cartesian_rtree_2d: RTree3DHolder[np.float64],
        query_coords_2d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test that config object and kwargs produce same results.

        Args:
            cartesian_rtree_2d: 2D Cartesian RTree fixture
            query_coords_2d: Query coordinates fixture

        """
        # IDW with kwargs
        idw_kwargs, _ = inverse_distance_weighting(
            cartesian_rtree_2d, query_coords_2d, k=8, p=2, radius=5.0
        )

        # IDW with config
        config = (
            rtree.InverseDistanceWeighting()
            .with_k(8)
            .with_p(2)
            .with_radius(5.0)
        )
        idw_config, _ = inverse_distance_weighting(
            cartesian_rtree_2d, query_coords_2d, config=config
        )

        # Results should be identical
        np.testing.assert_array_equal(idw_kwargs, idw_config)

    def test_geographic_rtree_2d_only(
        self,
        geographic_rtree: GeographicRTree,
        query_coords_geographic: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test that geographic RTree works with 2D coordinates only.

        Args:
            geographic_rtree: Geographic RTree fixture
            query_coords_geographic: Query coordinates fixture

        """
        # Should work with 2D
        result, _ = inverse_distance_weighting(
            geographic_rtree,
            query_coords_geographic,
            k=5,
            p=2,
        )

        assert result.shape == (query_coords_geographic.shape[0],)


class TestErrorHandling:
    """Tests for proper error handling and validation."""

    def test_invalid_boundary_check_raises_error(
        self,
        cartesian_rtree_2d: RTree3DHolder[np.float64],
        query_coords_2d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test that invalid boundary check raises error.

        Args:
            cartesian_rtree_2d: 2D Cartesian RTree fixture
            query_coords_2d: Query coordinates fixture

        """
        with pytest.raises(KeyError):
            inverse_distance_weighting(
                cartesian_rtree_2d,
                query_coords_2d,
                k=4,
                p=2,
                boundary_check="invalid",  # type: ignore[arg-type]
            )

    def test_invalid_rbf_kernel_raises_error(
        self,
        cartesian_rtree_2d: RTree3DHolder[np.float64],
        query_coords_2d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test that invalid RBF kernel raises error.

        Args:
            cartesian_rtree_2d: 2D Cartesian RTree fixture
            query_coords_2d: Query coordinates fixture

        """
        with pytest.raises(KeyError):
            radial_basis_function(
                cartesian_rtree_2d,
                query_coords_2d,
                k=8,
                rbf="invalid",  # type: ignore[arg-type]
            )

    def test_invalid_covariance_model_raises_error(
        self,
        cartesian_rtree_2d: RTree3DHolder[np.float64],
        query_coords_2d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test that invalid covariance model raises error.

        Args:
            cartesian_rtree_2d: 2D Cartesian RTree fixture
            query_coords_2d: Query coordinates fixture

        """
        with pytest.raises(KeyError):
            kriging(
                cartesian_rtree_2d,
                query_coords_2d,
                k=8,
                covariance_model="invalid",  # type: ignore[arg-type]
            )

    def test_invalid_window_kernel_raises_error(
        self,
        cartesian_rtree_2d: RTree3DHolder[np.float64],
        query_coords_2d: np.ndarray[TwoDims, np.dtype[np.float64]],
    ) -> None:
        """Test that invalid window kernel raises error.

        Args:
            cartesian_rtree_2d: 2D Cartesian RTree fixture
            query_coords_2d: Query coordinates fixture

        """
        with pytest.raises(KeyError):
            window_function(
                cartesian_rtree_2d,
                query_coords_2d,
                k=8,
                wf="invalid",  # type: ignore[arg-type]
            )
