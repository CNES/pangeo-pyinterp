# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""RTree3D Interpolation Wrapper.

This module provides convenient wrapper functions around pyinterp.core.RTree3D
interpolation methods that accept both configuration objects and simplified
keyword arguments.

For most use cases, you can simply pass keyword arguments:
>>> result, counts = inverse_distance_weighting(tree, coords, k=10, p=2)

For advanced configuration, pass a config object:
>>> from pyinterp.config import rtree
>>> config = rtree.InverseDistanceWeighting().with_k(10).with_p(2)
>>> result, counts = inverse_distance_weighting(tree, coords, config)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from .core.config import rtree


if TYPE_CHECKING:
    import numpy as np

    from . import core
    from .type_hints import NDArray1DUInt32, OneDim, TwoDims

__all__ = [
    "inverse_distance_weighting",
    "kriging",
    "query",
    "radial_basis_function",
    "window_function",
]

#: Boundary check modes
BoundaryCheck = Literal["convex_hull", "envelope", "none"]

#: RBF kernel types
RBFKernel = Literal[
    "cubic",
    "gaussian",
    "inverse_multiquadric",
    "linear",
    "multiquadric",
    "thin_plate",
]

#: Window kernel types
WindowKernel = Literal[
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
]

#: Covariance function types
CovarianceFunction = Literal[
    "cauchy",
    "gaussian",
    "matern_12",
    "matern_32",
    "matern_52",
    "spherical",
    "wendland",
]

#: Drift function types
DriftFunction = Literal["linear", "quadratic"]

# Boundary check mapping
_BOUNDARY_MAP = {
    "convex_hull": rtree.BoundaryCheck.CONVEX_HULL,
    "envelope": rtree.BoundaryCheck.ENVELOPE,
    "none": rtree.BoundaryCheck.NONE,
}

# RBF kernel mapping
_RBF_MAP = {
    "cubic": rtree.RBFKernel.CUBIC,
    "gaussian": rtree.RBFKernel.GAUSSIAN,
    "inverse_multiquadric": rtree.RBFKernel.INVERSE_MULTIQUADRIC,
    "linear": rtree.RBFKernel.LINEAR,
    "multiquadric": rtree.RBFKernel.MULTIQUADRIC,
    "thin_plate": rtree.RBFKernel.THIN_PLATE,
}

# Window kernel mapping
_WINDOW_MAP = {
    "blackman": rtree.WindowKernel.BLACKMAN,
    "blackman_harris": rtree.WindowKernel.BLACKMAN_HARRIS,
    "boxcar": rtree.WindowKernel.BOXCAR,
    "flat_top": rtree.WindowKernel.FLAT_TOP,
    "gaussian": rtree.WindowKernel.GAUSSIAN,
    "hamming": rtree.WindowKernel.HAMMING,
    "lanczos": rtree.WindowKernel.LANCZOS,
    "nuttall": rtree.WindowKernel.NUTTALL,
    "parzen": rtree.WindowKernel.PARZEN,
    "parzen_swot": rtree.WindowKernel.PARZEN_SWOT,
}

# Covariance function mapping
_COVARIANCE_MAP = {
    "cauchy": rtree.CovarianceFunction.CAUCHY,
    "gaussian": rtree.CovarianceFunction.GAUSSIAN,
    "matern_12": rtree.CovarianceFunction.MATERN_12,
    "matern_32": rtree.CovarianceFunction.MATERN_32,
    "matern_52": rtree.CovarianceFunction.MATERN_52,
    "spherical": rtree.CovarianceFunction.SPHERICAL,
    "wendland": rtree.CovarianceFunction.WENDLAND,
}

# Drift function mapping
_DRIFT_MAP = {
    "linear": rtree.DriftFunction.LINEAR,
    "quadratic": rtree.DriftFunction.QUADRATIC,
}


def _apply_common_config(
    cfg: _RTreeConfig,
    radius: float | None = None,
    boundary_check: BoundaryCheck | None = None,
    num_threads: int | None = None,
) -> _RTreeConfig:
    """Apply common configuration parameters to a config object.

    Args:
        cfg: Configuration object to modify
        radius: Search radius (None = no limit)
        boundary_check: Boundary checking mode
        num_threads: Number of threads (0 = auto)

    Returns:
        Updated configuration object

    """
    if radius is not None:
        cfg = cfg.with_radius(radius)
    if boundary_check is not None:
        cfg = cfg.with_boundary_check(_BOUNDARY_MAP[boundary_check])
    if num_threads is not None:
        cfg = cfg.with_num_threads(num_threads)
    return cast("_RTreeConfig", cfg)


# TypeVars for generic config builder
_RTreeConfig = TypeVar(
    "_RTreeConfig",
    rtree.InverseDistanceWeighting,
    rtree.RadialBasisFunction,
    rtree.Kriging,
    rtree.InterpolationWindow,
    rtree.Query,
)


def _build_config(
    config_class: type[_RTreeConfig],
    method_map: dict | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> _RTreeConfig:
    """Build a configuration object from keyword arguments.

    Args:
        config_class: The configuration class to instantiate
        method_map: Dict mapping parameter names to method names and optional
            value maps
        **kwargs: Keyword arguments to apply

    Returns:
        Configured object

    """
    cfg = config_class()
    method_map = method_map or {}

    for param_name, value in kwargs.items():
        if value is None or param_name not in method_map:
            continue

        method_name, value_map = method_map[param_name]
        method = getattr(cfg, method_name)
        cfg = method(value_map[value] if value_map else value)

    return cast("_RTreeConfig", cfg)


def inverse_distance_weighting(
    tree: core.RTree3DHolder | core.geometry.geographic.RTree,
    coordinates: np.ndarray[TwoDims, np.dtype],
    config: rtree.InverseDistanceWeighting | None = None,
    *,
    k: int | None = None,
    p: int | None = None,
    radius: float | None = None,
    boundary_check: BoundaryCheck | None = None,
    num_threads: int | None = None,
) -> tuple[np.ndarray[OneDim, np.dtype], NDArray1DUInt32]:
    """Inverse Distance Weighting interpolation.

    Args:
        tree: R*Tree containing the scattered data.
        coordinates: Query coordinates as a NumPy array with shape:

            * (N, 2): For geographic.RTree, representing (longitude, latitude).
            * (N, 2) or (N, 3): For RTree3DHolder, representing
              (longitude, latitude[, altitude]).

        config: Configuration object (if provided, keyword args are ignored)
        k: Number of nearest neighbors to use
        p: Power parameter for IDW
        radius: Search radius (None = no limit)
        boundary_check: Boundary checking mode
        num_threads: Number of threads (0 = auto)

    Returns:
        Tuple of (interpolated values, neighbor counts)

    Examples:
        Simple usage with keyword arguments:

        >>> result, counts = inverse_distance_weighting(
        ....    tree,
        ....    coords,
        ....    k=10,
        ....    p=2,
        .... )

        Advanced usage with config object:

        >>> from pyinterp.core.config import rtree
        >>> config = rtree.InverseDistanceWeighting().with_k(10).with_p(2)
        >>> result, counts = inverse_distance_weighting(tree, coords, config)

    """
    # If config is provided, use it directly
    if config is not None:
        return tree.inverse_distance_weighting(coordinates, config)

    # Create config from keyword arguments
    method_map = {
        "k": ("with_k", None),
        "p": ("with_p", None),
    }
    cfg = _build_config(rtree.InverseDistanceWeighting, method_map, k=k, p=p)
    cfg = _apply_common_config(cfg, radius, boundary_check, num_threads)

    return tree.inverse_distance_weighting(coordinates, cfg)


def radial_basis_function(
    tree: core.RTree3DHolder | core.geometry.geographic.RTree,
    coordinates: np.ndarray[TwoDims, np.dtype],
    config: rtree.RadialBasisFunction | None = None,
    *,
    k: int | None = None,
    rbf: RBFKernel | None = None,
    epsilon: float | None = None,
    smooth: float | None = None,
    radius: float | None = None,
    boundary_check: BoundaryCheck | None = None,
    num_threads: int | None = None,
) -> tuple[np.ndarray[OneDim, np.dtype], NDArray1DUInt32]:
    """Radial Basis Function interpolation.

    Args:
        tree: R*Tree containing the scattered data.
        coordinates: Query coordinates as a NumPy array with shape:

            * (N, 2): For geographic.RTree, representing (longitude, latitude).
            * (N, 2) or (N, 3): For RTree3DHolder, representing
              (longitude, latitude[, altitude]).

        config: Configuration object (if provided, keyword args are ignored)
        k: Number of nearest neighbors to use
        rbf: RBF kernel type
        epsilon: Shape parameter (None = auto)
        smooth: Smoothing parameter
        radius: Search radius (None = no limit)
        boundary_check: Boundary checking mode
        num_threads: Number of threads (0 = auto)

    Returns:
        Tuple of (interpolated values, neighbor counts)

    Examples:
        >>> result, counts = radial_basis_function(
        ...     tree, coords, k=10, rbf="thin_plate"
        ... )

    """
    # If config is provided, use it directly
    if config is not None:
        return tree.radial_basis_function(coordinates, config)

    # Create config from keyword arguments
    method_map = {
        "k": ("with_k", None),
        "rbf": ("with_rbf", _RBF_MAP),
        "epsilon": ("with_epsilon", None),
        "smooth": ("with_smooth", None),
    }
    cfg = _build_config(
        rtree.RadialBasisFunction,
        method_map,
        k=k,
        rbf=rbf,
        epsilon=epsilon,
        smooth=smooth,
    )
    cfg = _apply_common_config(cfg, radius, boundary_check, num_threads)

    return tree.radial_basis_function(coordinates, cfg)


def kriging(
    tree: core.RTree3DHolder | core.geometry.geographic.RTree,
    coordinates: np.ndarray[TwoDims, np.dtype],
    config: rtree.Kriging | None = None,
    *,
    k: int | None = None,
    covariance_model: CovarianceFunction | None = None,
    sigma: float | None = None,
    lambda_: float | None = None,
    nugget: float | None = None,
    drift_function: DriftFunction | None = None,
    radius: float | None = None,
    boundary_check: BoundaryCheck | None = None,
    num_threads: int | None = None,
) -> tuple[np.ndarray[OneDim, np.dtype], NDArray1DUInt32]:
    """Kriging interpolation.

    Args:
        tree: R*Tree containing the scattered data.
        coordinates: Query coordinates as a NumPy array with shape:

            * (N, 2): For geographic.RTree, representing (longitude, latitude).
            * (N, 2) or (N, 3): For RTree3DHolder, representing
              (longitude, latitude[, altitude]).

        config: Configuration object (if provided, keyword args are ignored)
        k: Number of nearest neighbors to use
        covariance_model: Covariance function type
        sigma: Variance parameter
        lambda_: Length scale parameter
        nugget: Nugget effect
        drift_function: Drift/trend function
        radius: Search radius (None = no limit)
        boundary_check: Boundary checking mode
        num_threads: Number of threads (0 = auto)

    Returns:
        Tuple of (interpolated values, neighbor counts)

    Examples:
        >>> result, counts = kriging(
        ...     tree, coords, k=10, covariance_model="matern_32"
        ... )

    """
    # If config is provided, use it directly
    if config is not None:
        return tree.kriging(coordinates, config)

    # Create config from keyword arguments
    method_map = {
        "k": ("with_k", None),
        "covariance_model": ("with_covariance_model", _COVARIANCE_MAP),
        "sigma": ("with_sigma", None),
        "lambda_": ("with_lambda", None),
        "nugget": ("with_nugget", None),
        "drift_function": ("with_drift_function", _DRIFT_MAP),
    }
    cfg = _build_config(
        rtree.Kriging,
        method_map,
        k=k,
        covariance_model=covariance_model,
        sigma=sigma,
        lambda_=lambda_,
        nugget=nugget,
        drift_function=drift_function,
    )
    cfg = _apply_common_config(cfg, radius, boundary_check, num_threads)

    return tree.kriging(coordinates, cfg)


def window_function(
    tree: core.RTree3DHolder | core.geometry.geographic.RTree,
    coordinates: np.ndarray[TwoDims, np.dtype],
    config: rtree.InterpolationWindow | None = None,
    *,
    k: int | None = None,
    wf: WindowKernel | None = None,
    arg: float | None = None,
    radius: float | None = None,
    boundary_check: BoundaryCheck | None = None,
    num_threads: int | None = None,
) -> tuple[np.ndarray[OneDim, np.dtype], NDArray1DUInt32]:
    """Window function interpolation.

    Args:
        tree: R*Tree containing the scattered data.
        coordinates: Query coordinates as a NumPy array with shape:

            * (N, 2): For geographic.RTree, representing (longitude, latitude).
            * (N, 2) or (N, 3): For RTree3DHolder, representing
              (longitude, latitude[, altitude]).

        config: Configuration object (if provided, keyword args are ignored)
        k: Number of nearest neighbors to use
        wf: Window kernel type
        arg: Window function parameter (kernel-specific)
        radius: Search radius (None = no limit)
        boundary_check: Boundary checking mode
        num_threads: Number of threads (0 = auto)

    Returns:
        Tuple of (interpolated values, neighbor counts)

    Examples:
        >>> result, counts = window_function(tree, coords, k=10, wf="gaussian")

    """
    # If config is provided, use it directly
    if config is not None:
        return tree.window_function(coordinates, config)

    # Create config from keyword arguments
    method_map = {
        "k": ("with_k", None),
        "wf": ("with_wf", _WINDOW_MAP),
        "arg": ("with_arg", None),
    }
    cfg = _build_config(
        rtree.InterpolationWindow, method_map, k=k, wf=wf, arg=arg
    )
    cfg = _apply_common_config(cfg, radius, boundary_check, num_threads)

    return tree.window_function(coordinates, cfg)


def query(
    tree: core.RTree3DHolder | core.geometry.geographic.RTree,
    coordinates: np.ndarray[TwoDims, np.dtype],
    config: rtree.Query | None = None,
    *,
    k: int | None = None,
    radius: float | None = None,
    boundary_check: BoundaryCheck | None = None,
    num_threads: int | None = None,
) -> tuple[
    np.ndarray[TwoDims, np.dtype],
    np.ndarray[TwoDims, np.dtype],
]:
    """Query nearest neighbors.

    Args:
        tree: R*Tree containing the scattered data.
        coordinates: Query coordinates as a NumPy array with shape:

            * (N, 2): For geographic.RTree, representing (longitude, latitude).
            * (N, 2) or (N, 3): For RTree3DHolder, representing
              (longitude, latitude[, altitude]).

        config: Configuration object (if provided, keyword args are ignored)
        k: Number of nearest neighbors to return
        radius: Search radius (None = no limit)
        boundary_check: Boundary checking mode
        num_threads: Number of threads (0 = auto)

    Returns:
        Tuple of (distances, values) arrays

    Examples:
        >>> distances, values = query(tree, coords, k=5)

    """
    # If config is provided, use it directly
    if config is not None:
        return tree.query(coordinates, config)

    # Create config from keyword arguments
    method_map = {
        "k": ("with_k", None),
    }
    cfg = _build_config(rtree.Query, method_map, k=k)
    cfg = _apply_common_config(cfg, radius, boundary_check, num_threads)

    return tree.query(coordinates, cfg)
