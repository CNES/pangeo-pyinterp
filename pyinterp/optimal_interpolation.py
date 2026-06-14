# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
r"""Optimal Interpolation (OI / BLUE) for scattered 4D observations.

This module provides an Optimal Interpolation estimator with an anisotropic
covariance kernel and a per-observation error variance. Decorrelation length
scales and the field standard deviation may be either scalars or
:class:`pyinterp.Grid2D` objects sampled (bilinear) at each query point,
which lets the analysis be tuned regionally — a common requirement in
operational altimetry products (DUACS / AVISO).

Two coordinate-system modes are available:

* ``"cartesian"`` (default, backward-compatible) — observations are stored as
  ``(x, y, t)`` in user units, and the analysis is run in that frame. Good
  for regional studies after a manual projection.
* ``"geographic"`` — observations are passed as ``(lon, lat, t)`` in
  degrees/seconds; the wrapper internally converts ``(lon, lat)`` to ECEF
  coordinates via the WGS-84 spheroid, indexes them in a 4D R-tree, and
  runs the OI in ECEF + time. The ``Grid2D`` length-scale and sigma fields
  remain naturally indexed by ``(lon, lat)`` and are sampled at the query
  position. The spatial decorrelation scale ``l_spatial`` is expressed in
  meters and applied isotropically to the three ECEF axes — appropriate for
  global isotropic spatial covariances such as those used in DUACS-like
  products.

For each query point the algorithm:

1. Retrieves up to ``k`` nearest observations from the internal 4D R-tree.
2. Builds the obs-obs covariance ``C_oo`` and the obs-grid covariance
   ``c_og`` with an anisotropic kernel
   :math:`C(\Delta) = \sigma^2 K(r)` where
   :math:`r^2 = \sum_d (\Delta_d / L_d)^2`.
3. Adds the per-observation error variance to the diagonal,
   :math:`R = \mathrm{diag}(\sigma^2_\mathrm{obs})`.
4. Solves :math:`w = (C_{oo} + R)^{-1} c_{og}` (Cholesky, LDLT fallback).
5. Returns the analysed value :math:`f = w^\top \cdot \mathrm{obs}` and
   the formal error standard deviation
   :math:`e = \sqrt{\max(\sigma^2 - c_{og}^\top w, 0)}`.

Heavy lifting is delegated to the C++ backend exposed by
:class:`pyinterp.RTree4D.optimal_interpolation` — this module is a thin
shim that handles coordinate conversion and ``Grid2D`` sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Union

import numpy as np

from . import core
from .core import RTree4D
from .core.config.rtree import CovarianceFunction as _CovEnum
from .core.config.rtree import OptimalInterpolation as _OIConfig
from .core.geometry.geographic import Coordinates
from .core.geometry.geographic import Spheroid as _S
from .regular_grid_interpolator import bivariate


if TYPE_CHECKING:
    from .core import GridHolder
    from .core.geometry.geographic import Spheroid
    from .type_hints import NDArray1DFloat64, NDArray1DUInt32, NDArray2DFloat64


__all__ = [
    "CoordinateSystem",
    "CovarianceFunction",
    "OIResult",
    "OptimalInterpolation",
]


#: Anisotropic covariance kernels supported by the OI.
CovarianceFunction = Literal[
    "cauchy",
    "gaussian",
    "matern_12",
    "matern_32",
    "matern_52",
    "spherical",
    "wendland",
]

#: Coordinate-system options for the OI.
CoordinateSystem = Literal["cartesian", "geographic"]

#: A length-scale or sigma parameter: scalar or 2D grid sampled bilinearly.
ScalarOrGrid = Union[float, "GridHolder"]


_SQRT3 = np.sqrt(3.0)
_SQRT5 = np.sqrt(5.0)


_KERNEL_NAME_TO_ENUM = {
    "cauchy": _CovEnum.CAUCHY,
    "gaussian": _CovEnum.GAUSSIAN,
    "matern_12": _CovEnum.MATERN_12,
    "matern_32": _CovEnum.MATERN_32,
    "matern_52": _CovEnum.MATERN_52,
    "spherical": _CovEnum.SPHERICAL,
    "wendland": _CovEnum.WENDLAND,
}


# A length scale large enough that any reasonable obs/query distance becomes
# negligible in the anisotropic squared distance — used to neutralise the
# z axis in 4D ⟵ 3D padding.
_HUGE_LENGTH = 1.0e30


def _kernel_from_r2(r2: np.ndarray, kernel: CovarianceFunction) -> np.ndarray:
    """Evaluate an anisotropic covariance kernel from squared scaled distance.

    Used by the regression tests as a pure-Python reference.

    Args:
        r2: Squared anisotropic distance.
        kernel: Kernel name.

    Returns:
        Covariance values with the same shape as ``r2``.

    """
    if kernel == "gaussian":
        return np.exp(-r2)
    if kernel == "cauchy":
        return 1.0 / (1.0 + r2)
    if kernel == "matern_12":
        return np.exp(-np.sqrt(r2))
    if kernel == "matern_32":
        d = np.sqrt(r2)
        return (1.0 + _SQRT3 * d) * np.exp(-_SQRT3 * d)
    if kernel == "matern_52":
        d = np.sqrt(r2)
        return (1.0 + _SQRT5 * d + (5.0 / 3.0) * r2) * np.exp(-_SQRT5 * d)
    if kernel == "spherical":
        d = np.sqrt(r2)
        return np.where(d < 1.0, 1.0 - 1.5 * d + 0.5 * d**3, 0.0)
    if kernel == "wendland":
        d = np.sqrt(r2)
        return np.where(d < 1.0, (1.0 - d) ** 2, 0.0)
    msg = f"Unknown covariance kernel: {kernel!r}"
    raise ValueError(msg)


def _sample_scalar_or_grid(
    param: ScalarOrGrid,
    lon: np.ndarray,
    lat: np.ndarray,
    name: str,
) -> np.ndarray:
    """Resolve a scalar-or-Grid2D parameter into per-query values."""
    if np.isscalar(param) or (
        isinstance(param, np.ndarray) and param.ndim == 0
    ):
        value = float(param)  # type: ignore[arg-type]
        if not np.isfinite(value) or value <= 0.0:
            msg = f"{name} must be a positive finite scalar, got {value!r}"
            raise ValueError(msg)
        return np.full(lon.shape, value, dtype=np.float64)

    if isinstance(param, core.GridHolder):
        result = np.asarray(
            bivariate(param, lon, lat, "bilinear"), dtype=np.float64
        )
        # Out-of-domain query points yield NaN (bivariate uses
        # bounds_error=False); masked / fill cells can yield non-positive
        # values. Either would silently propagate into 1/L or sigma**2 and
        # produce NaN/garbage analyses, so reject them up front with a clear
        # message rather than at the C++ layer.
        if not np.all(np.isfinite(result)):
            msg = (
                f"{name} sampled from the Grid2D contains non-finite "
                "values; this usually means some query points fall outside "
                "the grid domain. Extend the grid to cover all query points."
            )
            raise ValueError(msg)
        if np.any(result <= 0.0):
            msg = (
                f"{name} sampled from the Grid2D contains non-positive "
                "values; length scales and sigma must be strictly positive."
            )
            raise ValueError(msg)
        return result

    msg = (
        f"{name} must be a positive scalar or a pyinterp.Grid2D, "
        f"got type {type(param).__name__}"
    )
    raise TypeError(msg)


def _lla_to_ecef(
    lon: np.ndarray, lat: np.ndarray, spheroid: Spheroid | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert (lon, lat, alt=0) → (x, y, z) ECEF in meters."""

    sph = spheroid if spheroid is not None else _S()
    coords = Coordinates(sph)
    alt = np.zeros_like(lon)
    return coords.lla_to_ecef(lon, lat, alt)


@dataclass
class OIResult:
    """Result of an Optimal Interpolation analysis at a set of query points.

    Attributes:
        value: Analysed field, shape ``(M,)``. ``NaN`` where insufficient
            neighbours were found.
        error: Formal error standard deviation, shape ``(M,)``. ``NaN``
            where insufficient neighbours were found.
        neighbors: Number of observations actually used at each query
            point, shape ``(M,)``, ``uint32``.

    """

    value: NDArray1DFloat64
    error: NDArray1DFloat64
    neighbors: NDArray1DUInt32


class OptimalInterpolation:
    """Optimal Interpolation with anisotropic covariance and per-obs error.

    The estimator indexes scattered observations and computes the BLUE
    (Best Linear Unbiased Estimator) at arbitrary query points.

    .. important::
        This is **simple kriging**: the BLUE about a *known, zero* mean. The
        weights are not constrained to sum to one, so **observations must be
        anomalies relative to a zero background mean** — subtract a mean /
        reference field before passing them in (e.g. use sea-level
        *anomalies*, not absolute sea-surface height). Where a query has no
        nearby, well-correlated observations the analysis relaxes toward
        ``0`` (the prior mean) and the formal error toward ``sigma``, *not*
        toward the local data mean. Feeding non-centred values therefore
        produces a systematic bias toward zero in data-sparse regions.

        The field standard deviation ``sigma`` is also evaluated only at the
        query point and assumed locally constant across that query's
        neighbourhood; strong ``sigma`` gradients within one neighbourhood
        introduce a small modelling bias.

    Args:
        obs_coords: Observation coordinates of shape ``(N, 3)``. In
            ``cartesian`` mode this is ``(x, y, t)`` in user units. In
            ``geographic`` mode this is ``(lon_deg, lat_deg, t_seconds)``.
        obs_values: Observed values, shape ``(N,)``.
        obs_sigma2: Per-observation error variance, shape ``(N,)``. Must
            be strictly positive.
        covariance: Anisotropic covariance kernel name. Defaults to
            ``"gaussian"``.
        coordinate_system: ``"cartesian"`` (default) or ``"geographic"``.
            See module docstring for the trade-offs.
        spheroid: Optional :class:`pyinterp.geometry.geographic.Spheroid`
            used in ``geographic`` mode. Defaults to WGS-84.
        time_scale: Scale factor applied to ``t`` when packing the internal
            R-tree, so the k-NN search (and any ``radius`` filter) treat
            space and time comparably. In ``cartesian`` mode it is in
            user-space-units per time-unit; in ``geographic`` mode it is in
            **meters per second** (time becomes a meters-equivalent before
            being mixed with the ECEF axes). It cancels out of the
            covariance — only neighbour selection is affected, so results
            with the default ``1.0`` are unchanged. Default ``1.0``.

    Note:
        The internal R-tree uses the 4D Euclidean metric on the packed
        coordinates, mixing the three spatial axes with the
        ``time_scale``-scaled time axis. In ``geographic`` mode the spatial
        part is the geodetic chord distance (very close to the great-circle
        distance for short ranges); set ``time_scale`` to a meters-per-second
        value so the k-NN search balances space and time. In ``cartesian``
        mode the user is responsible for unit consistency between space and
        time.

    Choosing ``time_scale``:
        The k-NN search ranks neighbours by the *raw* packed Euclidean
        distance, which mixes spatial units with time. With the default
        ``time_scale = 1`` that ranking depends on the arbitrary ratio of
        spatial units to seconds, not on how correlated the neighbours
        actually are, so the retrieved set can be skewed toward one axis. A
        neighbour one decorrelation length away in space and one away in
        time are equally correlated with the query; to make the search treat
        them equally, scale time so those two steps have the same packed
        length::

            time_scale ≈ L_spatial / L_t

        (typical spatial decorrelation length over the temporal one). Then
        the ``k`` nearest neighbours are the ``k`` *most correlated*
        observations. Worked example (geographic): ``L_spatial = 150 km``
        and ``L_t = 6 h`` give ``time_scale ≈ 150000 m / 21600 s ≈ 7 m/s``.
        ``time_scale`` is baked into the tree at construction, so choose a
        value representative of the scales you query with; it never changes
        the covariance (hence the analysis is identical when all neighbours
        are used), only which neighbours are retrieved.

    """

    def __init__(
        self,
        obs_coords: NDArray2DFloat64,
        obs_values: NDArray1DFloat64,
        obs_sigma2: NDArray1DFloat64,
        *,
        covariance: CovarianceFunction = "gaussian",
        coordinate_system: CoordinateSystem = "cartesian",
        spheroid: Spheroid | None = None,
        time_scale: float = 1.0,
    ) -> None:
        obs_coords = np.ascontiguousarray(obs_coords, dtype=np.float64)
        obs_values = np.ascontiguousarray(obs_values, dtype=np.float64)
        obs_sigma2 = np.ascontiguousarray(obs_sigma2, dtype=np.float64)

        if obs_coords.ndim != 2 or obs_coords.shape[1] != 3:
            msg = (
                "obs_coords must have shape (N, 3); got "
                f"{obs_coords.shape}"
            )
            raise ValueError(msg)
        n = obs_coords.shape[0]
        if obs_values.shape != (n,):
            msg = (
                f"obs_values shape {obs_values.shape} does not match "
                f"obs_coords ({n},)"
            )
            raise ValueError(msg)
        if obs_sigma2.shape != (n,):
            msg = (
                f"obs_sigma2 shape {obs_sigma2.shape} does not match "
                f"obs_coords ({n},)"
            )
            raise ValueError(msg)
        if np.any(obs_sigma2 <= 0.0):
            msg = "obs_sigma2 must be strictly positive everywhere"
            raise ValueError(msg)
        if coordinate_system not in ("cartesian", "geographic"):
            msg = (
                "coordinate_system must be 'cartesian' or 'geographic', "
                f"got {coordinate_system!r}"
            )
            raise ValueError(msg)
        if time_scale <= 0.0 or not np.isfinite(time_scale):
            msg = (
                "time_scale must be a positive finite number, got "
                f"{time_scale!r}"
            )
            raise ValueError(msg)

        self._covariance: CovarianceFunction = covariance
        # Validate kernel name early.
        _ = _kernel_from_r2(np.zeros(1), covariance)

        self._coordinate_system = coordinate_system
        self._spheroid = spheroid
        self._time_scale = float(time_scale)
        self._obs_coords = obs_coords  # kept for introspection / tests

        # Build the 4D packed coordinates.
        packed = np.empty((n, 4), dtype=np.float64)
        if coordinate_system == "cartesian":
            # (x, y, 0, t * time_scale) — z axis neutralised via large Lz.
            packed[:, 0] = obs_coords[:, 0]
            packed[:, 1] = obs_coords[:, 1]
            packed[:, 2] = 0.0
            packed[:, 3] = obs_coords[:, 2] * self._time_scale
        else:  # geographic
            x, y, z = _lla_to_ecef(
                obs_coords[:, 0], obs_coords[:, 1], spheroid
            )
            packed[:, 0] = x
            packed[:, 1] = y
            packed[:, 2] = z
            # Time is scaled like the cartesian path so that ``time_scale``
            # rebalances space vs. time in the k-NN / radius search. It
            # cancels out of the covariance (lt is scaled identically below).
            packed[:, 3] = obs_coords[:, 2] * self._time_scale

        tree = RTree4D(dtype="float64")
        tree.packing(packed, obs_values, obs_sigma2)
        self._tree = tree

    @property
    def covariance(self) -> CovarianceFunction:
        """Covariance kernel name."""
        return self._covariance

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """Coordinate system in use."""
        return self._coordinate_system

    @property
    def n_observations(self) -> int:
        """Number of indexed observations."""
        return self._obs_coords.shape[0]

    def __call__(
        self,
        query_coords: NDArray2DFloat64,
        *,
        lx: ScalarOrGrid | None = None,
        ly: ScalarOrGrid | None = None,
        lt: ScalarOrGrid,
        sigma: ScalarOrGrid,
        l_spatial: ScalarOrGrid | None = None,
        k: int = 24,
        radius: float | None = None,
        num_threads: int = 0,
    ) -> OIResult:
        """Run the OI analysis at a set of query points.

        Args:
            query_coords: Query coordinates of shape ``(M, 3)``.
                Cartesian mode: ``(x, y, t)``. Geographic mode:
                ``(lon_deg, lat_deg, t_seconds)``.
            lx: Decorrelation length along the first axis (cartesian mode).
                Scalar or :class:`pyinterp.Grid2D` sampled at the query
                ``(x, y)`` (cartesian) or ``(lon, lat)`` (geographic).
            ly: Decorrelation length along the second axis (cartesian
                mode). Same options as ``lx``.
            lt: Temporal decorrelation length (same unit as ``t``).
                Required in both modes.
            sigma: Field standard deviation. Same options as ``lx``.
            l_spatial: Spatial decorrelation length **in meters**
                (geographic mode only). Applied isotropically to the
                three ECEF axes. Scalar or Grid2D sampled at
                ``(lon, lat)``. Mutually exclusive with ``lx``/``ly``.
            k: Maximum number of nearest neighbours to use.
            radius: Optional maximum search radius in the packed metric —
                the *combined* space+time Euclidean distance, not a pure
                spatial distance. With ``time_scale`` chosen so time is a
                meters-equivalent this is metres in geographic mode, and
                user units in cartesian.
            num_threads: Number of worker threads. ``0`` uses
                ``os.cpu_count()``.

        Returns:
            :class:`OIResult`.

        """
        query_coords = np.ascontiguousarray(query_coords, dtype=np.float64)
        if query_coords.ndim != 2 or query_coords.shape[1] != 3:
            msg = (
                "query_coords must have shape (M, 3); got "
                f"{query_coords.shape}"
            )
            raise ValueError(msg)
        if k < 1:
            msg = f"k must be >= 1, got {k}"
            raise ValueError(msg)

        m = query_coords.shape[0]
        qx = query_coords[:, 0]  # cartesian x OR longitude
        qy = query_coords[:, 1]  # cartesian y OR latitude

        sigma_arr = _sample_scalar_or_grid(sigma, qx, qy, "sigma")
        lt_arr = _sample_scalar_or_grid(lt, qx, qy, "lt")

        if self._coordinate_system == "cartesian":
            if l_spatial is not None:
                msg = "l_spatial is only valid in geographic mode"
                raise ValueError(msg)
            if lx is None or ly is None:
                msg = "lx and ly are required in cartesian mode"
                raise ValueError(msg)
            lx_arr = _sample_scalar_or_grid(lx, qx, qy, "lx")
            ly_arr = _sample_scalar_or_grid(ly, qx, qy, "ly")
            lz_arr = np.full(m, _HUGE_LENGTH, dtype=np.float64)
            lt_packed = lt_arr * self._time_scale

            packed_query = np.empty((m, 4), dtype=np.float64)
            packed_query[:, 0] = qx
            packed_query[:, 1] = qy
            packed_query[:, 2] = 0.0
            packed_query[:, 3] = query_coords[:, 2] * self._time_scale
        else:  # geographic
            if lx is not None or ly is not None:
                msg = (
                    "lx / ly are not valid in geographic mode; use "
                    "l_spatial (isotropic horizontal scale in meters)"
                )
                raise ValueError(msg)
            if l_spatial is None:
                msg = "l_spatial is required in geographic mode"
                raise ValueError(msg)
            ls_arr = _sample_scalar_or_grid(l_spatial, qx, qy, "l_spatial")
            lx_arr = ls_arr
            ly_arr = ls_arr
            lz_arr = ls_arr
            lt_packed = lt_arr * self._time_scale

            x, y, z = _lla_to_ecef(qx, qy, self._spheroid)
            packed_query = np.empty((m, 4), dtype=np.float64)
            packed_query[:, 0] = x
            packed_query[:, 1] = y
            packed_query[:, 2] = z
            packed_query[:, 3] = query_coords[:, 2] * self._time_scale

        cfg = (
            _OIConfig()
            .with_covariance_model(_KERNEL_NAME_TO_ENUM[self._covariance])
            .with_k(int(k))
            .with_num_threads(int(num_threads))
        )
        if radius is not None:
            cfg = cfg.with_radius(float(radius))

        values, errors, neighbors = self._tree.optimal_interpolation(
            packed_query, lx_arr, ly_arr, lz_arr, lt_packed, sigma_arr, cfg
        )
        return OIResult(value=values, error=errors, neighbors=neighbors)
