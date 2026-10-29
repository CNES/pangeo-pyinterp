# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
r"""Conservative regridding between rectilinear grids.

The point interpolators of :mod:`pyinterp` sample a field at the target
points. Conservative regridding instead gives each target cell the mean of the
source values weighted by the area of their overlap with the cell, which
preserves the integral of the field: it is the method of choice to coarsen a
grid, or to regrid fluxes and budgets.

The cells of a longitude/latitude grid are boxes whose area factorizes,
:math:`R^2 (\lambda_2 - \lambda_1)(\sin\beta_2 - \sin\beta_1)`, with
:math:`\beta` the authalic latitude (the latitude itself on a sphere). The
overlap of two cells is therefore the product of the overlaps of their sides,
and the regridding is applied along each axis in turn with two sparse 1D
operators, without building the 2D weight matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from . import core


if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from .core.geometry.geographic import Spheroid

__all__ = ["cell_bounds", "cells", "conservative"]

#: Kind of axis whose cell bounds are computed
AxisKind = Literal["longitude", "latitude", "cartesian"]


def cell_bounds(
    centers: ArrayLike,
    kind: AxisKind = "cartesian",
) -> NDArray[np.float64]:
    """Compute the bounds of the cells of an axis from their centers.

    The inner bounds are the midpoints between consecutive centers; the outer
    bounds are extrapolated by half a step. Latitude bounds are clamped to
    [-90, 90]. A longitude axis whose cells cover the whole circle gets the
    midpoint between its last and first centers (modulo 360) as outer bounds,
    so that the cells cover exactly 360 degrees.

    Args:
        centers: Centers of the cells, monotonic.
        kind: Kind of the axis.

    Returns:
        The bounds of the cells, shape (n, 2), in the CF convention.

    """
    x = np.asarray(centers, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:  # noqa: PLR2004
        msg = "the centers must be a 1D array of at least two values"
        raise ValueError(msg)
    edges = np.empty(x.size + 1)
    edges[1:-1] = 0.5 * (x[1:] + x[:-1])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    if kind == "longitude":
        span = abs(edges[-1] - edges[0])
        step = max(abs(x[1] - x[0]), abs(x[-1] - x[-2]))
        if abs(span - 360.0) < 0.5 * step:
            # Global axis: the outer bounds are the same meridian
            sign = 1.0 if x[-1] > x[0] else -1.0
            edge = 0.5 * (x[-1] + x[0] + sign * 360.0)
            edges[-1] = edge
            edges[0] = edge - sign * 360.0
    elif kind == "latitude":
        edges = np.clip(edges, -90.0, 90.0)
    return np.stack((edges[:-1], edges[1:]), axis=-1)


def cells(
    value: ArrayLike,
    kind: AxisKind = "cartesian",
    name: str = "the axis",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the centers and the bounds of the cells of an axis.

    Args:
        value: Centers of the cells, a 1D array of at least two values, or
            their bounds, an (n, 2) array as the CF bounds variables.
        kind: Kind of the axis.
        name: Name of the axis, used in the error messages.

    Returns:
        The centers, shape (n,), and the bounds, shape (n, 2), of the cells.
        The centers of cells given by their bounds are their midpoints.

    Raises:
        ValueError: If the bounds of the cells cannot be inferred from their
            centers, because there is only one, or if the shape of the array
            is invalid.

    """
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 2:  # noqa: PLR2004
        if kind == "latitude":
            array = np.clip(array, -90.0, 90.0)
        return array.mean(axis=1), np.ascontiguousarray(array)
    if array.ndim <= 1 and array.size == 1:
        msg = (
            f"the cell of {name} cannot be inferred from a single center: "
            "give its bounds instead, as an array of shape (1, 2), e.g. "
            "[[lower, upper]]"
        )
        raise ValueError(msg)
    if array.ndim != 1 or array.size == 0:
        msg = (
            f"{name} must be the centers of the cells, a 1D array, or their "
            f"bounds, an (n, 2) array, not an array of shape {array.shape}"
        )
        raise ValueError(msg)
    return array, cell_bounds(array, kind)


def conservative(
    data: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    target_x: ArrayLike,
    target_y: ArrayLike,
    *,
    axes: tuple[int, int] = (0, 1),
    geodetic: bool = True,
    spheroid: Spheroid | None = None,
    min_valid_fraction: float = 0.0,
    normalize: bool = True,
    num_threads: int = 0,
) -> NDArray[np.floating]:
    """Regrid a field between rectilinear grids, conserving its integral.

    Each target cell receives the mean of the source values weighted by the
    area of their overlap with the cell (first order conservative
    regridding). This is the NumPy layer of
    :py:meth:`pyinterp.backends.xarray.DataArrayAccessor.regrid`.

    The cells of an axis are given by their centers, a 1D array, or by their
    bounds, an (n, 2) array as the CF bounds variables. The bounds of cells
    given by their centers are the midpoints between consecutive centers (see
    :func:`cell_bounds`), so an axis of a single cell must be given by its
    bounds. Edges of contiguous cells, an array of n + 1 values, are converted
    with ``np.column_stack((edges[:-1], edges[1:]))``.

    The areas are computed on a sphere unless ``spheroid`` is given; the
    cells are longitude/latitude boxes, bounded by meridians and parallels.
    ESMF and xESMF draw the edges of the cells as great circles instead,
    which bulge towards the poles by an amount proportional to the square of
    the width of the cell and to sin 2φ: the results differ most at
    mid-latitudes, not at the poles. Coarsening a smooth global field from
    0.25° to 1°, they differ by up to 2·10⁻⁵ of its amplitude between 15° and
    75° of latitude, 3·10⁻⁶ between 75° and 85° and 3·10⁻⁷ beyond; to 5°, by
    up to 4·10⁻⁴ between 15° and 75°. Next to undefined values, the slivers
    of the great circles reach defined values that the boxes do not: some
    target cells get a value from xESMF and NaN from pyinterp, and the means
    of partly undefined cells differ by up to 3·10⁻³ at 5°.

    Args:
        data: Field to regrid. Its other dimensions (time, depth...) are
            regridded independently.
        x: Cells of the source grid along the first axis: longitudes in
            degrees if ``geodetic``.
        y: Cells of the source grid along the second axis: latitudes in
            degrees if ``geodetic``.
        target_x: Cells of the target grid along the first axis.
        target_y: Cells of the target grid along the second axis.
        axes: Dimensions of ``data`` along the first and second axes.
        geodetic: If true, the first axis is a longitude, periodic with a
            period of 360 degrees: the source and target grids can use
            different conventions (e.g. [0, 360) and [-180, 180)). The second
            axis is a latitude, clamped to [-90, 90]. Otherwise, both axes are
            Cartesian and the cells are weighted by their surface.
        spheroid: Ellipsoid on which the areas of the cells are computed.
            Defaults to a sphere, as ESMF. Pass
            :py:class:`pyinterp.geometry.geographic.Spheroid` () for WGS-84:
            the areas are then exact on the ellipsoid, using the authalic
            latitude. Coarsening to 1°, the results differ from those on the
            sphere by about 10⁻⁶ of the field; to 5°, by about 10⁻⁵.
        min_valid_fraction: Minimum fraction of the area of the target cell
            covered by defined source values (and by the source grid); below,
            the result is NaN. With the default, a target cell gets a value as
            soon as it overlaps a defined source value.
        normalize: If true (``conservative_normed`` in xESMF), the undefined
            source values (NaN) and the parts of the target cell outside the
            source grid are ignored: the result is the weighted mean of the
            defined values. Otherwise (``conservative``), the weighted sum is
            divided by the area of the whole target cell, as if the undefined
            values were zero.
        num_threads: Number of threads, 0 for all CPUs.

    Returns:
        The regridded field, of the same dimensions as ``data``, with the
        lengths of the target axes. Its type is float32 if ``data`` is
        float32, float64 otherwise.

    Raises:
        ValueError: If the cells of an axis are invalid, or if the shape of
            ``data`` does not match the source grid.

    Examples:
        Coarsen a global 0.25° field, stored as (lon, lat), to 1°:

        >>> lon = np.arange(0, 360, 0.25) + 0.125
        >>> lat = np.arange(-90, 90, 0.25) + 0.125
        >>> field = np.ones((lon.size, lat.size))
        >>> result = conservative(
        ...     field, lon, lat, np.arange(-179.5, 180), np.arange(-89.5, 90)
        ... )
        >>> result.shape
        (360, 180)

        A single target cell, given by its bounds, on a (time, lat, lon)
        array:

        >>> mean = conservative(
        ...     np.ones((3, lat.size, lon.size)),
        ...     lon,
        ...     lat,
        ...     [[-10.0, 10.0]],
        ...     [[40.0, 50.0]],
        ...     axes=(2, 1),
        ... )
        >>> mean.shape
        (3, 1, 1)

    """
    values = np.asarray(data)
    if values.dtype != np.float32:
        values = values.astype(np.float64, copy=False)
    if values.ndim < 2:  # noqa: PLR2004
        msg = "data must have at least two dimensions"
        raise ValueError(msg)
    x_kind: AxisKind = "longitude" if geodetic else "cartesian"
    y_kind: AxisKind = "latitude" if geodetic else "cartesian"
    source_x = cells(x, x_kind, "x")[1]
    source_y = cells(y, y_kind, "y")[1]
    target_xb = cells(target_x, x_kind, "target_x")[1]
    target_yb = cells(target_y, y_kind, "target_y")[1]

    # The first and second axes are moved to the end, the other dimensions
    # stack the fields to regrid.
    values = np.moveaxis(values, axes, (-2, -1))
    shape = values.shape[:-2]
    nx, ny = values.shape[-2:]
    if (nx, ny) != (source_x.shape[0], source_y.shape[0]):
        msg = (
            f"the shape of data along the axes, {(nx, ny)}, does not match "
            f"the source grid, {(source_x.shape[0], source_y.shape[0])}"
        )
        raise ValueError(msg)
    values = np.ascontiguousarray(values).reshape(-1, ny)

    result = core.regrid.conservative(
        values,
        source_x,
        source_y,
        target_xb,
        target_yb,
        geodetic=geodetic,
        spheroid=spheroid,
        min_valid_fraction=min_valid_fraction,
        normalize=normalize,
        num_threads=num_threads,
    )
    result = result.reshape(*shape, target_xb.shape[0], target_yb.shape[0])
    return np.moveaxis(result, (-2, -1), axes)
