# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests of the conservative regridding."""

from __future__ import annotations

import numpy as np
import pytest

from .. import core, regrid
from ..core.geometry.geographic import Spheroid


def _global_axes(step: float) -> tuple[np.ndarray, np.ndarray]:
    """Centers of a global grid in [0, 360) x [-90, 90]."""
    lon = np.arange(0, 360, step) + step / 2
    lat = np.arange(-90, 90, step) + step / 2
    return lon, lat


def _areas(
    lon: np.ndarray, lat: np.ndarray, spheroid: Spheroid | None = None
) -> np.ndarray:
    """Areas of the cells, up to the square of the radius."""
    x_bounds = regrid.cell_bounds(lon, "longitude")
    y_bounds = core.regrid.latitude_measure(
        regrid.cell_bounds(lat, "latitude"), spheroid
    )
    return np.abs(
        np.outer(
            np.radians(np.diff(x_bounds, axis=1)[:, 0]),
            np.diff(y_bounds, axis=1)[:, 0],
        )
    )


def _field(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    x, y = np.meshgrid(np.radians(lon), np.radians(lat), indexing="ij")
    return np.cos(y) * np.sin(3 * x) + 0.5 * np.sin(2 * y) + 2


def test_cell_bounds() -> None:
    """Bounds from the centers of the cells."""
    lon, lat = _global_axes(1.0)
    bounds = regrid.cell_bounds(lon, "longitude")
    assert bounds.shape == (360, 2)
    assert bounds[0, 0] == 0 and bounds[-1, 1] == 360
    bounds = regrid.cell_bounds(lat[::-1], "latitude")
    assert bounds[0, 0] == 90 and bounds[-1, 1] == -90
    # Latitudes of the nodes: the outer cells are clamped at the poles
    bounds = regrid.cell_bounds(np.arange(-90, 91), "latitude")
    assert bounds[0, 0] == -90 and bounds[0, 1] == -89.5
    # Global longitude axis of irregular spacing
    bounds = regrid.cell_bounds([10, 100, 200, 300], "longitude")
    assert bounds[0, 0] == pytest.approx(-25) and bounds[-1, 1] == 335
    with pytest.raises(ValueError, match="two values"):
        regrid.cell_bounds([1.0])


def test_overlap_matrix() -> None:
    """Weights of the overlaps along one axis."""
    source = regrid.cell_bounds(np.arange(0.5, 360), "longitude")
    target = np.array([[-1.5, 1.5], [10, 12], [359, 361], [5.25, 5.75]])
    indptr, indices, weights = core.regrid.overlap_matrix(source, target, 360)
    assert indptr.tolist() == [0, 4, 6, 8, 9]
    rows = [
        dict(
            zip(
                indices[indptr[ix] : indptr[ix + 1]].tolist(),
                weights[indptr[ix] : indptr[ix + 1]].tolist(),
                strict=True,
            )
        )
        for ix in range(4)
    ]
    assert rows[0] == pytest.approx(
        {358: 1 / 6, 359: 1 / 3, 0: 1 / 3, 1: 1 / 6}
    )
    assert rows[1] == pytest.approx({10: 0.5, 11: 0.5})
    assert rows[2] == pytest.approx({359: 0.5, 0: 0.5})
    assert rows[3] == pytest.approx({5: 1.0})

    # Not periodic: the target cell is half covered
    indptr, indices, weights = core.regrid.overlap_matrix(
        source, np.array([[-1.0, 1.0]])
    )
    assert indices.tolist() == [0] and weights.tolist() == [0.5]

    with pytest.raises(ValueError, match="overlap"):
        core.regrid.overlap_matrix(
            np.array([[0.0, 2.0], [1.0, 3.0]]), np.array([[0.0, 1.0]])
        )


def test_latitude_measure() -> None:
    """Sines of the authalic latitudes."""
    lat = np.array([[-90.0, 0.0], [30.0, 90.0]])
    sphere = core.regrid.latitude_measure(lat)
    np.testing.assert_allclose(sphere, np.sin(np.radians(lat)), atol=1e-15)
    wgs84 = Spheroid()
    ellipsoid = core.regrid.latitude_measure(lat, wgs84)
    assert ellipsoid[0, 0] == -1 and ellipsoid[1, 1] == 1
    # The authalic latitude is smaller than the geodetic latitude, by at most
    # about 0.13 degrees.
    authalic = np.degrees(np.arcsin(ellipsoid[1, 0]))
    assert 29.85 < authalic < 30.0
    # Area of the zone between the equator and 30° on WGS84, from the
    # closed form of the area of the ellipsoid.
    radius = wgs84.authalic_radius()
    area = 2 * np.pi * radius**2 * ellipsoid[1, 0]
    a, e2 = wgs84.semi_major_axis, wgs84.first_eccentricity_squared()
    e = np.sqrt(e2)
    s = 0.5
    expected = (
        np.pi * a**2 * (1 - e2) * (s / (1 - e2 * s**2) + np.arctanh(e * s) / e)
    )
    assert area == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_constant(dtype: type) -> None:
    """A constant field is preserved."""
    lon, lat = _global_axes(0.25)
    data: np.ndarray = np.full((lon.size, lat.size), 3.5, dtype=dtype)
    result = regrid.conservative(
        data, lon, lat, np.arange(-180, 180) + 0.5, np.arange(-89.5, 90)
    )
    assert result.dtype == dtype
    assert result.shape == (360, 180)
    np.testing.assert_allclose(
        result, 3.5, rtol=1e-6 if dtype == np.float32 else 1e-13
    )


@pytest.mark.parametrize("spheroid", [None, Spheroid()])
@pytest.mark.parametrize("step", [0.7, 1.0, 3.0])
def test_global_integral(spheroid: Spheroid | None, step: float) -> None:
    """The integral over the globe is preserved, coarsening and refining."""
    lon, lat = _global_axes(0.25)
    data = _field(lon, lat)
    target_lon = np.arange(-180, 180, 360 / round(360 / step))
    target_lon += (target_lon[1] - target_lon[0]) / 2
    target_lat = np.linspace(-90, 90, round(180 / step) + 1)
    target_lat = (target_lat[1:] + target_lat[:-1]) / 2
    result = regrid.conservative(
        data, lon, lat, target_lon, target_lat, spheroid=spheroid
    )
    expected = np.sum(data * _areas(lon, lat, spheroid))
    assert np.sum(result * _areas(target_lon, target_lat, spheroid)) == (
        pytest.approx(expected, rel=1e-12)
    )
    back = regrid.conservative(
        result, target_lon, target_lat, lon, lat, spheroid=spheroid
    )
    assert np.sum(back * _areas(lon, lat, spheroid)) == pytest.approx(
        expected, rel=1e-12
    )


def test_block_mean() -> None:
    """Coarsening by an integer factor is the area-weighted block mean."""
    lon, lat = _global_axes(0.25)
    data = np.random.default_rng(0).normal(size=(lon.size, lat.size))
    areas = _areas(lon, lat)
    result = regrid.conservative(data, lon, lat, *_global_axes(1.0))
    sums = (data * areas).reshape(360, 4, 180, 4).sum(axis=(1, 3))
    expected = sums / areas.reshape(360, 4, 180, 4).sum(axis=(1, 3))
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)


def test_periodic() -> None:
    """The target grid can use another longitude convention."""
    lon, lat = _global_axes(0.25)
    data = _field(lon, lat)
    target_lon, target_lat = _global_axes(1.0)
    expected = regrid.conservative(data, lon, lat, target_lon, target_lat)
    result = regrid.conservative(data, lon, lat, target_lon - 180, target_lat)
    np.testing.assert_allclose(result, np.roll(expected, 180, axis=0))
    # Source in [-180, 180), decreasing latitudes, target in [0, 360)
    result = regrid.conservative(
        np.roll(data, 720, axis=0)[:, ::-1],
        lon - 180,
        lat[::-1],
        target_lon,
        target_lat,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12)
    # A target cell straddling the origin of the source axis
    result = regrid.conservative(data, lon, lat, [[-0.5, 0.5]], target_lat)
    block = np.concatenate((data[-2:], data[:2]))
    areas = _areas(lon, lat)
    areas = np.concatenate((areas[-2:], areas[:2]))
    expected = (block * areas).reshape(4, 180, 4).sum(
        axis=(0, 2)
    ) / areas.reshape(4, 180, 4).sum(axis=(0, 2))
    np.testing.assert_allclose(result[0], expected, rtol=1e-12)


def test_undefined_values() -> None:
    """NaN are ignored, or count as zero without normalization."""
    lon, lat = _global_axes(0.25)
    data = np.random.default_rng(1).uniform(1, 2, size=(lon.size, lat.size))
    data[:4, :4] = np.nan  # First target cell: no data
    data[4:8, :2] = np.nan  # Second target cell: half the cells are undefined
    target_lon, target_lat = _global_axes(1.0)
    result = regrid.conservative(data, lon, lat, target_lon, target_lat)
    assert np.isnan(result[0, 0])
    areas = _areas(lon, lat)[4:8, :4]
    block = data[4:8, :4]
    defined = ~np.isnan(block)
    expected = np.sum(np.where(defined, block * areas, 0)) / np.sum(
        areas[defined]
    )
    assert result[1, 0] == pytest.approx(expected, rel=1e-12)
    assert np.all(np.isfinite(result[2:]))

    # The undefined cells, the two southernmost rows, have less than half the
    # area of the target cell.
    fraction = np.sum(areas[defined]) / np.sum(areas)
    assert 0.5 < fraction < 1
    for threshold, undefined in (
        (fraction * 0.999, False),
        (fraction * 1.001, True),
    ):
        result = regrid.conservative(
            data,
            lon,
            lat,
            target_lon,
            target_lat,
            min_valid_fraction=threshold,
        )
        assert np.isnan(result[1, 0]) == undefined
        assert np.all(np.isfinite(result[2:]))

    result = regrid.conservative(
        data, lon, lat, target_lon, target_lat, normalize=False
    )
    assert np.isnan(result[0, 0])
    assert result[1, 0] == pytest.approx(expected * fraction, rel=1e-12)


def test_partial_coverage() -> None:
    """Target cells partially outside a regional source grid."""
    x = np.arange(10.5, 20)
    y = np.arange(0.5, 10)
    data = np.ones((x.size, y.size))
    y_bounds = [[4.0, 6.0]]
    result = regrid.conservative(
        data,
        x,
        y,
        [[9.0, 11.0], [12.0, 18.0], [19.0, 21.0]],
        y_bounds,
    )
    np.testing.assert_allclose(result, 1.0)
    result = regrid.conservative(
        data,
        x,
        y,
        [[9.0, 11.0], [12.0, 18.0]],
        y_bounds,
        min_valid_fraction=0.75,
    )
    assert np.isnan(result[0, 0]) and result[1, 0] == pytest.approx(1)
    result = regrid.conservative(
        data,
        x,
        y,
        [[9.0, 11.0], [12.0, 18.0], [30.0, 31.0]],
        y_bounds,
        normalize=False,
    )
    assert result[0, 0] == pytest.approx(0.5) and np.isnan(result[2, 0])


def test_cartesian() -> None:
    """Cartesian axes: no periodicity, lengths along both axes."""
    x = np.arange(0.5, 10)
    y = np.arange(0.5, 6)
    data = np.add.outer(x, 10 * y)
    result = regrid.conservative(
        data, x, y, np.arange(1, 10, 2), np.arange(1.5, 6, 3), geodetic=False
    )
    expected = data.reshape(5, 2, 2, 3).mean(axis=(1, 3))
    np.testing.assert_allclose(result, expected, rtol=1e-14)
    # 359.5 would wrap around in geodetic mode
    result = regrid.conservative(
        data,
        x,
        y,
        [[-1.5, -0.5]],
        [[1.0, 2.0]],
        geodetic=False,
    )
    assert np.isnan(result[0, 0])


def test_bounds() -> None:
    """Cells given by their bounds."""
    lon, lat = _global_axes(1.0)
    data = _field(lon, lat)
    target_lon, target_lat = _global_axes(5.0)
    expected = regrid.conservative(data, lon, lat, target_lon, target_lat)
    edges = np.arange(0, 361, 5.0)
    result = regrid.conservative(
        data,
        regrid.cell_bounds(lon, "longitude"),
        regrid.cell_bounds(lat, "latitude"),
        np.column_stack((edges[:-1], edges[1:])),
        regrid.cell_bounds(target_lat, "latitude"),
    )
    np.testing.assert_allclose(result, expected, rtol=1e-14)
    # Bounds in the reverse order (upper, lower) are accepted
    result = regrid.conservative(
        data,
        lon,
        regrid.cell_bounds(lat, "latitude")[:, ::-1],
        target_lon,
        target_lat,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-14)
    with pytest.raises(ValueError, match="give its bounds instead"):
        regrid.conservative(data, lon, lat, [0.0], target_lat)
    with pytest.raises(ValueError, match="give its bounds instead"):
        regrid.conservative(data, lon, lat, target_lon, 45.0)
    with pytest.raises(ValueError, match=r"an \(n, 2\) array"):
        regrid.conservative(data, lon, lat, np.zeros((3, 3)), target_lat)
    with pytest.raises(ValueError, match="does not match"):
        regrid.conservative(data.T, lon, lat, target_lon, target_lat)


def test_extra_dimensions() -> None:
    """Leading and trailing dimensions are regridded independently."""
    lon, lat = _global_axes(1.0)
    target_lon, target_lat = _global_axes(2.5)
    fields = np.stack(
        [_field(lon, lat) * (ix + 1) for ix in range(6)]
    ).reshape(2, 3, lon.size, lat.size)
    fields[1, 2, 10:20, 30:40] = np.nan
    # (time, lat, depth, lon)
    data = np.moveaxis(fields, (2, 3), (3, 1))
    result = regrid.conservative(
        data, lon, lat, target_lon, target_lat, axes=(3, 1)
    )
    assert result.shape == (2, target_lat.size, 3, target_lon.size)
    for ix in range(2):
        for jx in range(3):
            expected = regrid.conservative(
                fields[ix, jx], lon, lat, target_lon, target_lat
            )
            np.testing.assert_allclose(
                result[ix, :, jx, :], expected.T, rtol=1e-14
            )
    single = regrid.conservative(
        data,
        lon,
        lat,
        target_lon,
        target_lat,
        axes=(-1, 1),
        num_threads=1,
    )
    np.testing.assert_array_equal(single, result)


def test_float32() -> None:
    """Single precision values are accumulated in double precision."""
    lon, lat = _global_axes(0.25)
    data = _field(lon, lat)
    target_lon, target_lat = _global_axes(1.0)
    expected = regrid.conservative(data, lon, lat, target_lon, target_lat)
    result = regrid.conservative(
        data.astype(np.float32), lon, lat, target_lon, target_lat
    )
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize("descending", ["source", "target", "both"])
def test_descending_latitudes(descending: str) -> None:
    """Latitudes from north to south, in the source or in the target."""
    lon, lat = _global_axes(0.25)
    data = _field(lon, lat)
    target_lon, target_lat = _global_axes(2.0)
    expected = regrid.conservative(data, lon, lat, target_lon, target_lat)
    if descending in ("source", "both"):
        data, lat = data[:, ::-1], lat[::-1]
    if descending in ("target", "both"):
        target_lat = target_lat[::-1]
        expected = expected[:, ::-1]
    result = regrid.conservative(data, lon, lat, target_lon, target_lat)
    np.testing.assert_allclose(result, expected, rtol=1e-13)
    areas = _areas(target_lon, target_lat)
    assert np.sum(result * areas) == pytest.approx(
        np.sum(data * _areas(lon, lat)), rel=1e-12
    )


@pytest.mark.parametrize("origin", [-10.0, 350.0])
def test_regional_source_across_origin(origin: float) -> None:
    """A regional source crossing the prime meridian, in any convention."""
    lon = origin + np.arange(0, 20, 0.25) + 0.125
    lat = np.arange(30, 50, 0.25) + 0.125
    data = _field(lon, lat)
    areas = _areas(lon, lat)
    # Target cells of 2°, in [-180, 180) and in [0, 360)
    for shift in (0.0, 360.0):
        target_lon = np.arange(-10, 10, 2.0) + 1 + shift
        result = regrid.conservative(
            data, lon, lat, target_lon, np.arange(31, 50, 2.0)
        )
        expected = (data * areas).reshape(10, 8, 10, 8).sum(
            axis=(1, 3)
        ) / areas.reshape(10, 8, 10, 8).sum(axis=(1, 3))
        np.testing.assert_allclose(result, expected, rtol=1e-12)
    # A target cell half outside the source: normalized, or not
    for normalize, factor in ((True, 1.0), (False, 0.5)):
        result = regrid.conservative(
            np.ones_like(data),
            lon,
            lat,
            [[5.0, 15.0]],
            [[30.0, 50.0]],
            normalize=normalize,
        )
        assert result[0, 0] == pytest.approx(factor)
    result = regrid.conservative(
        np.ones_like(data),
        lon,
        lat,
        [[5.0, 15.0]],
        [[30.0, 50.0]],
        min_valid_fraction=0.6,
    )
    assert np.isnan(result[0, 0])
