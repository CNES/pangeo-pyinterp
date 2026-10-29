# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests of the interpolation of reduced Gaussian grids."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from .. import reduced_gaussian as rg
from . import load_reduced_gaussian


N = 32


@pytest.fixture(scope="module")
def mesh() -> rg.Mesh:
    """Build an octahedral O32 mesh."""
    return rg.Mesh(rg.gaussian_latitudes(N), rg.octahedral_points(N))


def _nodes(mesh: rg.Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates of the points of the mesh, in storage order."""
    pl = mesh.points_per_parallel
    j = np.repeat(np.arange(mesh.ny), pl)
    i = np.arange(mesh.size) - mesh.offsets[j]
    return i * 360.0 / pl[j], mesh.latitudes[j]


def _field(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    return np.cos(np.radians(lat)) * np.cos(np.radians(lon))


def _queries(
    mesh: rg.Mesh, n: int = 20000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Random points inside the mesh, longitudes in several turns."""
    rs = np.random.default_rng(seed)
    lat = rs.uniform(mesh.latitudes[-1], mesh.latitudes[0], n)
    return rs.uniform(-400, 760, n), lat


def test_helpers() -> None:
    """Latitudes and points per parallel of standard meshes."""
    lats = rg.gaussian_latitudes(4)
    assert lats.shape == (8,)
    assert np.all(np.diff(lats) < 0)
    np.testing.assert_allclose(lats, -lats[::-1])
    assert list(rg.octahedral_points(2)) == [20, 24, 24, 20]
    assert list(rg.regular_points(2)) == [8, 8, 8, 8]


def test_mesh_validation() -> None:
    """Inconsistent meshes are refused."""
    with pytest.raises(ValueError, match="same size"):
        rg.Mesh(np.array([10.0, -10.0]), np.array([4], dtype=np.int64))
    with pytest.raises(ValueError, match="decreasing"):
        rg.Mesh(np.array([-10.0, 10.0]), np.array([4, 4], dtype=np.int64))
    with pytest.raises(ValueError, match="at least one point"):
        rg.Mesh(np.array([10.0, -10.0]), np.array([4, 0], dtype=np.int64))


def test_mesh_pickle(mesh: rg.Mesh) -> None:
    """A mesh survives pickling."""
    other = pickle.loads(pickle.dumps(mesh))
    assert other == mesh
    assert other.size == mesh.size == int(rg.octahedral_points(N).sum())


def test_locate(mesh: rg.Mesh) -> None:
    """Weights sum to one; points outside the mesh have no cell."""
    lon, lat = _queries(mesh)
    index, weight = mesh.locate(lon, lat)
    assert np.all(index >= 0)
    np.testing.assert_allclose(weight.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    index, weight = mesh.locate(np.array([0.0, np.nan]), np.array([90.0, 0.0]))
    assert np.all(index == -1)
    assert np.all(np.isnan(weight))


def test_nodes_reproduced(mesh: rg.Mesh) -> None:
    """The value at a grid point is that of the point."""
    lon, lat = _nodes(mesh)
    values = _field(lon, lat)
    grid = rg.Grid(mesh, {"f": values})
    result = rg.interpolate(grid, lon, lat)["f"]
    np.testing.assert_allclose(result, values, rtol=0, atol=1e-14)


def test_linear_in_latitude(mesh: rg.Mesh) -> None:
    """A field linear in latitude is interpolated exactly."""
    _, lat = _nodes(mesh)
    grid = rg.Grid(mesh, {"lat": lat.copy()})
    qlon, qlat = _queries(mesh)
    result = rg.interpolate(grid, qlon, qlat)["lat"]
    np.testing.assert_allclose(result, qlat, rtol=0, atol=1e-12)


def test_smooth_field_and_seam(mesh: rg.Mesh) -> None:
    """A smooth field, queried in any turn and across the seam."""
    lon, lat = _nodes(mesh)
    grid = rg.Grid(mesh, {"f": _field(lon, lat)})
    qlon, qlat = _queries(mesh)
    result = rg.interpolate(grid, qlon, qlat)["f"]
    np.testing.assert_allclose(result, _field(qlon, qlat), atol=5e-3)
    seam = np.array([0.0, np.nextafter(360.0, 0), 360.0, -1e-9, 359.99])
    result = rg.interpolate(grid, seam, np.full(seam.shape, 10.0))["f"]
    np.testing.assert_allclose(
        result, _field(seam, np.full(seam.shape, 10.0)), atol=1e-4
    )
    # The same point read in another turn gives the same value.
    a = rg.interpolate(grid, qlon, qlat)["f"]
    b = rg.interpolate(grid, qlon + 720.0, qlat)["f"]
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def test_outside_mesh(mesh: rg.Mesh) -> None:
    """Points beyond the first and last parallels are NaN or an error."""
    _, lat = _nodes(mesh)
    grid = rg.Grid(mesh, {"f": lat.copy()})
    result = rg.interpolate(grid, np.array([0.0, 0.0]), np.array([89.9, 0.0]))
    assert np.isnan(result["f"][0])
    assert np.isfinite(result["f"][1])
    with pytest.raises(IndexError, match="outside the mesh"):
        rg.interpolate(
            grid, np.array([0.0]), np.array([89.9]), bounds_error=True
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_3d_fields(mesh: rg.Mesh, dtype: type[np.floating]) -> None:
    """Columns of levels, with NaN values renormalized level by level."""
    lon, lat = _nodes(mesh)
    base = _field(lon, lat)
    levels: np.ndarray = np.stack(
        [base * k for k in range(1, 6)], axis=1
    ).astype(dtype)
    levels[:, 4] = np.nan  # a level missing everywhere
    surface: np.ndarray = base.astype(dtype)
    grid = rg.Grid(mesh, {"t": levels, "sp": surface})
    assert grid.field_names == ["t", "sp"]
    qlon, qlat = _queries(mesh, 500)
    result = rg.interpolate(grid, qlon, qlat)
    assert result["t"].shape == (500, 5)
    assert result["sp"].shape == (500,)
    assert result["t"].dtype == np.float64
    np.testing.assert_allclose(
        result["t"][:, 2],
        3 * result["sp"],
        rtol=1e-6 if dtype == np.float32 else 1e-12,
    )
    assert np.all(np.isnan(result["t"][:, 4]))
    only = rg.interpolate(grid, qlon, qlat, fields=["sp"])
    assert list(only) == ["sp"]
    with pytest.raises(ValueError, match="no field"):
        rg.interpolate(grid, qlon, qlat, fields=["q"])


def test_nan_corner_renormalized(mesh: rg.Mesh) -> None:
    """A NaN corner is left out and the other weights renormalized."""
    values = np.ones(mesh.size)
    qlon, qlat = np.array([10.3]), np.array([20.1])
    index, _ = mesh.locate(qlon, qlat)
    values[index[0, 0]] = np.nan
    grid = rg.Grid(mesh, {"f": values})
    assert rg.interpolate(grid, qlon, qlat)["f"][0] == pytest.approx(1.0)


def test_grid_validation(mesh: rg.Mesh) -> None:
    """Fields must match the mesh or the zone, and be contiguous floats."""
    with pytest.raises(ValueError, match="points, expected"):
        rg.Grid(mesh, {"f": np.zeros(10)})
    with pytest.raises(TypeError, match="float32 or float64"):
        rg.Grid(
            mesh,
            {"f": np.zeros(mesh.size, dtype=np.int32)},  # type: ignore[dict-item]
        )
    with pytest.raises(TypeError, match="C-contiguous"):
        rg.Grid(mesh, {"f": np.zeros((3, mesh.size)).T})


def test_grid_keeps_arrays_alive(mesh: rg.Mesh) -> None:
    """The grid reads the arrays without copy and keeps them alive."""
    values = np.arange(mesh.size, dtype=np.float64)
    grid = rg.Grid(mesh, {"f": values})
    assert grid.fields["f"] is values
    del values
    lon, lat = _nodes(mesh)
    result = rg.interpolate(grid, lon[:3], lat[:3])["f"]
    np.testing.assert_array_equal(result, [0.0, 1.0, 2.0])


def test_zone_from_points(mesh: rg.Mesh) -> None:
    """A zone from points interpolates as the global grid."""
    lon, lat = _nodes(mesh)
    values = np.stack([_field(lon, lat), lat], axis=1)
    qlon, qlat = _queries(mesh, 300)
    zone = rg.Zone.from_points(mesh, qlon, qlat)
    index, _ = mesh.locate(qlon, qlat)
    np.testing.assert_array_equal(zone.indices(), np.unique(index))
    ranges = zone.ranges()
    assert np.all(ranges[1:, 0] > ranges[:-1, 1])
    assert pickle.loads(pickle.dumps(zone)) == zone
    regional = rg.Grid(mesh, {"v": values[zone.indices()]}, zone=zone)
    full = rg.Grid(mesh, {"v": values})
    np.testing.assert_array_equal(
        rg.interpolate(regional, qlon, qlat)["v"],
        rg.interpolate(full, qlon, qlat)["v"],
    )
    # A point the zone was not built for reads a corner it does not hold.
    cand_lon, cand_lat = _queries(mesh, 1000, seed=7)
    cand, _ = mesh.locate(cand_lon, cand_lat)
    missing = np.flatnonzero(
        (zone.rows(cand.ravel()) < 0).reshape(-1, 4).any(axis=1)
    )
    other_lon, other_lat = cand_lon[missing[:1]], cand_lat[missing[:1]]
    assert other_lon.size == 1
    assert np.isnan(rg.interpolate(regional, other_lon, other_lat)["v"]).all()
    with pytest.raises(IndexError, match="zone of step 0"):
        rg.interpolate(regional, other_lon, other_lat, bounds_error=True)


@pytest.mark.parametrize(
    "box", [(-6.0, 30.0, 36.0, 46.0), (-30.0, 20.0, 20.0, 60.0)]
)
def test_zone_from_box(mesh: rg.Mesh, box: tuple[float, ...]) -> None:
    """Every point of a box, those of its edges included, is in its zone."""
    lon_min, lat_min, lon_max, lat_max = box
    zone = rg.Zone.from_box(mesh, *box)
    rs = np.random.default_rng(1)
    qlon = np.concatenate(
        (rs.uniform(lon_min, lon_max, 2000), [lon_min, lon_max] * 2)
    )
    qlat = np.concatenate(
        (
            rs.uniform(lat_min, lat_max, 2000),
            [lat_min, lat_min, lat_max, lat_max],
        )
    )
    index, _ = mesh.locate(qlon, qlat)
    assert np.all(zone.rows(index.ravel()) >= 0)


def test_zone_validation(mesh: rg.Mesh) -> None:
    """Ranges are sorted and merged; invalid ones are refused."""
    zone = rg.Zone(mesh, np.array([[10, 20], [0, 5], [5, 8]], dtype=np.int64))
    np.testing.assert_array_equal(zone.ranges(), [[0, 8], [10, 20]])
    assert zone.size == 18
    with pytest.raises(ValueError, match="outside the mesh"):
        rg.Zone(mesh, np.array([[0, mesh.size + 1]], dtype=np.int64))


def test_coalesce() -> None:
    """Merged ranges read the zone values back through the gather index."""
    values = np.arange(100) * 10
    ranges = np.array([[2, 5], [7, 9], [30, 31], [33, 40], [90, 100]])
    zone_values = np.concatenate([values[s:e] for s, e in ranges])
    for io_gap in (0, 2, 3, 30, 1000):
        merged, gather = rg.coalesce(ranges, io_gap)
        buffer = np.concatenate([values[s:e] for s, e in merged])
        np.testing.assert_array_equal(buffer[gather], zone_values)
    assert len(rg.coalesce(ranges, 0)[0]) == 5
    assert len(rg.coalesce(ranges, 1000)[0]) == 1


def _series(mesh: rg.Mesh, times: np.ndarray) -> rg.Series:
    """Build a series whose field at step k is (k + 1) * f."""
    lon, lat = _nodes(mesh)
    base = _field(lon, lat)
    series = rg.Series(mesh)
    for k, time in enumerate(times):
        series.append(time, rg.Grid(mesh, {"f": (k + 1) * base}))
    return series


def test_series_linear_and_nearest(mesh: rg.Mesh) -> None:
    """Linear blend between two steps, and the nearest one."""
    times = np.array(["2018-02-13T12", "2018-02-13T18"], dtype="M8[h]")
    series = _series(mesh, times)
    np.testing.assert_array_equal(series.dates, times.astype("M8[ns]"))
    qlon, qlat = _queries(mesh, 1000)
    ref = rg.interpolate(
        rg.Grid(mesh, {"f": _field(*_nodes(mesh))}), qlon, qlat
    )["f"]
    rs = np.random.default_rng(2)
    dates = times[0] + rs.integers(0, 6 * 60, qlon.size).astype("m8[m]")
    w = (dates - times[0]) / np.timedelta64(6, "h")
    result = rg.interpolate(series, qlon, qlat, dates)["f"]
    np.testing.assert_allclose(result, ref * (1 + w), rtol=1e-12, atol=1e-14)
    nearest = rg.interpolate(series, qlon, qlat, dates, time_method="nearest")
    np.testing.assert_allclose(
        nearest["f"], ref * np.where(w <= 0.5, 1, 2), atol=1e-14
    )
    outside = rg.interpolate(
        series, qlon[:1], qlat[:1], np.array(["2018-02-14"], dtype="M8[s]")
    )
    assert np.isnan(outside["f"]).all()


def test_series_rules(mesh: rg.Mesh) -> None:
    """Interval lattice and largest step gap."""
    times = np.array([0, 6, 18], dtype=np.int64)  # the 12h step is missing
    series = _series(mesh, times)
    qlon, qlat = np.full(4, 10.0), np.full(4, 20.0)
    dates = np.array([3, 9, 12, 15], dtype=np.int64)
    linear = rg.interpolate(series, qlon, qlat, dates)["f"]
    assert np.isfinite(linear).all()  # 6 -> 18 bracket 9, 12 and 15
    f0 = linear[0] / 1.5  # step k holds (k + 1) * f0; 3 is halfway 0 -> 6
    lattice = rg.interpolate(series, qlon, qlat, dates, interval=6)["f"]
    # [6, 12) and [12, 18) lost their 12h bound: each is read off the other.
    np.testing.assert_allclose(lattice, f0 * np.array([1.5, 2.0, 3.0, 3.0]))
    lattice = rg.interpolate(
        series, qlon, qlat, np.array([30] * 4), interval=6
    )["f"]
    assert np.isnan(lattice).all()  # [30, 36) has neither bound
    gap = rg.interpolate(series, qlon, qlat, dates, max_step_gap=6)["f"]
    assert np.isfinite(gap[0])
    assert np.isnan(gap[1:]).all()
    with pytest.raises(IndexError, match="outside the series"):
        rg.interpolate(
            series, qlon, qlat, dates, max_step_gap=6, bounds_error=True
        )


def test_series_validation(mesh: rg.Mesh) -> None:
    """Steps must share the mesh, the fields and the kind of dates."""
    series = _series(mesh, np.array([0, 6], dtype=np.int64))
    assert len(series) == 2
    with pytest.raises(ValueError, match="already has a step"):
        series.append(0, rg.Grid(mesh, {"f": np.zeros(mesh.size)}))
    with pytest.raises(ValueError, match="other fields"):
        series.append(12, rg.Grid(mesh, {"g": np.zeros(mesh.size)}))
    other = rg.Mesh(rg.gaussian_latitudes(4), rg.octahedral_points(4))
    with pytest.raises(ValueError, match="another mesh"):
        series.append(12, rg.Grid(other, {"f": np.zeros(other.size)}))
    with pytest.raises(TypeError, match="integer"):
        series.append(
            np.datetime64("2018-01-01"),
            rg.Grid(mesh, {"f": np.zeros(mesh.size)}),
        )
    grid = rg.Grid(mesh, {"f": np.zeros(mesh.size)})
    with pytest.raises(ValueError, match="use a Series"):
        rg.interpolate(grid, [0.0], [0.0], np.array([0]))
    with pytest.raises(ValueError, match="needs the dates"):
        rg.interpolate(series, [0.0], [0.0])
    series.drop_before(6)
    np.testing.assert_array_equal(series.times, [6])


def test_map_availability() -> None:
    """Missing bounds of the analysis interval, as c3po counts them."""
    hours = np.timedelta64(1, "h")
    origin = np.datetime64("2025-01-01T00", "ns")
    # The 06:00 step could not be fetched.
    steps = origin + np.array([0, 12, 18]) * hours
    dates = origin + np.array([3, 6, 9, 12, 15, 21, 25]) * hours
    flags = rg.map_availability(steps, dates, np.timedelta64(6, "h"))
    np.testing.assert_array_equal(flags, [1, 1, 1, 0, 0, 1, 2])
    # A forecast series is read on its own lattice (03, 09, 15...).
    steps = origin + np.array([9, 15]) * hours
    flags = rg.map_availability(
        steps, origin + np.array([12, 14]) * hours, np.timedelta64(6, "h")
    )
    np.testing.assert_array_equal(flags, [0, 0])
    assert rg.map_availability([], np.array([0]), 6)[0] == 2


def test_time_zones(mesh: rg.Mesh) -> None:
    """Per-step zones give the same result as global grids."""
    rs = np.random.default_rng(3)
    qlon, qlat = _queries(mesh, 2000)
    lon, lat = _nodes(mesh)
    base = np.stack([_field(lon, lat), lat], axis=1)
    times = np.array([0, 6, 18, 24], dtype=np.int64)  # 12h is missing
    dates = rs.integers(0, 25, qlon.size)
    cases: tuple[tuple[rg.TimeMethod, int | None], ...] = (
        ("linear", None),
        ("nearest", None),
        ("linear", 6),
        ("nearest", 6),
    )
    for method, interval in cases:
        zones = rg.time_zones(
            mesh,
            times,
            dates,
            qlon,
            qlat,
            time_method=method,
            interval=interval,
        )
        assert all(zone.size < mesh.size for zone in zones)
        regional = rg.Series(mesh)
        full = rg.Series(mesh)
        for k, (time, zone) in enumerate(zip(times, zones, strict=True)):
            values = (k + 1) * base
            regional.append(
                time, rg.Grid(mesh, {"v": values[zone.indices()]}, zone=zone)
            )
            full.append(time, rg.Grid(mesh, {"v": values}))
        np.testing.assert_array_equal(
            rg.interpolate(
                regional,
                qlon,
                qlat,
                dates,
                time_method=method,
                interval=interval,
                bounds_error=True,
            )["v"],
            rg.interpolate(
                full, qlon, qlat, dates, time_method=method, interval=interval
            )["v"],
        )


def test_apply_python(mesh: rg.Mesh) -> None:
    """A batched callback sees the interpolated columns and the points."""
    lon, lat = _nodes(mesh)
    base = _field(lon, lat)
    levels = np.stack([base * k for k in range(1, 4)], axis=1)
    a, b = np.arange(4.0), np.ones(4)
    grid = rg.Grid(mesh, {"t": levels, "sp": base}, a=a, b=b)
    qlon, qlat = _queries(mesh, 1000)
    alt = np.linspace(0, 1, qlon.size)
    calls = []

    def callback(
        columns: dict[str, np.ndarray],
        points: dict[str, np.ndarray],
        ca: np.ndarray | None,
        cb: np.ndarray | None,
    ) -> np.ndarray:
        calls.append(len(points["lon"]))
        np.testing.assert_array_equal(ca, a)
        np.testing.assert_array_equal(cb, b)
        return np.stack(
            [columns["t"].sum(axis=1) + points["alt"], columns["sp"]], axis=1
        )

    result = rg.apply(
        grid, callback, qlon, qlat, args={"alt": alt}, n_out=2, chunk_size=300
    )
    assert calls == [300, 300, 300, 100]
    ref = rg.interpolate(grid, qlon, qlat)
    np.testing.assert_allclose(result[:, 0], ref["t"].sum(axis=1) + alt)
    np.testing.assert_allclose(result[:, 1], ref["sp"])

    def wrong(
        columns: dict[str, np.ndarray],
        points: dict[str, np.ndarray],
        ca: np.ndarray | None,
        cb: np.ndarray | None,
    ) -> np.ndarray:
        return np.zeros((len(points["lon"]), 3))

    with pytest.raises(ValueError, match="expected n_out=2"):
        rg.apply(grid, wrong, qlon, qlat, n_out=2)

    def failing(
        columns: dict[str, np.ndarray],
        points: dict[str, np.ndarray],
        ca: np.ndarray | None,
        cb: np.ndarray | None,
    ) -> np.ndarray:
        raise RuntimeError("from the callback")

    with pytest.raises(RuntimeError, match="from the callback"):
        rg.apply(grid, failing, qlon, qlat)


def test_apply_blend(mesh: rg.Mesh) -> None:
    """Blend the results of a nonlinear callback rather than its columns.

    Blending the results matches the blend of the callback on each step.
    """
    times = np.array([0, 6], dtype=np.int64)
    series = _series(mesh, times)
    qlon, qlat = _queries(mesh, 500)
    dates = np.full(qlon.size, 3, dtype=np.int64)

    def square(
        columns: dict[str, np.ndarray],
        points: dict[str, np.ndarray],
        a: np.ndarray | None,
        b: np.ndarray | None,
    ) -> np.ndarray:
        return columns["f"] ** 2

    columns = rg.apply(series, square, qlon, qlat, dates)
    results = rg.apply(series, square, qlon, qlat, dates, blend="results")
    f = rg.interpolate(series, qlon, qlat, dates)["f"]  # 1.5 * f0
    np.testing.assert_allclose(columns, f**2)
    f0 = f / 1.5
    np.testing.assert_allclose(results, 0.5 * f0**2 + 0.5 * (2 * f0) ** 2)


def test_column_kernel(mesh: rg.Mesh) -> None:
    """A numba kernel gives the results of the equivalent numpy callback."""
    pytest.importorskip("numba")
    times = np.array([0, 6], dtype=np.int64)
    lon, lat = _nodes(mesh)
    base = _field(lon, lat)
    series = rg.Series(mesh)
    for k, time in enumerate(times):
        levels = (k + 1) * np.stack([base * j for j in range(1, 4)], axis=1)
        series.append(
            time,
            rg.Grid(
                mesh,
                {"t": levels, "sp": (k + 1) * base},
                a=np.arange(4.0),
                b=np.ones(4),
            ),
        )
    qlon, qlat = _queries(mesh, 5000)
    dates = np.random.default_rng(4).integers(0, 7, qlon.size)
    alt = np.linspace(0, 1, qlon.size)

    @rg.column_kernel(n_out=2)
    def kernel(
        columns: np.ndarray,
        widths: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        point: np.ndarray,
        out: np.ndarray,
    ) -> int:
        total = 0.0
        for jx in range(widths[0]):
            total += columns[jx] ** 2
        out[0] = total * a[1] + point[3]  # point: lon, lat, time, alt
        out[1] = columns[widths[0]] * b[0] + point[2]
        return 0

    def callback(
        columns: dict[str, np.ndarray],
        points: dict[str, np.ndarray],
        a: np.ndarray | None,
        b: np.ndarray | None,
    ) -> np.ndarray:
        assert a is not None and b is not None
        return np.stack(
            [
                (columns["t"] ** 2).sum(axis=1) * a[1] + points["alt"],
                columns["sp"] * b[0] + points["time"],
            ],
            axis=1,
        )

    for blend in ("columns", "results"):
        native = rg.apply(
            series, kernel, qlon, qlat, dates, args={"alt": alt}, blend=blend
        )
        python = rg.apply(
            series,
            callback,
            qlon,
            qlat,
            dates,
            args={"alt": alt},
            n_out=2,
            blend=blend,
        )
        np.testing.assert_allclose(native, python, rtol=1e-12, atol=1e-12)
    single = rg.apply(
        series, kernel, qlon, qlat, dates, args={"alt": alt}, num_threads=1
    )
    np.testing.assert_array_equal(
        single,
        rg.apply(series, kernel, qlon, qlat, dates, args={"alt": alt}),
    )

    @rg.column_kernel()
    def failing(
        columns: np.ndarray,
        widths: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        point: np.ndarray,
        out: np.ndarray,
    ) -> int:
        out[0] = 1.0
        return 1 if point[1] > 0 else 0

    result = rg.apply(series, failing, qlon, qlat, dates)
    np.testing.assert_array_equal(np.isnan(result), qlat > 0)
    with pytest.raises(RuntimeError, match="kernel returned 1"):
        rg.apply(series, failing, qlon, qlat, dates, bounds_error=True)


def test_column_kernel_division_by_zero(mesh: rg.Mesh) -> None:
    """Give inf for a division by zero in a kernel, as in C.

    The default numba error model raises, which a cfunc cannot propagate.
    """
    pytest.importorskip("numba")
    grid = rg.Grid(mesh, {"f": np.ones(mesh.size)})

    @rg.column_kernel()
    def kernel(
        columns: np.ndarray,
        widths: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        point: np.ndarray,
        out: np.ndarray,
    ) -> int:
        out[0] = columns[0] / (point[1] - point[1])
        return 0

    result = rg.apply(grid, kernel, np.array([10.0]), np.array([20.0]))
    assert np.isposinf(result).all()


def test_native_kernel_needs_its_abi(mesh: rg.Mesh) -> None:
    """An object with an address but without the kernel ABI is refused."""

    class Foreign:
        address = 1

    grid = rg.Grid(mesh, {"f": np.ones(mesh.size)})
    with pytest.raises(TypeError, match="abi"):
        rg.apply(grid, Foreign(), np.array([10.0]), np.array([20.0]))  # type: ignore[arg-type]


def test_units_must_match(mesh: rg.Mesh) -> None:
    """Durations and dates of another kind than the series are refused."""
    series = _series(mesh, np.array([0, 6], dtype=np.int64))
    with pytest.raises(TypeError, match="integer in the unit"):
        rg.interpolate(
            series,
            [0.0],
            [0.0],
            np.array([3]),
            interval=np.timedelta64(6, "h"),
        )
    with pytest.raises(TypeError, match="as step_times"):
        rg.time_zones(
            mesh,
            np.array([0, 6]),
            np.array(["2018-01-01"], dtype="M8[s]"),
            [0.0],
            [0.0],
        )
    dated = _series(
        mesh, np.array(["2018-02-13T12", "2018-02-13T18"], dtype="M8[h]")
    )
    with pytest.raises(ValueError, match="resolution"):
        rg.interpolate(
            dated, [0.0], [0.0], np.array(["3000-01-01"], dtype="M8[s]")
        )


def test_huge_longitudes(mesh: rg.Mesh) -> None:
    """A longitude of any magnitude is reduced to a turn first."""
    index, weight = mesh.locate(np.array([1e300, -1e300]), np.zeros(2))
    assert np.all((index >= 0) & (index < mesh.size))
    np.testing.assert_allclose(weight.sum(axis=1), 1.0)
    zone = rg.Zone.from_box(mesh, 1e300, -10.0, 1e300 + 1.0, 10.0)
    assert 0 < zone.size < mesh.size
    with pytest.raises(ValueError, match="finite"):
        rg.Zone.from_box(mesh, np.inf, -10.0, np.inf, 10.0)


def test_mesh_signed_zero() -> None:
    """Latitudes of 0.0 and -0.0 describe the same mesh."""
    pl = np.array([4, 4, 4], dtype=np.int64)
    a = rg.Mesh(np.array([10.0, 0.0, -10.0]), pl)
    b = rg.Mesh(np.array([10.0, -0.0, -10.0]), pl)
    assert a == b
    series = rg.Series(a)
    series.append(0, rg.Grid(b, {"f": np.zeros(12)}))


def test_threads_agree(mesh: rg.Mesh) -> None:
    """One thread and all of them give the same results."""
    lon, lat = _nodes(mesh)
    grid = rg.Grid(mesh, {"f": _field(lon, lat)})
    qlon, qlat = _queries(mesh, 50000)
    np.testing.assert_array_equal(
        rg.interpolate(grid, qlon, qlat, num_threads=1)["f"],
        rg.interpolate(grid, qlon, qlat, num_threads=0)["f"],
    )


# ---------------------------------------------------------------------------
# A real grid: regional subset of the ECMWF N640 analyses of 2018-02-13
# ---------------------------------------------------------------------------

BOX = (-4.0, 42.0, 4.0, 48.0)


@pytest.fixture(scope="module")
def ecmwf() -> dict:
    """Load the dataset, its mesh, its zone and the 2D and 3D series."""
    ds = load_reduced_gaussian()
    mesh = rg.Mesh(ds["latitude"].values, ds["pl"].values.astype(np.int64))
    zone = rg.Zone(mesh, ds["ranges"].values.astype(np.int64))
    a, b = ds["a"].values, ds["b"].values
    surface = rg.Series(mesh)
    levels = rg.Series(mesh)
    for k, time in enumerate(ds["time"].values):
        surface.append(
            time,
            rg.Grid(
                mesh,
                {"sp": np.ascontiguousarray(ds["sp"].values[k])},
                zone=zone,
            ),
        )
        levels.append(
            time,
            rg.Grid(
                mesh,
                {
                    "t": np.ascontiguousarray(ds["t"].values[k]),
                    "q": np.ascontiguousarray(ds["q"].values[k]),
                    "sp": np.ascontiguousarray(ds["sp"].values[k]),
                },
                zone=zone,
                a=a,
                b=b,
            ),
        )
    return {
        "ds": ds,
        "mesh": mesh,
        "zone": zone,
        "surface": surface,
        "levels": levels,
    }


def _track(n: int = 2000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a pass through the box between 12h and 18h."""
    s = np.linspace(0.0, 1.0, n)
    lon = -3.5 + 7.0 * s
    lat = 42.5 + 5.0 * s
    time = np.datetime64("2018-02-13T12") + (s * 6 * 3600e9).astype("m8[ns]")
    return lon, lat, time


def test_ecmwf_mesh_and_zone(ecmwf: dict) -> None:
    """The mesh is the N640 one and the stored zone is that of the box."""
    mesh, zone = ecmwf["mesh"], ecmwf["zone"]
    assert mesh.ny == 1280
    assert mesh.size == 2140702
    assert zone == rg.Zone.from_box(mesh, *BOX)
    # The box crosses the Greenwich meridian: the end of a parallel and the
    # start of the next one touch, and make one range.
    ranges = zone.ranges()
    offsets = mesh.offsets
    assert np.any(
        (offsets[None, :] > ranges[:, :1]) & (offsets[None, :] < ranges[:, 1:])
    )
    lon, lat, _ = _track()
    sub = rg.Zone.from_points(mesh, lon, lat)
    assert np.all(zone.rows(sub.indices()) >= 0)


def test_ecmwf_nodes(ecmwf: dict) -> None:
    """The grid points of the zone are reproduced exactly."""
    mesh, zone, ds = ecmwf["mesh"], ecmwf["zone"], ecmwf["ds"]
    index = zone.indices()
    j = np.searchsorted(mesh.offsets, index, side="right") - 1
    pl = mesh.points_per_parallel[j]
    lon = (index - mesh.offsets[j]) * 360.0 / pl
    lat = mesh.latitudes[j]
    grid = ecmwf["levels"]
    result = rg.interpolate(
        grid, lon, lat, np.full(lon.size, ds["time"].values[0])
    )
    # Points on the edge of the zone lack a neighbour: keep the inner ones.
    ok = np.isfinite(result["sp"])
    assert ok.mean() > 0.8
    np.testing.assert_allclose(
        result["sp"][ok], ds["sp"].values[0][ok], rtol=1e-12
    )
    np.testing.assert_allclose(
        result["t"][ok], ds["t"].values[0][ok], rtol=1e-12
    )


def test_ecmwf_2d_series(ecmwf: dict) -> None:
    """Surface pressure along a pass, blended between 12h and 18h."""
    lon, lat, time = _track()
    series = ecmwf["surface"]
    result = rg.interpolate(series, lon, lat, time, bounds_error=True)["sp"]
    assert np.all((result > 80000) & (result < 105000))
    at12 = rg.interpolate(
        series, lon, lat, np.full(lon.size, series.dates[0])
    )["sp"]
    at18 = rg.interpolate(
        series, lon, lat, np.full(lon.size, series.dates[1])
    )["sp"]
    w = (time - series.dates[0]) / np.timedelta64(6, "h")
    np.testing.assert_allclose(result, (1 - w) * at12 + w * at18, rtol=1e-12)
    # Outside the box the zone holds no corner.
    outside = rg.interpolate(
        series, np.array([10.0]), np.array([45.0]), time[:1]
    )["sp"]
    assert np.isnan(outside).all()


def test_ecmwf_3d_series(ecmwf: dict) -> None:
    """Columns of temperature and humidity along a pass."""
    lon, lat, time = _track()
    result = rg.interpolate(ecmwf["levels"], lon, lat, time, bounds_error=True)
    assert result["t"].shape == (lon.size, 9)
    assert result["q"].shape == (lon.size, 9)
    assert np.all((result["t"] > 180) & (result["t"] < 320))
    assert np.all((result["q"] >= 0) & (result["q"] < 0.03))
    # The air is colder aloft than near the surface.
    assert np.all(result["t"][:, 0] < result["t"][:, -1])


def _dry_correction(sp: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Saastamoinen dry tropospheric correction at sea level, in metres."""
    return -2.2768e-5 * sp / (1.0 - 0.00266 * np.cos(2.0 * np.radians(lat)))


def test_ecmwf_apply_numpy(ecmwf: dict) -> None:
    """A user-side correction from the columns, as a batched callback."""
    lon, lat, time = _track()
    series = ecmwf["levels"]

    def callback(
        columns: dict[str, np.ndarray],
        points: dict[str, np.ndarray],
        a: np.ndarray | None,
        b: np.ndarray | None,
    ) -> np.ndarray:
        assert a is not None and b is not None
        sp = columns["sp"]
        # Pressure of the lowest half level is the surface pressure.
        bottom = a[-1] + b[-1] * sp
        return np.stack(
            [_dry_correction(sp, points["lat"]), bottom - sp], axis=1
        )

    result = rg.apply(
        series, callback, lon, lat, time, n_out=2, chunk_size=512
    )
    sp = rg.interpolate(series, lon, lat, time)["sp"]
    np.testing.assert_allclose(result[:, 0], _dry_correction(sp, lat))
    np.testing.assert_allclose(result[:, 1], 0.0, atol=1e-6)
    assert np.all((result[:, 0] > -2.4) & (result[:, 0] < -1.8))


def test_ecmwf_apply_kernel(ecmwf: dict) -> None:
    """The same correction as a numba kernel."""
    pytest.importorskip("numba")
    lon, lat, time = _track()
    series = ecmwf["levels"]

    @rg.column_kernel()
    def kernel(
        columns: np.ndarray,
        widths: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        point: np.ndarray,
        out: np.ndarray,
    ) -> int:
        sp = columns[widths[0] + widths[1]]  # fields t, q, then sp
        out[0] = (
            -2.2768e-5
            * sp
            / (1.0 - 0.00266 * np.cos(2.0 * np.radians(point[1])))
        )
        return 0

    for blend in ("columns", "results"):
        result = rg.apply(series, kernel, lon, lat, time, blend=blend)
        sp = rg.interpolate(series, lon, lat, time)["sp"]
        np.testing.assert_allclose(
            result, _dry_correction(sp, lat), rtol=1e-12
        )


def test_ecmwf_time_zones(ecmwf: dict) -> None:
    """Per-step zones of a pass are subsets of the stored zone."""
    lon, lat, time = _track()
    mesh, zone = ecmwf["mesh"], ecmwf["zone"]
    zones = rg.time_zones(mesh, ecmwf["surface"].dates, time, lon, lat)
    for step in zones:
        assert 0 < step.size < zone.size
        assert np.all(zone.rows(step.indices()) >= 0)
