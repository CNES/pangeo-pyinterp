# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Compare pyinterp with scipy, xarray and xESMF on common tasks.

Each scenario times the libraries that are installed on the same inputs and
prints their speed relative to the first one. scipy, xarray and xESMF are
optional: the cases of a missing library are skipped.

pyinterp is timed on one thread and on all the cores. The other libraries
run on one thread, except the k-d tree of scipy, which is also timed with all
its workers. scipy and xarray do not wrap longitudes: they are given points
inside the longitudes of the grid, which pyinterp does not need.

Examples:
    Run every scenario with the sizes of the documentation::

        python -m benchmarks.compare

    Smaller sizes, and the results as JSON::

        python -m benchmarks.compare --quick -o compare.json

"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import importlib.util
import pathlib
import time
from typing import TYPE_CHECKING, Any

import numpy as np

import pyinterp
from pyinterp.geometry import geographic

from . import _runner


if TYPE_CHECKING:
    from collections.abc import Callable

#: Step of the source grid, in degrees.
STEP = 0.25

#: Number of neighbors of the inverse distance weighting.
K = 11


@dataclasses.dataclass(frozen=True)
class Scenario:
    """A task timed with several libraries."""

    #: Name of the scenario.
    name: str
    #: Description printed above the results.
    title: str
    #: Library label and case, the first one being the reference.
    cases: list[tuple[str, _runner.Case]]
    #: Measurements made once, printed below the results.
    notes: list[str] = dataclasses.field(default_factory=list)


def _has(module: str) -> bool:
    """Tell whether an optional library is installed."""
    return importlib.util.find_spec(module) is not None


def _source(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global grid of 0.25 degree, stored as (lon, lat)."""
    lon = np.arange(0, 360, STEP)
    lat = np.arange(-90 + STEP / 2, 90, STEP)
    data = (
        np.sin(np.radians(2 * lon))[:, None] * np.cos(np.radians(lat))[None, :]
    )
    return lon, lat, data + 0.01 * rng.standard_normal(data.shape)


def _grid(lon: np.ndarray, lat: np.ndarray, data: np.ndarray) -> Any:  # noqa: ANN401
    """Build the pyinterp grid, with a periodic longitude."""
    return pyinterp.Grid(
        pyinterp.Axis(lon, period=360.0), pyinterp.Axis(lat), data
    )


def _data_array(lon: np.ndarray, lat: np.ndarray, data: np.ndarray) -> Any:  # noqa: ANN401
    """Build the xarray data array, stored as (lat, lon)."""
    import xarray as xr  # noqa: PLC0415

    return xr.DataArray(
        data.T,
        dims=("lat", "lon"),
        coords={
            "lon": ("lon", lon, {"units": "degrees_east"}),
            "lat": ("lat", lat, {"units": "degrees_north"}),
        },
        name="field",
    )


def _ecef(
    spheroid: geographic.Spheroid, lon: np.ndarray, lat: np.ndarray
) -> np.ndarray:
    """Convert positions on the ellipsoid to ECEF coordinates, with NumPy."""
    lon, lat = np.radians(lon), np.radians(lat)
    e2 = spheroid.first_eccentricity_squared()
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    radius = spheroid.semi_major_axis / np.sqrt(1 - e2 * sin_lat**2)
    return np.column_stack(
        (
            radius * cos_lat * np.cos(lon),
            radius * cos_lat * np.sin(lon),
            radius * (1 - e2) * sin_lat,
        )
    )


def _case(name: str, items: int, func: Callable[[], object]) -> _runner.Case:
    """Wrap a callable whose inputs are already built."""
    return _runner.Case(name, lambda: func, items)


def _points(ctx: _runner.Context) -> Scenario:
    """Bilinear interpolation at scattered points."""
    rng = ctx.rng()
    lon, lat, data = _source(rng)
    n = ctx.size(1_000_000, 100_000)
    x = rng.uniform(lon[0], lon[-1], n)
    y = rng.uniform(lat[0], lat[-1], n)
    grid = _grid(lon, lat, data)
    cases: list[tuple[str, _runner.Case]] = []

    if _has("scipy"):
        import scipy.interpolate  # noqa: PLC0415
        import scipy.ndimage  # noqa: PLC0415

        rgi = scipy.interpolate.RegularGridInterpolator(
            (lon, lat), data, bounds_error=False
        )
        xy = np.column_stack((x, y))
        cases.append(
            ("scipy RegularGridInterpolator", _case("rgi", n, lambda: rgi(xy)))
        )
        cases.append(
            (
                "scipy ndimage.map_coordinates, order=1",
                _case(
                    "map_coordinates",
                    n,
                    lambda: scipy.ndimage.map_coordinates(
                        data,
                        [(x - lon[0]) / STEP, (y - lat[0]) / STEP],
                        order=1,
                    ),
                ),
            )
        )
    if _has("xarray") and _has("scipy"):
        import xarray as xr  # noqa: PLC0415

        array = _data_array(lon, lat, data)
        xi, yi = xr.DataArray(x, dims="p"), xr.DataArray(y, dims="p")
        cases.append(
            (
                "xarray DataArray.interp",
                _case("xarray", n, lambda: array.interp(lon=xi, lat=yi)),
            )
        )
    for label, threads in (("1 thread", 1), ("all cores", 0)):
        cases.append(
            (
                f"pyinterp bivariate, {label}",
                _case(
                    f"pyinterp/{threads}",
                    n,
                    functools.partial(
                        pyinterp.bivariate, grid, x, y, num_threads=threads
                    ),
                ),
            )
        )
    return Scenario(
        "points",
        f"Bilinear interpolation at {n:,} scattered points of a global "
        "0.25 degree grid",
        cases,
    )


def _regrid(ctx: _runner.Context) -> Scenario:
    """Bilinear interpolation of a grid onto a finer grid."""
    rng = ctx.rng()
    lon, lat, data = _source(rng)
    step = ctx.size(0.1, 0.5)
    new_lon = np.arange(-180, 180, step) + step / 2
    new_lat = np.arange(-80, 80, step) + step / 2
    n = new_lon.size * new_lat.size
    # scipy and xarray do not wrap the longitudes of the grid.
    wrapped = new_lon % 360
    cases: list[tuple[str, _runner.Case]] = []
    notes: list[str] = []

    if _has("xesmf"):
        import xarray as xr  # noqa: PLC0415
        import xesmf  # noqa: PLC0415

        source = _data_array(lon, lat, data).to_dataset()
        target = xr.Dataset(
            coords={"lon": ("lon", new_lon), "lat": ("lat", new_lat)}
        )
        # The weights are computed once, too slowly to be repeated.
        start = time.perf_counter()
        regridder = xesmf.Regridder(source, target, "bilinear", periodic=True)
        notes.append(
            "xESMF computed its weights in "
            f"{(time.perf_counter() - start) * 1e3:.1f} ms"
        )
        cases.append(
            (
                "xESMF, applying the weights",
                _case("xesmf", n, lambda: regridder(source.field)),
            )
        )
    if _has("scipy"):
        import scipy.interpolate  # noqa: PLC0415

        rgi = scipy.interpolate.RegularGridInterpolator(
            (lon, lat), data, bounds_error=False
        )
        mx, my = np.meshgrid(wrapped, new_lat, indexing="ij")
        cases.append(
            (
                "scipy RegularGridInterpolator",
                _case("rgi", n, lambda: rgi((mx, my))),
            )
        )
    if _has("xarray"):
        array = _data_array(lon, lat, data)
        if _has("scipy"):
            cases.append(
                (
                    "xarray DataArray.interp",
                    _case(
                        "xarray",
                        n,
                        lambda: array.interp(lon=wrapped, lat=new_lat),
                    ),
                )
            )
        import pyinterp.backends.xarray  # noqa: F401, PLC0415

        for label, threads in (("1 thread", 1), ("all cores", 0)):
            cases.append(
                (
                    f"pyinterp accessor, {label}",
                    _case(
                        f"pyinterp/{threads}",
                        n,
                        functools.partial(
                            array.pyinterp.interp,
                            lon=new_lon,
                            lat=new_lat,
                            num_threads=threads,
                        ),
                    ),
                )
            )
    return Scenario(
        "regrid",
        f"Bilinear regridding of a global 0.25 degree grid onto a {step} "
        f"degree grid ({n:,} points)",
        cases,
        notes,
    )


def _conservative(ctx: _runner.Context) -> Scenario:
    """Conservative regridding of a grid onto a coarser grid."""
    rng = ctx.rng()
    lon, lat, data = _source(rng)
    step = ctx.size(1.0, 2.0)
    new_lon = np.arange(-180, 180, step) + step / 2
    new_lat = np.arange(-90, 90, step) + step / 2
    n = data.size
    cases: list[tuple[str, _runner.Case]] = []
    notes: list[str] = []

    if _has("xesmf"):
        import xarray as xr  # noqa: PLC0415
        import xesmf  # noqa: PLC0415

        source = _data_array(lon, lat, data).to_dataset()
        source = source.assign_coords(
            lon_b=("lon_b", np.arange(-STEP / 2, 360, STEP)),
            lat_b=("lat_b", np.linspace(-90, 90, lat.size + 1)),
        )
        target = xr.Dataset(
            coords={
                "lon": ("lon", new_lon),
                "lat": ("lat", new_lat),
                "lon_b": ("lon_b", np.arange(-180, 180 + step / 2, step)),
                "lat_b": ("lat_b", np.arange(-90, 90 + step / 2, step)),
            }
        )
        start = time.perf_counter()
        regridder = xesmf.Regridder(
            source, target, "conservative", periodic=True
        )
        notes.append(
            "xESMF computed its weights in "
            f"{(time.perf_counter() - start) * 1e3:.1f} ms"
        )
        cases.append(
            (
                "xESMF, applying the weights",
                _case("xesmf", n, lambda: regridder(source.field)),
            )
        )
    if _has("xarray"):
        from pyinterp.backends import xarray as _  # noqa: F401, PLC0415

        array = _data_array(lon, lat, data)
        regrid = functools.partial(
            array.pyinterp.regrid, lon=new_lon, lat=new_lat
        )
        label = "da.pyinterp.regrid"
    else:
        regrid = functools.partial(
            pyinterp.regrid.conservative, data, lon, lat, new_lon, new_lat
        )
        label = "pyinterp.regrid.conservative"
    for name, threads in (("one thread", 1), ("all cores", 0)):
        cases.append(
            (
                f"{label}, {name}",
                _case(
                    f"pyinterp/{threads}",
                    n,
                    functools.partial(regrid, num_threads=threads),
                ),
            )
        )
    return Scenario(
        "conservative",
        f"Conservative regridding of a global 0.25 degree grid onto a {step} "
        f"degree grid ({n:,} source cells)",
        cases,
        notes,
    )


def _cubic(ctx: _runner.Context) -> Scenario:
    """Cubic interpolation at scattered points."""
    rng = ctx.rng()
    lon, lat, data = _source(rng)
    n = ctx.size(1_000_000, 100_000)
    x = rng.uniform(lon[0], lon[-1], n)
    y = rng.uniform(lat[0] + 1, lat[-1] - 1, n)
    grid = _grid(lon, lat, data)
    cases: list[tuple[str, _runner.Case]] = []

    if _has("scipy"):
        import scipy.ndimage  # noqa: PLC0415

        # The B-spline coefficients are computed once for the whole grid.
        coefficients = scipy.ndimage.spline_filter(data, order=3)
        cases.append(
            (
                "scipy ndimage.map_coordinates, order=3",
                _case(
                    "map_coordinates",
                    n,
                    lambda: scipy.ndimage.map_coordinates(
                        coefficients,
                        [(x - lon[0]) / STEP, (y - lat[0]) / STEP],
                        order=3,
                        prefilter=False,
                    ),
                ),
            )
        )
    for size in (2, 3):
        for label, threads in (("1 thread", 1), ("all cores", 0)):
            cases.append(
                (
                    f"pyinterp bicubic, {2 * size}x{2 * size} window, {label}",
                    _case(
                        f"pyinterp/{size}/{threads}",
                        n,
                        functools.partial(
                            pyinterp.bivariate,
                            grid,
                            x,
                            y,
                            method="bicubic",
                            half_window_size_x=size,
                            half_window_size_y=size,
                            num_threads=threads,
                        ),
                    ),
                )
            )
    return Scenario(
        "cubic",
        f"Cubic interpolation at {n:,} scattered points of a global "
        "0.25 degree grid",
        cases,
    )


def _idw(ctx: _runner.Context, *, cartesian: bool) -> Scenario:
    """Inverse distance weighting of scattered observations.

    In geodetic mode, both libraries are given longitudes and latitudes: the
    time of SciPy includes their conversion to ECEF coordinates, which
    pyinterp does in its own time. In Cartesian mode, both are given the ECEF
    coordinates.
    """
    rng = ctx.rng()
    n = ctx.size(1_000_000, 100_000)
    lon, lat = rng.uniform(-180, 180, n), rng.uniform(-80, 80, n)
    values = np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    query_lon, query_lat = rng.uniform(-180, 180, n), rng.uniform(-80, 80, n)
    spheroid = geographic.Spheroid()
    ecef = _ecef(spheroid, lon, lat)
    if cartesian:
        tree = pyinterp.RTree3D()
        tree.packing(ecef, values)
        queries = _ecef(spheroid, query_lon, query_lat)
    else:
        tree = pyinterp.RTree3D(spheroid=spheroid)
        tree.packing(np.column_stack((lon, lat)), values)
        queries = np.column_stack((query_lon, query_lat))
    cases: list[tuple[str, _runner.Case]] = []

    if _has("scipy"):
        import scipy.spatial  # noqa: PLC0415

        # The k-d tree works on the same ECEF coordinates as pyinterp.
        kdtree = scipy.spatial.cKDTree(ecef)

        def idw(workers: int) -> np.ndarray:
            points = (
                queries if cartesian else _ecef(spheroid, query_lon, query_lat)
            )
            distance, index = kdtree.query(points, k=K, workers=workers)
            weight = 1.0 / np.maximum(distance, 1e-12) ** 2
            return (weight * values[index]).sum(axis=1) / weight.sum(axis=1)

        for label, workers in (("1 thread", 1), ("all cores", -1)):
            cases.append(
                (
                    f"scipy cKDTree + NumPy, {label}",
                    _case(
                        f"ckdtree/{workers}",
                        n,
                        functools.partial(idw, workers),
                    ),
                )
            )
    mode = "Cartesian" if cartesian else "geodetic"
    for label, threads in (("1 thread", 1), ("all cores", 0)):
        cases.append(
            (
                f"pyinterp RTree3D, {mode}, {label}",
                _case(
                    f"pyinterp/{threads}",
                    n,
                    functools.partial(
                        pyinterp.inverse_distance_weighting,
                        tree,
                        queries,
                        k=K,
                        num_threads=threads,
                    ),
                ),
            )
        )
    given = "ECEF coordinates" if cartesian else "longitudes and latitudes"
    return Scenario(
        "cartesian" if cartesian else "scattered",
        f"Inverse distance weighting ({K} neighbors) of {n:,} scattered "
        f"observations at {n:,} points, given their {given}",
        cases,
    )


SCENARIOS: dict[str, Callable[[_runner.Context], Scenario]] = {
    "points": _points,
    "regrid": _regrid,
    "conservative": _conservative,
    "cubic": _cubic,
    "scattered": functools.partial(_idw, cartesian=False),
    "cartesian": functools.partial(_idw, cartesian=True),
}


def _usage() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.compare",
        description=__doc__.split("\n", 1)[0],
    )
    parser.add_argument(
        "scenarios",
        nargs="*",
        metavar="scenario",
        help=f"scenarios to run (default: all): {', '.join(SCENARIOS)}",
    )
    parser.add_argument(
        "-q", "--quick", action="store_true", help="use reduced sizes"
    )
    parser.add_argument(
        "-o", "--output", type=pathlib.Path, help="write the results as JSON"
    )
    args = parser.parse_args()
    unknown = set(args.scenarios) - set(SCENARIOS)
    if unknown:
        parser.error(f"unknown scenario: {', '.join(sorted(unknown))}")
    return args


def main() -> int:
    """Run the selected scenarios."""
    args = _usage()
    ctx = _runner.Context(quick=args.quick, threads=0)
    timer = _runner.Timer(min_repeat=3, min_time=1.0)
    results: list[_runner.Result] = []
    for name in args.scenarios or SCENARIOS:
        scenario = SCENARIOS[name](ctx)
        print(f"\n{scenario.title}")
        reference: float | None = None
        for label, case in scenario.cases:
            result = timer.run(scenario.name, case)
            results.append(result)
            reference = reference or result.best
            print(
                f"  {label:<50} {result.best * 1e3:10.1f} ms "
                f"{result.throughput / 1e6:9.1f} Mpts/s "
                f"{reference / result.best:8.1f}x",
                flush=True,
            )
        for note in scenario.notes:
            print(f"  {note}")
    if args.output is not None:
        _runner.save(args.output, _runner.metadata(ctx, timer), results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
