# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Interpolation of reduced Gaussian grids.

A reduced Gaussian grid, as ECMWF distributes its analyses, holds a fixed set
of parallels (the Gaussian latitudes) with a number of equally spaced points
that varies from parallel to parallel. This module interpolates such grids at
arbitrary points from numeric inputs only: the latitudes, the number of points
per parallel and the values, so no GRIB library is needed.

The workflow separates the mesh, which is metadata, from the values:

1. Build the :class:`Mesh` from the latitudes and the points per parallel.
2. Optionally build a :class:`Zone` from the points to interpolate, and read
   only the values of the zone (:meth:`Zone.indices`, :meth:`Zone.ranges`).
3. Build a :class:`Grid` per time step, and a :class:`Series` of them to
   interpolate in time.
4. :func:`interpolate` the fields, or :func:`apply` a function to their
   interpolated columns, such as a tropospheric correction written in Python.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np

from . import core
from .core.config import reduced_gaussian as _config


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ColumnKernel",
    "Grid",
    "Mesh",
    "Series",
    "Zone",
    "apply",
    "coalesce",
    "column_kernel",
    "gaussian_latitudes",
    "interpolate",
    "map_availability",
    "octahedral_points",
    "regular_points",
    "time_zones",
]

Mesh = core.reduced_gaussian.Mesh
Zone = core.reduced_gaussian.Zone
Grid = core.reduced_gaussian.Grid

#: Interpolation between the two steps of a series.
TimeMethod = Literal["linear", "nearest"]

#: When a callback meets the time blend of a series.
Blend = Literal["columns", "results"]

#: A date: ``numpy.datetime64``, or an integer in the unit of the series.
DateLike = np.datetime64 | np.timedelta64 | int

#: A duration: ``numpy.timedelta64``, an integer in the unit of the series,
#: or None.
Duration = np.timedelta64 | int | None

#: Resolution of the dates handled by a series given datetime64 values.
_TIME_UNIT = "ns"


def gaussian_latitudes(n: int) -> NDArray[np.float64]:
    """Compute the latitudes of a Gaussian grid.

    Args:
        n: Number of parallels between a pole and the equator (the ``N`` of
            an ``N320`` or ``O1280`` grid).

    Returns:
        The ``2 n`` latitudes in degrees, north to south.

    """
    if n < 1:
        raise ValueError("n must be positive")
    roots, _ = np.polynomial.legendre.leggauss(2 * n)
    return np.degrees(np.arcsin(roots))[::-1].copy()


def octahedral_points(n: int) -> NDArray[np.int64]:
    """Compute the number of points per parallel of an octahedral grid.

    Args:
        n: Number of parallels between a pole and the equator.

    Returns:
        The ``2 n`` numbers of points, north to south: ``20, 24, ...,
        4 n + 16`` then back.

    """
    if n < 1:
        raise ValueError("n must be positive")
    half = 4 * np.arange(1, n + 1, dtype=np.int64) + 16
    return np.concatenate((half, half[::-1]))


def regular_points(n: int) -> NDArray[np.int64]:
    """Compute the number of points per parallel of a regular Gaussian grid.

    Args:
        n: Number of parallels between a pole and the equator.

    Returns:
        ``2 n`` times ``4 n``.

    """
    if n < 1:
        raise ValueError("n must be positive")
    return np.full(2 * n, 4 * n, dtype=np.int64)


def _to_int64(values: ArrayLike, name: str) -> NDArray[np.int64]:
    """Convert dates to int64 in the unit of the series."""
    array = np.asarray(values)
    if array.dtype.kind in "Mm":
        error = ValueError(
            f"{name} cannot be represented at {_TIME_UNIT} resolution: "
            "dates must lie within 1678-2262 and use a unit no finer than "
            f"{_TIME_UNIT}"
        )
        try:
            converted = array.astype(f"{array.dtype.kind}8[{_TIME_UNIT}]")
        except OverflowError:
            raise error from None
        # A conversion to a coarser unit drops what the finer one holds, and
        # older numpy releases wrap around on an overflow: a value that does
        # not come back is not representable.
        defined = ~np.isnat(array)
        if np.any(converted.astype(array.dtype)[defined] != array[defined]):
            raise error
        return converted.view(np.int64)
    if array.dtype.kind in "iu" or array.size == 0:
        return array.astype(np.int64)
    raise TypeError(
        f"{name} must hold datetime64, timedelta64 or integer values, "
        f"got {array.dtype}"
    )


def _is_datetime(values: ArrayLike) -> bool | None:
    """Tell datetime64 dates (True) from integers (False); None if empty."""
    array = np.asarray(values)
    return None if array.size == 0 else array.dtype.kind == "M"


def _to_duration(value: Duration, name: str, datetime: bool | None) -> int:
    """Convert a duration (timedelta64 or integer) to the series unit.

    A duration is a timedelta64 for datetime64 dates, and an integer in the
    unit of the dates for integer dates: either other combination would be
    read in the wrong unit.
    """
    if value is None:
        return 0
    is_timedelta = np.asarray(value).dtype.kind == "m"
    if datetime is not None and datetime != is_timedelta:
        raise TypeError(
            f"{name} must be "
            + (
                "a timedelta64, as the dates are datetime64"
                if datetime
                else "an integer in the unit of the dates, as they are "
                "integers"
            )
        )
    result = int(_to_int64(np.asarray(value).reshape(()), name))
    if result < 0:
        raise ValueError(f"{name} must not be negative")
    return result


class Series(core.reduced_gaussian.Series):
    """A time series of grids on one mesh.

    The dates may be ``numpy.datetime64`` values, handled internally at
    nanosecond resolution, or integers in a unit of your choice; a series
    holds one kind or the other. Each step may hold its own zone: see
    :func:`time_zones`.

    Args:
        mesh: The mesh shared by every step.

    """

    def __init__(self, mesh: Mesh) -> None:
        """Initialize the series."""
        super().__init__(mesh)
        self._datetime: bool | None = None

    def _check_kind(self, values: ArrayLike) -> None:
        is_datetime = np.asarray(values).dtype.kind == "M"
        if self._datetime is None:
            self._datetime = is_datetime
        elif self._datetime != is_datetime:
            raise TypeError(
                "the series holds "
                + ("datetime64" if self._datetime else "integer")
                + " dates"
            )

    def append(self, time: DateLike, grid: Grid) -> None:  # type: ignore[override]
        """Add a step.

        Args:
            time: Date of the step.
            grid: The grid of the step.

        """
        self._check_kind(time)
        super().append(
            int(_to_int64(np.asarray(time).reshape(()), "time")), grid
        )

    def drop_before(self, time: DateLike) -> None:  # type: ignore[override]
        """Remove the steps strictly before a date."""
        self._check_kind(time)
        super().drop_before(
            int(_to_int64(np.asarray(time).reshape(()), "time"))
        )

    @property
    def dates(self) -> NDArray[Any]:
        """Dates of the steps, sorted, as given to :meth:`append`."""
        times = np.asarray(self.times)
        if self._datetime:
            return times.view(f"datetime64[{_TIME_UNIT}]")
        return times

    def _dates_of(self, time: ArrayLike) -> NDArray[np.int64]:
        if self._datetime is not None:
            is_datetime = np.asarray(time).dtype.kind == "M"
            if self._datetime != is_datetime:
                raise TypeError(
                    "the dates of the points must be "
                    + ("datetime64" if self._datetime else "integers")
                    + ", as those of the series"
                )
        return _to_int64(time, "time")


def _make_config(
    *,
    datetime: bool | None = None,
    bounds_error: bool,
    num_threads: int,
    time_method: TimeMethod = "linear",
    interval: Duration = None,
    max_step_gap: Duration = None,
    blend: Blend = "columns",
    chunk_size: int = _config.ReducedGaussian().chunk_size,
) -> _config.ReducedGaussian:
    return (
        _config.ReducedGaussian()
        .with_bounds_error(bounds_error)
        .with_num_threads(num_threads)
        .with_time_method(time_method)
        .with_interval(_to_duration(interval, "interval", datetime))
        .with_max_step_gap(
            _to_duration(max_step_gap, "max_step_gap", datetime)
        )
        .with_blend(blend)
        .with_chunk_size(chunk_size)
    )


def _series_kind(source: Grid | Series) -> bool | None:
    """Whether a series holds datetime64 dates; None if unknown."""
    return source._datetime if isinstance(source, Series) else None


def _prepare_time(
    source: Grid | Series, time: ArrayLike | None
) -> NDArray[np.int64] | None:
    if time is None:
        return None
    if isinstance(source, Series):
        return source._dates_of(time)
    return _to_int64(time, "time")


def interpolate(
    source: Grid | Series,
    lon: ArrayLike,
    lat: ArrayLike,
    time: ArrayLike | None = None,
    *,
    fields: Sequence[str] | None = None,
    time_method: TimeMethod = "linear",
    interval: Duration = None,
    max_step_gap: Duration = None,
    bounds_error: bool = False,
    num_threads: int = 0,
) -> dict[str, NDArray[np.float64]]:
    """Interpolate the fields of a grid or a series at points.

    Each point is located once on the mesh: its cell then serves every
    field, every level and both time steps it reads.

    Args:
        source: A :class:`Grid`, or a :class:`Series` of grids in time.
        lon: Longitudes of the points, in degrees, in any turn.
        lat: Latitudes of the points, in degrees.
        time: Dates of the points, required for a series.
        fields: Names of the fields to interpolate; all by default.
        time_method: ``"linear"`` blends the two steps that bracket a date,
            ``"nearest"`` reads the nearest one.
        interval: Width of the analysis intervals (``timedelta64``, or an
            integer in the unit of the series). A date is then read between
            the two ends of the interval it falls in, on a lattice whose
            phase is the first step; an interval that lost one of its ends
            is read off the other alone, one that lost both is NaN. By
            default the two steps that bracket the date are read.
        max_step_gap: Largest time between the two steps a date reads; a
            wider pair is treated as a missing step.
        bounds_error: Raise an :class:`IndexError` for a point outside the
            mesh or the series, or reading a corner a zone does not hold,
            instead of returning NaN.
        num_threads: Number of threads; 0 uses all the cores.

    Returns:
        A dict from field name to values: shape ``(n,)`` for a field given as
        ``(points,)``, ``(n, levels)`` for one given as ``(points, levels)``.

    """
    return core.reduced_gaussian.interpolate(
        source,
        np.ascontiguousarray(lon, dtype=np.float64).ravel(),
        np.ascontiguousarray(lat, dtype=np.float64).ravel(),
        time=_prepare_time(source, time),
        fields=list(fields) if fields is not None else None,
        config=_make_config(
            datetime=_series_kind(source),
            bounds_error=bounds_error,
            num_threads=num_threads,
            time_method=time_method,
            interval=interval,
            max_step_gap=max_step_gap,
        ),
    )


def apply(
    source: Grid | Series,
    callback: Callable[..., ArrayLike] | ColumnKernel,
    lon: ArrayLike,
    lat: ArrayLike,
    time: ArrayLike | None = None,
    *,
    args: Mapping[str, ArrayLike] | None = None,
    fields: Sequence[str] | None = None,
    n_out: int | None = None,
    blend: Blend = "columns",
    chunk_size: int = _config.ReducedGaussian().chunk_size,
    time_method: TimeMethod = "linear",
    interval: Duration = None,
    max_step_gap: Duration = None,
    bounds_error: bool = False,
    num_threads: int = 0,
) -> NDArray[np.float64]:
    """Compute values per point from the interpolated columns of the fields.

    The columns are interpolated in C++, in parallel and without the GIL.
    Two kinds of callbacks consume them:

    * A Python callable, called on blocks of ``chunk_size`` points as
      ``callback(columns, points, a, b)``: ``columns`` maps each field to its
      interpolated values (as :func:`interpolate` returns them), ``points``
      holds ``lon``, ``lat``, ``time`` (for a series) and the arrays of
      ``args``, ``a`` and ``b`` are the hybrid coefficients of the half
      levels (or None). It returns an array of shape ``(block,)`` or
      ``(block, n_out)``. Write it with numpy operations over the block.
    * A :class:`ColumnKernel` made with :func:`column_kernel`: a numba
      function called for each point from several threads, without the GIL,
      for computations that do not vectorize over points.

    Args:
        source: A :class:`Grid`, or a :class:`Series` of grids in time.
        callback: The function applied to the columns.
        lon: Longitudes of the points, in degrees.
        lat: Latitudes of the points, in degrees.
        time: Dates of the points, required for a series.
        args: Extra values per point, passed to the callback by name (in
            this order in the ``point`` array of a kernel).
        fields: Names of the fields given to the callback; all by default.
        n_out: Number of values computed per point; by default that of the
            kernel, or 1.
        blend: For a series, ``"columns"`` blends the columns of the two
            steps then calls the callback once; ``"results"`` calls it on
            the columns of each step and blends its results, which costs
            twice as much: it interpolates in time the result rather than
            the fields, which differs for a callback nonlinear in them.
        chunk_size: Number of points per call of a Python callable.
        time_method: See :func:`interpolate`.
        interval: See :func:`interpolate`.
        max_step_gap: See :func:`interpolate`.
        bounds_error: Raise instead of returning NaN for a point that cannot
            be interpolated, or for which a kernel returns a non-zero code.
        num_threads: Number of threads; 0 uses all the cores.

    Returns:
        The values, shape ``(n,)`` if ``n_out`` is 1, else ``(n, n_out)``;
        NaN where a point cannot be interpolated.

    """
    if n_out is None:
        n_out = callback.n_out if isinstance(callback, ColumnKernel) else 1
    elif isinstance(callback, ColumnKernel) and n_out != callback.n_out:
        raise ValueError(
            f"n_out={n_out} differs from that of the kernel ({callback.n_out})"
        )
    lon = np.ascontiguousarray(lon, dtype=np.float64).ravel()
    point_args = (
        {
            name: np.ascontiguousarray(value, dtype=np.float64).ravel()
            for name, value in args.items()
        }
        if args
        else None
    )
    return core.reduced_gaussian.apply(
        source,
        callback,
        lon,
        np.ascontiguousarray(lat, dtype=np.float64).ravel(),
        time=_prepare_time(source, time),
        args=point_args,
        fields=list(fields) if fields is not None else None,
        n_out=n_out,
        config=_make_config(
            datetime=_series_kind(source),
            bounds_error=bounds_error,
            num_threads=num_threads,
            time_method=time_method,
            interval=interval,
            max_step_gap=max_step_gap,
            blend=blend,
            chunk_size=chunk_size,
        ),
    )


def _step_dates(
    step_times: ArrayLike, time: ArrayLike
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Sorted step dates, their order, and the dates of the points.

    Raises if the steps are not distinct, or if the dates of the points are
    not of the kind of those of the steps.
    """
    steps_kind, dates_kind = _is_datetime(step_times), _is_datetime(time)
    if None not in (steps_kind, dates_kind) and steps_kind != dates_kind:
        raise TypeError(
            "time must hold "
            + ("datetime64" if steps_kind else "integer")
            + " dates, as step_times"
        )
    steps = _to_int64(step_times, "step_times").ravel()
    order = np.argsort(steps, kind="stable")
    times = steps[order]
    if np.any(np.diff(times) == 0):
        raise ValueError("step_times holds duplicates")
    return times, order, _to_int64(time, "time").ravel()


def time_zones(
    mesh: Mesh,
    step_times: ArrayLike,
    time: ArrayLike,
    lon: ArrayLike,
    lat: ArrayLike,
    *,
    time_method: TimeMethod = "linear",
    interval: Duration = None,
    max_step_gap: Duration = None,
    num_threads: int = 0,
) -> list[Zone]:
    """Compute the zone each time step of a series needs.

    The zone of a step holds exactly the corners read by the points whose
    date reads that step, with the same rule as :func:`interpolate`. Build
    the zones before reading any file, read for each step only the values of
    its zone, and give it to the :class:`Grid` of the step. A step whose zone
    is empty is read by no point and can be left out of the series.

    Args:
        mesh: The mesh.
        step_times: Dates of the steps of the series.
        time: Dates of the points.
        lon: Longitudes of the points, in degrees.
        lat: Latitudes of the points, in degrees.
        time_method: See :func:`interpolate`.
        interval: See :func:`interpolate`.
        max_step_gap: See :func:`interpolate`.
        num_threads: Number of threads; 0 uses all the cores.

    Returns:
        One zone per step, in the order of ``step_times``.

    """
    times, order, dates = _step_dates(step_times, time)
    lon = np.ascontiguousarray(lon, dtype=np.float64).ravel()
    lat = np.ascontiguousarray(lat, dtype=np.float64).ravel()
    if not dates.shape == lon.shape == lat.shape:
        raise ValueError("time, lon and lat must have the same size")
    datetime = _is_datetime(step_times)
    # The rule of the series itself, so that a zone holds what the
    # interpolation will read.
    i0, i1, _ = core.reduced_gaussian.bracket(
        times,
        dates,
        _make_config(
            datetime=datetime,
            bounds_error=False,
            num_threads=num_threads,
            time_method=time_method,
            interval=interval,
            max_step_gap=max_step_gap,
        ),
    )
    zones: list[Zone] = [None] * len(times)  # type: ignore[list-item]
    for k, original in enumerate(order):
        mask = (i0 == k) | (i1 == k)
        zones[original] = Zone.from_points(
            mesh, lon[mask], lat[mask], num_threads=num_threads
        )
    return zones


def map_availability(
    step_times: ArrayLike, time: ArrayLike, interval: Duration
) -> NDArray[np.int8]:
    """Count the steps missing from the analysis interval of each date.

    The flag the altimetry products publish beside the meteorological
    corrections: a date reads the two ends of the interval it falls in, on
    the lattice of the series (see ``interval`` in :func:`interpolate`), and
    the flag counts those that are not in the series.

    * 0: both ends are present and the value is interpolated, or the date
      sits on a step that is present and is read off it.
    * 1: one end is missing; the value is read off the other alone.
    * 2: both ends are missing; nothing is computed.

    Args:
        step_times: Dates of the steps of the series.
        time: Dates of the points.
        interval: Width of the analysis intervals (``timedelta64``, or an
            integer in the unit of the dates).

    Returns:
        The count of missing steps per date.

    """
    times, _, dates = _step_dates(step_times, time)
    if (
        interval is None
        or _to_duration(interval, "interval", _is_datetime(step_times)) == 0
    ):
        raise ValueError("interval must be positive")
    if len(times) == 0:
        return np.full(dates.shape, 2, dtype=np.int8)
    # The steps the series reads for each date, by its own rule: none when
    # both ends are missing, one when an end is missing or the date sits on
    # a step, two otherwise.
    i0, i1, _ = core.reduced_gaussian.bracket(
        times,
        dates,
        _make_config(
            datetime=_is_datetime(step_times),
            bounds_error=False,
            num_threads=1,
            interval=interval,
        ),
    )
    on_step = np.isin(dates, times)
    flags = np.where(i0 < 0, 2, np.where((i1 != i0) | on_step, 0, 1))
    return flags.astype(np.int8)


def coalesce(
    ranges: ArrayLike, io_gap: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Merge the ranges of a zone for reading, without changing the zone.

    Many small reads cost more than a few larger ones: merging ranges closer
    than ``io_gap`` points reads the gaps too, then ``gather`` picks the
    values of the zone from what was read.

    Args:
        ranges: The ranges of a zone, as :meth:`Zone.ranges` returns them.
        io_gap: Largest gap, in points, read to merge two ranges.

    Returns:
        ``(merged, gather)``: the ranges to read, and the positions of the
        values of the zone in the concatenation of what they read.

    Example:
        >>> merged, gather = coalesce(zone.ranges(), io_gap=256)
        >>> buffer = np.concatenate(
        ...     [values[start:stop] for start, stop in merged]
        ... )
        >>> zone_values = buffer[gather]

    """
    ranges = np.asarray(ranges, dtype=np.int64).reshape(-1, 2)
    if len(ranges) == 0:
        return ranges.copy(), np.empty(0, dtype=np.int64)
    starts, stops = ranges[:, 0], ranges[:, 1]
    new = np.concatenate(([True], starts[1:] - stops[:-1] > io_gap))
    group = np.cumsum(new) - 1
    merged = np.stack(
        (starts[new], np.maximum.reduceat(stops, np.flatnonzero(new))), axis=1
    )
    offset = np.concatenate(([0], np.cumsum(merged[:, 1] - merged[:, 0])))
    lengths = stops - starts
    base = offset[group] + starts - merged[group, 0]
    first = np.repeat(
        base - np.concatenate(([0], np.cumsum(lengths)))[:-1], lengths
    )
    gather = first + np.arange(lengths.sum())
    return merged, gather


@dataclasses.dataclass(frozen=True)
class ColumnKernel:
    """A native column kernel, called per point without the GIL.

    Made by :func:`column_kernel`; ``address`` is that of a C function
    following the ABI documented in
    :func:`pyinterp.core.reduced_gaussian.apply`.
    """

    #: Tag of the calling convention, checked before the kernel is called.
    abi: ClassVar[str] = "pyinterp.reduced_gaussian.column_kernel/1"
    #: Address of the C function.
    address: int
    #: Number of values computed per point.
    n_out: int
    #: The numba ``cfunc``, kept alive with the kernel.
    cfunc: Any = dataclasses.field(repr=False, compare=False)


def column_kernel(
    n_out: int = 1,
) -> Callable[[Callable[..., int]], ColumnKernel]:
    """Compile a Python function into a native column kernel with numba.

    The function is compiled in nopython mode and called once per point from
    the threads of :func:`apply`, without the GIL: use it for computations
    that loop over the levels and do not vectorize over points. It receives
    numpy views and returns 0, or a non-zero code to report a failure (the
    point is then NaN). A kernel cannot raise: an exception is printed and
    lost, so report failures through the return code. The function, and any
    function it calls, is compiled with numba's ``error_model="numpy"``: a
    division by zero gives ``inf`` or ``NaN`` as in C; decorate the helpers
    it calls with ``numba.njit(error_model="numpy")`` too::

        @column_kernel(n_out=1)
        def kernel(columns, widths, a, b, point, out):
            # columns: the fields concatenated, widths[k] values for field k
            # a, b: hybrid coefficients (empty if not given)
            # point: lon, lat, [time,] then the extra args, in order
            # out: the n_out values to compute
            out[0] = columns[0] * point[1]
            return 0

    Args:
        n_out: Number of values the kernel computes per point.

    Returns:
        A decorator turning the function into a :class:`ColumnKernel`.

    Raises:
        ImportError: If numba is not installed.

    """
    try:
        import numba  # noqa: PLC0415 (optional dependency)
    except ImportError as error:
        raise ImportError(
            "column_kernel needs numba: install it with "
            "'conda install numba' or 'pip install numba'"
        ) from error

    if n_out < 1:
        raise ValueError("n_out must be positive")

    types = numba.types
    signature = types.intc(
        types.CPointer(types.float64),
        types.int64,
        types.CPointer(types.int64),
        types.CPointer(types.float64),
        types.CPointer(types.float64),
        types.int64,
        types.CPointer(types.float64),
        types.int64,
        types.CPointer(types.float64),
        types.int64,
    )

    def decorator(function: Callable[..., int]) -> ColumnKernel:
        # The numpy error model gives IEEE results (inf, NaN) on a division
        # by zero, as C does: the default raises ZeroDivisionError, which a
        # cfunc cannot propagate and only prints, leaving `out` unset.
        jitted = numba.njit(nogil=True, error_model="numpy")(function)
        carray = numba.carray

        # Compiled by numba from `signature`: the annotations only document
        # the C types (the pointers are opaque at the Python level).
        def wrapper(
            columns: object,
            n_fields: int,
            widths: object,
            a: object,
            b: object,
            n_half: int,
            point: object,
            n_point: int,
            out: object,
            n_values: int,
        ) -> int:
            width = carray(widths, (n_fields,))
            total = 0
            for ix in range(n_fields):
                total += width[ix]
            return jitted(
                carray(columns, (total,)),
                width,
                carray(a, (n_half,)),
                carray(b, (n_half,)),
                carray(point, (n_point,)),
                carray(out, (n_values,)),
            )

        compiled = numba.cfunc(signature, nopython=True, error_model="numpy")(
            wrapper
        )
        return ColumnKernel(
            address=compiled.address, n_out=n_out, cfunc=compiled
        )

    return decorator
