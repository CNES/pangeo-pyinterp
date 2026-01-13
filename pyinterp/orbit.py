# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Orbit interpolation."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Iterator

    from .type_hints import (
        NDArray1DFloat64,
        NDArray1DTimeDelta64,
        NDArray2DBool,
        NDArray2DFloat64,
    )


from . import core
from .core.geometry.geographic import Box, Coordinates, LineString, Spheroid
from .core.geometry.geographic.algorithms import (
    Strategy,
    curvilinear_distance,
    for_each_point_covered_by,
    intersection,
)


#: Minimum number of points required to process a pass.
_MIN_POINTS = 5


#: Minimum points to compute satellite equator position.
_MIN_EQUATOR_POINTS = 2


#: Latitude threshold to consider that the satellite is at the equator.
_EQUATOR_LAT_THRESHOLD = 40.0


def interpolate(
    lon: NDArray1DFloat64,
    lat: NDArray1DFloat64,
    xp: NDArray1DFloat64,
    xi: NDArray1DFloat64,
    height: float = 0.0,
    coordinates: Coordinates | None = None,
    half_window_size: int = 3,
) -> tuple[NDArray1DFloat64, NDArray1DFloat64]:
    """Interpolate the given orbit at the given coordinates.

    Args:
        lon: Longitudes (in degrees).
        lat: Latitudes (in degrees).
        xp: The x-coordinates at which the orbit is defined.
        height: Height of the satellite above the Earth's surface (in meters).
        xi: The x-coordinates at which to evaluate the interpolated values.
        coordinates: The geographic coordinates system used to convert the
            coordinates from geodetic to ECEF and vice versa. If None, a
            WGS-84 coordinate system is used.
        half_window_size: Half size of the window used to interpolate the
            orbit.


    Returns:
        The interpolated longitudes and latitudes.


    """
    coordinates = coordinates or Coordinates()
    spheroid = coordinates.spheroid
    mz = spheroid.semi_major_axis / spheroid.semi_minor_axis()
    x, y, z = coordinates.lla_to_ecef(
        lon,
        lat,
        np.full_like(lon, height),
    )

    r = np.sqrt(x * x + y * y + z * z * mz * mz)
    x_axis = core.Axis(xp - xp[0], 1e-6)
    xi = xi - xp[0]

    config = (
        core.config.windowed.Univariate.c_spline()
        .with_half_window_size(half_window_size)
        .with_boundary_mode(
            core.config.windowed.BoundaryConfig.shrink(),
        )
    )

    x = core.univariate(  # type: ignore[assignment]
        core.Grid(x_axis, x),
        xi,  # type: ignore[arg-type]
        config=config,
    )
    y = core.univariate(  # type: ignore[assignment]
        core.Grid(x_axis, y),
        xi,  # type: ignore[arg-type]
        config=config,
    )
    z = core.univariate(  # type: ignore[assignment]
        core.Grid(x_axis, z),
        xi,  # type: ignore[arg-type]
        config=config,
    )
    r = core.univariate(
        core.Grid(x_axis, r),
        xi,  # type: ignore[arg-type]
        config=config,
    )

    r /= np.sqrt(x * x + y * y + z * z)
    x *= r
    y *= r
    z *= r * (1 / mz)

    lon, lat, _ = coordinates.ecef_to_lla(
        x,  # type: ignore[arg-type]
        y,  # type: ignore[arg-type]
        z,  # type: ignore[arg-type]
    )

    return lon, lat


def _rearrange_orbit(
    cycle_duration: np.timedelta64,
    lon: NDArray1DFloat64,
    lat: NDArray1DFloat64,
    time: NDArray1DTimeDelta64,
) -> tuple[NDArray1DFloat64, NDArray1DFloat64, NDArray1DTimeDelta64]:
    """Rearrange orbit to start from pass 1.

    Detect the beginning of pass 1 in the ephemeris and reorder the data
    accordingly. By definition, pass 1 starts at the first passage at
    southernmost latitude.


    Args:
        cycle_duration: Cycle time in seconds.
        lon: Longitudes (in degrees).
        lat: Latitudes (in degrees).
        time: Time since the beginning of the orbit.


    Returns:
        The orbit rearranged starting from pass 1.


    """
    dy = np.roll(lat, 1) - lat
    indexes = np.where((dy < 0) & (np.roll(dy, 1) >= 0))[0]

    # If the orbit is already starting from pass 1, nothing to do
    if indexes[0] < int(indexes.mean()):
        return lon, lat, time

    # Shift coordinates, so that the first point of the orbit is the beginning
    # of pass 1
    shift = indexes[-1]

    lon = np.hstack([lon[shift:], lon[:shift]])
    lat = np.hstack([lat[shift:], lat[:shift]])
    time = np.hstack([time[shift:], time[:shift]])
    time = (time - time[0]) % cycle_duration
    if np.any(time < np.timedelta64(0, "s")):
        raise ValueError("Time is negative")
    return lon, lat, time


def _calculate_pass_time(
    lat: NDArray1DFloat64, time: NDArray1DTimeDelta64
) -> NDArray1DTimeDelta64:
    """Compute the initial time of each pass.

    Args:
        lat: Latitudes (in degrees)
        time: Date of the latitudes (in seconds).


    Returns:
        Start date of half-orbits.


    """
    dy = np.roll(lat, 1) - lat
    indexes = np.where(
        ((dy < 0) & (np.roll(dy, 1) >= 0)) | ((dy > 0) & (np.roll(dy, 1) <= 0))
    )[0]
    # The duration of the first pass is zero.
    indexes[0] = 0
    return time[indexes]


@dataclasses.dataclass(frozen=True)
class Orbit:
    """Represent properties of the orbit.

    Store and manage orbital parameters including position, timing, and
    geodetic information.


    Args:
        height: Height of the satellite (in meters).
        latitude: Latitudes (in degrees).
        longitude: Longitudes (in degrees).
        pass_time: Start date of half-orbits.
        time: Time elapsed since the beginning of the orbit.
        x_al: Along track distance (in meters).
        wgs: World Geodetic System used.


    """

    #: Height of the satellite (in meters).
    height: float
    #: Latitudes (in degrees).
    latitude: NDArray1DFloat64
    #: Longitudes (in degrees).
    longitude: NDArray1DFloat64
    #: Start date of half-orbits.
    pass_time: NDArray1DTimeDelta64
    #: Time elapsed since the beginning of the orbit.
    time: NDArray1DTimeDelta64
    #: Along track distance (in meters).
    x_al: NDArray1DFloat64
    #: Spheroid model used.
    wgs: Spheroid | None

    def cycle_duration(self) -> np.timedelta64:
        """Get the cycle duration."""
        return self.time[-1]

    def passes_per_cycle(self) -> int:
        """Get the number of passes per cycle.

        Returns:
            The number of passes in one complete cycle.


        """
        return len(self.pass_time)

    def orbit_duration(self) -> np.timedelta64:
        """Get the orbit duration.

        Returns:
            The duration of one complete orbit.


        """
        duration = self.cycle_duration().astype(
            "timedelta64[us]"
        ) / np.timedelta64(int(self.passes_per_cycle() // 2), "us")
        return np.timedelta64(int(duration), "us")

    def curvilinear_distance(self) -> NDArray1DFloat64:
        """Get the curvilinear distance.

        Returns:
            The curvilinear distance along the orbit.


        """
        ls = LineString(
            self.longitude,
            self.latitude,
        )
        return curvilinear_distance(
            ls,
            spheroid=self.wgs,
            strategy=Strategy.THOMAS,
        )

    def pass_duration(self, number: int) -> np.timedelta64:
        """Get the duration of a given pass.

        Args:
            number: track number (must be in [1, passes_per_cycle()])


        Returns:
            numpy.datetime64: track duration


        """
        passes_per_cycle = self.passes_per_cycle()
        if number < 1 or number > passes_per_cycle:
            raise ValueError(f"number must be in [1, {passes_per_cycle}]")
        if number == passes_per_cycle:
            return (
                self.time[-1]
                - self.pass_time[-1]
                + self.time[1]
                - self.time[0]
            )
        return self.pass_time[number] - self.pass_time[number - 1]

    def decode_absolute_pass_number(self, number: int) -> tuple[int, int]:
        """Calculate cycle and pass numbers from an absolute pass number.

        Convert an absolute pass number into its corresponding cycle and pass
        number components.


        Args:
            number: Absolute pass number.


        Returns:
            A tuple containing the cycle number and pass number.


        """
        number -= 1
        return (
            int(number / self.passes_per_cycle()) + 1,
            (number % self.passes_per_cycle()) + 1,
        )

    def encode_absolute_pass_number(
        self, cycle_number: int, pass_number: int
    ) -> int:
        """Calculate the absolute pass number for a given half-orbit.

        Args:
            cycle_number (int): Cycle number
            pass_number (int): Pass number
        Returns:
            int: Absolute pass number


        """
        passes_per_cycle = self.passes_per_cycle()
        if not 1 <= pass_number <= passes_per_cycle:
            raise ValueError(f"pass_number must be in [1, {passes_per_cycle}")
        return (cycle_number - 1) * self.passes_per_cycle() + pass_number

    def delta_t(self) -> np.timedelta64:
        """Return the average time difference between two measurements.

        Calculate the mean time interval between consecutive measurements.


        Returns:
            Average time difference between measurements.


        """
        return np.diff(self.time).mean()

    def iterate(
        self,
        first_date: np.datetime64 | None = None,
        last_date: np.datetime64 | None = None,
        absolute_pass_number: int = 1,
    ) -> Iterator[tuple[int, int, np.datetime64]]:
        """Obtain all half-orbits within the defined time interval.

        Args:
            first_date: First date of the period to be considered.
                Defaults to the current date.
            last_date: Last date of the period to be considered.
                Defaults to the current date plus the orbit duration.
            absolute_pass_number (int, optional): Absolute number of the first
                pass to be returned.


        Returns:
            iterator: An iterator for all passes in the interval pointing to
            the cycle number, pass number and start date of the half-orbit.


        """
        date = first_date or np.datetime64("now")
        last_date = last_date or date + self.cycle_duration()
        while date <= last_date:
            cycle_number, pass_number = self.decode_absolute_pass_number(
                absolute_pass_number
            )

            yield cycle_number, pass_number, date

            # Shift the date of the duration of the generated pass
            date += self.pass_duration(pass_number)

            # Update of the number of the next pass to be generated
            absolute_pass_number += 1
        return StopIteration  # type: ignore[return-value]


@dataclasses.dataclass(frozen=True)
class EquatorCoordinates:
    """Represent coordinates of the satellite at the equator.

    Store the longitude and time when the satellite crosses the equator.
    """

    #: Longitude
    longitude: float
    #: Product dataset name
    time: np.datetime64

    @classmethod
    def undefined(cls) -> EquatorCoordinates:
        """Create an undefined instance."""
        return cls(np.nan, np.datetime64("NaT"))


@dataclasses.dataclass(frozen=True)
class Pass:
    """Represent a pass of an orbit.

    Store the properties of a single orbital pass including nadir coordinates,
    timing, and along-track distance.
    """

    #: Nadir longitude of the pass (degrees)
    lon_nadir: NDArray1DFloat64
    #: Nadir latitude of the pass (degrees)
    lat_nadir: NDArray1DFloat64
    #: Time of the pass
    time: NDArray1DTimeDelta64
    #: Along track distance of the pass (in meters)
    x_al: NDArray1DFloat64
    #: Coordinates of the satellite at the equator
    equator_coordinates: EquatorCoordinates

    def __len__(self) -> int:
        """Get the number of points in the pass."""
        return len(self.time)


@dataclasses.dataclass(frozen=True)
class Swath(Pass):
    """Represent a swath of an orbital pass.

    Extend the Pass class with additional swath-specific properties including
    cross-track coordinates and distances.
    """

    #: Longitude of the swath (degrees)
    lon: NDArray2DFloat64
    #: Latitude of the swath (degrees)
    lat: NDArray2DFloat64
    #: Across track distance of the pass (m)
    x_ac: NDArray2DFloat64

    def mask(self, requirement_bounds: tuple[float, float]) -> NDArray2DBool:
        """Obtain a mask to set NaN values outside the mission requirements.

        Args:
            requirement_bounds (tuple): Limits of SWOT swath requirements:
                absolute value of the minimum and maximum across track
                distance.


        Returns:
            Mask set true, if the swath is outside the requirements of the
            mission.


        """
        valid = np.full_like(self.x_ac, 0, dtype=np.bool_)
        valid[
            (np.abs(self.x_ac) >= requirement_bounds[0])
            & (np.abs(self.x_ac) <= requirement_bounds[1])
        ] = 1
        along_track = np.full(self.lon_nadir.shape, 1, dtype=np.bool_)
        return along_track[:, np.newaxis] * valid

    def insert_central_pixel(self) -> Swath:
        """Insert a central pixel dividing the swath in two.

        Return a new swath with a central pixel added at the reference ground
        track, effectively dividing the swath into two halves.


        Returns:
            A new Swath instance with the central pixel inserted.


        """

        def _insert(
            array: NDArray2DFloat64,
            central_pixel: int,
            fill_value: NDArray1DFloat64,
        ) -> NDArray2DFloat64:
            """Insert a central pixel in a given array."""
            return np.c_[
                array[:, :central_pixel],
                fill_value[:, np.newaxis],
                array[:, central_pixel:],
            ]

        num_pixels = self.lon.shape[1] + 1
        num_lines = self.lon.shape[0]
        central_pixel = num_pixels // 2

        return Swath(
            self.lon_nadir,
            self.lat_nadir,
            self.time,
            self.x_al,
            self.equator_coordinates,
            _insert(self.lon, central_pixel, self.lon_nadir),
            _insert(self.lat, central_pixel, self.lat_nadir),
            _insert(
                self.x_ac,
                central_pixel,
                np.zeros(num_lines, dtype=self.x_ac.dtype),
            ),
        )


def _equator_properties(
    lon_nadir: NDArray1DFloat64,
    lat_nadir: NDArray1DFloat64,
    time: NDArray1DTimeDelta64,
) -> EquatorCoordinates:
    """Calculate the position of the satellite at the equator.

    Determine where and when the satellite crosses the equator.


    Args:
        lon_nadir: Nadir longitudes (in degrees).
        lat_nadir: Nadir latitudes (in degrees).
        time: Time since the beginning of the orbit.


    Returns:
        The equator coordinates of the satellite.


    """
    if lon_nadir.size < _MIN_EQUATOR_POINTS:
        return EquatorCoordinates.undefined()

    # Search the nearest point to the equator
    i1 = (np.abs(lat_nadir)).argmin()
    i0 = i1 - 1 if i1 > 0 else 1
    if lat_nadir[i0] * lat_nadir[i1] > 0:
        i0, i1 = (i1, i1 + 1) if i1 < lat_nadir.size - 1 else (i1 - 1, i1)
    lon1 = lon_nadir[i0 : i1 + 1]
    lat1 = lat_nadir[i0 : i1 + 1]

    # Calculate the position of the satellite at the equator
    points = intersection(
        LineString(
            lon1,
            lat1,
        ),
        LineString(
            np.array([lon1[0] - 0.5, lon1[1] + 0.5]),
            np.array(
                [0, 0],
                dtype="float64",
            ),
        ),
    )
    if len(points) == 0:
        return EquatorCoordinates.undefined()

    point = points[0]

    # Calculate the time of the point on the equator
    lon1 = np.insert(lon1, 1, point.lon)
    lat1 = np.insert(lat1, 1, 0)
    x_al = curvilinear_distance(
        LineString(
            lon1,
            lat1,
        ),
        strategy=Strategy.THOMAS,
        spheroid=None,
    )

    # Pop the along track distance at the equator
    x_eq = x_al[1]
    x_al = np.delete(
        x_al,
        1,
    )

    return EquatorCoordinates(
        point.lon,
        np.interp(x_eq, x_al, time[i0 : i1 + 1].astype("i8")).astype(
            time.dtype
        ),
    )


def calculate_orbit(
    height: float,
    lon_nadir: NDArray1DFloat64,
    lat_nadir: NDArray1DFloat64,
    time: NDArray1DTimeDelta64,
    cycle_duration: np.timedelta64 | None = None,
    along_track_resolution: float | None = None,
    spheroid: Spheroid | None = None,
) -> Orbit:
    """Calculate the orbit at the given height.

    Args:
        height: Height of the orbit, in meters.
        lon_nadir: Nadir longitude of the orbit (degrees).
        lat_nadir: Nadir latitude of the orbit (degrees).
        time: Time elapsed since the start of the orbit.
        cycle_duration: Duration of the cycle.
        along_track_resolution: Resolution of the along-track interpolation in
            kilometers. Defaults to 2 kilometers.
        spheroid: Spheroid to use for the calculations. Defaults to WGS84.


    Returns:
        Orbit object.


    """
    coordinates = Coordinates(spheroid)

    # If the first point of the given orbit starts at the equator, we need to
    # skew this first pass.
    if -_EQUATOR_LAT_THRESHOLD <= lat_nadir[0] <= _EQUATOR_LAT_THRESHOLD:
        dy = np.roll(lat_nadir, 1) - lat_nadir
        indexes = np.where(
            ((dy < 0) & (np.roll(dy, 1) >= 0))
            | ((dy > 0) & (np.roll(dy, 1) <= 0))
        )[0]
        lat_nadir = lat_nadir[indexes[1:]]
        lon_nadir = lon_nadir[indexes[1:]]
        time = time[indexes[1:]]

    lon_nadir = (lon_nadir + 180) % 360 - 180
    time = time.astype("m8[ms]")

    if np.mean(np.diff(time)) > np.timedelta64(500, "ms"):
        time_hr = np.arange(
            time[0], time[-1], np.timedelta64(500, "ms"), dtype="m8[ms]"
        )
        lon_nadir, lat_nadir = interpolate(
            lon_nadir,
            lat_nadir,
            time.view("i8"),
            time_hr.view("i8"),
            height=height,
            coordinates=coordinates,
            half_window_size=50,
        )
        time = time_hr

    if cycle_duration is not None:
        indexes = np.where(time < cycle_duration)[0]
        lon_nadir = lon_nadir[indexes]
        lat_nadir = lat_nadir[indexes]
        time = time[indexes]
        del indexes

    # Rearrange orbit starting from pass 1
    lon_nadir, lat_nadir, time = _rearrange_orbit(
        time[-1] + time[1] - time[0],
        lon_nadir,
        lat_nadir,
        time,
    )

    # Calculates the along track distance (km)
    distance = (
        curvilinear_distance(
            LineString(
                lon_nadir,
                lat_nadir,
            ),
            spheroid,
            Strategy.THOMAS,
        )
        * 1e-3
    )

    # Interpolate the final orbit according the given along track resolution
    x_al = np.arange(
        distance[0],
        distance[-2],
        along_track_resolution or 2,
        dtype=distance.dtype,
    )
    lon_nadir, lat_nadir = interpolate(
        lon_nadir[:-1],
        lat_nadir[:-1],
        distance[:-1],
        x_al,
        height=height,
        coordinates=coordinates,
        half_window_size=10,
    )

    time = np.interp(x_al, distance[:-1], time[:-1].astype("i8")).astype(
        time.dtype
    )

    return Orbit(
        height,
        lat_nadir,
        lon_nadir,
        np.sort(_calculate_pass_time(lat_nadir, time)),
        time,
        x_al,
        coordinates.spheroid,  # type: ignore[arg-type]
    )


def calculate_pass(
    pass_number: int,
    orbit: Orbit,
    *,
    bbox: Box | None = None,
) -> Pass | None:
    """Get the properties of a swath of an half-orbit.

    Args:
        pass_number: Pass number
        orbit: Orbit describing the pass to be calculated.
        bbox: Bounding box of the pass. Defaults to the whole Earth.


    Returns:
        The properties of the pass.


    """
    index = pass_number - 1
    # Selected indexes corresponding to the current pass
    if index == len(orbit.pass_time) - 1:
        indexes = np.where(orbit.time >= orbit.pass_time[-1])[0]
    else:
        indexes = np.where(
            (orbit.time >= orbit.pass_time[index])
            & (orbit.time < orbit.pass_time[index + 1])
        )[0]

    if len(indexes) < _MIN_POINTS:
        return None

    lon_nadir = orbit.longitude[indexes]
    lat_nadir = orbit.latitude[indexes]
    time = orbit.time[indexes]
    x_al = orbit.x_al[indexes]

    # Selects the orbit in the defined box
    if bbox is not None:
        mask = for_each_point_covered_by(
            LineString(
                lon_nadir,
                lat_nadir,
            ),
            bbox,
        )
        if np.all(~mask):
            return None
        if np.any(mask):
            lon_nadir = lon_nadir[mask]
            lat_nadir = lat_nadir[mask]
            time = time[mask]
            x_al = x_al[mask]

    equator_coordinates = _equator_properties(lon_nadir, lat_nadir, time)

    return Pass(
        lon_nadir,
        lat_nadir,
        time,
        x_al,
        equator_coordinates,
    )


def calculate_swath(
    half_orbit: Pass,
    *,
    across_track_resolution: float | None = None,
    along_track_resolution: float | None = None,
    half_swath: float | None = None,
    half_gap: float | None = None,
    spheroid: Spheroid | None = None,
) -> Swath:
    """Get the properties of a swath of an half-orbit.

    Args:
        half_orbit: Half-orbit used to calculate the swath.
        bbox: Bounding box of the pass. Defaults to the whole Earth.
        across_track_resolution: Distance, in km, between two points across
            track direction. Defaults to 2 km.
        along_track_resolution: Distance, in km, between two points along track
            direction. Defaults to 2 km.
        half_swath: Distance, in km, between the nadir and the center of the
            last pixel of the swath. Defaults to 70 km.
        half_gap: Distance, in km, between the nadir and the center of the
            first pixel of the swath. Defaults to 2 km.
        spheroid: The spheroid to use for the calculation. Defaults to
            ``None``, which means the WGS-84 spheroid is used.

    Returns:
        The properties of the pass.


    """
    across_track_resolution = across_track_resolution or 2.0
    along_track_resolution = along_track_resolution or 2
    half_swath = half_swath or 70.0
    half_gap = half_gap or 2.0

    # Compute across track distances from nadir
    # Number of points in half of the swath
    half_swath = int((half_swath - half_gap) / across_track_resolution) + 1
    x_ac = (
        np.arange(half_swath, dtype=float) * along_track_resolution + half_gap
    )
    x_ac = np.hstack((-np.flip(x_ac), x_ac)) * 1e3
    x_ac = np.full((len(half_orbit), x_ac.size), x_ac)

    lon, lat = core.geometry.satellite.calculate_swath(
        half_orbit.lon_nadir,
        half_orbit.lat_nadir,
        across_track_resolution * 1e3,
        half_gap * 1e3,
        half_swath,
        spheroid,
    )

    return Swath(
        half_orbit.lon_nadir,
        half_orbit.lat_nadir,
        half_orbit.time,
        half_orbit.x_al,
        half_orbit.equator_coordinates,
        lon,
        lat,
        x_ac,
    )
