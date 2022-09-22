# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Orbit interpolation.
====================
"""
from typing import Iterator, Optional, Tuple
import dataclasses

import numpy

from . import core, geodetic
from .typing import NDArray, NDArrayDateTime, NDArrayTimeDelta


def interpolate(
    lon: NDArray,
    lat: NDArray,
    xp: NDArray,
    xi: NDArray,
    height: float = 0.0,
    wgs: Optional[geodetic.Coordinates] = None,
    half_window_size: int = 10,
) -> Tuple[NDArray, NDArray]:
    """Interpolate the given orbit at the given coordinates.

    Args:
        lon: Longitudes (in degrees).
        lat: Latitudes (in degrees).
        xp: The x-coordinates at which the orbit is defined.
        height: Height of the satellite above the Earth's surface (in meters).
        xi: The x-coordinates at which to evaluate the interpolated values.
        wgs: The World Geodetic System used to convert the coordinates.
        half_window_size: Half size of the window used to interpolate the
            orbit.

    Returns:
        Tuple[NDArray, NDArray]: The interpolated longitudes and latitudes.
    """
    wgs = wgs or geodetic.Coordinates()
    mz = wgs.spheroid.semi_major_axis / wgs.spheroid.semi_minor_axis()
    x, y, z = wgs.lla_to_ecef(lon, lat, numpy.full_like(lon, height))

    r = numpy.sqrt(x * x + y * y + z * z * mz * mz)

    x_axis = core.Axis((xp - xp[0]).astype(numpy.float64), 1e-6, False)
    xi = (xi - xp[0]).astype(numpy.float64)

    x = core.interpolate1d(x_axis, x, xi, half_window_size=half_window_size)
    y = core.interpolate1d(x_axis, y, xi, half_window_size=half_window_size)
    z = core.interpolate1d(x_axis, z, xi, half_window_size=half_window_size)
    r = core.interpolate1d(x_axis, r, xi, half_window_size=half_window_size)

    r /= numpy.sqrt(x * x + y * y + z * z)
    x *= r
    y *= r
    z *= r * (1 / mz)

    lon, lat, _ = wgs.ecef_to_lla(x, y, z)

    return lon, lat


def _rearrange_orbit(
    cycle_duration: numpy.timedelta64,
    lon: NDArray,
    lat: NDArray,
    time: NDArrayTimeDelta,
) -> Tuple[NDArray, NDArray, NDArrayTimeDelta]:
    """Rearrange orbit starting from pass 1.

    Detect the beginning of pass 1 in the ephemeris. By definition, it is
    the first passage at southernmost latitude.

    Args:
        cycle_duration: Cycle time in seconds.
        lon: Longitudes (in degrees).
        lat: Latitudes (in degrees).
        time: Time since the beginning of the orbit.

    Returns:
        The orbit rearranged starting from pass 1.
    """
    dy = numpy.roll(lat, 1) - lat  # type: ignore
    indexes = numpy.where((dy < 0) & (numpy.roll(dy, 1) >= 0))[0]

    # Shift coordinates, so that the first point of the orbit is the beginning
    # of pass 1
    shift = indexes[-1]

    lon = numpy.hstack([lon[shift:], lon[:shift]])
    lat = numpy.hstack([lat[shift:], lat[:shift]])
    time = numpy.hstack([time[shift:], time[:shift]])
    time = (time - time[0]) % cycle_duration
    if numpy.any(time < numpy.timedelta64(0, 's')):
        raise ValueError('Time is negative')
    return lon, lat, time


def _calculate_pass_time(lat: NDArray,
                         time: NDArrayTimeDelta) -> NDArrayTimeDelta:
    """Compute the initial time of each pass.

    Args:
        lat: Latitudes (in degrees)
        time: Date of the latitudes (in seconds).

    Returns:
        Start date of half-orbits.
    """
    dy = numpy.roll(lat, 1) - lat  # type: ignore
    indexes = numpy.where(((dy < 0) & (numpy.roll(dy, 1) >= 0))
                          | ((dy > 0)  # type: ignore
                             & (numpy.roll(dy, 1) <= 0)))[0]
    # The duration of the first pass is zero.
    indexes[0] = 0
    return time[indexes]


@dataclasses.dataclass(frozen=True)
class Orbit:
    """Properties of the orbit.

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
    latitude: NDArray
    #: Longitudes (in degrees).
    longitude: NDArray
    #: Start date of half-orbits.
    pass_time: NDArrayTimeDelta
    #: Time elapsed since the beginning of the orbit.
    time: NDArrayTimeDelta
    #: Along track distance (in meters).
    x_al: NDArray
    #: Spheroid model used.
    wgs: Optional[geodetic.Spheroid]

    def cycle_duration(self) -> numpy.timedelta64:
        """Get the cycle duration."""
        return self.time[-1]

    def passes_per_cycle(self) -> int:
        """Get the number of passes per cycle."""
        return len(self.pass_time)

    def orbit_duration(self) -> numpy.timedelta64:
        """Get the orbit duration."""
        duration = self.cycle_duration().astype(
            'timedelta64[us]') / numpy.timedelta64(
                int(self.passes_per_cycle() // 2), 'us')
        return numpy.timedelta64(int(duration), 'us')

    def curvilinear_distance(self) -> numpy.ndarray:
        """Get the curvilinear distance."""
        return geodetic.LineString(self.longitude,
                                   self.latitude).curvilinear_distance(
                                       strategy='thomas', wgs=self.wgs)

    def pass_duration(self, number: int) -> numpy.timedelta64:
        """Get the duration of a given pass.

        Args:
            number: track number (must be in [1, passes_per_cycle()])

        Returns:
            numpy.datetime64: track duration
        """
        passes_per_cycle = self.passes_per_cycle()
        if number < 1 or number > passes_per_cycle:
            raise ValueError(f'number must be in [1, {passes_per_cycle}]')
        if number == passes_per_cycle:
            return (self.time[-1] - self.pass_time[-1] + self.time[1] -
                    self.time[0])
        return self.pass_time[number] - self.pass_time[number - 1]

    def decode_absolute_pass_number(self, number: int) -> Tuple[int, int]:
        """Calculate the cycle and pass number from a given absolute pass
        number.

        Args:
            number (int): absolute pass number
        Returns:
            tuple: cycle and pass number
        """
        number -= 1
        return (int(number / self.passes_per_cycle()) + 1,
                (number % self.passes_per_cycle()) + 1)

    def encode_absolute_pass_number(self, cycle_number: int,
                                    pass_number: int) -> int:
        """Calculate the absolute pass number for a given half-orbit.

        Args:
            cycle_number (int): Cycle number
            pass_number (int): Pass number
        Returns:
            int: Absolute pass number
        """
        passes_per_cycle = self.passes_per_cycle()
        if not 1 <= pass_number <= passes_per_cycle:
            raise ValueError(f'pass_number must be in [1, {passes_per_cycle}')
        return (cycle_number - 1) * self.passes_per_cycle() + pass_number

    def delta_t(self) -> numpy.timedelta64:
        """Returns the average time difference between two measurements.

        Returns:
            int: average time difference
        """
        return numpy.diff(self.time).mean()

    def iterate(
        self,
        first_date: Optional[numpy.datetime64] = None,
        last_date: Optional[numpy.datetime64] = None,
        absolute_pass_number: int = 1
    ) -> Iterator[Tuple[int, int, numpy.datetime64]]:
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
        date = first_date or numpy.datetime64('now')
        last_date = last_date or date + self.cycle_duration()
        while date <= last_date:
            cycle_number, pass_number = self.decode_absolute_pass_number(
                absolute_pass_number)

            yield cycle_number, pass_number, date

            # Shift the date of the duration of the generated pass
            date += self.pass_duration(pass_number)

            # Update of the number of the next pass to be generated
            absolute_pass_number += 1
        return StopIteration


@dataclasses.dataclass(frozen=True)
class EquatorCoordinates:
    """Coordinates of the satellite at the equator."""
    #: Longitude
    longitude: float
    #: Product dataset name
    time: numpy.datetime64

    @classmethod
    def undefined(cls) -> 'EquatorCoordinates':
        """Create an undefined instance."""
        return cls(numpy.nan, numpy.datetime64('NaT'))


@dataclasses.dataclass(frozen=True)
class Pass:
    """Class representing a pass of an orbit."""
    #: Nadir longitude of the pass (degrees)
    lon_nadir: NDArray
    #: Nadir latitude of the pass (degrees)
    lat_nadir: NDArray
    #: Time of the pass
    time: NDArrayDateTime
    #: Along track distance of the pass (in meters)
    x_al: NDArray
    #: Coordinates of the satellite at the equator
    equator_coordinates: EquatorCoordinates

    def __len__(self) -> int:
        """Get the number of points in the pass."""
        return len(self.time)


@dataclasses.dataclass(frozen=True)
class Swath(Pass):
    """Class representing a pass of an orbit."""
    #: Longitude of the swath (degrees)
    lon: NDArray
    #: Latitude of the swath (degrees)
    lat: NDArray
    #: Across track distance of the pass (m)
    x_ac: NDArray

    def mask(self, requirement_bounds: Tuple[float, float]) -> NDArray:
        """Obtain a mask to set NaN values outside the mission requirements.

        Args:
            requirement_bounds (tuple): Limits of SWOT swath requirements:
                absolute value of the minimum and maximum across track
                distance.

        Returns:
            Mask set true, if the swath is outside the requirements of the
            mission.
        """
        valid = numpy.full_like(self.x_ac, numpy.nan)
        valid[(numpy.abs(self.x_ac) >= requirement_bounds[0])
              & (numpy.abs(self.x_ac) <= requirement_bounds[1])] = 1
        along_track = numpy.full(self.lon_nadir.shape, 1, dtype=numpy.float64)
        return along_track[:, numpy.newaxis] * valid

    def insert_central_pixel(self) -> 'Swath':
        """Return a swath with a central pixel dividing the swath in two by the
        reference ground track."""

        def _insert(array: NDArray, central_pixel: int,
                    fill_value: NDArray) -> NDArray:
            """Insert a central pixel in a given array."""
            return numpy.c_[array[:, :central_pixel],
                            fill_value[:, numpy.newaxis],
                            array[:, central_pixel:]]

        num_pixels = self.lon.shape[1] + 1
        num_lines = self.lon.shape[0]
        central_pixel = num_pixels // 2

        return Swath(
            self.lon_nadir, self.lat_nadir, self.time, self.x_al,
            self.equator_coordinates,
            _insert(self.lon, central_pixel, self.lon_nadir),
            _insert(self.lat, central_pixel, self.lat_nadir),
            _insert(self.x_ac, central_pixel,
                    numpy.zeros(num_lines, dtype=self.x_ac.dtype)))


def _equator_properties(lon_nadir: NDArray, lat_nadir: NDArray,
                        time: NDArrayTimeDelta) -> EquatorCoordinates:
    """Calculate the position of the satellite at the equator."""
    if lon_nadir.size < 2:
        return EquatorCoordinates.undefined()

    # Search the nearest point to the equator
    i1 = (numpy.abs(lat_nadir)).argmin()
    i0 = i1 - 1
    if lat_nadir[i0] * lat_nadir[i1] > 0:
        i0, i1 = i1, i1 + 1
    lon1 = lon_nadir[i0:i1 + 1]
    lat1 = lat_nadir[i0:i1 + 1]

    # Calculate the position of the satellite at the equator
    intersection = geodetic.LineString(lon1, lat1).intersection(
        geodetic.LineString(numpy.array([lon1[0] - 0.5, lon1[1] + 0.5]),
                            numpy.array([0, 0], dtype='float64')))
    if len(intersection) == 0:
        return EquatorCoordinates.undefined()

    point = intersection[0]

    # Calculate the time of the point on the equator
    lon1 = numpy.insert(lon1, 1, point.lon)
    lat1 = numpy.insert(lat1, 1, 0)
    x_al = geodetic.LineString(lon1,
                               lat1).curvilinear_distance(strategy='thomas')

    # Pop the along track distance at the equator
    x_eq = x_al[1]
    x_al = numpy.delete(x_al, 1)

    return EquatorCoordinates(
        point.lon,
        numpy.interp(x_eq, x_al, time[i0:i1 + 1].astype('i8')).astype(
            time.dtype),  # type: ignore
    )


def calculate_orbit(
    height: float,
    lon_nadir: NDArray,
    lat_nadir: NDArray,
    time: NDArrayTimeDelta,
    cycle_duration: Optional[numpy.timedelta64] = None,
    along_track_resolution: Optional[float] = None,
    spheroid: Optional[geodetic.Spheroid] = None,
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
    wgs = geodetic.Coordinates(spheroid)

    lon_nadir = geodetic.normalize_longitudes(lon_nadir)
    time = time.astype('m8[ns]')

    if numpy.mean(numpy.diff(time)) > numpy.timedelta64(500, 'ms'):
        time_hr = numpy.arange(time[0],
                               time[-1],
                               numpy.timedelta64(500, 'ms'),
                               dtype=time.dtype)
        lon_nadir, lat_nadir = interpolate(lon_nadir,
                                           lat_nadir,
                                           time.astype('i8'),
                                           time_hr.astype('i8'),
                                           height=height,
                                           wgs=wgs,
                                           half_window_size=50)
        time = time_hr

    if cycle_duration is not None:
        indexes = numpy.where(time < cycle_duration)[0]
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
    distance = geodetic.LineString(lon_nadir, lat_nadir).curvilinear_distance(
        strategy='thomas', wgs=spheroid) * 1e-3

    # Interpolate the final orbit according the given along track resolution
    x_al = numpy.arange(distance[0],
                        distance[-2],
                        along_track_resolution or 2,
                        dtype=distance.dtype)
    lon_nadir, lat_nadir = interpolate(lon_nadir[:-1],
                                       lat_nadir[:-1],
                                       distance[:-1],
                                       x_al,
                                       height=height,
                                       wgs=wgs,
                                       half_window_size=10)

    time = numpy.interp(
        x_al,  # type: ignore
        distance[:-1],  # type: ignore
        time[:-1].astype('i8')).astype(time.dtype)

    return Orbit(height, lat_nadir, lon_nadir,
                 numpy.sort(_calculate_pass_time(lat_nadir, time)), time, x_al,
                 wgs.spheroid)  # type: ignore


def calculate_pass(
    pass_number: int,
    orbit: Orbit,
    *,
    bbox: Optional[geodetic.Box] = None,
) -> Optional[Pass]:
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
        indexes = numpy.where(orbit.time >= orbit.pass_time[-1])[0]
    else:
        indexes = numpy.where((orbit.time >= orbit.pass_time[index])
                              & (orbit.time < orbit.pass_time[index + 1]))[0]

    if len(indexes) < 5:
        return None

    lon_nadir = orbit.longitude[indexes]
    lat_nadir = orbit.latitude[indexes]
    time = orbit.time[indexes]
    x_al = orbit.x_al[indexes]

    # Selects the orbit in the defined box
    if bbox is not None:
        mask = bbox.covered_by(lon_nadir, lat_nadir)
        if numpy.all(~mask):
            return None
        if numpy.any(mask):
            lon_nadir = lon_nadir[mask]
            lat_nadir = lat_nadir[mask]
            time = time[mask]
            x_al = x_al[mask]

    equator_coordinates = _equator_properties(lon_nadir, lat_nadir, time)

    return Pass(lon_nadir, lat_nadir, time, x_al, equator_coordinates)


def calculate_swath(
    half_orbit: Pass,
    *,
    across_track_resolution: Optional[float] = None,
    along_track_resolution: Optional[float] = None,
    half_swath: Optional[float] = None,
    half_gap: Optional[float] = None,
    spheroid: Optional[geodetic.Spheroid] = None,
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
        half_gap: Distance, in km, between the nadir and the center of the first
            pixel of the swath. Defaults to 2 km.
        spheroid: The spheroid to use for the calculation. Defaults to ``None``,
            which means the WGS-84 spheroid is used.

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
    x_ac = numpy.arange(half_swath,
                        dtype=float) * along_track_resolution + half_gap
    x_ac = numpy.hstack((-numpy.flip(x_ac), x_ac)) * 1e3
    x_ac = numpy.full((len(half_orbit), x_ac.size), x_ac)

    lon, lat = core.geodetic.calculate_swath(
        half_orbit.lon_nadir,
        half_orbit.lat_nadir,
        across_track_resolution * 1e3,
        half_gap * 1e3,
        half_swath,
        spheroid,
    )

    return Swath(half_orbit.lon_nadir, half_orbit.lat_nadir, half_orbit.time,
                 half_orbit.x_al, half_orbit.equator_coordinates, lon, lat,
                 x_ac)
