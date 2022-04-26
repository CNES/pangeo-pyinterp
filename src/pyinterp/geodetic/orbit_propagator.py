# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
from typing import Iterator, Optional, Tuple
import dataclasses
import functools

import numpy as np

from .. import core, geodetic
from ..typing import NDArray, NDArrayDateTime, NDArrayTimeDelta


def _satellite_direction(location: NDArray) -> NDArray:
    """Calculate satellite direction."""
    direction = np.empty_like(location)

    denominator = np.sqrt(
        np.power(location[1:-1, 0], 2) + np.power(location[1:-1, 1], 2) +
        np.power(location[1:-1, 2], 2))
    direction[1:-1, 0] = (location[2:, 0] -
                          location[:-2, 0]) / denominator  # type: ignore
    direction[1:-1, 1] = (location[2:, 1] -
                          location[:-2, 1]) / denominator  # type: ignore
    direction[1:-1, 2] = (location[2:, 2] -
                          location[:-2, 2]) / denominator  # type: ignore

    direction[0, :] = direction[1, :]
    direction[0, :] = direction[1, :]
    direction[0, :] = direction[1, :]
    direction[-1, :] = direction[-2, :]
    direction[-1, :] = direction[-2, :]
    direction[-1, :] = direction[-2, :]

    return direction


def _interpolate(
    lon: NDArray,
    lat: NDArray,
    xi: NDArray,
    xp: NDArray,
    wgs: geodetic.Coordinates,
) -> Tuple[NDArray, NDArray]:
    """Interpolate the given orbit at the given coordinates.

    Args:
        lon: Longitudes (in degrees).
        lat: Latitudes (in degrees).
        xi: The x-coordinates at which to evaluate the interpolated values.
        xp: The x-coordinates at which the orbit is defined.
        wgs: The World Geodetic System used to convert the coordinates.

    Returns:
        Tuple[NDArray, NDArray]: The interpolated longitudes and latitudes.
    """
    mz = wgs.system.semi_major_axis / wgs.system.semi_minor_axis()
    x, y, z = wgs.lla_to_ecef(lon, lat, np.full_like(lon, 0))

    r = np.sqrt(x * x + y * y + z * z * mz * mz)
    x = np.interp(xi, xp, x)
    y = np.interp(xi, xp, y)
    z = np.interp(xi, xp, z)
    r = np.interp(xi, xp, r)

    r /= np.sqrt(x * x + y * y + z * z)
    x *= r
    y *= r
    z *= r * (1 / mz)

    lon, lat, _ = wgs.ecef_to_lla(x, y, z)

    return lon, lat


def _rearrange_orbit(
    cycle_duration: np.timedelta64,
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
    dy = np.roll(lat, 1) - lat  # type: ignore
    indexes = np.where((dy < 0) & (np.roll(dy, 1) >= 0))[0]

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


def _calculate_pass_time(lat: NDArray,
                         time: NDArrayTimeDelta) -> NDArrayTimeDelta:
    """Compute the initial time of each pass.

    Args:
        lat: Latitudes (in degrees)
        time: Date of the latitudes (in seconds).

    Returns:
        Start date of half-orbits.
    """
    dy = np.roll(lat, 1) - lat  # type: ignore
    indexes = np.where(((dy < 0) & (np.roll(dy, 1) >= 0))
                       | ((dy > 0)  # type: ignore
                          & (np.roll(dy, 1) <= 0)))[0]
    # The duration of the first pass is zero.
    indexes[0] = 0
    return time[indexes]


@dataclasses.dataclass(frozen=True)
class Ephemeris:
    time: NDArrayDateTime
    lon: NDArray
    lat: NDArray


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
    height: float
    latitude: NDArray
    longitude: NDArray
    pass_time: NDArrayTimeDelta
    time: NDArrayTimeDelta
    x_al: NDArray
    wgs: geodetic.System

    def cycle_duration(self) -> np.timedelta64:
        """Get the cycle duration."""
        return self.time[-1]

    def passes_per_cycle(self) -> int:
        """Get the number of passes per cycle."""
        return len(self.pass_time)

    def orbit_duration(self) -> np.timedelta64:
        """Get the orbit duration."""
        duration = self.cycle_duration().astype(
            "timedelta64[us]") / np.timedelta64(
                int(self.passes_per_cycle() // 2), 'us')
        return np.timedelta64(int(duration), "us")

    @functools.cached_property
    def curvilinear_distance(self) -> np.ndarray:
        """Get the curvilinear distance."""
        return geodetic.LineString(self.longitude,
                                   self.latitude).curvilinear_distance(
                                       strategy="thomas", wgs=self.wgs)

    def pass_duration(self, number: int) -> np.timedelta64:
        """Get the duration of a given pass.

        Args:
            number: track number (must be in [1, passes_per_cycle()])

        Returns:
            np.datetime64: track duration
        """
        passes_per_cycle = self.passes_per_cycle()
        if number < 1 or number > passes_per_cycle:
            raise ValueError(f"number must be in [1, {passes_per_cycle}]")
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
            raise ValueError(f"pass_number must be in [1, {passes_per_cycle}")
        return (cycle_number - 1) * self.passes_per_cycle() + pass_number

    def delta_t(self) -> np.timedelta64:
        """Returns the average time difference between two measurements.

        Returns:
            int: average time difference
        """
        return np.diff(self.time).mean()

    def iterate(
        self,
        first_date: Optional[np.datetime64] = None,
        last_date: Optional[np.datetime64] = None,
        absolute_pass_number: int = 1
    ) -> Iterator[Tuple[int, int, np.datetime64]]:
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
    time: np.datetime64

    @classmethod
    def undefined(cls) -> "EquatorCoordinates":
        """Create an undefined instance."""
        return cls(np.nan, np.datetime64("NaT"))


@dataclasses.dataclass(frozen=True)
class Pass:
    """Class representing a pass of an orbit."""
    #: Nadir longitude of the pass (degrees)
    lon_nadir: NDArray
    #: Nadir latitude of the pass (degrees)
    lat_nadir: NDArray
    #: Time of the pass
    time: NDArrayDateTime
    #: Along track distance of the pass (km)
    x_al: NDArray
    #: Coordinates of the satellite at the equator
    equator_coordinates: EquatorCoordinates


@dataclasses.dataclass(frozen=True)
class Swath(Pass):
    """Class representing a pass of an orbit."""
    #: Longitude of the swath (degrees)
    lon: NDArray
    #: Latitude of the swath (degrees)
    lat: NDArray
    #: Across track distance of the pass (km)
    x_ac: NDArray


def equator_properties(lon_nadir: NDArray, lat_nadir: NDArray,
                       time: NDArrayTimeDelta,
                       wgs: geodetic.System) -> EquatorCoordinates:
    """Calculate the position of the satellite at the equator."""
    if lon_nadir.size < 2:
        return EquatorCoordinates.undefined()

    # Search the nearest point to the equator
    i1 = (np.abs(lat_nadir)).argmin()
    i0 = i1 - 1
    if lat_nadir[i0] * lat_nadir[i1] > 0:
        i0, i1 = i1, i1 + 1
    lon1 = lon_nadir[i0:i1 + 1]
    lat1 = lat_nadir[i0:i1 + 1]

    # Calculate the position of the satellite at the equator
    point = geodetic.LineString(lon1, lat1).intersection(
        geodetic.LineString(lon1, np.array([0, 0], dtype="float64")))
    if point is None:
        return EquatorCoordinates.undefined()

    # Calculate the time of the point on the equator
    lon1 = np.insert(lon1, 1, point.lon)
    lat1 = np.insert(lat1, 1, 0)
    x_al = geodetic.LineString(lon1,
                               lat1).curvilinear_distance(strategy="thomas",
                                                          wgs=wgs)

    # Pop the along track distance at the equator
    x_eq = x_al[1]
    x_al = np.delete(x_al, 1)

    return EquatorCoordinates(point.lon,
                              np.interp(x_eq, x_al,
                                        time[i0:i1 + 1].astype("i8")).astype(
                                            time.dtype))  # type: ignore


def calculate_orbit(
    height: float,
    ephemeris: Ephemeris,
    cycle_duration: Optional[np.timedelta64] = None,
    along_track_resolution: Optional[float] = None,
    system: Optional[geodetic.System] = None,
) -> Orbit:
    """Calculate the orbit at the given height.

    Args:
        height: Height of the orbit.
        ephemeris: Ephemeris to propagate.
        cycle_duration: Duration of the cycle.
        along_track_resolution: Resolution of the along-track interpolation in
            kilometers. Defaults to 2 kilometers.
        wgs: Geodetic system to use. Defaults to WGS84.

    Returns:
        Orbit object.
    """
    wgs = geodetic.Coordinates(system)

    lon = geodetic.normalize_longitudes(ephemeris.lon)
    lat = ephemeris.lat
    time = ephemeris.time.astype("m8[ns]")

    if np.mean(np.diff(time)) > np.timedelta64(500, "ms"):
        time_hr = np.arange(time[0],
                            time[-1],
                            np.timedelta64(500, "ms"),
                            dtype=time.dtype)
        lon, lat = _interpolate(lon, lat, time_hr.astype("i8"),
                                time.astype("i8"), wgs)
        time = time_hr

    if cycle_duration is not None:
        indexes = np.where(time < cycle_duration)[0]
        lon = lon[indexes]
        lat = lat[indexes]
        time = time[indexes]
        del indexes

    # Rearrange orbit starting from pass 1
    lon, lat, time = _rearrange_orbit(
        time[-1] + time[1] - time[0],
        lon,
        lat,
        time,
    )

    # Calculates the along track distance (km)
    distance = geodetic.LineString(lon, lat).curvilinear_distance(
        strategy="thomas", wgs=system) * 1e-3

    # Interpolate the final orbit according the given along track resolution
    x_al = np.arange(distance[0],
                     distance[-2],
                     along_track_resolution or 2,
                     dtype=distance.dtype)
    lon, lat = _interpolate(lon[:-1], lat[:-1], x_al, distance[:-1], wgs)

    time = np.interp(
        x_al,  # type: ignore
        distance[:-1],  # type: ignore
        time[:-1].astype("i8")).astype(time.dtype)

    return Orbit(height, lat, lon, np.sort(_calculate_pass_time(lat, time)),
                 time, x_al, wgs.system)  # type: ignore


def calculate_pass(
    pass_number: int,
    orbit: Orbit,
    *,
    bbox: Optional[geodetic.Box] = None,
    along_track_resolution: Optional[float] = None,
) -> Optional[Pass]:
    """Get the properties of a swath of an half-orbit.

    Args:
        pass_number: Pass number
        orbit: Orbit describing the pass to be calculated.
        bbox: Bounding box of the pass. Defaults to the whole Earth.
        along_track_resolution: Distance, in km, between two points along track
            direction. Defaults to 2 km.

    Returns:
        The properties of the pass.
    """
    along_track_resolution = along_track_resolution or 2.0
    index = pass_number - 1
    # Selected indexes corresponding to the current pass
    if index == len(orbit.pass_time) - 1:
        indexes = np.where(orbit.time >= orbit.pass_time[-1])[0]
    else:
        indexes = np.where((orbit.time >= orbit.pass_time[index])
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
        if np.all(mask == 0):
            return None
        lon_nadir = lon_nadir[mask]
        lat_nadir = lat_nadir[mask]
        time = time[mask]
        x_al = x_al[mask]

    equator_coordinates = equator_properties(lon_nadir, lat_nadir, time,
                                             orbit.wgs)

    return Pass(lon_nadir, lat_nadir, time, x_al, equator_coordinates)


def calculate_swath(
    pass_number: int,
    orbit: Orbit,
    *,
    bbox: Optional[geodetic.Box] = None,
    across_track_resolution: Optional[float] = None,
    along_track_resolution: Optional[float] = None,
    half_swath: Optional[float] = None,
    half_gap: Optional[float] = None,
) -> Optional[Swath]:
    """Get the properties of a swath of an half-orbit.

    Args:
        pass_number: Pass number
        orbit: Orbit describing the pass to be calculated.
        bbox: Bounding box of the pass. Defaults to the whole Earth.
        across_track_resolution: Distance, in km, between two points across
            track direction. Defaults to 2 km.
        along_track_resolution: Distance, in km, between two points along track
            direction. Defaults to 2 km.
        half_swath: Distance, in km, between the nadir and the center of the
            last pixel of the swath. Defaults to 70 km.
        half_gap: Distance, in km, between the nadir and the center of the first
            pixel of the swath. Defaults to 2 km.

    Returns:
        The properties of the pass.
    """
    pass_ = calculate_pass(pass_number,
                           orbit,
                           bbox=bbox,
                           along_track_resolution=along_track_resolution)
    if pass_ is None:
        return None
    across_track_resolution = across_track_resolution or 2.0
    half_swath = half_swath or 70.0
    half_gap = half_gap or 2.0

    # Compute across track distances from nadir
    # Number of points in half of the swath
    half_swath = int((half_swath - half_gap) / across_track_resolution) + 1
    x_ac = np.arange(half_swath,
                     dtype=float) * along_track_resolution + half_gap
    x_ac = np.hstack((-np.flip(x_ac), x_ac))

    lon, lat = core.geodetic.calculate_swath(
        pass_.lon_nadir,
        pass_.lat_nadir,
        across_track_resolution,
        half_gap,
        half_swath,
        orbit.wgs.mean_radius() * 1e-3,
    )

    return Swath(pass_.lon_nadir, pass_.lat_nadir, pass_.time, pass_.x_al,
                 pass_.equator_coordinates, lon, lat, x_ac)
