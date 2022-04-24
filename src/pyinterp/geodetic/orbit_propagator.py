# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
from typing import Iterator, Optional, Tuple
import dataclasses

import numpy as np

from .. import core, geodetic


def _spher2cart(lon: np.ndarray,
                lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert spherical coordinates to cartesian coordinates."""
    rlon = np.radians(lon)
    rlat = np.radians(lat)
    x = np.cos(rlon) * np.cos(rlat)
    y = np.sin(rlon) * np.cos(rlat)
    z = np.sin(rlat)
    return x, y, z


def _cart2spher(x: np.ndarray, y: np.ndarray,
                z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert cartesian coordinates to spherical coordinates."""
    indexes = np.where((x == 0) & (y == 0))[0]
    if indexes.size:
        x[indexes] = np.nan
        y[indexes] = np.nan
    lat = np.arctan2(z, np.sqrt(x * x + y * y))  # type: ignore
    lon = np.arctan2(y, x)
    if indexes.size:
        lon[indexes] = 0
        lat[indexes] = np.pi * 0.5 * np.sign(z[indexes])
    return np.degrees(lon), np.degrees(lat)


def _satellite_direction(location: np.ndarray) -> np.ndarray:
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


Ephemeris = np.dtype([
    ("time", "timedelta64[ns]"),
    ("longitude", "float64"),
    ("latitude", "float64"),
])


def _interpolate(ephemeris: np.ndarray) -> np.ndarray:
    """Interpolate the given orbit at high resolution (0.5 seconds)
    Args:
        ephemeris: Ephemeris to interpolate.
    Returns:
        Ephemeris interpolated at 0.5 seconds.
    """
    x, y, z = _spher2cart(ephemeris["longitude"], ephemeris["latitude"])
    time_hr = np.arange(ephemeris["time"][0], ephemeris["time"][-1],
                        np.timedelta64(500, "ms"))

    x = np.interp(time_hr.astype("i8"), ephemeris["time"].astype("i8"), x)
    y = np.interp(time_hr.astype("i8"), ephemeris["time"].astype("i8"), y)
    z = np.interp(time_hr.astype("i8"), ephemeris["time"].astype("i8"), z)

    lon, lat = _cart2spher(x, y, z)

    return np.rec.fromarrays([time_hr, lon, lat], dtype=Ephemeris)


def _rearrange_orbit(
    cycle_duration: np.timedelta64,
    ephemeris: np.ndarray,
) -> np.ndarray:
    """Rearrange orbit starting from pass 1.

    Detect the beginning of pass 1 in the ephemeris. By definition, it is
    the first passage at southernmost latitude.

    Args:
        cycle_duration: Cycle time in seconds.
        ephemeris: Ephemeris to rearrange.
    Returns:
        Ephemeris rearranged.
    """
    lon, lat, time = (ephemeris["longitude"], ephemeris["latitude"],
                      ephemeris["time"])
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
    return np.rec.fromarrays([time, lon, lat], dtype=Ephemeris)


def _calculate_pass_time(lat: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Compute the initial time of each pass.

    Args:
        lat (np.ndarray): Latitudes (in degrees)
        time (np.ndarray): Date of the latitudes (in seconds).

    Returns:
        np.ndarray: Start date of half-orbits.
    """
    dy = np.roll(lat, 1) - lat  # type: ignore
    indexes = np.where(((dy < 0) & (np.roll(dy, 1) >= 0))
                       | ((dy > 0)  # type: ignore
                          & (np.roll(dy, 1) <= 0)))[0]
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
        curvlinear_distance: Curvilinear distance (in meters).
        wgs: World Geodetic System used.
    """
    height: float
    latitude: np.ndarray
    longitude: np.ndarray
    pass_time: np.ndarray
    time: np.ndarray
    x_al: np.ndarray
    curvlinear_distance: np.ndarray
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
class Pass:
    """Class representing a pass of an orbit."""
    #: Nadir longitude of the pass (degrees)
    lon_nadir: np.ndarray
    #: Nadir latitude of the pass (degrees)
    lat_nadir: np.ndarray
    #: Longitude of the pass (degrees)
    lon: np.ndarray
    #: Latitude of the pass (degrees)
    lat: np.ndarray
    #: Time of the pass
    time: np.ndarray
    #: Along track distance of the pass (km)
    x_al: np.ndarray
    #: Across track distance of the pass (km)
    x_ac: np.ndarray


def calculate_orbit(
    height: float,
    ephemeris: np.ndarray,
    cycle_duration: Optional[np.timedelta64] = None,
    along_track_resolution: Optional[float] = None,
    wgs: Optional[geodetic.System] = None,
):
    """Calculate the orbit at the given height.

    Args:
        height: Height of the orbit.
        ephemeris: Ephemeris to propagate.
        cycle_duration: Duration of the cycle.
        along_track_resolution: Resolution of the along-track interpolation in
            kilometers. Defaults to 2 kilometers.
        wgs: Geodetic system to use. Defaults to WGS84.

    Returns:
        Ephemeris of the orbit.
    """
    wgs = wgs or geodetic.System()

    if np.mean(np.diff(ephemeris["time"])) > np.timedelta64(500, "ms"):
        ephemeris = _interpolate(ephemeris)

    lon = geodetic.normalize_longitudes(ephemeris["longitude"])

    if cycle_duration is not None:
        indexes = np.where(ephemeris["time"] < cycle_duration)[0]
        ephemeris[indexes] = ephemeris[indexes]
        del indexes

    cycle_duration = (ephemeris["time"][-1] + ephemeris["time"][1] -
                      ephemeris["time"][0])

    # Rearrange orbit starting from pass 1
    ephemeris = _rearrange_orbit(cycle_duration, ephemeris)  # type: ignore

    # Calculates the along track distance (km)
    distance = geodetic.LineString(ephemeris["longitude"],
                                   ephemeris["latitude"]).curvilinear_distance(
                                       strategy="andoyer", wgs=wgs) * 1e-3

    # Interpolate the final orbit according the given along track resolution
    x_al = np.arange(distance[0],
                     distance[-2],
                     along_track_resolution or 2,
                     dtype=distance.dtype)

    x, y, z = _spher2cart(ephemeris["longitude"], ephemeris["latitude"])
    x = np.interp(x_al, distance[:-1], x[:-1])  # type: ignore
    y = np.interp(x_al, distance[:-1], y[:-1])  # type: ignore
    z = np.interp(x_al, distance[:-1], z[:-1])  # type: ignore
    lon, lat = _cart2spher(x, y, z)

    time = ephemeris["time"]
    time = np.interp(
        x_al,  # type: ignore
        distance[:-1],  # type: ignore
        time[:-1].astype("i8")).astype(time.dtype)

    return Orbit(height, lat, lon, np.sort(_calculate_pass_time(lat, time)),
                 time, x_al, distance, wgs)


def calculate_pass(
    pass_number: int,
    orbit: Orbit,
    *,
    bbox: Optional[geodetic.Box] = None,
    across_track_resolution: Optional[float] = None,
    along_track_resolution: Optional[float] = None,
    half_swath: Optional[float] = None,
    half_gap: Optional[float] = None,
) -> Optional[Pass]:
    """Get the properties of an half-orbit.

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
    across_track_resolution = across_track_resolution or 2.0
    along_track_resolution = along_track_resolution or 2.0
    half_swath = half_swath or 70.0
    half_gap = half_gap or 2.0
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
        if np.all(~mask):
            return None
        lon_nadir = lon_nadir[mask]
        lat_nadir = lat_nadir[mask]
        time = time[mask]
        x_al = x_al[mask]

    # Compute across track distances from nadir
    # Number of points in half of the swath
    half_swath = int((half_swath - half_gap) / across_track_resolution) + 1
    x_ac = np.arange(half_swath,
                     dtype=float) * along_track_resolution + half_gap
    x_ac = np.hstack((-np.flip(x_ac), x_ac))

    location = np.ascontiguousarray(
        np.vstack(_spher2cart(lon_nadir, lat_nadir)).T)
    satellite_direction = _satellite_direction(location)
    lon, lat = core.geodetic.calculate_swath(
        across_track_resolution,
        half_gap,
        half_swath,
        orbit.wgs.mean_radius() * 1e-3,
        location,
        satellite_direction,
    )

    return Pass(lon_nadir, lat_nadir, lon, lat, time, x_al, x_ac)
