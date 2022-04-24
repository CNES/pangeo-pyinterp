# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
from typing import Dict, Tuple
import os

import numpy as np

from . import swot_calval_ephemeris_path
from ..geodetic import orbit_propagator


def load_ephemeris(
        filename: os.PathLike) -> Tuple[float, np.ndarray, np.timedelta64]:
    """Loads the ephemeris from a text file.

    Args:
        filename: Name of the file to be loaded.

    Returns:
        A tuple containing the height of the orbit, the ephemeris and the
        duration of the cycle.
    """
    with open(filename, "r") as stream:
        lines = stream.readlines()

    def to_dict(comments) -> Dict[str, float]:
        """Returns a dictionary describing the parameters of the orbit."""
        result = dict()
        for item in comments:
            assert item.startswith("#"), "Comments must start with #"
            key, value = item[1:].split("=")
            result[key.strip()] = float(value)
        return result

    # The two first lines are the header and contain the height and the
    # duration of the cycle in fractional days.
    settings = to_dict(lines[:2])
    del lines[:2]

    # The rest of the lines are the ephemeris
    ephemeris = np.loadtxt(lines,
                           delimiter=" ",
                           dtype={
                               "names":
                               ("time", "longitude", "latitude", "height"),
                               "formats": ("f8", "f8", "f8", "f8")
                           })

    return (
        settings["height"],
        np.rec.fromarrays(
            [
                ephemeris["time"].astype("timedelta64[s]"),
                ephemeris["longitude"],
                ephemeris["latitude"],
            ],
            dtype=orbit_propagator.Ephemeris,
        ),
        np.timedelta64(int(settings["cycle_duration"] * 86400.0 * 1e9), "ns"),
    )


def load_test_ephemeris() -> Tuple[float, np.ndarray, np.timedelta64]:
    """Loads the test ephemeris."""
    return load_ephemeris(swot_calval_ephemeris_path())


def test_calculate_orbit():
    """Test the calculation of the orbit."""
    height, ephemeris, cycle_duration = load_test_ephemeris()
    orbit = orbit_propagator.calculate_orbit(height, ephemeris, cycle_duration)


def test_calculate_pass():
    """Test the calculation of the pass."""
    height, ephemeris, cycle_duration = load_test_ephemeris()
    orbit = orbit_propagator.calculate_orbit(height, ephemeris, cycle_duration)
    pass_ = orbit_propagator.calculate_pass(1, orbit)
