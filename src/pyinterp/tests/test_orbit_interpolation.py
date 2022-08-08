# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
from typing import Dict, Tuple
import os

import numpy as np

from . import swot_calval_ephemeris_path
from ..orbit import calculate_orbit, calculate_pass, calculate_swath
from ..typing import NDArray


def load_test_ephemeris(
    filename: os.PathLike
) -> Tuple[float, NDArray, NDArray, NDArray, np.timedelta64]:
    """Loads the ephemeris from a text file.

    Args:
        filename: Name of the file to be loaded.

    Returns:
        A tuple containing the height of the orbit, the ephemeris and the
        duration of the cycle.
    """
    with open(filename) as stream:
        lines = stream.readlines()

    def to_dict(comments) -> Dict[str, float]:
        """Returns a dictionary describing the parameters of the orbit."""
        result = dict()
        for item in comments:
            assert item.startswith('#'), 'Comments must start with #'
            key, value = item[1:].split('=')
            result[key.strip()] = float(value)
        return result

    # The two first lines are the header and contain the height and the
    # duration of the cycle in fractional days.
    settings = to_dict(lines[:2])
    del lines[:2]

    # The rest of the lines are the ephemeris
    ephemeris = np.loadtxt(lines,
                           delimiter=' ',
                           dtype={
                               'names':
                               ('time', 'longitude', 'latitude', 'height'),
                               'formats': ('f8', 'f8', 'f8', 'f8')
                           })

    return (
        settings['height'],
        ephemeris['longitude'],
        ephemeris['latitude'],
        ephemeris['time'].astype('timedelta64[s]'),
        np.timedelta64(int(settings['cycle_duration'] * 86400.0 * 1e9), 'ns'),
    )


def test_calculate_orbit():
    """Test the calculation of the orbit."""
    orbit = calculate_orbit(*load_test_ephemeris(swot_calval_ephemeris_path()))
    assert orbit.passes_per_cycle() == 28


def test_calculate_pass():
    """Test the calculation of the pass."""
    orbit = calculate_orbit(*load_test_ephemeris(swot_calval_ephemeris_path()))
    pass_ = calculate_pass(2, orbit)
    assert pass_ is not None
    swath = calculate_swath(pass_)
