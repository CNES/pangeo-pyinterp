# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
CF (Climate and Forecast)
-------------------------
"""
from __future__ import annotations


class AxisUnit(list):
    """Units management for axes."""

    @property
    def units(self) -> list:
        """Get the list of known units.

        Returns:
            list: The known units.
        """
        return self


class AxisLatitudeUnit(AxisUnit):
    """Units known to the axis defining the latitude."""

    def __init__(self):
        super().__init__((
            'degrees_north',
            'degree_north',
            'degree_N',
            'degrees_N',
            'degreeN',
            'degreesN',
        ))


class AxisLongitudeUnit(AxisUnit):
    """Units known to the axis defining the longitude."""

    def __init__(self):
        super().__init__((
            'degrees_east',
            'degree_east',
            'degree_E',
            'degrees_E',
            'degreeE',
            'degreesE',
        ))


class AxisTimeUnit(AxisUnit):
    """Units known to the axis defining the time."""

    def __init__(self):
        super().__init__(('days', 'seconds'))
