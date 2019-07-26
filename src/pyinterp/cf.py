# Copyright (c) 2019 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
CF (Climate and Forecast)
-------------------------
"""
from typing import List


class AxisUnit(list):
    """Units management for axes"""

    @property
    def units(self) -> List:
        """Get the list of known units

        Returns:
            list: The known units
        """
        return self


class AxisLatitudeUnit(AxisUnit):
    """Units known to the axis defining the latitude"""

    def __init__(self):
        super(AxisLatitudeUnit, self).__init__()
        self.append("degrees_north")
        self.append("degree_north")
        self.append("degree_N")
        self.append("degrees_N")
        self.append("degreeN")
        self.append("degreesN")


class AxisLongitudeUnit(AxisUnit):
    """Units known to the axis defining the longitude"""

    def __init__(self):
        super(AxisLongitudeUnit, self).__init__()
        self.append("degrees_east")
        self.append("degree_east")
        self.append("degree_E")
        self.append("degrees_E")
        self.append("degreeE")
        self.append("degreesE")


class AxisTimeUnit(AxisUnit):
    """Units known to the axis defining the time"""

    def __init__(self):
        super(AxisTimeUnit, self).__init__()
        self.append("days")
        self.append("seconds")
