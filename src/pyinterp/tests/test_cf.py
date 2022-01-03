# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
from .. import cf


def test_longitude():
    assert isinstance(cf.AxisLongitudeUnit().units, list)


def test_latitude():
    assert isinstance(cf.AxisLatitudeUnit().units, list)


def test_time():
    assert isinstance(cf.AxisTimeUnit().units, list)
