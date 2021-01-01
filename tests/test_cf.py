# Copyright (c) 2021 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import pyinterp.cf


def test_longitude():
    assert isinstance(pyinterp.cf.AxisLongitudeUnit().units, list)


def test_latitude():
    assert isinstance(pyinterp.cf.AxisLatitudeUnit().units, list)


def test_time():
    assert isinstance(pyinterp.cf.AxisTimeUnit().units, list)
