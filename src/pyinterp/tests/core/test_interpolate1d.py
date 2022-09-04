# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import numpy
import pytest

from ... import core


def test_interpolate1d():
    x = numpy.linspace(0, 100, endpoint=True)
    y = numpy.cos(-x**2 / 9.0)
    xi = numpy.linspace(0, 100, num=200, endpoint=True)
    yi = core.interpolate1d(core.Axis(x), y, xi, half_window_size=20)

    mask = xi == x

    assert pytest.approx(numpy.cos(-xi[mask]**2 / 9.0), rel=1e-6) == yi[mask]

    with pytest.raises(RuntimeError):
        core.interpolate1d(core.Axis(x), y, xi, half_window_size=0)
