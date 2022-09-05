# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import numpy
import pytest

from ... import core


def test_interpolate1d():
    xi = numpy.linspace(0, 100, num=200, endpoint=True)
    x = numpy.concatenate((xi[::4], xi[-1:]))
    y = numpy.cos(-x**2 / 9.0)
    yi = core.interpolate1d(core.Axis(x), y, xi, half_window_size=20)
    index = numpy.searchsorted(xi, x)

    assert pytest.approx(numpy.cos(-x**2 / 9.0), rel=1e-6) == yi[index]

    with pytest.raises(RuntimeError):
        core.interpolate1d(core.Axis(x), y, xi, half_window_size=0)
