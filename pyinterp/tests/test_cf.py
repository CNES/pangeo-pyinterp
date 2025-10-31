# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""CF-compliant axes."""
from .. import cf


def test_longitude() -> None:
    """Test CF AxisLongitudeUnit class."""
    assert isinstance(cf.AxisLongitudeUnit().units, list)


def test_latitude() -> None:
    """Test CF AxisLatitudeUnit class."""
    assert isinstance(cf.AxisLatitudeUnit().units, list)


def test_time() -> None:
    """Test CF AxisTimeUnit class."""
    assert isinstance(cf.AxisTimeUnit().units, list)
