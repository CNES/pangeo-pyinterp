# Copyright (c) 2021 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
GeoHash
-------

Geohashing is a geocoding method used to encode geographic coordinates (latitude
and longitude) into a short string of digits and letters delineating an area on
a map, which is called a cell, with varying resolutions. The more characters in
the string, the more precise the location. The table below gives the
correspondence between the number of characters, the size of the boxes of the
grid at the equator and the total number of boxes.

            =========  ===============  ==========
            precision  lng/lat (km)     samples
            =========  ===============  ==========
            1          4950/4950        32
            2          618.75/1237.50   1024
            3          154.69/154.69    32768
            4          19.34/38.67      1048576
            5          4.83/4.83        33554432
            6          0.60/1.21        1073741824
            =========  ===============  ==========

Geohashes use Base-32 alphabet encoding (characters can be ``0`` to ``9`` and
``A`` to ``Z``, excl ``A``, ``I``, ``L`` and ``O``).

The geohash is a compact way of representing a location, and is useful for
storing a location in a database, or for indexing a location in a database.
"""
from typing import Optional, Tuple, Type

#
import numpy
import xarray

#
from ..core.geohash import (
    area,
    bounding_boxes,
    decode,
    encode,
    int64,
    transform,
    where,
)
from .. import geodetic
from .converter import to_xarray
