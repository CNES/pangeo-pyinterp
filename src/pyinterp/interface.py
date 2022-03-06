# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Interface with the library core
===============================
"""
import re

import numpy as np

from . import core

#: Regular expression to extract the grid type from the class name.
PATTERN = re.compile(r"((?:Float|Int)\d+)").search


def _core_class_suffix(x: np.ndarray, handle_integer: bool = False) -> str:
    """Get the suffix of the class handling the numpy data type.

    Args:
        x: array to process
        handle_integer: if True, the integer type is handled
    Returns:
        str: the class suffix
    """
    dtype = x.dtype.type
    if dtype == np.float64:
        return 'Float64'
    if dtype == np.float32:
        return 'Float32'
    if dtype == np.int64:
        return 'Float64'
    if dtype == np.uint64:
        return 'Float64'
    if dtype == np.int32:
        return 'Float32'
    if dtype == np.uint32:
        return 'Float32'
    if dtype == np.int16:
        return 'Float32'
    if dtype == np.uint16:
        return 'Float32'
    if dtype == np.int8:
        return 'Float32' if not handle_integer else 'Int8'
    if dtype == np.uint8:
        return 'Float32'
    raise ValueError("Unhandled dtype: " + str(dtype))


def _core_function(function: str, instance: object) -> str:
    """Get the suffix of the function handling the grid instance.

    Args:
        function: function name
        instance: grid instance
    Returns:
        str: the class suffix
    """
    if not isinstance(instance, (
            core.Grid2DFloat64,
            core.Grid2DFloat32,
            core.Grid2DInt8,
            core.Grid3DFloat64,
            core.Grid3DFloat32,
            core.Grid4DFloat64,
            core.Grid4DFloat32,
            core.TemporalGrid3DFloat64,
            core.TemporalGrid3DFloat32,
            core.TemporalGrid4DFloat64,
            core.TemporalGrid4DFloat32,
    )):
        raise TypeError("instance is not an object handling a grid.")
    name = instance.__class__.__name__
    match = PATTERN(name)
    assert match is not None
    suffix = match.group(1).lower()
    return f"{function}_{suffix}"
