# Copyright (c) 2019 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Interface with the library core
===============================
"""
import numpy as np
from . import core


def _core_class_suffix(x: np.ndarray) -> str:
    """Get the suffix of the class handling the numpy data type.

    Args:
        x (numpy.ndarray): array to process
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
        return 'Float32'
    if dtype == np.uint8:
        return 'Float32'
    raise ValueError("Unhandled dtype: " + str(dtype))


def _core_function_suffix(instance: object) -> str:
    """Get the suffix of the function handling the grid instance.

    Args:
        instance (object): grid instance
    Returns:
        str: the class suffix
    """
    if not isinstance(instance, (core.Grid2DFloat64, core.Grid2DFloat32,
                                 core.Grid3DFloat64, core.Grid3DFloat32)):
        raise TypeError("instance is not an object handling a grid.")
    name = instance.__class__.__name__
    if name.endswith("Float64"):
        return "float64"
    return "float32"
