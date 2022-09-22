# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Interface with the library core
===============================
"""
from typing import Optional
import re

import numpy

from . import core

#: Regular expression to extract the grid type from the class name.
PATTERN = re.compile(r'((?:Float|Int)\d+)').search

__all__ = [
    '_core_class_suffix',
    '_core_function',
    '_core_radial_basis_function',
    '_core_window_function',
]


def _core_class_suffix(x: numpy.ndarray, handle_integer: bool = False) -> str:
    """Get the suffix of the class handling the numpy data type.

    Args:
        x: array to process
        handle_integer: if True, the integer type is handled
    Returns:
        str: the class suffix
    """
    dtype = x.dtype.type
    if dtype == numpy.float64:
        return 'Float64'
    if dtype == numpy.float32:
        return 'Float32'
    if dtype == numpy.int64:
        return 'Float64'
    if dtype == numpy.uint64:
        return 'Float64'
    if dtype == numpy.int32:
        return 'Float32'
    if dtype == numpy.uint32:
        return 'Float32'
    if dtype == numpy.int16:
        return 'Float32'
    if dtype == numpy.uint16:
        return 'Float32'
    if dtype == numpy.int8:
        return 'Float32' if not handle_integer else 'Int8'
    if dtype == numpy.uint8:
        return 'Float32'
    raise ValueError('Unhandled dtype: ' + str(dtype))


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
        raise TypeError('instance is not an object handling a grid.')
    name = instance.__class__.__name__
    match = PATTERN(name)
    assert match is not None
    suffix = match.group(1).lower()
    return f'{function}_{suffix}'


def _core_radial_basis_function(
        rbf: Optional[str],
        epsilon: Optional[float]) -> core.RadialBasisFunction:
    """Get the radial basis function."""
    adjustable = ['gaussian', 'inverse_multiquadric', 'multiquadric']
    non_adjustable = ['cubic', 'linear', 'thin_plate']
    rbf = rbf or adjustable[-1]
    if epsilon is not None and rbf in non_adjustable:
        raise ValueError(
            f"epsilon must be None for {', '.join(non_adjustable)} RBF")
    if rbf not in adjustable + non_adjustable:
        raise ValueError(f'Radial basis function {rbf!r} is not defined')
    rbf = ''.join(item.capitalize() for item in rbf.split('_'))
    return getattr(core.RadialBasisFunction, rbf)


def _core_window_function(wf: Optional[str],
                          arg: Optional[float]) -> core.WindowFunction:
    """Get the window function."""
    wf = wf or 'blackman'
    if wf not in [
            'blackman',
            'blackman_harris',
            'boxcar',
            'flat_top',
            'gaussian',
            'hamming',
            'lanczos',
            'nuttall',
            'parzen',
            'parzen_swot',
    ]:
        raise ValueError(f'Window function {wf!r} is not defined')

    if wf in ['gaussian', 'lanczos', 'parzen']:
        if arg is None:
            defaults = dict(gaussian=None, lanczos=1, parzen=0)
            arg = defaults[wf]

        if wf == 'lanczos' and arg < 1:  # type: ignore
            raise ValueError(f'The argument of the function {wf!r} must be '
                             'greater than 1')

        if wf == 'parzen' and arg < 0:  # type: ignore
            raise ValueError(f'The argument of the function {wf!r} must be '
                             'greater than 0')

        if wf == 'gaussian' and arg is None:
            raise ValueError(f'The argument of the function {wf!r} must be '
                             'specified')
    else:
        if arg is not None:
            raise ValueError(f'The function {wf!r} does not support the '
                             'optional argument')

    wf = ''.join(item.capitalize() for item in wf.split('_'))
    wf = wf.replace('Swot', 'SWOT')
    return getattr(core.WindowFunction, wf)
