# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Interface with the library core."""
from __future__ import annotations

import re
import warnings

import numpy

from . import core

#: Regular expression to extract the grid type from the class name.
PATTERN = re.compile(r'((?:Float|Int|UInt)\d+)').search

__all__ = [
    '_core_class_suffix',
    '_core_covariance_function',
    '_core_drift_function',
    '_core_function',
    '_core_radial_basis_function',
    '_core_window_function',
]

#: Default mappings for all handled dtypes
_DEFAULT_SUFFIX_MAP = {
    numpy.float64: 'Float64',
    numpy.int64: 'Float64',
    numpy.uint64: 'Float64',
    numpy.float32: 'Float32',
    numpy.int32: 'Float32',
    numpy.uint32: 'Float32',
    numpy.int16: 'Float32',
    numpy.uint16: 'Float32',
    numpy.int8: 'Float32',
    numpy.uint8: 'Float32',
}

#: Special mappings when handle_integer=True (overrides defaults)
_INT_SUFFIX_MAP = {
    numpy.int8: 'Int8',
    numpy.uint8: 'UInt8',
}


def _core_class_suffix(x: numpy.ndarray, handle_integer: bool = False) -> str:
    """Get the suffix of the class handling the numpy data type.

    Args:
        x: array to process
        handle_integer: if True, the 8-bit integer types (int8, uint8)
                        are handled specifically. Otherwise, all integers
                        are mapped to float equivalents.

    Returns:
        str: the class suffix

    Raises:
        ValueError: if the array's dtype is not handled.

    """
    dtype = x.dtype.type

    if handle_integer:
        # First, try the special integer map.
        suffix = _INT_SUFFIX_MAP.get(dtype)
        if suffix:
            return suffix

    # If not handling integers specially, or if the type wasn't
    # in the special map, fall back to the default map.
    suffix = _DEFAULT_SUFFIX_MAP.get(dtype)
    if suffix:
        return suffix

    # If the dtype wasn't in any applicable map, it's unhandled.
    raise ValueError(f'Unhandled dtype: {dtype}')


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
            core.Grid2DUInt8,
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


def _core_covariance_function(
        covariance: str | None) -> core.CovarianceFunction:
    """Get the covariance function."""
    covariance = covariance or 'matern_32'
    if covariance not in [
            'exponential',
            'gaussian',
            'linear',
            'matern_12',
            'matern_32',
            'matern_52',
            'spherical',
            'whittle_matern',
    ]:
        raise ValueError(f'Covariance function {covariance!r} is not defined')
    if covariance == 'whittle_matern':
        warnings.warn(
            'Covariance function "whittle_matern" is '
            'deprecated. Use "matern_32" instead.',
            DeprecationWarning,
            stacklevel=3)
    if covariance == 'exponential':
        warnings.warn(
            'Covariance function "exponential" is '
            'deprecated. Use "matern_12" instead.',
            DeprecationWarning,
            stacklevel=3)
    covariance = '_'.join(item.capitalize() for item in covariance.split('_'))
    return getattr(core.CovarianceFunction, covariance)


def _core_drift_function(drift: str | None) -> core.DriftFunction | None:
    if drift is None:
        return None

    if drift not in [
            'linear',
            'quadratic',
    ]:
        raise ValueError(f'Drift function {drift!r} is not defined')
    drift = '_'.join(item.capitalize() for item in drift.split('_'))
    return getattr(core.DriftFunction, drift)


def _core_radial_basis_function(
        rbf: str | None, epsilon: float | None) -> core.RadialBasisFunction:
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


def _core_window_function(wf: str | None,
                          arg: float | None) -> core.WindowFunction:
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
            defaults = {'gaussian': None, 'lanczos': 1, 'parzen': 0}
            arg = defaults[wf]

        if wf == 'lanczos' and arg < 1:  # type: ignore[operator]
            raise ValueError(f'The argument of the function {wf!r} must be '
                             'greater than 1')

        if wf == 'parzen' and arg < 0:  # type: ignore[operator]
            raise ValueError(f'The argument of the function {wf!r} must be '
                             'greater than 0')

        if wf == 'gaussian' and arg is None:
            raise ValueError(f'The argument of the function {wf!r} must be '
                             'specified')
    elif arg is not None:
        raise ValueError(f'The function {wf!r} does not support the '
                         'optional argument')

    wf = ''.join(item.capitalize() for item in wf.split('_'))
    wf = wf.replace('Swot', 'SWOT')
    return getattr(core.WindowFunction, wf)
