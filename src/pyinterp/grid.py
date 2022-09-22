# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Regular grids
=============
"""
from typing import Optional, Union

import numpy

from . import core, interface


class Grid2D:
    """2D Cartesian Grid.

    Args:
        x (pyinterp.Axis): X-Axis.
        y (pyinterp.Axis): Y-Axis.
        array (numpy.ndarray): Discrete representation of a continuous function
            on a uniform 2-dimensional grid.
        increasing_axes: Optional string indicating how to ensure that the grid
            axes are increasing. If axes are decreasing, the axes and grid
            provided will be flipped in place or copied before being flipped. By
            default, the decreasing axes are not modified.

    Examples:

        >>> import numpy as np
        >>> import pyinterp
        >>> x_axis = pyinterp.Axis(numpy.arange(-180.0, 180.0, 1.0),
        ...                        is_circle=True)
        >>> y_axis = pyinterp.Axis(numpy.arange(-80.0, 80.0, 1.0),
        ...                        is_circle=False)
        >>> array = numpy.zeros((len(x_axis), len(y_axis)))
        >>> grid = pyinterp.Grid2D(x_axis, y_axis, array)
        >>> grid
        <pyinterp.grid.Grid2D>
        array([[0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.],
                ...,
                [0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.]])
        Axis:
            * x: <pyinterp.axis.Axis>
                min_value: -180.0
                max_value: 179.0
                step: 1.0
                is_circle: True
            * y: <pyinterp.axis.Axis>
                min_value: -80.0
                max_value: 79.0
                step: 1.0
                is_circle: False
    """
    #: The number of grid dimensions handled by this object.
    _DIMENSIONS = 2

    def __init__(self, *args, increasing_axes: Optional[str] = None):
        prefix = ''
        for idx, item in enumerate(args):
            if isinstance(item, core.TemporalAxis):
                prefix = 'Temporal'
                break
        _class = f'{prefix}Grid{self._DIMENSIONS}D' + \
            interface._core_class_suffix(args[-1], handle_integer=True)
        if increasing_axes is not None:
            if increasing_axes not in ['inplace', 'copy']:
                raise ValueError('increasing_axes '
                                 f'{increasing_axes!r} is not defined')
            inplace = increasing_axes == 'inplace'
            # Tuple does not support item assignment
            args = list(args)
            for idx, item in enumerate(args):
                if isinstance(item,
                              (core.Axis,
                               core.TemporalAxis)) and not item.is_ascending():
                    args[idx] = item.flip(inplace=inplace)
                    args[-1] = numpy.flip(args[-1], axis=idx)
        self._instance = getattr(core, _class)(*args)
        self._prefix = prefix

    def __repr__(self):
        """Called by the ``repr()`` built-in function to compute the string
        representation of this instance."""

        def pad(string, length):
            """Pad a string to a given length."""
            return '\n'.join([(' ' * length if ix else '') + line
                              for ix, line in enumerate(string.split('\n'))])

        result = [
            f'<{self.__module__}.{self.__class__.__name__}>',
            repr(self.array),
        ]
        result.append('Axis:')
        for item in dir(self):
            attr = getattr(self, item)
            if isinstance(attr, (core.Axis, core.TemporalAxis)):
                prefix = f'* {item}: '
                result.append(f' {prefix}{pad(repr(attr), len(prefix))}')
        return '\n'.join(result)

    @property
    def x(self) -> core.Axis:
        """Gets the X-Axis handled by this instance.

        Returns:
            X-Axis.
        """
        return self._instance.x

    @property
    def y(self) -> core.Axis:
        """Gets the Y-Axis handled by this instance.

        Returns:
            Y-Axis.
        """
        return self._instance.y

    @property
    def array(self) -> numpy.ndarray:
        """Gets the values handled by this instance.

        Returns:
            numpy.ndarray: values.
        """
        return self._instance.array


class Grid3D(Grid2D):
    """3D Cartesian Grid.

    Args:
        x (pyinterp.Axis): X-Axis.
        y (pyinterp.Axis, pyinterp.TemporalAxis): Y-Axis.
        z (pyinterp.Axis): Z-Axis.
        array (numpy.ndarray): Discrete representation of a continuous function
            on a uniform 3-dimensional grid.
        increasing_axes: Ensure that the axes of the grid are increasing. If
            this is not the case, the axes and grid provided will be flipped.
            Default to False.

    .. note::

        If the Z axis is a :py:class:`temporal axis
        <pyinterp.TemporalAxis>`, the grid will handle this axis during
        interpolations as a time axis.

    Examples:

        >>> import numpy as np
        >>> import pyinterp
        >>> x_axis = pyinterp.Axis(numpy.arange(-180.0, 180.0, 1.0),
        ...                        is_circle=True)
        >>> y_axis = pyinterp.Axis(numpy.arange(-80.0, 80.0, 1.0),
        ...                        is_circle=False)
        >>> z_axis = pyinterp.TemporalAxis(
        ...     numpy.array(['2000-01-01'], dtype="datetime64[s]"))
        >>> array = numpy.zeros((len(x_axis), len(y_axis), len(z_axis)))
        >>> grid = pyinterp.Grid3D(x_axis, y_axis, z_axis, array)
    """
    _DIMENSIONS = 3

    def __init__(self, *args, increasing_axes: Optional[str] = None):
        super().__init__(*args, increasing_axes=increasing_axes)

    @property
    def z(self) -> Union[core.Axis, core.TemporalAxis]:
        """Gets the Z-Axis handled by this instance.

        Returns:
            Z-Axis.
        """
        return self._instance.z


class Grid4D(Grid3D):
    """4D Cartesian Grid.

    Args:
        x (pyinterp.Axis): X-Axis.
        y (pyinterp.Axis): Y-Axis.
        z (pyinterp.Axis, pyinterp.TemporalAxis): Z-Axis.
        u (pyinterp.Axis): U-Axis.
        array (numpy.ndarray): Discrete representation of a continuous
            function on a uniform 4-dimensional grid.
        increasing_axes: Ensure that the axes of the grid are increasing.
            If this is not the case, the axes and grid provided will be
            flipped. Default to False.

    .. note::

        If the Z axis is a temporal axis, the grid will handle this axis
        during interpolations as a time axis.
    """
    _DIMENSIONS = 4

    def __init__(self, *args, increasing_axes: Optional[str] = None):
        super().__init__(*args, increasing_axes=increasing_axes)

    @property
    def u(self) -> core.Axis:
        """Gets the U-Axis handled by this instance.

        Returns:
            U-Axis.
        """
        return self._instance.u


def _core_variate_interpolator(instance: object, interpolator: str, **kwargs):
    """Obtain the interpolator from the string provided."""
    if isinstance(instance, Grid2D):
        dimensions = instance._DIMENSIONS
        # 4D interpolation uses the 3D interpolator
        if dimensions > 3:
            dimensions -= 1
    else:
        raise TypeError('instance is not an object handling a grid.')

    prefix = instance._prefix

    if interpolator == 'bilinear':
        return getattr(core, f'{prefix}Bilinear{dimensions}D')(**kwargs)
    if interpolator == 'nearest':
        return getattr(core, f'{prefix}Nearest{dimensions}D')(**kwargs)
    if interpolator == 'inverse_distance_weighting':
        return getattr(
            core, f'{prefix}InverseDistanceWeighting{dimensions}D')(**kwargs)

    raise ValueError(f'interpolator {interpolator!r} is not defined')
