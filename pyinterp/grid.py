# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Regular grids."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy

from . import core, interface

if TYPE_CHECKING:
    from .typing import NDArray2D, NDArray3D, NDArray4D

#: Two dimensional type variable
NUM_DIMS_2 = 2

#: Three dimensional type variable
NUM_DIMS_3 = 3

#: Four dimensional type variable
NUM_DIMS_4 = 4


def _configure_grid_class(
    self: Grid2D | Grid3D | Grid4D,
    *args: Any,  # noqa: ANN401
    increasing_axes: str | None = None,
) -> None:
    """Initialize a Grid instance."""
    prefix = ''
    for item in args:
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
        args = list(args)  # type: ignore[assignment]
        for idx, item in enumerate(args):
            if isinstance(item, (
                    core.Axis,
                    core.TemporalAxis,
            )) and not item.is_ascending():
                args[idx] = item.flip(  # type: ignore[index]
                    inplace=inplace)
                args[-1] = numpy.flip(  # type: ignore[index]
                    args[-1], axis=idx)
    self._instance = getattr(core, _class)(*args)
    self._prefix = prefix


def _format_instance(self: Grid2D | Grid3D | Grid4D) -> str:
    """Get the string representation of this instance."""

    def pad(string: str, length: int) -> str:
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


class Grid2D:
    """2D Cartesian Grid.

    Args:
        x: X-Axis.
        y: Y-Axis.
        array: Discrete representation of a continuous function
            on a uniform 2-dimensional grid.
        increasing_axes: Optional string indicating how to ensure that the grid
            axes are increasing. If axes are decreasing, the axes and grid
            provided will be flipped in place or copied before being flipped. By
            default, the decreasing axes are not modified.

    Examples:
        >>> import numpy as np
        >>> import pyinterp
        >>> x_axis = pyinterp.Axis(
        >>>     np.arange(-180.0, 180.0, 1.0),
        >>>     is_circle=True,
        >>> )
        >>> y_axis = pyinterp.Axis(
        >>>     np.arange(-80.0, 80.0, 1.0),
        >>>     is_circle=False,
        >>> )
        >>> array = np.zeros((len(x_axis), len(y_axis)))
        >>> grid = pyinterp.Grid2D(x_axis, y_axis, array)
        <pyinterp.grid.Grid2D>
        array([[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]], shape=(360, 160))
        Axis:
        * x: <pyinterp.core.Axis>
            min_value: -180
            max_value: 179
            step     : 1
            is_circle: true
        * y: <pyinterp.core.Axis>
            min_value: -80
            max_value: 79
            step     : 1
            is_circle: false

    """

    #: The number of grid dimensions handled by this object.
    _DIMENSIONS = NUM_DIMS_2

    def __init__(
        self,
        x: core.Axis,
        y: core.Axis,
        array: NDArray2D,
        increasing_axes: str | None = None,
    ) -> None:
        """Initialize a Grid2D instance."""
        self._instance: core.Grid2DFloat32 | core.Grid2DFloat64
        self._prefix: str
        _configure_grid_class(
            self,
            x,
            y,
            array,
            increasing_axes=increasing_axes,
        )

    def __repr__(self) -> str:
        """Get the string representation of this instance."""
        return _format_instance(self)

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
    def array(self) -> NDArray2D:
        """Gets the values handled by this instance.

        Returns:
            numpy.ndarray: values.

        """
        return self._instance.array


class Grid3D:
    """3D Cartesian Grid.

    Args:
        x: X-Axis.
        y: Y-Axis.
        z: Z-Axis.
        array: Discrete representation of a continuous function on a uniform
            3-dimensional grid.
        increasing_axes: Ensure that the axes of the grid are increasing. If
            this is not the case, the axes and grid provided will be flipped.
            Default to False.

    Notes:
        If the Z axis is a :py:class:`temporal axis
        <pyinterp.TemporalAxis>`, the grid will handle this axis during
        interpolations as a time axis.

    Examples:
        >>> import numpy as np
        >>> import pyinterp
        >>>
        >>> x_axis = pyinterp.Axis(
        >>>     np.arange(-180.0, 180.0, 1.0),
        >>>     is_circle=True,
        >>> )
        >>> y_axis = pyinterp.Axis(
        >>>     np.arange(-80.0, 80.0, 1.0),
        >>>     is_circle=False,
        >>> )
        >>> z_axis = pyinterp.TemporalAxis(
        >>>     np.array(
        >>>         ['2000-01-01'],
        >>>         dtype="datetime64[s]",
        >>>     ))
        >>> array = np.zeros((len(x_axis), len(y_axis), len(z_axis)))
        >>> grid = pyinterp.Grid3D(x_axis, y_axis, z_axis, array)
        <pyinterp.grid.Grid3D>
        array([[[0.], [0.], [0.], ..., [0.], [0.], [0.]],
            [[0.], [0.], [0.], ..., [0.], [0.], [0.]],
            [[0.], [0.], [0.], ..., [0.], [0.], [0.]],
            ...,
            [[0.], [0.], [0.], ..., [0.], [0.], [0.]],
            [[0.], [0.], [0.], ..., [0.], [0.], [0.]],
            [[0.], [0.], [0.], ..., [0.], [0.], [0.]]],shape=(360, 160, 1))
        Axis:
        * x: <pyinterp.core.Axis>
            min_value: -180
            max_value: 179
            step     : 1
            is_circle: true
        * y: <pyinterp.core.Axis>
            min_value: -80
            max_value: 79
            step     : 1
            is_circle: false
        * z: <pyinterp.core.TemporalAxis>
            values   : ['2000-01-01T00:00:00']

    """

    _DIMENSIONS = NUM_DIMS_3

    def __init__(
        self,
        x: core.Axis,
        y: core.Axis,
        z: core.Axis | core.TemporalAxis,
        array: NDArray3D,
        increasing_axes: str | None = None,
    ) -> None:
        """Initialize a Grid3D instance."""
        self._instance: (core.Grid3DInt8 | core.Grid3DUInt8
                         | core.Grid3DFloat32 | core.Grid3DFloat64
                         | core.TemporalGrid3DFloat32
                         | core.TemporalGrid3DFloat64)
        self._prefix: str
        self._dtype = None if isinstance(y, core.Axis) else y.dtype()
        _configure_grid_class(
            self,
            x,
            y,
            z,
            array,
            increasing_axes=increasing_axes,
        )

    def __repr__(self) -> str:
        """Get the string representation of this instance."""
        return _format_instance(self)

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
    def z(self) -> core.Axis | core.TemporalAxis:
        """Gets the Z-Axis handled by this instance.

        Returns:
            Z-Axis.

        """
        return self._instance.z  # type: ignore[return-value]

    @property
    def array(self) -> NDArray3D:
        """Gets the values handled by this instance.

        Returns:
            numpy.ndarray: values.

        """
        return self._instance.array


class Grid4D:
    """4D Cartesian Grid.

    Args:
        x: X-Axis.
        y: Y-Axis.
        z: Z-Axis.
        u: U-Axis.
        array: Discrete representation of a continuous function on a
            uniform 4-dimensional grid.
        increasing_axes: Ensure that the axes of the grid are increasing.
            If this is not the case, the axes and grid provided will be
            flipped. Default to False.

    Notes:
        If the Z axis is a temporal axis, the grid will handle this axis
        during interpolations as a time axis.

    """

    _DIMENSIONS = NUM_DIMS_4

    def __init__(
        self,
        x: core.Axis,
        y: core.Axis,
        z: core.Axis | core.TemporalAxis,
        u: core.Axis,
        array: NDArray4D,
        increasing_axes: str | None = None,
    ) -> None:
        """Initialize a Grid4D instance."""
        self._instance: (core.Grid4DInt8 | core.Grid4DUInt8
                         | core.Grid4DFloat32 | core.Grid4DFloat64
                         | core.TemporalGrid4DFloat32
                         | core.TemporalGrid4DFloat64)
        self._prefix: str
        self._dtype = None if isinstance(y, core.Axis) else y.dtype()
        _configure_grid_class(
            self,
            x,
            y,
            z,
            u,
            array,
            increasing_axes=increasing_axes,
        )

    def __repr__(self) -> str:
        """Get the string representation of this instance."""
        return _format_instance(self)

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
    def z(self) -> core.Axis | core.TemporalAxis:
        """Gets the Z-Axis handled by this instance.

        Returns:
            Z-Axis.

        """
        return self._instance.z  # type: ignore[return-value]

    @property
    def u(self) -> core.Axis:
        """Gets the U-Axis handled by this instance.

        Returns:
            U-Axis.

        """
        return self._instance.u

    @property
    def array(self) -> NDArray4D:
        """Gets the values handled by this instance.

        Returns:
            numpy.ndarray: values.

        """
        return self._instance.array


def _core_variate_interpolator(
        instance: Grid2D | Grid3D | Grid4D,
        interpolator: str,
        **kwargs: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Obtain the interpolator from the string provided."""
    dimensions = instance._DIMENSIONS
    # 4D interpolation uses the 3D interpolator
    if dimensions > NUM_DIMS_3:
        dimensions -= 1

    prefix = instance._prefix

    if interpolator == 'bilinear':
        return getattr(core, f'{prefix}Bilinear{dimensions}D')(**kwargs)
    if interpolator == 'nearest':
        return getattr(core, f'{prefix}Nearest{dimensions}D')(**kwargs)
    if interpolator == 'inverse_distance_weighting':
        return getattr(
            core, f'{prefix}InverseDistanceWeighting{dimensions}D')(**kwargs)

    raise ValueError(f'interpolator {interpolator!r} is not defined')
