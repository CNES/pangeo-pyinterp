# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Typing definitions.

.. data:: NDArray

    A numpy array of any data type. This alias corresponds to
    numpy.typing.NDArray[Any].

.. data:: NDArray1D

    A numpy array specifically of one dimension.

.. data:: NDArray1DBool

    A numpy array specifically of one dimension with numpy.bool_ elements.

.. data:: NDArray1DDateTime

    A numpy array specifically of one dimension with numpy.datetime64 elements.

.. data:: NDArray1DFloat32

    A numpy array specifically of one dimension with numpy.float32 elements.

.. data:: NDArray1DFloat64

    A numpy array specifically of one dimension with numpy.float64 elements.

.. data:: NDArray1DInt64

    A numpy array specifically of one dimension with numpy.int64 elements.

.. data:: NDArray1DStr

    A numpy array specifically of one dimension with numpy.str_ elements.

.. data:: NDArray1DTimeDelta

    A numpy array specifically of one dimension with numpy.timedelta64 elements.

.. data:: NDArray1DUInt64

    A numpy array specifically of one dimension with numpy.uint64 elements.

.. data:: NDArray2D

    A numpy array specifically of two dimensions.

.. data:: NDArray2DBool

    A numpy array specifically of two dimensions with numpy.bool_ elements.

.. data:: NDArray2DDateTime

    A numpy array specifically of two dimensions with numpy.datetime64 elements.

.. data:: NDArray2DFloat32

    A numpy array specifically of two dimensions with numpy.float32 elements.

.. data:: NDArray2DFloat64

    A numpy array specifically of two dimensions with numpy.float64 elements.

.. data:: NDArray2DInt8

    A numpy array specifically of two dimensions with numpy.int8 elements.

.. data:: NDArray2DUInt8

    A numpy array specifically of two dimensions with numpy.uint8 elements.

.. data:: NDArray2DUInt64

    A numpy array specifically of two dimensions with numpy.uint64 elements.

.. data:: NDArray3D

    A numpy array specifically of three dimensions.

.. data:: NDArray3DFloat32

    A numpy array specifically of three dimensions with numpy.float32 elements.

.. data:: NDArray3DFloat64

    A numpy array specifically of three dimensions with numpy.float64 elements.

.. data:: NDArray3DInt8

    A numpy array specifically of three dimensions with numpy.int8 elements.

.. data:: NDArray3DUInt8

    A numpy array specifically of three dimensions with numpy.uint8 elements.

.. data:: NDArray4D

    A numpy array specifically of four dimensions.

.. data:: NDArray4DFloat32

    A numpy array specifically of four dimensions with numpy.float32 elements.

.. data:: NDArray4DFloat64

    A numpy array specifically of four dimensions with numpy.float64 elements.

.. data:: NDArray4DInt8

    A numpy array specifically of four dimensions with numpy.int8 elements.

.. data:: NDArray4DUInt8

    A numpy array specifically of four dimensions with numpy.uint8 elements.

.. data:: NDArrayDateTime

    A numpy array of any dimension with numpy.datetime64 elements.

.. data:: NDArrayFloat32

    A numpy array of any dimension with numpy.float32 elements.

.. data:: NDArrayFloat64

    A numpy array of any dimension with numpy.float64 elements.

.. data:: NDArrayInt64

    A numpy array of any dimension with numpy.int64 elements.

.. data:: NDArrayStructured

    A numpy array of any dimension with structured elements.

.. data:: NDArrayTimeDelta

    A numpy array of any dimension with numpy.timedelta64 elements.

.. data:: NDArrayUInt64

    A numpy array of any dimension with numpy.uint64 elements.

"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar
from types import GenericAlias

import numpy
import numpy.typing

NDArray = numpy.typing.NDArray[Any]

__all__ = [
    'NDArray',
    'NDArray1D',
    'NDArray1DBool',
    'NDArray1DDateTime',
    'NDArray1DFloat32',
    'NDArray1DFloat64',
    'NDArray1DInt64',
    'NDArray1DStr',
    'NDArray1DTimeDelta',
    'NDArray1DUInt64',
    'NDArray2D',
    'NDArray2DBool',
    'NDArray2DDateTime',
    'NDArray2DFloat32',
    'NDArray2DFloat64',
    'NDArray2DInt8',
    'NDArray2DUInt8',
    'NDArray2DUInt64',
    'NDArray3D',
    'NDArray3DFloat32',
    'NDArray3DFloat64',
    'NDArray3DInt8',
    'NDArray3DUInt8',
    'NDArray4D',
    'NDArray4DFloat32',
    'NDArray4DFloat64',
    'NDArray4DInt8',
    'NDArray4DUInt8',
    'NDArrayDateTime',
    'NDArrayFloat32',
    'NDArrayFloat64',
    'NDArrayInt64',
    'NDArrayStructured',
    'NDArrayTimeDelta',
    'NDArrayUInt64',
]

if TYPE_CHECKING:  # pragma: no cover

    OneDim = tuple[int]
    TwoDims = tuple[int, int]
    ThreeDims = tuple[int, int, int]
    FourDims = tuple[int, int, int, int]

    NDArray1D = numpy.ndarray[OneDim, numpy.dtype[Any]]
    NDArray1DBool = numpy.ndarray[OneDim, numpy.dtype[numpy.bool_]]
    NDArray1DDateTime = numpy.ndarray[OneDim, numpy.dtype[numpy.datetime64]]
    NDArray1DFloat32 = numpy.ndarray[OneDim, numpy.dtype[numpy.float32]]
    NDArray1DFloat64 = numpy.ndarray[OneDim, numpy.dtype[numpy.float64]]
    NDArray1DInt64 = numpy.ndarray[OneDim, numpy.dtype[numpy.int64]]
    NDArray1DStr = numpy.ndarray[OneDim, numpy.dtype[numpy.str_]]
    NDArray1DTimeDelta = numpy.ndarray[OneDim, numpy.dtype[numpy.timedelta64]]
    NDArray1DUInt64 = numpy.ndarray[OneDim, numpy.dtype[numpy.uint64]]
    NDArray2D = numpy.ndarray[TwoDims, numpy.dtype[Any]]
    NDArray2DBool = numpy.ndarray[TwoDims, numpy.dtype[numpy.bool_]]
    NDArray2DDateTime = numpy.ndarray[TwoDims, numpy.dtype[numpy.datetime64]]
    NDArray2DFloat32 = numpy.ndarray[TwoDims, numpy.dtype[numpy.float32]]
    NDArray2DFloat64 = numpy.ndarray[TwoDims, numpy.dtype[numpy.float64]]
    NDArray2DInt8 = numpy.ndarray[TwoDims, numpy.dtype[numpy.int8]]
    NDArray2DUInt64 = numpy.ndarray[TwoDims, numpy.dtype[numpy.uint64]]
    NDArray2DUInt8 = numpy.ndarray[TwoDims, numpy.dtype[numpy.uint8]]
    NDArray3D = numpy.ndarray[ThreeDims, numpy.dtype[Any]]
    NDArray3DFloat32 = numpy.ndarray[ThreeDims, numpy.dtype[numpy.float32]]
    NDArray3DFloat64 = numpy.ndarray[ThreeDims, numpy.dtype[numpy.float64]]
    NDArray3DInt8 = numpy.ndarray[ThreeDims, numpy.dtype[numpy.int8]]
    NDArray3DUInt8 = numpy.ndarray[ThreeDims, numpy.dtype[numpy.uint8]]
    NDArray4D = numpy.ndarray[FourDims, numpy.dtype[Any]]
    NDArray4DFloat32 = numpy.ndarray[FourDims, numpy.dtype[numpy.float32]]
    NDArray4DFloat64 = numpy.ndarray[FourDims, numpy.dtype[numpy.float64]]
    NDArray4DInt8 = numpy.ndarray[FourDims, numpy.dtype[numpy.int8]]
    NDArray4DUInt8 = numpy.ndarray[FourDims, numpy.dtype[numpy.uint8]]
    NDArrayDateTime = numpy.ndarray[Any, numpy.dtype[numpy.datetime64]]
    NDArrayFloat32 = numpy.ndarray[Any, numpy.dtype[numpy.float32]]
    NDArrayFloat64 = numpy.ndarray[Any, numpy.dtype[numpy.float64]]
    NDArrayInt64 = numpy.ndarray[Any, numpy.dtype[numpy.int64]]
    NDArrayStructured = numpy.ndarray[Any, numpy.dtype[numpy.void]]
    NDArrayTimeDelta = numpy.ndarray[Any, numpy.dtype[numpy.timedelta64]]
    NDArrayUInt64 = numpy.ndarray[Any, numpy.dtype[numpy.uint64]]

else:  # pragma: no cover
    ScalarType_co = TypeVar('ScalarType_co',
                            bound=numpy.generic,
                            covariant=True)
    _DType = GenericAlias(
        numpy.dtype,
        (ScalarType_co, ),
    )
    NDArray1D = GenericAlias(
        numpy.ndarray,
        (tuple[int], _DType),
    )
    NDArray1DBool = GenericAlias(
        numpy.ndarray,
        (tuple[int], numpy.dtype[numpy.bool_]),
    )
    NDArray1DDateTime = GenericAlias(
        numpy.ndarray,
        (tuple[int], numpy.dtype[numpy.datetime64]),
    )
    NDArray1DFloat32 = GenericAlias(
        numpy.ndarray,
        (tuple[int], numpy.dtype[numpy.float32]),
    )
    NDArray1DFloat64 = GenericAlias(
        numpy.ndarray,
        (tuple[int], numpy.dtype[numpy.float64]),
    )
    NDArray1DInt64 = GenericAlias(
        numpy.ndarray,
        (tuple[int], numpy.dtype[numpy.int64]),
    )
    NDArray1DStr = GenericAlias(
        numpy.ndarray,
        (tuple[int], numpy.dtype[numpy.str_]),
    )
    NDArray1DTimeDelta = GenericAlias(
        numpy.ndarray,
        (tuple[int], numpy.dtype[numpy.timedelta64]),
    )
    NDArray1DUInt64 = GenericAlias(
        numpy.ndarray,
        (tuple[int], numpy.dtype[numpy.uint64]),
    )
    NDArray2D = GenericAlias(
        numpy.ndarray,
        (tuple[int, int], _DType),
    )
    NDArray2DBool = GenericAlias(
        numpy.ndarray,
        (tuple[int, int], numpy.dtype[numpy.bool_]),
    )
    NDArray2DDateTime = GenericAlias(
        numpy.ndarray,
        (tuple[int, int], numpy.dtype[numpy.datetime64]),
    )
    NDArray2DFloat32 = GenericAlias(
        numpy.ndarray,
        (tuple[int, int], numpy.dtype[numpy.float32]),
    )
    NDArray2DFloat64 = GenericAlias(
        numpy.ndarray,
        (tuple[int, int], numpy.dtype[numpy.float64]),
    )
    NDArray2DInt8 = GenericAlias(
        numpy.ndarray,
        (tuple[int, int], numpy.dtype[numpy.int8]),
    )
    NDArray2DUInt64 = GenericAlias(
        numpy.ndarray,
        (tuple[int, int], numpy.dtype[numpy.uint64]),
    )
    NDArray2DUInt8 = GenericAlias(
        numpy.ndarray,
        (tuple[int, int], numpy.dtype[numpy.uint8]),
    )
    NDArray3D = GenericAlias(
        numpy.ndarray,
        (tuple[int, int, int], _DType),
    )
    NDArray3DFloat32 = GenericAlias(
        numpy.ndarray,
        (tuple[int, int, int], numpy.dtype[numpy.float32]),
    )
    NDArray3DFloat64 = GenericAlias(
        numpy.ndarray,
        (tuple[int, int, int], numpy.dtype[numpy.float64]),
    )
    NDArray3DInt8 = GenericAlias(
        numpy.ndarray,
        (tuple[int, int, int], numpy.dtype[numpy.int8]),
    )
    NDArray3DUInt8 = GenericAlias(
        numpy.ndarray,
        (tuple[int, int, int], numpy.dtype[numpy.uint8]),
    )
    NDArray4D = GenericAlias(
        numpy.ndarray,
        (tuple[int, int, int, int], _DType),
    )
    NDArray4DFloat32 = GenericAlias(
        numpy.ndarray,
        (tuple[int, int, int, int], numpy.dtype[numpy.float32]),
    )
    NDArray4DFloat64 = GenericAlias(
        numpy.ndarray,
        (tuple[int, int, int, int], numpy.dtype[numpy.float64]),
    )
    NDArray4DInt8 = GenericAlias(
        numpy.ndarray,
        (tuple[int, int, int, int], numpy.dtype[numpy.int8]),
    )
    NDArray4DUInt8 = GenericAlias(
        numpy.ndarray,
        (tuple[int, int, int, int], numpy.dtype[numpy.uint8]),
    )
    NDArrayDateTime = GenericAlias(
        numpy.ndarray,
        (Any, numpy.dtype[numpy.datetime64]),
    )
    NDArrayFloat32 = GenericAlias(
        numpy.ndarray,
        (Any, numpy.dtype[numpy.float32]),
    )
    NDArrayFloat64 = GenericAlias(
        numpy.ndarray,
        (Any, numpy.dtype[numpy.float64]),
    )
    NDArrayInt64 = GenericAlias(
        numpy.ndarray,
        (Any, numpy.dtype[numpy.int64]),
    )
    NDArrayStructured = GenericAlias(
        numpy.ndarray,
        (Any, numpy.dtype[numpy.void]),
    )
    NDArrayTimeDelta = GenericAlias(
        numpy.ndarray,
        (Any, numpy.dtype[numpy.timedelta64]),
    )
    NDArrayUInt64 = GenericAlias(
        numpy.ndarray,
        (Any, numpy.dtype[numpy.uint64]),
    )
