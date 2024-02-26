from typing import Any, Literal

import numpy

_1D = Literal['N']
_2D = Literal['N', 'M']
_3D = Literal['N', 'M', 'O']
_4D = Literal['N', 'M', 'O', 'P']

Array1DBool = numpy.ndarray[_1D, numpy.dtype[numpy.bool_]]
Array1DFloat32 = numpy.ndarray[_1D, numpy.dtype[numpy.float32]]
Array1DFloat64 = numpy.ndarray[_1D, numpy.dtype[numpy.float64]]
Array1DInt64 = numpy.ndarray[_1D, numpy.dtype[numpy.int64]]
Array1DStr = numpy.ndarray[_1D, numpy.dtype[numpy.str_]]
Array1DUInt64 = numpy.ndarray[_1D, numpy.dtype[numpy.uint64]]
Array2DFloat32 = numpy.ndarray[_2D, numpy.dtype[numpy.float32]]
Array2DFloat64 = numpy.ndarray[_2D, numpy.dtype[numpy.float64]]
Array2DInt64 = numpy.ndarray[_2D, numpy.dtype[numpy.int64]]
Array2DInt8 = numpy.ndarray[_2D, numpy.dtype[numpy.int8]]
Array2DUInt64 = numpy.ndarray[_2D, numpy.dtype[numpy.uint64]]
Array2DUInt8 = numpy.ndarray[_2D, numpy.dtype[numpy.uint8]]
Array3DFloat32 = numpy.ndarray[_3D, numpy.dtype[numpy.float32]]
Array3DFloat64 = numpy.ndarray[_3D, numpy.dtype[numpy.float64]]
Array3DInt8 = numpy.ndarray[_3D, numpy.dtype[numpy.int8]]
Array4DFloat32 = numpy.ndarray[_4D, numpy.dtype[numpy.float32]]
Array4DFloat64 = numpy.ndarray[_4D, numpy.dtype[numpy.float64]]
Array4DInt8 = numpy.ndarray[_4D, numpy.dtype[numpy.int8]]
ArrayFloat32 = numpy.ndarray[Any, numpy.dtype[numpy.float32]]
ArrayFloat64 = numpy.ndarray[Any, numpy.dtype[numpy.float64]]
ArrayInt64 = numpy.ndarray[Any, numpy.dtype[numpy.int64]]
ArrayUInt64 = numpy.ndarray[Any, numpy.dtype[numpy.uint64]]