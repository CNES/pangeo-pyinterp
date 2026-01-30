# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Type hints for pyinterp package."""

from __future__ import annotations

from types import GenericAlias
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

import numpy as np


OneDim: TypeAlias = tuple[int]
TwoDims: TypeAlias = tuple[int, int]
ThreeDims: TypeAlias = tuple[int, int, int]
FourDims: TypeAlias = tuple[int, int, int, int]

if TYPE_CHECKING:
    NDArray1D: TypeAlias = np.ndarray[
        OneDim,
        np.dtype[Any],
    ]
    NDArray1DNumeric: TypeAlias = np.ndarray[
        OneDim,
        np.dtype[np.integer[Any] | np.floating[Any]],
    ]
    NDArray1DNumericWithTime: TypeAlias = np.ndarray[
        OneDim,
        np.dtype[
            np.integer[Any] | np.floating[Any] | np.datetime64 | np.timedelta64
        ],
    ]
    NDArray1DUInt8: TypeAlias = np.ndarray[
        OneDim,
        np.dtype[np.uint8],
    ]
    NDArray1DBool: TypeAlias = np.ndarray[
        OneDim,
        np.dtype[np.int8],
    ]
    NDArray1DInt32: TypeAlias = np.ndarray[
        OneDim,
        np.dtype[np.int32],
    ]
    NDArray1DInt64: TypeAlias = np.ndarray[
        OneDim,
        np.dtype[np.int64],
    ]
    NDArray1DFloat32: TypeAlias = np.ndarray[
        OneDim,
        np.dtype[np.float32],
    ]
    NDArray1DFloat64: TypeAlias = np.ndarray[
        OneDim,
        np.dtype[np.float64],
    ]
    NDArray1DStr: TypeAlias = np.ndarray[
        OneDim,
        np.dtype[np.str_],
    ]
    NDArray1DDateTime64: TypeAlias = np.ndarray[
        OneDim,
        np.dtype[np.datetime64],
    ]
    NDArray1DTimeDelta64: TypeAlias = np.ndarray[
        OneDim,
        np.dtype[np.timedelta64],
    ]
    NDArray1DUInt32: TypeAlias = np.ndarray[
        OneDim,
        np.dtype[np.uint32],
    ]

    NDArray2DBool: TypeAlias = np.ndarray[
        TwoDims,
        np.dtype[np.bool],
    ]
    NDArray2DInt64: TypeAlias = np.ndarray[
        TwoDims,
        np.dtype[np.int64],
    ]
    NDArray2DFloat32: TypeAlias = np.ndarray[
        TwoDims,
        np.dtype[np.float32],
    ]
    NDArray2DFloat64: TypeAlias = np.ndarray[
        TwoDims,
        np.dtype[np.float64],
    ]
    NDArray2DStr: TypeAlias = np.ndarray[
        TwoDims,
        np.dtype[np.str_],
    ]
    NDArray2DUInt64: TypeAlias = np.ndarray[
        TwoDims,
        np.dtype[np.int64],
    ]
    NDArray2DFloating: TypeAlias = np.ndarray[
        TwoDims,
        np.dtype[np.floating[Any]],
    ]
else:
    ScalarType_co = TypeVar("ScalarType_co", bound=np.generic, covariant=True)
    _DType = GenericAlias(
        np.dtype,
        (ScalarType_co,),
    )
    NDArray1D = GenericAlias(
        np.ndarray,
        (OneDim, _DType),
    )
    NDArray1DUInt8 = GenericAlias(
        np.ndarray,
        (OneDim, np.dtype[np.uint8]),
    )
    NDArray1DBool = GenericAlias(
        np.ndarray,
        (OneDim, np.dtype[np.int8]),
    )
    NDArray1DInt32 = GenericAlias(
        np.ndarray,
        (OneDim, np.dtype[np.int32]),
    )
    NDArray1DInt64 = GenericAlias(
        np.ndarray,
        (OneDim, np.dtype[np.int64]),
    )
    NDArray1DFloat32 = GenericAlias(
        np.ndarray,
        (OneDim, np.dtype[np.float32]),
    )
    NDArray1DFloat64 = GenericAlias(
        np.ndarray,
        (OneDim, np.dtype[np.float64]),
    )
    NDArray1DStr = GenericAlias(
        np.ndarray,
        (OneDim, np.dtype[np.str_]),
    )
    NDArray1DDateTime64 = GenericAlias(
        np.ndarray,
        (OneDim, np.dtype[np.datetime64]),
    )
    NDArray1DTimeDelta64 = GenericAlias(
        np.ndarray,
        (OneDim, np.dtype[np.timedelta64]),
    )
    NDArray1DUInt32 = GenericAlias(
        np.ndarray,
        (OneDim, np.dtype[np.uint32]),
    )

    NDArray2DBool = GenericAlias(
        np.ndarray,
        (TwoDims, np.dtype[np.bool]),
    )
    NDArray2DInt64 = GenericAlias(
        np.ndarray,
        (TwoDims, np.dtype[np.int64]),
    )
    NDArray2DFloat32 = GenericAlias(
        np.ndarray,
        (TwoDims, np.dtype[np.float32]),
    )
    NDArray2DFloat64 = GenericAlias(
        np.ndarray,
        (TwoDims, np.dtype[np.float64]),
    )
    NDArray2DStr = GenericAlias(
        np.ndarray,
        (TwoDims, np.dtype[np.str_]),
    )
    NDArray2DUInt64 = GenericAlias(
        np.ndarray,
        (TwoDims, np.dtype[np.int64]),
    )
    NDArray2DFloating = GenericAlias(
        np.ndarray,
        (TwoDims, np.dtype[np.floating[Any]]),
    )
