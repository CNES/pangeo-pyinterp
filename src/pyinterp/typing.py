# Copyright (c) 2024 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Typing
======

.. py:data:: NDArray

    A numpy tensor with any type.

"""
from typing import TYPE_CHECKING, Any, Tuple
import sys

import numpy


def numpy_version() -> Tuple[int, int]:
    """Returns the version of the installed numpy library.

    Returns:
        Tuple[int, int]: The version of the installed numpy library.
    """
    return tuple(map(int, numpy.__version__.split('.')[:2]))


if TYPE_CHECKING and numpy_version() >= (1, 20) and sys.version_info > (
        3, 8):  # pragma: no cover
    import numpy.typing

    NDArray = numpy.typing.NDArray
    NDArrayDateTime = numpy.ndarray[Any, numpy.dtype[numpy.datetime64]]
    NDArrayStructured = numpy.ndarray[Any, numpy.dtype[numpy.void]]
    NDArrayTimeDelta = numpy.ndarray[Any, numpy.dtype[numpy.timedelta64]]

else:  # pragma: no cover
    NDArray = numpy.ndarray
    NDArrayDateTime = NDArray
    NDArrayStructured = NDArray
    NDArrayTimeDelta = NDArray
