# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Typing
======

.. py:data:: NDArray

    A numpy tensor with any type.

"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

import numpy
import numpy.typing

try:
    from types import GenericAlias  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    # pylint: disable=ungrouped-imports
    # For Python < 3.9 we use a backport of GenericAlias provided by
    # numpy
    # isort: off
    from numpy._typing._generic_alias import (  # type: ignore[misc,no-redef]
        _GenericAlias as GenericAlias,  # yapf: disable
    )
    # isort: on
    # pylint: enable=ungrouped-imports

NDArray: TypeAlias = numpy.typing.NDArray[Any]


def numpy_version() -> tuple[int, ...]:
    """Returns the version of the installed numpy library.

    Returns:
        Tuple[int, int]: The version of the installed numpy library.
    """
    return tuple(map(int, numpy.__version__.split('.')[:2]))


if TYPE_CHECKING:  # pragma: no cover
    NDArrayDateTime = numpy.ndarray[Any, numpy.dtype[numpy.datetime64]]
    NDArrayStructured = numpy.ndarray[Any, numpy.dtype[numpy.void]]
    NDArrayTimeDelta = numpy.ndarray[Any, numpy.dtype[numpy.timedelta64]]

else:  # pragma: no cover
    ScalarType_co = TypeVar('ScalarType_co',
                            bound=numpy.generic,
                            covariant=True)
    _DType = GenericAlias(
        numpy.dtype,
        (ScalarType_co, ),
    )
    NDArrayDateTime = GenericAlias(
        numpy.ndarray,
        (Any, numpy.dtype(numpy.datetime64)),
    )
    NDArrayStructured = GenericAlias(
        numpy.ndarray,
        (Any, _DType),
    )
    NDArrayTimeDelta = GenericAlias(
        numpy.ndarray,
        (Any, numpy.dtype(numpy.timedelta64)),
    )
