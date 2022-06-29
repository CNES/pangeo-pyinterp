from typing import ClassVar, Tuple, overload

import numpy

from . import (
    Grid2DFloat32,
    Grid2DFloat64,
    Grid3DFloat32,
    Grid3DFloat64,
    TemporalGrid3DFloat32,
    TemporalGrid3DFloat64,
)

class FirstGuess:
    __members__: ClassVar[dict] = ...  # read-only
    Zero: ClassVar[FirstGuess] = ...
    ZonalAverage: ClassVar[FirstGuess] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, value: int) -> None:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __getstate__(self) -> int:
        ...

    def __hash__(self) -> int:
        ...

    def __index__(self) -> int:
        ...

    def __int__(self) -> int:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __setstate__(self, state: int) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def value(self) -> int:
        ...


class ValueType:
    __members__: ClassVar[dict] = ...  # read-only
    All: ClassVar[ValueType] = ...
    Defined: ClassVar[ValueType] = ...
    Undefined: ClassVar[ValueType] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, value: int) -> None:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __getstate__(self) -> int:
        ...

    def __hash__(self) -> int:
        ...

    def __index__(self) -> int:
        ...

    def __int__(self) -> int:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __setstate__(self, state: int) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def value(self) -> int:
        ...


def gauss_seidel_float32(grid: numpy.ndarray[numpy.float32],
                         first_guess: FirstGuess = ...,
                         is_circle: bool = ...,
                         max_iterations: int = ...,
                         epsilon: float = ...,
                         relaxation: float = ...,
                         num_thread: int = ...) -> Tuple[int, float]:
    ...


def gauss_seidel_float64(grid: numpy.ndarray[numpy.float64],
                         first_guess: FirstGuess = ...,
                         is_circle: bool = ...,
                         max_iterations: int = ...,
                         epsilon: float = ...,
                         relaxation: float = ...,
                         num_thread: int = ...) -> Tuple[int, float]:
    ...


@overload
def loess_float32(grid: Grid2DFloat32,
                  nx: int = ...,
                  ny: int = ...,
                  value_type: ValueType = ...,
                  num_threads: int = ...) -> numpy.ndarray[numpy.float32]:
    ...


@overload
def loess_float32(grid: Grid3DFloat32,
                  nx: int = ...,
                  ny: int = ...,
                  value_type: ValueType = ...,
                  num_threads: int = ...) -> numpy.ndarray[numpy.float32]:
    ...


@overload
def loess_float32(grid: TemporalGrid3DFloat32,
                  nx: int = ...,
                  ny: int = ...,
                  value_type: ValueType = ...,
                  num_threads: int = ...) -> numpy.ndarray[numpy.float32]:
    ...


@overload
def loess_float64(grid: Grid2DFloat64,
                  nx: int = ...,
                  ny: int = ...,
                  value_type: ValueType = ...,
                  num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def loess_float64(grid: Grid3DFloat64,
                  nx: int = ...,
                  ny: int = ...,
                  value_type: ValueType = ...,
                  num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def loess_float64(grid: TemporalGrid3DFloat64,
                  nx: int = ...,
                  ny: int = ...,
                  value_type: ValueType = ...,
                  num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...
