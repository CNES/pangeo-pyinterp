from typing import ClassVar, Tuple, overload

from . import (
    Grid2DFloat32,
    Grid2DFloat64,
    Grid3DFloat32,
    Grid3DFloat64,
    Grid4DFloat32,
    Grid4DFloat64,
    TemporalGrid3DFloat32,
    TemporalGrid3DFloat64,
    TemporalGrid4DFloat32,
    TemporalGrid4DFloat64,
)
from .array import (
    Array1DFloat32,
    Array1DFloat64,
    Array1DInt64,
    Array2DFloat32,
    Array2DFloat64,
    Array3DFloat32,
    Array3DFloat64,
    Array4DFloat32,
    Array4DFloat64,
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


def gauss_seidel_float32(grid: Array2DFloat32,
                         first_guess: FirstGuess = ...,
                         is_circle: bool = ...,
                         max_iterations: int = ...,
                         epsilon: float = ...,
                         relaxation: float = ...,
                         num_threads: int = ...) -> Tuple[int, float]:
    ...


def gauss_seidel_float64(grid: Array2DFloat64,
                         first_guess: FirstGuess = ...,
                         is_circle: bool = ...,
                         max_iterations: int = ...,
                         epsilon: float = ...,
                         relaxation: float = ...,
                         num_threads: int = ...) -> Tuple[int, float]:
    ...


@overload
def loess_float32(grid: Grid2DFloat32,
                  nx: int = ...,
                  ny: int = ...,
                  value_type: ValueType = ...,
                  num_threads: int = ...) -> Array2DFloat32:
    ...


@overload
def loess_float32(grid: Grid3DFloat32,
                  nx: int = ...,
                  ny: int = ...,
                  value_type: ValueType = ...,
                  num_threads: int = ...) -> Array3DFloat32:
    ...


@overload
def loess_float32(grid: TemporalGrid3DFloat32,
                  nx: int = ...,
                  ny: int = ...,
                  value_type: ValueType = ...,
                  num_threads: int = ...) -> Array3DFloat32:
    ...


@overload
def loess_float32(grid: Grid4DFloat32,
                  nx: int = ...,
                  ny: int = ...,
                  value_type: ValueType = ...,
                  num_threads: int = ...) -> Array4DFloat32:
    ...


@overload
def loess_float32(grid: TemporalGrid4DFloat32,
                  nx: int = ...,
                  ny: int = ...,
                  value_type: ValueType = ...,
                  num_threads: int = ...) -> Array4DFloat32:
    ...


@overload
def loess_float64(grid: Grid2DFloat64,
                  nx: int = ...,
                  ny: int = ...,
                  value_type: ValueType = ...,
                  num_threads: int = ...) -> Array2DFloat64:
    ...


@overload
def loess_float64(grid: Grid3DFloat64,
                  nx: int = ...,
                  ny: int = ...,
                  value_type: ValueType = ...,
                  num_threads: int = ...) -> Array3DFloat64:
    ...


@overload
def loess_float64(grid: TemporalGrid3DFloat64,
                  nx: int = ...,
                  ny: int = ...,
                  value_type: ValueType = ...,
                  num_threads: int = ...) -> Array3DFloat64:
    ...


@overload
def loess_float64(grid: Grid4DFloat64,
                  nx: int = ...,
                  ny: int = ...,
                  value_type: ValueType = ...,
                  num_threads: int = ...) -> Array4DFloat64:
    ...


@overload
def loess_float64(grid: TemporalGrid4DFloat64,
                  nx: int = ...,
                  ny: int = ...,
                  value_type: ValueType = ...,
                  num_threads: int = ...) -> Array4DFloat64:
    ...


def matrix_float32(x: Array2DFloat32, fill_value: float = ...) -> None:
    ...


def matrix_float64(x: Array2DFloat64, fill_value: float = ...) -> None:
    ...


def vector_float32(x: Array1DFloat32, fill_value: float = ...) -> None:
    ...


def vector_float64(x: Array1DFloat64, fill_value: float = ...) -> None:
    ...


def vector_int64(x: Array1DInt64, fill_value: int) -> None:
    ...
