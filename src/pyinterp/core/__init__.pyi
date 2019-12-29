from typing import Any, Optional, Tuple, Union
from types import ModuleType
import numpy
from . import geodetic
from . import fill


class AxisBoundary:
    Expand: 'AxisBoundary'
    Sym: 'AxisBoundary'
    Undef: 'AxisBoundary'
    Wrap: 'AxisBoundary'


class Axis:
    is_circle: bool

    def __init__(self,
                 values: numpy.ndarray[numpy.float64],
                 epsilon: float = 1e-6,
                 is_circle: bool = False) -> None:
        ...

    def __eq__(self, other: 'Axis') -> bool:
        ...

    def __getstate__(self) -> Tuple:
        ...

    def __ne__(self, other: 'Axis') -> bool:
        ...

    def __repr__(self) -> str:
        ...

    def __setstate__(self, state: Tuple) -> None:
        ...

    def __getitem__(
            self,
            arg0: Union[int,
                        Tuple[Optional[int], Optional[int], Optional[int]]]
    ) -> Union[float, numpy.ndarray[numpy.float64]]:
        ...

    def back(self) -> float:
        ...

    def front(self) -> float:
        ...

    def max_value(self) -> float:
        ...

    def min_value(self) -> float:
        ...

    def flip(self, inplace: bool = False) -> 'Axis':
        ...

    def is_ascending(self) -> bool:
        ...

    def is_regular(self) -> bool:
        ...

    def find_index(self,
                   coordinates: numpy.ndarray[numpy.float64],
                   bounded: bool = False) -> numpy.ndarray[numpy.float64]:
        ...

    def __len__(self) -> int:
        ...


class TemporalAxis:
    is_circle: bool

    def __init__(self,
                 values: numpy.ndarray[numpy.int64],
                 epsilon: int = 0,
                 is_circle: bool = False) -> None:
        ...

    def back(self) -> int:
        ...

    def front(self) -> int:
        ...

    def max_value(self) -> int:
        ...

    def min_value(self) -> int:
        ...

    def flip(self, inplace: bool = False) -> 'Axis':
        ...

    def is_ascending(self) -> bool:
        ...

    def is_regular(self) -> bool:
        ...

    def find_index(self,
                   coordinates: numpy.ndarray[numpy.int64],
                   bounded: bool = False) -> numpy.ndarray[numpy.int64]:
        ...

    def increment(self) -> int:
        ...

    def __setstate__(self, state: Tuple) -> None:
        ...

    def __eq__(self, other: 'Axis') -> bool:
        ...

    def __ne__(self, other: 'Axis') -> bool:
        ...

    def __getitem__(
            self,
            arg0: Union[int,
                        Tuple[Optional[int], Optional[int], Optional[int]]]
    ) -> Union[int, numpy.ndarray[numpy.int64]]:
        ...

    def __getstate__(self) -> Tuple:
        ...

    def __len__(self) -> int:
        ...

    def __repr__(self) -> str:
        ...


class FittingModel:
    Akima: 'FittingModel'
    AkimaPeriodic: 'FittingModel'
    CSpline: 'FittingModel'
    CSplinePeriodic: 'FittingModel'
    Linear: 'FittingModel'
    Polynomial: 'FittingModel'
    Steffen: 'FittingModel'


class RadialBasisFunction:
    pass


class Grid2DFloat64:
    array: numpy.ndarray[numpy.float64]
    x: Axis
    y: Axis

    def __getstate__(self) -> tuple:
        ...

    def __init__(self, x: Axis, y: Axis,
                 array: numpy.ndarray[numpy.float64]) -> None:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Grid2DFloat32:
    array: numpy.ndarray[numpy.float32]
    x: Axis
    y: Axis

    def __getstate__(self) -> tuple:
        ...

    def __init__(self, x: Axis, y: Axis,
                 array: numpy.ndarray[numpy.float32]) -> None:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Grid3DFloat64:
    array: numpy.ndarray[numpy.float64]
    x: Axis
    y: Axis
    z: Axis

    def __getstate__(self) -> tuple:
        ...

    def __init__(self, x: Axis, y: Axis, z: Axis,
                 array: numpy.ndarray[numpy.float64]) -> None:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Grid3DFloat32:
    array: numpy.ndarray[numpy.float32]
    x: Axis
    y: Axis
    z: Axis

    def __getstate__(self) -> tuple:
        ...

    def __init__(self, x: Axis, y: Axis, z: Axis,
                 array: numpy.ndarray[numpy.float32]) -> None:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Grid4DFloat64:
    array: numpy.ndarray[numpy.float64]
    x: Axis
    y: Axis
    z: Axis
    u: Axis

    def __getstate__(self) -> tuple:
        ...

    def __init__(self, x: Axis, y: Axis, z: Axis, u: Axis,
                 array: numpy.ndarray[numpy.float64]) -> None:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Grid4DFloat32:
    array: numpy.ndarray[numpy.float32]
    x: Axis
    y: Axis
    z: Axis
    u: Axis

    def __getstate__(self) -> tuple:
        ...

    def __init__(self, x: Axis, y: Axis, z: Axis, u: Axis,
                 array: numpy.ndarray[numpy.float32]) -> None:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class TemporalGrid3DFloat64:
    array: numpy.ndarray[numpy.float64]
    x: Axis
    y: Axis
    z: TemporalAxis

    def __getstate__(self) -> tuple:
        ...

    def __init__(self, x: Axis, y: Axis, z: TemporalAxis,
                 array: numpy.ndarray[numpy.float64]) -> None:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class TemporalGrid3DFloat32:
    array: numpy.ndarray[numpy.float32]
    x: Axis
    y: Axis
    z: TemporalAxis

    def __getstate__(self) -> tuple:
        ...

    def __init__(self, x: Axis, y: Axis, z: TemporalAxis,
                 array: numpy.ndarray[numpy.float32]) -> None:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class TemporalGrid4DFloat64:
    array: numpy.ndarray[numpy.float64]
    x: Axis
    y: Axis
    z: TemporalAxis
    u: Axis

    def __getstate__(self) -> tuple:
        ...

    def __init__(self, x: Axis, y: Axis, z: TemporalAxis, u: Axis,
                 array: numpy.ndarray[numpy.float64]) -> None:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class TemporalGrid4DFloat32:
    array: numpy.ndarray[numpy.float32]
    x: Axis
    y: Axis
    z: TemporalAxis
    u: Axis

    def __getstate__(self) -> tuple:
        ...

    def __init__(self, x: Axis, y: Axis, z: TemporalAxis, u: Axis,
                 array: numpy.ndarray[numpy.float32]) -> None:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Binning2D:
    x: Axis
    y: Axis

    def __init__(self, x: Axis, y: Axis,
                 wgs: Optional[geodetic.System] = None) -> None:
        ...

    def clear(self) -> None:
        ...

    def push(self,
             x: numpy.ndarray,
             y: numpy.ndarray,
             z: numpy.ndarray,
             simple: Optional[bool] = True) -> None:
        ...

    def count(self) -> numpy.ndarray:
        ...

    def kurtosis(self) -> numpy.ndarray:
        ...

    def max(self) -> numpy.ndarray:
        ...

    def mean(self) -> numpy.ndarray:
        ...

    def median(self) -> numpy.ndarray:
        ...

    def min(self) -> numpy.ndarray:
        ...

    def skewness(self) -> numpy.ndarray:
        ...

    def sum(self) -> numpy.ndarray:
        ...

    def sum_of_weights(self) -> numpy.ndarray:
        ...

    def variance(self) -> numpy.ndarray:
        ...


class Binning2DFloat64(Binning2D):
    ...


class Binning2DFloat32(Binning2D):
    ...
