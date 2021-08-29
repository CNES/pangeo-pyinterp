from typing import Optional, Tuple, Union
import numpy
from . import (Grid2DFloat32, Grid2DFloat64, Grid3DFloat32, Grid3DFloat32,
               Grid3DFloat64, TemporalGrid3DFloat64, TemporalGrid3DFloat32)


class FirstGuess:
    Zero: 'FirstGuess'
    ZonalAverage: 'FirstGuess'


class ValueType:
    Undefined: 'ValueType'
    Defined: 'ValueType'
    All: 'ValueType'


def loess_float64(
        grid: Union[Grid2DFloat64, Grid3DFloat64, TemporalGrid3DFloat64],
        nx: int = 3,
        ny: int = 3,
        processing_mode: Optional[str] = None,
        num_threads: int = 0) -> numpy.ndarray[numpy.float64]:
    ...


def loess_float32(
        grid: Union[Grid2DFloat32, Grid3DFloat32, TemporalGrid3DFloat32],
        nx: int = 3,
        ny: int = 3,
        processing_mode: Optional[str] = None,
        num_threads: int = 0) -> numpy.ndarray[numpy.float64]:
    ...


def gauss_seidel_float64(grid: numpy.ndarray[numpy.float64],
                         first_guess: FirstGuess = FirstGuess.ZonalAverage,
                         is_circle: bool = True,
                         max_iterations: int = 2000,
                         epsilon: float = 0.0001,
                         relaxation: float = 1.0,
                         num_thread: int = 0) -> Tuple[int, float]:
    ...


def gauss_seidel_float32(grid: numpy.ndarray[numpy.float32],
                         first_guess: FirstGuess = FirstGuess.ZonalAverage,
                         is_circle: bool = True,
                         max_iterations: int = 2000,
                         epsilon: float = 0.0001,
                         relaxation: float = 1.0,
                         num_thread: int = 0) -> Tuple[int, float]:
    ...
