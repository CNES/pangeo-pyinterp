from typing import Tuple, overload

import numpy

from .. import geodetic

def decode(
    hash: numpy.ndarray[numpy.uint64],
    precision: int = ...,
    round: bool = ...
) -> Tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
    ...


def encode(lon: numpy.ndarray[numpy.float64],
           lat: numpy.ndarray[numpy.float64],
           precision: int = ...) -> numpy.ndarray[numpy.uint64]:
    ...


def neighbors(hash: int, precision: int = ...) -> numpy.ndarray[numpy.uint64]:
    ...
