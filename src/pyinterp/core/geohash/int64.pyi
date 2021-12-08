from typing import Tuple, overload
import numpy
from .. import geodetic


def bounding_box(hash: int, precision: int = ...) -> geodetic.Box:
    ...


@overload
def decode(hash: int,
           precision: int = ...,
           round: bool = ...) -> geodetic.Point:
    ...


@overload
def decode(
    hash: numpy.ndarray[numpy.uint64],
    precision: int = ...,
    round: bool = ...
) -> Tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
    ...


@overload
def encode(point: geodetic.Point, precision: int = ...) -> int:
    ...


@overload
def encode(lon: numpy.ndarray[numpy.float64],
           lat: numpy.ndarray[numpy.float64],
           precision: int = ...) -> numpy.ndarray[numpy.uint64]:
    ...


def error(precision: int) -> tuple:
    ...


def grid_properties(box: geodetic.Box = ...,
                    precision: int = ...) -> Tuple[int, int, int]:
    ...


def neighbors(hash: int, precision: int = ...) -> numpy.ndarray[numpy.uint64]:
    ...
