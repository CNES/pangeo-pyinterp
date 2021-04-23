from typing import Dict, Optional, Tuple, overload
import numpy
from .. import geodetic


def bounding_box(hash: int, precision: int = 64) -> geodetic.Box:
    ...


def bounding_boxes(box: Optional[geodetic.Box] = None,
                   precision: int = 5) -> numpy.ndarray[numpy.uint64]:
    ...


@overload
def decode(hash: int,
           precision: int = 64,
           round: bool = False) -> geodetic.Point:
    ...


@overload
def decode(
    hash: numpy.ndarray[numpy.uint64],
    precision: int = 64,
    round: bool = False
) -> Tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
    ...


def encode(point: geodetic.Point, precision: int = 64) -> int:
    ...


@overload
def encode(lon: numpy.ndarray[numpy.float64],
           lat: numpy.ndarray[numpy.float64],
           precision: int = 64) -> numpy.ndarray[numpy.uint64]:
    ...


def error(precision: int) -> tuple:
    ...


def grid_properties(box: geodetic.Box = None,
                    precision: int = 64) -> Tuple[int, int, int]:
    ...


def neighbors(hash: int, precision: int = 64) -> numpy.ndarray[numpy.uint64]:
    ...


def where(
    hash: numpy.ndarray[numpy.uint64]
) -> Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]]:
    ...
