from typing import Dict, Optional, Tuple, overload
import numpy
from .. import geodetic


def bounding_box(hash: str) -> geodetic.Box:
    ...


def bounding_boxes(box: Optional[geodetic.Box] = None,
                   precision: int = 1) -> numpy.ndarray[numpy.bytes_]:
    ...


def decode(hash: str, round: bool = False) -> geodetic.Point:
    ...


@overload
def decode(
    hash: numpy.ndarray[numpy.bytes_],
    round: bool = False
) -> Tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
    ...


def encode(point: geodetic.Point, precision: int = 12) -> bytes:
    ...


@overload
def encode(lon: numpy.ndarray[numpy.float64],
           lat: numpy.ndarray[numpy.float64],
           precision: int = 12) -> numpy.ndarray[numpy.bytes_]:
    ...


def error(precision: int) -> tuple:
    ...


def grid_properties(box: geodetic.Box = None,
                    precision: int = 12) -> Tuple[int, int, int]:
    ...


def neighbors(hash: str) -> numpy.ndarray[numpy.bytes_]:
    ...


def where(
    hash: numpy.ndarray[numpy.bytes_]
) -> Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]]:
    ...
