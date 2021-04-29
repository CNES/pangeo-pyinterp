from typing import Any, Dict, Iterable, Optional, Tuple, overload
import numpy
from .. import geodetic


@overload
def area(hash: str, wgs: Optional[geodetic.System] = None) -> float:
    ...


@overload
def area(
        hash: numpy.ndarray,
        wgs: Optional[geodetic.System] = None) -> numpy.ndarray[numpy.float64]:
    ...


def bounding_box(hash: str) -> geodetic.Box:
    ...


def bounding_boxes(box: Optional[geodetic.Box] = None,
                   precision: int = 1) -> numpy.ndarray[numpy.bytes_]:
    ...


@overload
def decode(hash: str, round: bool = False) -> geodetic.Point:
    ...


@overload
def decode(
    hash: numpy.ndarray[numpy.bytes_],
    round: bool = False
) -> Tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
    ...


@overload
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
) -> Dict[bytes, Tuple[Tuple[int, int], Tuple[int, int]]]:
    ...


def update_dict(dictionnary: Dict, others: Iterable[Tuple[Any, Any]]) -> None:
    ...
