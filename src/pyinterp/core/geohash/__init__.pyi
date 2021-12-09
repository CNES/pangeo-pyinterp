from typing import Iterable, Optional, Tuple, overload
import numpy
from .. import geodetic


@overload
def area(hash: str, wgs: Optional[geodetic.System] = ...) -> float:
    ...


@overload
def area(hash: numpy.ndarray,
         wgs: Optional[geodetic.System] = ...) -> numpy.ndarray[numpy.float64]:
    ...


def bounding_box(hash: str) -> geodetic.Box:
    ...


@overload
def bounding_boxes(box: Optional[geodetic.Box] = ...,
                   precision: int = ...) -> numpy.ndarray:
    ...


@overload
def bounding_boxes(box: geodetic.Polygon = ...,
                   precision: int = ...) -> numpy.ndarray:
    ...


@overload
def decode(hash: str, round: bool = ...) -> geodetic.Point:
    ...


@overload
def decode(
    hash: numpy.ndarray,
    round: bool = ...
) -> Tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
    ...


@overload
def encode(point: geodetic.Point, precision: int = ...) -> handle:
    ...


@overload
def encode(lon: numpy.ndarray[numpy.float64],
           lat: numpy.ndarray[numpy.float64],
           precision: int = ...) -> numpy.ndarray:
    ...


def error(precision: int) -> tuple:
    ...


def grid_properties(box: geodetic.Box = ...,
                    precision: int = ...) -> Tuple[int, int, int]:
    ...


def neighbors(hash: str) -> numpy.ndarray:
    ...


def update_dict(dictionary: dict, others: Iterable) -> None:
    ...


def where(hash: numpy.ndarray) -> dict:
    ...


def transform(hash: numpy.ndarray, precision: int = ...) -> numpy.ndarray:
    ...
