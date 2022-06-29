from typing import Optional, Tuple, overload

import numpy

from . import int64
from .. import geodetic

def area(
        hash: numpy.ndarray,
        wgs: Optional[geodetic.Spheroid] = None
) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def bounding_boxes(box: Optional[geodetic.Box] = ...,
                   precision: int = ...) -> numpy.ndarray:
    ...


@overload
def bounding_boxes(polygon: geodetic.Polygon,
                   precision: int = ...,
                   num_threads: int = ...) -> numpy.ndarray:
    ...


@overload
def bounding_boxes(polygons: geodetic.MultiPolygon,
                   precision: int = ...,
                   num_threads: int = ...) -> numpy.ndarray:
    ...


def decode(
    hash: numpy.ndarray,
    round: bool = ...
) -> Tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
    ...


def encode(lon: numpy.ndarray[numpy.float64],
           lat: numpy.ndarray[numpy.float64],
           precision: int = ...) -> numpy.ndarray:
    ...


def transform(hash: numpy.ndarray, precision: int = ...) -> numpy.ndarray:
    ...


def where(hash: numpy.ndarray) -> dict:
    ...
