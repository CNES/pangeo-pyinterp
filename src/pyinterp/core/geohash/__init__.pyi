from typing import Optional, Tuple, overload

import numpy

from .. import geodetic
from ..array import Array1DFloat64, Array1DStr

def area(hash: numpy.ndarray,
         wgs: Optional[geodetic.Spheroid] = None) -> Array1DFloat64:
    ...


@overload
def bounding_boxes(box: Optional[geodetic.Box] = ...,
                   precision: int = ...) -> Array1DStr:
    ...


@overload
def bounding_boxes(polygon: geodetic.Polygon,
                   precision: int = ...,
                   num_threads: int = ...) -> Array1DStr:
    ...


@overload
def bounding_boxes(polygons: geodetic.MultiPolygon,
                   precision: int = ...,
                   num_threads: int = ...) -> Array1DStr:
    ...


def decode(hash: Array1DStr,
           round: bool = ...) -> Tuple[Array1DFloat64, Array1DFloat64]:
    ...


def encode(lon: Array1DFloat64,
           lat: Array1DFloat64,
           precision: int = ...) -> Array1DStr:
    ...


def transform(hash: Array1DStr, precision: int = ...) -> Array1DStr:
    ...


def where(hash: Array1DStr) -> dict:
    ...
