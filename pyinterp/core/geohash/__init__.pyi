from typing import Optional, Tuple, overload

import numpy

from .. import geodetic
from ...typing import NDArray1DFloat64, NDArray1DStr

def area(hash: numpy.ndarray,
         wgs: Optional[geodetic.Spheroid] = None) -> NDArray1DFloat64:
    ...


@overload
def bounding_boxes(box: Optional[geodetic.Box] = ...,
                   precision: int = ...) -> NDArray1DStr:
    ...


@overload
def bounding_boxes(polygon: geodetic.Polygon,
                   precision: int = ...,
                   num_threads: int = ...) -> NDArray1DStr:
    ...


@overload
def bounding_boxes(polygons: geodetic.MultiPolygon,
                   precision: int = ...,
                   num_threads: int = ...) -> NDArray1DStr:
    ...


def decode(hash: NDArray1DStr,
           round: bool = ...) -> Tuple[NDArray1DFloat64, NDArray1DFloat64]:
    ...


def encode(lon: NDArray1DFloat64,
           lat: NDArray1DFloat64,
           precision: int = ...) -> NDArray1DStr:
    ...


def transform(hash: NDArray1DStr, precision: int = ...) -> NDArray1DStr:
    ...


def where(hash: NDArray1DStr) -> dict:
    ...
