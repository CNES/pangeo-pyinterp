from typing import Any, ClassVar, Optional, overload
import numpy
from .. import geodetic


class Box:
    __hash__: ClassVar[None] = ...
    max_corner: Point
    min_corner: Point

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, min_corner: Point, max_corner: Point) -> None:
        ...

    def area(self, wgs: Optional[System] = ...) -> float:
        ...

    @overload
    def covered_by(self, point: Point) -> bool:
        ...

    @overload
    def covered_by(self,
                   lon: numpy.ndarray[numpy.float64],
                   lat: numpy.ndarray[numpy.float64],
                   num_theads: int = ...) -> numpy.ndarray[numpy.int8]:
        ...

    @overload
    def distance(self, other: Box) -> float:
        ...

    @overload
    def distance(self, point: Point) -> float:
        ...

    @staticmethod
    def read_wkt(wkt: str) -> Box:
        ...

    @staticmethod
    def whole_earth() -> Box:
        ...

    def wkt(self) -> str:
        ...

    def __eq__(self, other: Box) -> bool:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __ne__(self, other: Box) -> bool:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...


class Coordinates:
    def __init__(self, system: Optional[System]) -> None:
        ...

    def ecef_to_lla(self,
                    x: numpy.ndarray[numpy.float64],
                    y: numpy.ndarray[numpy.float64],
                    z: numpy.ndarray[numpy.float64],
                    num_threads: int = ...) -> tuple:
        ...

    def lla_to_ecef(self,
                    lon: numpy.ndarray[numpy.float64],
                    lat: numpy.ndarray[numpy.float64],
                    alt: numpy.ndarray[numpy.float64],
                    num_threads: int = ...) -> tuple:
        ...

    def transform(self,
                  target: Coordinates,
                  lon: numpy.ndarray[numpy.float64],
                  lat: numpy.ndarray[numpy.float64],
                  alt: numpy.ndarray[numpy.float64],
                  num_threads: int = ...) -> tuple:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...


class Point:
    __hash__: ClassVar[None] = ...
    lat: float
    lon: float

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, lon: float, lat: float) -> None:
        ...

    def distance(self,
                 other: Point,
                 strategy: str = ...,
                 wgs: Optional[System] = ...) -> float:
        ...

    @staticmethod
    def read_wkt(wkt: str) -> Point:
        ...

    def wkt(self) -> str:
        ...

    def __eq__(self, other: Point) -> bool:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __ne__(self, other: Point) -> bool:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...


class Polygon:
    __hash__: ClassVar[None] = ...

    def __init__(self, outer: list, inners: Optional[list] = ...) -> None:
        ...

    def area(self, wgs: Optional[System] = ...) -> float:
        ...

    @overload
    def covered_by(self, point: Point) -> bool:
        ...

    @overload
    def covered_by(self,
                   lon: numpy.ndarray[numpy.float64],
                   lat: numpy.ndarray[numpy.float64],
                   num_theads: int = ...) -> numpy.ndarray[numpy.int8]:
        ...

    @overload
    def distance(self, other: Polygon) -> float:
        ...

    @overload
    def distance(self, point: Point) -> float:
        ...

    def envelope(self) -> Box:
        ...

    @staticmethod
    def read_wkt(wkt: str) -> Polygon:
        ...

    def wkt(self) -> str:
        ...

    def __eq__(self, other: Polygon) -> bool:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __ne__(self, other: Polygon) -> bool:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...


class System(_System):
    __hash__: ClassVar[None] = ...

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, semi_major_axis: float, flattening: float) -> None:
        ...

    def authalic_radius(self) -> float:
        ...

    def axis_ratio(self) -> float:
        ...

    def equatorial_circumference(self, semi_major_axis: bool = ...) -> float:
        ...

    def equatorial_radius_of_curvature(self) -> float:
        ...

    def first_eccentricity_squared(self) -> float:
        ...

    def linear_eccentricity(self) -> float:
        ...

    def mean_radius(self) -> float:
        ...

    def polar_radius_of_curvature(self) -> float:
        ...

    def second_eccentricity_squared(self) -> float:
        ...

    def semi_minor_axis(self) -> float:
        ...

    def volumetric_radius(self) -> float:
        ...

    def __eq__(self, other: _System) -> bool:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __ne__(self, other: _System) -> bool:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def flattening(self) -> float:
        ...

    @property
    def semi_major_axis(self) -> float:
        ...


class _System:
    def __init__(self, *args, **kwargs) -> None:
        ...


def coordinate_distances(
        lon1: numpy.ndarray[numpy.float64],
        lat1: numpy.ndarray[numpy.float64],
        lon2: numpy.ndarray[numpy.float64],
        lat2: numpy.ndarray[numpy.float64],
        strategy: str = ...,
        wgs: Optional[System] = ...,
        num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...
