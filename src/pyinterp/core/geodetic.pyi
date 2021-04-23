from typing import List, Tuple, Optional, overload
import numpy


class System:
    flattening: float
    semi_major_axis: float

    def __init__(self,
                 semi_major_axis: Optional[float] = None,
                 flattening: Optional[float] = None) -> None:
        ...

    def __eq__(self, arg0: 'System') -> bool:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __ne__(self, arg0: 'System') -> bool:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    def authalic_radius(self) -> float:
        ...

    def axis_ratio(self) -> float:
        ...

    def equatorial_circumference(self, semi_major_axis: bool = True) -> float:
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


class Coordinates:
    ...

    def __getstate__(self) -> Tuple:
        ...

    def __init__(self, system: Optional[System]) -> None:
        ...

    def __setstate__(self, tuple: tuple) -> None:
        ...

    def ecef_to_lla(self,
                    x: numpy.ndarray[numpy.float64],
                    y: numpy.ndarray[numpy.float64],
                    z: numpy.ndarray[numpy.float64],
                    num_threads: int = 0) -> tuple:
        ...

    def lla_to_ecef(self,
                    lon: numpy.ndarray[numpy.float64],
                    lat: numpy.ndarray[numpy.float64],
                    alt: numpy.ndarray[numpy.float64],
                    num_threads: int = 0) -> tuple:
        ...

    def transform(self,
                  target: 'Coordinates',
                  lon: numpy.ndarray[numpy.float64],
                  lat: numpy.ndarray[numpy.float64],
                  alt: numpy.ndarray[numpy.float64],
                  num_threads: int = 0) -> tuple:
        ...


class Point:
    lat: float
    lon: float

    def __init__(self,
                 lon: Optional[float] = None,
                 lat: Optional[float] = None) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    def distance(self,
                 other: "Point",
                 strategy: str = 'thomas',
                 wgs: Optional[System] = None) -> float:
        ...

    def wtk(self) -> str:
        ...

    @staticmethod
    def read_wkt(wtk: str) -> 'Point':
        ...


class Box:
    min_corner: Point
    max_corner: Point

    def __init__(self,
                 min_corner: Optional[Point] = None,
                 max_corner: Optional[Point] = None) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    def area(self, wgs: Optional[System] = None) -> float:
        ...

    @overload
    def distance(self, other: 'Box') -> float:
        ...

    @overload
    def distance(self, other: Point) -> float:
        ...

    @overload
    def covered_by(self, point: Point) -> bool:
        ...

    @overload
    def covered_by(self,
                   lon: numpy.ndarray[numpy.float64],
                   lat: numpy.ndarray[numpy.float64],
                   num_theads: int = 1) -> numpy.ndarray[numpy.int8]:
        ...

    def wtk(self) -> str:
        ...

    @staticmethod
    def read_wkt(wkt: str) -> 'Box':
        ...

    @staticmethod
    def whole_earth() -> 'Box':
        ...


class Polygon:
    def __init__(self,
                 outer: List[Point],
                 inners: Optional[List[List[Point]]] = None) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def area(self, wgs: Optional[System] = None) -> float:
        ...

    @overload
    def covered_by(self, point: Point) -> bool:
        ...

    @overload
    def covered_by(self,
                   lon: numpy.ndarray[numpy.float64],
                   lat: numpy.ndarray[numpy.float64],
                   num_theads: int = 1) -> numpy.ndarray[numpy.int8]:
        ...

    @overload
    def distance(self, other: 'Polygon') -> float:
        ...

    @overload
    def distance(self, other: Point) -> float:
        ...

    def envelope(self) -> Box:
        ...

    def wkt(self) -> str:
        ...

    @staticmethod
    def read_wkt(wkt: str) -> "Polygon":
        ...


def coordinate_distances(lon1: numpy.ndarray,
                         lat1: numpy.ndarray,
                         lon2: numpy.ndarray,
                         lat2: numpy.ndarray,
                         strategy: str = 'thomas',
                         wgs: Optional[System] = None,
                         num_threads: int = 0) -> numpy.ndarray:
    ...
