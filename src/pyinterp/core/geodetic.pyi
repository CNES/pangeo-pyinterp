from typing import Tuple, Optional, Union
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


class Point2D:
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


class Box2D:
    min_corner: Point2D
    max_corner: Point2D

    def __init__(self,
                 min_corner: Optional[Point2D] = None,
                 max_corner: Optional[Point2D] = None) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    def covered_by(self,
                   lon: numpy.ndarray[numpy.float64],
                   lat: numpy.ndarray[numpy.float64],
                   num_theads: int = 1) -> numpy.ndarray[numpy.int8]:
        ...

    @staticmethod
    def entire_earth() -> 'Box2D':
        ...
