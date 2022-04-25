from typing import (
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    overload,
)

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

    def as_polygon(self, *args, **kwargs) -> Any:
        ...

    def centroid(self) -> Point:
        ...

    @overload
    def covered_by(self, point: Point) -> bool:
        ...

    @overload
    def covered_by(self,
                   lon: numpy.ndarray[numpy.float64],
                   lat: numpy.ndarray[numpy.float64],
                   num_threads: int = ...) -> numpy.ndarray[numpy.int8]:
        ...

    @overload
    def distance(self, other: Box) -> float:
        ...

    @overload
    def distance(self, point: Point) -> float:
        ...

    @staticmethod
    def from_geojson(array: List[float]) -> Box:
        ...

    @staticmethod
    def read_wkt(wkt: str) -> Box:
        ...

    def to_geojson(self) -> Dict[str, Any]:
        ...

    @staticmethod
    def whole_earth() -> Box:
        ...

    def wkt(self) -> str:
        ...

    def __copy__(self) -> Box:
        ...

    def __eq__(self, other: Box) -> bool:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __ne__(self, other: Box) -> bool:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Coordinates:

    def __init__(self, system: Optional[System] = None) -> None:
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

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def system(self) -> System:
        ...


class Crossover:

    def __init__(self, half_orbit_1: LineString,
                 half_orbit_2: LineString) -> None:
        ...

    def exists(self) -> bool:
        ...

    def nearest(self,
                point: Point,
                predicate: Optional[float] = ...,
                strategy: str = ...,
                wgs: Optional[System] = ...) -> Optional[Tuple[int, int]]:
        ...

    def search(self) -> Optional[Point]:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def half_orbit_1(self) -> LineString:
        ...

    @property
    def half_orbit_2(self) -> LineString:
        ...


class LineString:
    __hash__: ClassVar[None] = ...

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, points: list) -> None:
        ...

    @overload
    def __init__(self, lon: numpy.ndarray[numpy.float64],
                 lat: numpy.ndarray[numpy.float64]) -> None:
        ...

    def append(self, point: Point) -> None:
        ...

    def curvilinear_distance(
            self,
            strategy: str = 'thomas',
            wgs: Optional[System] = None) -> numpy.ndarray[numpy.float64]:
        ...

    @staticmethod
    def from_geojson(array: List[List[float]]) -> LineString:
        ...

    def intersection(self, rhs: LineString) -> Optional[Point]:
        ...

    def intersects(self, rhs: LineString) -> bool:
        ...

    @staticmethod
    def read_wkt(wkt: str) -> LineString:
        ...

    def to_geojson(self) -> Dict[str, Any]:
        ...

    def wkt(self) -> str:
        ...

    def __copy__(self) -> LineString:
        ...

    def __eq__(self, other: LineString) -> bool:
        ...

    def __getitem__(self, index: int) -> Point:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iter__(self) -> Iterator:
        ...

    def __len__(self) -> int:
        ...

    def __ne__(self, other: LineString) -> bool:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class MultiPolygon:
    __hash__: ClassVar[None] = ...

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, polygons: list) -> None:
        ...

    def append(self, polygon: Polygon) -> None:
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
                   num_threads: int = ...) -> numpy.ndarray[numpy.int8]:
        ...

    @overload
    def distance(self, other: MultiPolygon) -> float:
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
    def from_geojson(array: List[List[List[List[float]]]]) -> MultiPolygon:
        ...

    @overload
    def intersection(self, other: Polygon) -> MultiPolygon:
        ...

    @overload
    def intersection(self, other: MultiPolygon) -> MultiPolygon:
        ...

    @overload
    def intersects(self, other: Polygon) -> bool:
        ...

    @overload
    def intersects(self, other: MultiPolygon) -> bool:
        ...

    def num_interior_rings(self) -> int:
        ...

    @staticmethod
    def read_wkt(wkt: str) -> MultiPolygon:
        ...

    def to_geojson(self) -> Dict[str, Any]:
        ...

    @overload
    def touches(self, other: Polygon) -> bool:
        ...

    @overload
    def touches(self, other: MultiPolygon) -> bool:
        ...

    @overload
    def union(self, other: Polygon) -> MultiPolygon:
        ...

    @overload
    def union(self, other: MultiPolygon) -> MultiPolygon:
        ...

    def wkt(self) -> str:
        ...

    def __add__(self, other: MultiPolygon) -> MultiPolygon:
        ...

    def __contains__(self, polygon: Polygon) -> bool:
        ...

    def __copy__(self) -> MultiPolygon:
        ...

    def __eq__(self, other: MultiPolygon) -> bool:
        ...

    def __getitem__(self, index: int) -> Polygon:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iadd__(self, other: MultiPolygon) -> MultiPolygon:
        ...

    def __iter__(self) -> Iterator:
        ...

    def __len__(self) -> int:
        ...

    def __ne__(self, other: MultiPolygon) -> bool:
        ...

    def __setstate__(self, state: tuple) -> None:
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

    def to_geojson(self) -> Dict[str, Any]:
        ...

    def wkt(self) -> str:
        ...

    def __copy__(self) -> Point:
        ...

    def __eq__(self, other: Point) -> bool:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __ne__(self, other: Point) -> bool:
        ...

    def __setstate__(self, state: tuple) -> None:
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
                   num_threads: int = ...) -> numpy.ndarray[numpy.int8]:
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
    def from_geojson(array: List[List[List[float]]]) -> Polygon:
        ...

    def intersection(self, *args, **kwargs) -> Any:
        ...

    def intersects(self, other: Polygon) -> bool:
        ...

    def num_interior_rings(self) -> int:
        ...

    @staticmethod
    def read_wkt(wkt: str) -> Polygon:
        ...

    def to_geojson(self) -> Dict[str, Any]:
        ...

    def touches(self, other: Polygon) -> bool:
        ...

    def union(self, *args, **kwargs) -> Any:
        ...

    def wkt(self) -> str:
        ...

    def __copy__(self) -> Polygon:
        ...

    def __eq__(self, other: Polygon) -> bool:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __ne__(self, other: Polygon) -> bool:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def inners(self) -> List[LineString]:
        ...

    @property
    def outer(self) -> LineString:
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

    def __setstate__(self, state: tuple) -> None:
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


def calculate_swath(
    lon_nadir: numpy.ndarray[numpy.float64],
    lat_nadir: numpy.ndarray[numpy.float64], delta_ac: float, half_gap: float,
    half_swath: int, radius: float
) -> Tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
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


def normalize_longitudes(
        lon: numpy.ndarray[numpy.float64],
        min_lon: float = -180.0) -> numpy.ndarray[numpy.float64]:
    ...
