from typing import (
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Self,
    Tuple,
    overload,
)

from sympy import Array

from .. import core
from ..typing import NDArray1DBool, NDArray1DFloat64, NDArray2DFloat64

class Box:
    __hash__: ClassVar[None] = ...  # type: ignore[assignment]
    max_corner: Point
    min_corner: Point

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, min_corner: Point, max_corner: Point) -> None:
        ...

    def area(self, wgs: Optional[Spheroid] = None) -> float:
        ...

    def as_polygon(self) -> Polygon:
        ...

    def centroid(self) -> Point:
        ...

    @overload
    def covered_by(self, point: Point) -> bool:
        ...

    @overload
    def covered_by(self,
                   lon: NDArray1DFloat64,
                   lat: NDArray1DFloat64,
                   num_threads: int = ...) -> NDArray1DBool:
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

    def __eq__(self, other: Box) -> bool:  # type: ignore[override]
        ...

    def __getstate__(self) -> tuple:
        ...

    def __ne__(self, other: Box) -> bool:  # type: ignore[override]
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Coordinates:

    def __init__(self, spheroid: Optional[Spheroid] = None) -> None:
        ...

    def ecef_to_lla(self,
                    x: NDArray1DFloat64,
                    y: NDArray1DFloat64,
                    z: NDArray1DFloat64,
                    num_threads: int = ...,) -> tuple[NDArray1DFloat64, NDArray1DFloat64, NDArray1DFloat64]:
        ...

    def lla_to_ecef(self,
                    lon: NDArray1DFloat64,
                    lat: NDArray1DFloat64,
                    alt: NDArray1DFloat64,
                    num_threads: int = ...,) -> tuple[NDArray1DFloat64, NDArray1DFloat64, NDArray1DFloat64]:
        ...

    def transform(self,
                  target: Coordinates,
                  lon: NDArray1DFloat64,
                  lat: NDArray1DFloat64,
                  alt: NDArray1DFloat64,
                  num_threads: int = ...,) -> tuple[NDArray1DFloat64, NDArray1DFloat64, NDArray1DFloat64]:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def spheroid(self) -> Spheroid:
        ...


class Crossover:

    def __init__(self, half_orbit_1: LineString,
                 half_orbit_2: LineString) -> None:
        ...

    def exists(self, wgs: Optional[Spheroid] = None) -> bool:
        ...

    def nearest(self,
                point: Point,
                predicate: Optional[float] = ...,
                strategy: str = ...,
                wgs: Optional[Spheroid] = None) -> Optional[Tuple[int, int]]:
        ...

    def search(self, wgs: Optional[Spheroid] = None) -> Optional[Point]:
        ...

    def search_all(self, wgs: Optional[Spheroid] = None) -> List[Point]:
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
    __hash__: ClassVar[None] = ...  # type: ignore[assignment]

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, points: list) -> None:
        ...

    @overload
    def __init__(self, lon: NDArray1DFloat64, lat: NDArray1DFloat64) -> None:
        ...

    def append(self, point: Point) -> None:
        ...

    @overload
    def closest_point(self,
                      point: Point,
                      wgs: Optional[Spheroid] = None) -> Point:
        ...

    @overload
    def closest_point(
        self,
        lon: NDArray1DFloat64,
        lat: NDArray1DFloat64,
        wgs: Optional[Spheroid] = None,
        num_threads: int = 0,
    ) -> Tuple[NDArray1DFloat64, NDArray1DFloat64]:
        ...

    def curvilinear_distance(self,
                             strategy: str = 'thomas',
                             wgs: Optional[Spheroid] = None) -> NDArray1DFloat64:
        ...

    @staticmethod
    def from_geojson(array: List[List[float]]) -> LineString:
        ...

    @overload
    def intersection(self,
                     rhs: LineString,
                     wgs: Optional[Spheroid] = None) -> LineString:
        ...

    @overload
    def intersection(self,
                     rhs: Polygon,
                     wgs: Optional[Spheroid] = None) -> List[LineString]:
        ...

    def intersects(self,
                   rhs: LineString,
                   wgs: Optional[Spheroid] = None) -> bool:
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

    def __eq__(self, other: LineString) -> bool:  # type: ignore[override]
        ...

    def __getitem__(self, index: int) -> Point:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iter__(self) -> Iterator:
        ...

    def __len__(self) -> int:
        ...

    def __ne__(self, other: LineString) -> bool:  # type: ignore[override]
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class MultiPolygon:
    __hash__: ClassVar[None] = ...  # type: ignore[assignment]

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, polygons: list) -> None:
        ...

    def append(self, polygon: Polygon) -> None:
        ...

    def area(self, wgs: Optional[Spheroid] = None) -> float:
        ...

    @overload
    def covered_by(self, point: Point) -> bool:
        ...

    @overload
    def covered_by(self,
                   lon: NDArray1DFloat64,
                   lat: NDArray1DFloat64,
                   num_threads: int = ...) -> NDArray1DBool:
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

    def __eq__(self, other: MultiPolygon) -> bool:  # type: ignore[override]
        ...

    def __getitem__(self, index: int) -> Polygon:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iadd__(self, other: MultiPolygon) -> Self:
        ...

    def __iter__(self) -> Iterator:
        ...

    def __len__(self) -> int:
        ...

    def __ne__(self, other: MultiPolygon) -> bool:  # type: ignore[override]
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Point:
    __hash__: ClassVar[None] = ...  # type: ignore[assignment]
    lat: float
    lon: float

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, lon: float, lat: float) -> None:
        ...

    def azimuth(self, other: Point, wgs: Optional[Spheroid] = None) -> float:
        ...

    def distance(self,
                 other: Point,
                 strategy: str = ...,
                 wgs: Optional[Spheroid] = None) -> float:
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

    def __eq__(self, other: Point) -> bool:  # type: ignore[override]
        ...

    def __getstate__(self) -> tuple:
        ...

    def __ne__(self, other: Point) -> bool:  # type: ignore[override]
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Polygon:
    __hash__: ClassVar[None] = ...  # type: ignore[assignment]

    def __init__(self, outer: list, inners: Optional[list] = ...) -> None:
        ...

    def area(self, wgs: Optional[Spheroid] = None) -> float:
        ...

    @overload
    def covered_by(self, point: Point) -> bool:
        ...

    @overload
    def covered_by(self,
                   lon: NDArray1DFloat64,
                   lat: NDArray1DFloat64,
                   num_threads: int = ...) -> NDArray1DBool:
        ...

    def difference(self, other: Polygon) -> MultiPolygon:
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

    @overload
    def intersection(self, other: Polygon) -> MultiPolygon:
        ...

    @overload
    def intersection(self, line_string: LineString) -> LineString:
        ...

    def intersects(self, other: Polygon) -> bool:
        ...

    def num_interior_rings(self) -> int:
        ...

    def perimeter(self, wgs: Optional[Spheroid] = None) -> float:
        ...

    @staticmethod
    def read_wkt(wkt: str) -> Polygon:
        ...

    def simplify(self, tolerance: float) -> Polygon:
        ...

    def to_geojson(self) -> Dict[str, Any]:
        ...

    def touches(self, other: Polygon) -> bool:
        ...

    def union(self, other: Polygon) -> MultiPolygon:
        ...

    def wkt(self) -> str:
        ...

    def __copy__(self) -> Polygon:
        ...

    def __eq__(self, other: Polygon) -> bool:  # type: ignore[override]
        ...

    def __getstate__(self) -> tuple:
        ...

    def __ne__(self, other: Polygon) -> bool:  # type: ignore[override]
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def inners(self) -> List[LineString]:
        ...

    @property
    def outer(self) -> LineString:
        ...


class RTree:

    def __init__(self, spheroid: Optional[Spheroid] = None) -> None:
        ...

    def clear(self) -> None:
        ...

    def insert(self, lon: NDArray1DFloat64, lat: NDArray1DFloat64,
               values: NDArray1DFloat64) -> None:
        ...

    def inverse_distance_weighting(self,
                                   lon: NDArray1DFloat64,
                                   lat: NDArray1DFloat64,
                                   radius: Optional[float] = ...,
                                   k: int = ...,
                                   p: int = ...,
                                   within: bool = ...,
                                   num_threads: int = ...) -> tuple:
        ...

    def packing(self, lon: NDArray1DFloat64, lat: NDArray1DFloat64,
                values: NDArray1DFloat64) -> None:
        ...

    def query(self,
              lon: NDArray1DBool,
              lat: NDArray1DBool,
              k: int = ...,
              within: bool = ...,
              num_threads: int = ...) -> tuple:
        ...

    def radial_basis_function(self,
                              lon: NDArray1DFloat64,
                              lat: NDArray1DFloat64,
                              radius: Optional[float] = ...,
                              k: int = ...,
                              rbf: core.RadialBasisFunction = ...,
                              epsilon: Optional[float] = ...,
                              smooth: float = ...,
                              within: bool = ...,
                              num_threads: int = ...) -> tuple:
        ...

    def window_function(self,
                        lon: NDArray1DFloat64,
                        lat: NDArray1DFloat64,
                        radius: float = ...,
                        k: int = ...,
                        wf: core.WindowFunction = ...,
                        arg: Optional[float] = ...,
                        within: bool = ...,
                        num_threads: int = ...) -> tuple:
        ...

    def __bool__(self) -> bool:
        ...

    def __copy__(self) -> RTree:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __len__(self) -> int:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Spheroid(_Spheroid):
    __hash__: ClassVar[None] = ...  # type: ignore[assignment]

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

    def geocentric_radius(self, lat: float) -> float:
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

    def __eq__(self, other: _Spheroid) -> bool:  # type: ignore[override]
        ...

    def __getstate__(self) -> tuple:
        ...

    def __ne__(self, other: _Spheroid) -> bool:  # type: ignore[override]
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def flattening(self) -> float:
        ...

    @property
    def semi_major_axis(self) -> float:
        ...


class _Spheroid:

    def __init__(self, *args, **kwargs) -> None:
        ...


def calculate_crossover(
        lon1: NDArray1DFloat64,
        lat1: NDArray1DFloat64,
        lon2: NDArray1DFloat64,
        lat2: NDArray1DFloat64,
        predicate: Optional[float] = None,
        strategy: str = "thomas",
        wgs: Optional[Spheroid] = None,
        cartesian_plane: bool = True
) -> Optional[Tuple[Point, Tuple[int, int]]]:
    ...


def calculate_crossover_list(
        lon1: NDArray1DFloat64,
        lat1: NDArray1DFloat64,
        lon2: NDArray1DFloat64,
        lat2: NDArray1DFloat64,
        predicate: Optional[float] = None,
        strategy: str = "thomas",
        wgs: Optional[Spheroid] = None,
        cartesian_plane: bool = True) -> List[Tuple[Point, Tuple[int, int]]]:
    ...


def calculate_swath(
    lon_nadir: NDArray1DFloat64,
    lat_nadir: NDArray1DFloat64,
    delta_ac: float,
    half_gap: float,
    half_swath: int,
    spheroid: Optional[Spheroid] = None,
) -> Tuple[NDArray2DFloat64, NDArray2DFloat64]:
    ...


def coordinate_distances(lon1: NDArray1DFloat64,
                         lat1: NDArray1DFloat64,
                         lon2: NDArray1DFloat64,
                         lat2: NDArray1DFloat64,
                         strategy: str = ...,
                         wgs: Optional[Spheroid] = None,
                         num_threads: int = ...) -> NDArray1DFloat64:
    ...


def normalize_longitudes(lon: NDArray1DFloat64,
                         min_lon: float = ...) -> NDArray1DFloat64 | None:
    ...
