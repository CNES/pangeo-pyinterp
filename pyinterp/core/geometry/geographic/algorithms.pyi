import enum
from typing import overload

from ....type_hints import NDArray1DBool, NDArray1DFloat64
from .. import cartesian
from . import (
    Box,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    Ring,
    Segment,
    Spheroid,
    _Concept,
)

ANDOYER: Strategy
KARNEY: Strategy
THOMAS: Strategy
VINCENTY: Strategy

def area(
    geometry: _Concept,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> float: ...
def azimuth(
    point1: Point,
    point2: Point,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> float: ...
def centroid(geometry: Box | Segment) -> Point: ...
def clear(geometry: _Concept) -> None: ...
@overload
def closest_points(
    geometry1: LineString,
    geometry2: LineString,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: LineString,
    geometry2: MultiLineString,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: LineString,
    geometry2: MultiPoint,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: LineString,
    geometry2: MultiPolygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: LineString,
    geometry2: Polygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: MultiPoint,
    geometry2: LineString,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: MultiPoint,
    geometry2: MultiLineString,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: MultiPoint,
    geometry2: MultiPoint,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: MultiPoint,
    geometry2: MultiPolygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: MultiPoint,
    geometry2: Polygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: MultiLineString,
    geometry2: LineString,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: MultiLineString,
    geometry2: MultiLineString,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: MultiLineString,
    geometry2: MultiPoint,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: MultiLineString,
    geometry2: MultiPolygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: MultiLineString,
    geometry2: Polygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: MultiPolygon,
    geometry2: LineString,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: MultiPolygon,
    geometry2: MultiLineString,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: MultiPolygon,
    geometry2: MultiPoint,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: MultiPolygon,
    geometry2: MultiPolygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: MultiPolygon,
    geometry2: Polygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: Polygon,
    geometry2: LineString,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: Polygon,
    geometry2: MultiLineString,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: Polygon,
    geometry2: MultiPoint,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: Polygon,
    geometry2: MultiPolygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def closest_points(
    geometry1: Polygon,
    geometry2: Polygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Segment: ...
@overload
def convert_to_cartesian(geometry: Point) -> cartesian.Point: ...
@overload
def convert_to_cartesian(geometry: Box) -> cartesian.Box: ...
@overload
def convert_to_cartesian(geometry: Segment) -> cartesian.Segment: ...
@overload
def convert_to_cartesian(geometry: LineString) -> cartesian.LineString: ...
@overload
def convert_to_cartesian(geometry: Ring) -> cartesian.Ring: ...
@overload
def convert_to_cartesian(geometry: Polygon) -> cartesian.Polygon: ...
@overload
def convert_to_cartesian(geometry: MultiPoint) -> cartesian.MultiPoint: ...
@overload
def convert_to_cartesian(
    geometry: MultiLineString,
) -> cartesian.MultiLineString: ...
@overload
def convert_to_cartesian(geometry: MultiPolygon) -> cartesian.MultiPolygon: ...
def convex_hull(
    geometry: _Concept,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Polygon: ...
def correct(geometry: _Concept) -> None: ...
@overload
def covered_by(geometry1: Point, geometry2: Point) -> bool: ...
@overload
def covered_by(geometry1: Point, geometry2: Segment) -> bool: ...
@overload
def covered_by(geometry1: Point, geometry2: Box) -> bool: ...
@overload
def covered_by(geometry1: Point, geometry2: LineString) -> bool: ...
@overload
def covered_by(geometry1: Point, geometry2: Ring) -> bool: ...
@overload
def covered_by(geometry1: Point, geometry2: Polygon) -> bool: ...
@overload
def covered_by(geometry1: Point, geometry2: MultiPoint) -> bool: ...
@overload
def covered_by(geometry1: Point, geometry2: MultiLineString) -> bool: ...
@overload
def covered_by(geometry1: Point, geometry2: MultiPolygon) -> bool: ...
@overload
def covered_by(geometry1: Box, geometry2: Box) -> bool: ...
@overload
def covered_by(geometry1: LineString, geometry2: Box) -> bool: ...
@overload
def covered_by(geometry1: LineString, geometry2: LineString) -> bool: ...
@overload
def covered_by(geometry1: LineString, geometry2: Ring) -> bool: ...
@overload
def covered_by(geometry1: LineString, geometry2: Polygon) -> bool: ...
@overload
def covered_by(geometry1: LineString, geometry2: MultiLineString) -> bool: ...
@overload
def covered_by(geometry1: LineString, geometry2: MultiPolygon) -> bool: ...
@overload
def covered_by(geometry1: Ring, geometry2: Box) -> bool: ...
@overload
def covered_by(geometry1: Ring, geometry2: Ring) -> bool: ...
@overload
def covered_by(geometry1: Ring, geometry2: Polygon) -> bool: ...
@overload
def covered_by(geometry1: Ring, geometry2: MultiPolygon) -> bool: ...
@overload
def covered_by(geometry1: Polygon, geometry2: Box) -> bool: ...
@overload
def covered_by(geometry1: Polygon, geometry2: Ring) -> bool: ...
@overload
def covered_by(geometry1: Polygon, geometry2: Polygon) -> bool: ...
@overload
def covered_by(geometry1: Polygon, geometry2: MultiPolygon) -> bool: ...
@overload
def covered_by(geometry1: MultiPoint, geometry2: Segment) -> bool: ...
@overload
def covered_by(geometry1: MultiPoint, geometry2: Box) -> bool: ...
@overload
def covered_by(geometry1: MultiPoint, geometry2: LineString) -> bool: ...
@overload
def covered_by(geometry1: MultiPoint, geometry2: Ring) -> bool: ...
@overload
def covered_by(geometry1: MultiPoint, geometry2: Polygon) -> bool: ...
@overload
def covered_by(geometry1: MultiPoint, geometry2: MultiPoint) -> bool: ...
@overload
def covered_by(geometry1: MultiPoint, geometry2: MultiLineString) -> bool: ...
@overload
def covered_by(geometry1: MultiPoint, geometry2: MultiPolygon) -> bool: ...
@overload
def covered_by(geometry1: MultiLineString, geometry2: Box) -> bool: ...
@overload
def covered_by(geometry1: MultiLineString, geometry2: LineString) -> bool: ...
@overload
def covered_by(geometry1: MultiLineString, geometry2: Ring) -> bool: ...
@overload
def covered_by(geometry1: MultiLineString, geometry2: Polygon) -> bool: ...
@overload
def covered_by(
    geometry1: MultiLineString, geometry2: MultiLineString
) -> bool: ...
@overload
def covered_by(
    geometry1: MultiLineString, geometry2: MultiPolygon
) -> bool: ...
@overload
def covered_by(geometry1: MultiPolygon, geometry2: Box) -> bool: ...
@overload
def covered_by(geometry1: MultiPolygon, geometry2: Ring) -> bool: ...
@overload
def covered_by(geometry1: MultiPolygon, geometry2: Polygon) -> bool: ...
@overload
def covered_by(geometry1: MultiPolygon, geometry2: MultiPolygon) -> bool: ...
@overload
def crosses(geometry1: Point, geometry2: LineString) -> bool: ...
@overload
def crosses(geometry1: Point, geometry2: Ring) -> bool: ...
@overload
def crosses(geometry1: Point, geometry2: Polygon) -> bool: ...
@overload
def crosses(geometry1: Point, geometry2: MultiLineString) -> bool: ...
@overload
def crosses(geometry1: Point, geometry2: MultiPolygon) -> bool: ...
@overload
def crosses(geometry1: LineString, geometry2: Point) -> bool: ...
@overload
def crosses(geometry1: LineString, geometry2: LineString) -> bool: ...
@overload
def crosses(geometry1: LineString, geometry2: Ring) -> bool: ...
@overload
def crosses(geometry1: LineString, geometry2: Polygon) -> bool: ...
@overload
def crosses(geometry1: LineString, geometry2: MultiLineString) -> bool: ...
@overload
def crosses(geometry1: LineString, geometry2: MultiPolygon) -> bool: ...
@overload
def crosses(geometry1: Ring, geometry2: Point) -> bool: ...
@overload
def crosses(geometry1: Polygon, geometry2: Point) -> bool: ...
@overload
def crosses(geometry1: MultiLineString, geometry2: Point) -> bool: ...
@overload
def crosses(geometry1: MultiLineString, geometry2: LineString) -> bool: ...
@overload
def crosses(geometry1: MultiLineString, geometry2: Ring) -> bool: ...
@overload
def crosses(geometry1: MultiLineString, geometry2: Polygon) -> bool: ...
@overload
def crosses(
    geometry1: MultiLineString, geometry2: MultiLineString
) -> bool: ...
@overload
def crosses(geometry1: MultiLineString, geometry2: MultiPolygon) -> bool: ...
@overload
def crosses(geometry1: MultiPolygon, geometry2: MultiPolygon) -> bool: ...
def curvilinear_distance(
    geometry: LineString | Ring,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> NDArray1DFloat64: ...
@overload
def densify(
    geometry: LineString,
    max_distance: float,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> LineString: ...
@overload
def densify(
    geometry: Ring,
    max_distance: float,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Ring: ...
@overload
def densify(
    geometry: Polygon,
    max_distance: float,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Polygon: ...
@overload
def densify(
    geometry: MultiLineString,
    max_distance: float,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> MultiLineString: ...
@overload
def densify(
    geometry: MultiPolygon,
    max_distance: float,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> MultiPolygon: ...
@overload
def difference(
    geometry1: Ring,
    geometry2: Ring,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def difference(
    geometry1: Ring,
    geometry2: Polygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def difference(
    geometry1: Polygon,
    geometry2: Ring,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def difference(
    geometry1: Polygon,
    geometry2: Polygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def difference(
    geometry1: MultiPolygon,
    geometry2: Ring,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def difference(
    geometry1: MultiPolygon,
    geometry2: Polygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def difference(
    geometry1: MultiPolygon,
    geometry2: MultiPolygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def difference(
    geometry1: Ring,
    geometry2: MultiPolygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def difference(
    geometry1: Polygon,
    geometry2: MultiPolygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def disjoint(geometry1: Point, geometry2: Point) -> bool: ...
@overload
def disjoint(geometry1: Point, geometry2: Box) -> bool: ...
@overload
def disjoint(geometry1: Point, geometry2: Polygon) -> bool: ...
@overload
def disjoint(geometry1: Point, geometry2: MultiPolygon) -> bool: ...
@overload
def disjoint(geometry1: Box, geometry2: Point) -> bool: ...
@overload
def disjoint(geometry1: Box, geometry2: Box) -> bool: ...
@overload
def disjoint(geometry1: Box, geometry2: Polygon) -> bool: ...
@overload
def disjoint(geometry1: LineString, geometry2: LineString) -> bool: ...
@overload
def disjoint(geometry1: LineString, geometry2: Polygon) -> bool: ...
@overload
def disjoint(geometry1: LineString, geometry2: Box) -> bool: ...
@overload
def disjoint(geometry1: Polygon, geometry2: Point) -> bool: ...
@overload
def disjoint(geometry1: Polygon, geometry2: Box) -> bool: ...
@overload
def disjoint(geometry1: Polygon, geometry2: Polygon) -> bool: ...
@overload
def disjoint(geometry1: Polygon, geometry2: MultiPolygon) -> bool: ...
@overload
def disjoint(geometry1: MultiPolygon, geometry2: Point) -> bool: ...
@overload
def disjoint(geometry1: MultiPolygon, geometry2: Polygon) -> bool: ...
@overload
def disjoint(geometry1: MultiPolygon, geometry2: MultiPolygon) -> bool: ...
@overload
def distance(
    geometry1: Point,
    geometry2: Point,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> float: ...
@overload
def distance(
    geometry1: Point,
    geometry2: Box,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> float: ...
@overload
def distance(
    geometry1: Point,
    geometry2: Polygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> float: ...
@overload
def distance(
    geometry1: Point,
    geometry2: MultiPolygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> float: ...
@overload
def distance(
    geometry1: Box,
    geometry2: Point,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> float: ...
@overload
def distance(
    geometry1: Box,
    geometry2: Box,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> float: ...
@overload
def distance(
    geometry1: Box,
    geometry2: Polygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> float: ...
@overload
def distance(
    geometry1: LineString,
    geometry2: LineString,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> float: ...
@overload
def distance(
    geometry1: LineString,
    geometry2: Polygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> float: ...
@overload
def distance(
    geometry1: LineString,
    geometry2: Box,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> float: ...
@overload
def distance(
    geometry1: Polygon,
    geometry2: Point,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> float: ...
@overload
def distance(
    geometry1: Polygon,
    geometry2: Box,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> float: ...
@overload
def distance(
    geometry1: Polygon,
    geometry2: Polygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> float: ...
@overload
def distance(
    geometry1: Polygon,
    geometry2: MultiPolygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> float: ...
@overload
def distance(
    geometry1: MultiPolygon,
    geometry2: Point,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> float: ...
@overload
def distance(
    geometry1: MultiPolygon,
    geometry2: Polygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> float: ...
@overload
def distance(
    geometry1: MultiPolygon,
    geometry2: MultiPolygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> float: ...
def envelope(geometry: _Concept) -> Box: ...
@overload
def equals(geometry1: Point, geometry2: Point) -> bool: ...
@overload
def equals(geometry1: Segment, geometry2: Segment) -> bool: ...
@overload
def equals(geometry1: Box, geometry2: Box) -> bool: ...
@overload
def equals(geometry1: LineString, geometry2: LineString) -> bool: ...
@overload
def equals(geometry1: Ring, geometry2: Ring) -> bool: ...
@overload
def equals(geometry1: Polygon, geometry2: Polygon) -> bool: ...
@overload
def equals(geometry1: MultiPoint, geometry2: MultiPoint) -> bool: ...
@overload
def equals(geometry1: MultiLineString, geometry2: MultiLineString) -> bool: ...
@overload
def equals(geometry1: MultiPolygon, geometry2: MultiPolygon) -> bool: ...
def for_each_point_covered_by(
    source: MultiPoint | LineString | Ring,
    container: Box | Ring | Polygon | MultiPolygon,
) -> NDArray1DBool: ...
def for_each_point_distance(
    source: MultiPoint | LineString | Ring,
    container: Box | Ring | Polygon | MultiPolygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> NDArray1DFloat64: ...
def for_each_point_within(
    source: MultiPoint | LineString | Ring,
    container: Box | Ring | Polygon | MultiPolygon,
) -> NDArray1DBool: ...
def from_geojson(
    geojson: str,
) -> (
    Point | LineString | MultiLineString | MultiPoint | MultiPolygon | Polygon
): ...
def from_wkt(
    wkt: str,
) -> (
    Point
    | LineString
    | Ring
    | Polygon
    | MultiPoint
    | MultiLineString
    | MultiPolygon
): ...
@overload
def intersection(
    geometry1: Ring,
    geometry2: Ring,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def intersection(
    geometry1: Ring,
    geometry2: Polygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def intersection(
    geometry1: Polygon,
    geometry2: Ring,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def intersection(
    geometry1: Polygon,
    geometry2: Polygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def intersection(
    geometry1: MultiPolygon,
    geometry2: Ring,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def intersection(
    geometry1: MultiPolygon,
    geometry2: Polygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def intersection(
    geometry1: MultiPolygon,
    geometry2: MultiPolygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def intersection(
    geometry1: Ring,
    geometry2: MultiPolygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def intersection(
    geometry1: Polygon,
    geometry2: MultiPolygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def intersection(
    geometry1: LineString,
    geometry2: Ring,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[LineString]: ...
@overload
def intersection(
    geometry1: LineString,
    geometry2: Polygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[LineString]: ...
@overload
def intersection(
    geometry1: LineString,
    geometry2: MultiPolygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[LineString]: ...
@overload
def intersection(
    geometry1: LineString,
    geometry2: LineString,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Point]: ...
@overload
def intersects(geometry1: Point, geometry2: Point) -> bool: ...
@overload
def intersects(geometry1: Point, geometry2: Box) -> bool: ...
@overload
def intersects(geometry1: Point, geometry2: Polygon) -> bool: ...
@overload
def intersects(geometry1: Point, geometry2: MultiPolygon) -> bool: ...
@overload
def intersects(geometry1: Box, geometry2: Point) -> bool: ...
@overload
def intersects(geometry1: Box, geometry2: Box) -> bool: ...
@overload
def intersects(geometry1: Box, geometry2: Polygon) -> bool: ...
@overload
def intersects(geometry1: LineString, geometry2: LineString) -> bool: ...
@overload
def intersects(geometry1: LineString, geometry2: Polygon) -> bool: ...
@overload
def intersects(geometry1: LineString, geometry2: Box) -> bool: ...
@overload
def intersects(geometry1: Polygon, geometry2: Point) -> bool: ...
@overload
def intersects(geometry1: Polygon, geometry2: Box) -> bool: ...
@overload
def intersects(geometry1: Polygon, geometry2: Polygon) -> bool: ...
@overload
def intersects(geometry1: Polygon, geometry2: MultiPolygon) -> bool: ...
@overload
def intersects(geometry1: MultiPolygon, geometry2: Point) -> bool: ...
@overload
def intersects(geometry1: MultiPolygon, geometry2: Polygon) -> bool: ...
@overload
def intersects(geometry1: MultiPolygon, geometry2: MultiPolygon) -> bool: ...
def line_interpolate(
    geometry: LineString | Segment,
    distance: float,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Point: ...
def is_empty(geometry: _Concept) -> bool: ...
def is_simple(geometry: _Concept) -> bool: ...
@overload
def is_valid(geometry: _Concept, return_reason: bool) -> tuple[bool, str]: ...
@overload
def is_valid(geometry: _Concept) -> bool: ...
def length(
    geometry: _Concept,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> float: ...
def num_geometries(geometry: _Concept) -> int: ...
def num_interior_rings(geometry: _Concept) -> int: ...
def num_points(geometry: _Concept) -> int: ...
def num_segments(geometry: _Concept) -> int: ...
@overload
def overlaps(geometry1: LineString, geometry2: LineString) -> bool: ...
@overload
def overlaps(geometry1: LineString, geometry2: MultiLineString) -> bool: ...
@overload
def overlaps(geometry1: Polygon, geometry2: Polygon) -> bool: ...
@overload
def overlaps(geometry1: Polygon, geometry2: MultiPolygon) -> bool: ...
@overload
def overlaps(geometry1: MultiLineString, geometry2: LineString) -> bool: ...
@overload
def overlaps(
    geometry1: MultiLineString, geometry2: MultiLineString
) -> bool: ...
@overload
def overlaps(geometry1: MultiPolygon, geometry2: Polygon) -> bool: ...
@overload
def overlaps(geometry1: MultiPolygon, geometry2: MultiPolygon) -> bool: ...
def perimeter(
    geometry: Ring | Polygon | MultiPolygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> float: ...
@overload
def relate(geometry1: Point, geometry2: Point, mask: str) -> bool: ...
@overload
def relate(
    geometry1: LineString, geometry2: LineString, mask: str
) -> bool: ...
@overload
def relate(geometry1: Ring, geometry2: Ring, mask: str) -> bool: ...
@overload
def relate(geometry1: Ring, geometry2: Polygon, mask: str) -> bool: ...
@overload
def relate(geometry1: Polygon, geometry2: Ring, mask: str) -> bool: ...
@overload
def relate(geometry1: Polygon, geometry2: Polygon, mask: str) -> bool: ...
@overload
def relation(geometry1: Point, geometry2: Point) -> str: ...
@overload
def relation(geometry1: LineString, geometry2: LineString) -> str: ...
@overload
def relation(geometry1: Ring, geometry2: Ring) -> str: ...
@overload
def relation(geometry1: Ring, geometry2: Polygon) -> str: ...
@overload
def relation(geometry1: Polygon, geometry2: Ring) -> str: ...
@overload
def relation(geometry1: Polygon, geometry2: Polygon) -> str: ...
def reverse(geometry: _Concept) -> None: ...
@overload
def simplify(
    geometry: LineString,
    max_distance: float,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> LineString: ...
@overload
def simplify(
    geometry: Ring,
    max_distance: float,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Ring: ...
@overload
def simplify(
    geometry: Polygon,
    max_distance: float,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> Polygon: ...
@overload
def simplify(
    geometry: MultiLineString,
    max_distance: float,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> MultiLineString: ...
@overload
def simplify(
    geometry: MultiPolygon,
    max_distance: float,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> MultiPolygon: ...
def to_geojson(
    geometry: Point
    | LineString
    | Polygon
    | MultiPoint
    | MultiLineString
    | MultiPolygon,
) -> str: ...
def to_wkt(
    geometry: Point
    | LineString
    | Ring
    | Polygon
    | MultiPoint
    | MultiLineString
    | MultiPolygon,
) -> str: ...
@overload
def touches(geometry1: Point, geometry2: Point) -> bool: ...
@overload
def touches(geometry1: Point, geometry2: LineString) -> bool: ...
@overload
def touches(geometry1: Point, geometry2: Ring) -> bool: ...
@overload
def touches(geometry1: Point, geometry2: Polygon) -> bool: ...
@overload
def touches(geometry1: Point, geometry2: MultiLineString) -> bool: ...
@overload
def touches(geometry1: Point, geometry2: MultiPolygon) -> bool: ...
@overload
def touches(geometry1: LineString, geometry2: LineString) -> bool: ...
@overload
def touches(geometry1: LineString, geometry2: Ring) -> bool: ...
@overload
def touches(geometry1: LineString, geometry2: Polygon) -> bool: ...
@overload
def touches(geometry1: LineString, geometry2: MultiLineString) -> bool: ...
@overload
def touches(geometry1: LineString, geometry2: MultiPolygon) -> bool: ...
@overload
def touches(geometry1: Ring, geometry2: LineString) -> bool: ...
@overload
def touches(geometry1: Ring, geometry2: Ring) -> bool: ...
@overload
def touches(geometry1: Ring, geometry2: Polygon) -> bool: ...
@overload
def touches(geometry1: Ring, geometry2: MultiLineString) -> bool: ...
@overload
def touches(geometry1: Ring, geometry2: MultiPolygon) -> bool: ...
@overload
def touches(geometry1: Polygon, geometry2: Point) -> bool: ...
@overload
def touches(geometry1: Polygon, geometry2: LineString) -> bool: ...
@overload
def touches(geometry1: Polygon, geometry2: Ring) -> bool: ...
@overload
def touches(geometry1: Polygon, geometry2: Polygon) -> bool: ...
@overload
def touches(geometry1: Polygon, geometry2: MultiLineString) -> bool: ...
@overload
def touches(geometry1: Polygon, geometry2: MultiPolygon) -> bool: ...
@overload
def touches(geometry1: MultiLineString, geometry2: Point) -> bool: ...
@overload
def touches(geometry1: MultiLineString, geometry2: LineString) -> bool: ...
@overload
def touches(geometry1: MultiLineString, geometry2: Ring) -> bool: ...
@overload
def touches(geometry1: MultiLineString, geometry2: Polygon) -> bool: ...
@overload
def touches(
    geometry1: MultiLineString, geometry2: MultiLineString
) -> bool: ...
@overload
def touches(geometry1: MultiLineString, geometry2: MultiPolygon) -> bool: ...
@overload
def touches(geometry1: MultiPolygon, geometry2: Point) -> bool: ...
@overload
def touches(geometry1: MultiPolygon, geometry2: LineString) -> bool: ...
@overload
def touches(geometry1: MultiPolygon, geometry2: Ring) -> bool: ...
@overload
def touches(geometry1: MultiPolygon, geometry2: Polygon) -> bool: ...
@overload
def touches(geometry1: MultiPolygon, geometry2: MultiLineString) -> bool: ...
@overload
def touches(geometry1: MultiPolygon, geometry2: MultiPolygon) -> bool: ...
@overload
def union(
    geometry1: Ring,
    geometry2: Ring,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def union(
    geometry1: Ring,
    geometry2: Polygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def union(
    geometry1: Polygon,
    geometry2: Ring,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def union(
    geometry1: Polygon,
    geometry2: Polygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def union(
    geometry1: MultiPolygon,
    geometry2: Ring,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def union(
    geometry1: MultiPolygon,
    geometry2: Polygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def union(
    geometry1: MultiPolygon,
    geometry2: MultiPolygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def union(
    geometry1: Ring,
    geometry2: MultiPolygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def union(
    geometry1: Polygon,
    geometry2: MultiPolygon,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[Polygon]: ...
@overload
def union(
    geometry1: LineString,
    geometry2: LineString,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[LineString]: ...
@overload
def union(
    geometry1: MultiLineString,
    geometry2: MultiLineString,
    spheroid: Spheroid | None = None,
    strategy: Strategy = ...,
) -> list[LineString]: ...
def unique(geometry: _Concept) -> None: ...
@overload
def within(geometry1: Point, geometry2: Point) -> bool: ...
@overload
def within(geometry1: Point, geometry2: Segment) -> bool: ...
@overload
def within(geometry1: Point, geometry2: Box) -> bool: ...
@overload
def within(geometry1: Point, geometry2: LineString) -> bool: ...
@overload
def within(geometry1: Point, geometry2: Ring) -> bool: ...
@overload
def within(geometry1: Point, geometry2: Polygon) -> bool: ...
@overload
def within(geometry1: Point, geometry2: MultiPoint) -> bool: ...
@overload
def within(geometry1: Point, geometry2: MultiLineString) -> bool: ...
@overload
def within(geometry1: Point, geometry2: MultiPolygon) -> bool: ...
@overload
def within(geometry1: Box, geometry2: Box) -> bool: ...
@overload
def within(geometry1: LineString, geometry2: LineString) -> bool: ...
@overload
def within(geometry1: LineString, geometry2: Ring) -> bool: ...
@overload
def within(geometry1: LineString, geometry2: Polygon) -> bool: ...
@overload
def within(geometry1: LineString, geometry2: MultiLineString) -> bool: ...
@overload
def within(geometry1: LineString, geometry2: MultiPolygon) -> bool: ...
@overload
def within(geometry1: Ring, geometry2: Ring) -> bool: ...
@overload
def within(geometry1: Ring, geometry2: Polygon) -> bool: ...
@overload
def within(geometry1: Ring, geometry2: MultiPolygon) -> bool: ...
@overload
def within(geometry1: Polygon, geometry2: Ring) -> bool: ...
@overload
def within(geometry1: Polygon, geometry2: Polygon) -> bool: ...
@overload
def within(geometry1: Polygon, geometry2: MultiPolygon) -> bool: ...
@overload
def within(geometry1: MultiLineString, geometry2: LineString) -> bool: ...
@overload
def within(geometry1: MultiLineString, geometry2: Ring) -> bool: ...
@overload
def within(geometry1: MultiLineString, geometry2: Polygon) -> bool: ...
@overload
def within(geometry1: MultiLineString, geometry2: MultiLineString) -> bool: ...
@overload
def within(geometry1: MultiLineString, geometry2: MultiPolygon) -> bool: ...
@overload
def within(geometry1: MultiPolygon, geometry2: Ring) -> bool: ...
@overload
def within(geometry1: MultiPolygon, geometry2: Polygon) -> bool: ...
@overload
def within(geometry1: MultiPolygon, geometry2: MultiPolygon) -> bool: ...

class Strategy(enum.Enum):
    ANDOYER = ...
    KARNEY = ...
    THOMAS = ...
    VINCENTY = ...
