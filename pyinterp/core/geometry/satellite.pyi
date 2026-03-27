from typing import Sequence
import enum

from ...type_hints import NDArray1DFloat64, NDArray2DFloat64
from .geographic import Box, Point, Spheroid
from .geographic.algorithms import Strategy

class CrossoverResult:
    @property
    def index1(self) -> int: ...
    @property
    def index2(self) -> int: ...
    @property
    def point(self) -> Point: ...

def find_crossovers(
    lon1: NDArray1DFloat64,
    lat1: NDArray1DFloat64,
    lon2: NDArray1DFloat64,
    lat2: NDArray1DFloat64,
    predicate: float,
    allow_multiple: bool = False,
    use_cartesian: bool = True,
    strategy: Strategy = ...,
    spheroid: Spheroid | None = None,
) -> list[CrossoverResult]: ...

class LatitudeZone(enum.Enum):
    SOUTH = 0
    MID = 1
    NORTH = 2

class OrbitDirection(enum.Enum):
    PROGRADE = 0
    RETROGRADE = 1

class DecompositionOptions:
    def __init__(self) -> None: ...
    @property
    def swath_width_km(self) -> float: ...
    @property
    def south_limit(self) -> float: ...
    @property
    def north_limit(self) -> float: ...
    @property
    def min_edge_size(self) -> int: ...
    @property
    def merge_area_ratio(self) -> float: ...
    @property
    def max_segments(self) -> int: ...
    def with_swath_width_km(
        self, swath_width_km: float
    ) -> DecompositionOptions: ...
    def with_south_limit(self, south_limit: float) -> DecompositionOptions: ...
    def with_north_limit(self, north_limit: float) -> DecompositionOptions: ...
    def with_min_edge_size(
        self, min_edge_size: int
    ) -> DecompositionOptions: ...
    def with_merge_area_ratio(
        self, merge_area_ratio: float
    ) -> DecompositionOptions: ...
    def with_max_segments(self, max_segments: int) -> DecompositionOptions: ...

class TrackSegment:
    @property
    def first_index(self) -> int: ...
    @property
    def last_index(self) -> int: ...
    @property
    def bbox(self) -> Box: ...
    @property
    def zone(self) -> LatitudeZone: ...
    @property
    def orbit(self) -> OrbitDirection: ...
    @property
    def size(self) -> int: ...
    def __repr__(self) -> str: ...

def decompose_track(
    lon: NDArray1DFloat64,
    lat: NDArray1DFloat64,
    strategy: str | None = None,
    opts: DecompositionOptions | None = None,
) -> list[TrackSegment]: ...
def infer_orbit_direction(lon: NDArray1DFloat64) -> OrbitDirection: ...
def filter_by_extent(
    segments: Sequence[TrackSegment], extent: Box
) -> list[TrackSegment]: ...
def calculate_swath(
    lon_nadir: NDArray1DFloat64,
    lat_nadir: NDArray1DFloat64,
    delta_ac: float,
    half_gap: float,
    half_swath: int,
    spheroid: Spheroid | None = None,
) -> tuple[NDArray2DFloat64, NDArray2DFloat64]: ...
