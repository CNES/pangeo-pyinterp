from ...type_hints import NDArray1DFloat64, NDArray2DFloat64
from .geographic import Point, Spheroid
from .geographic.algorithms import Strategy

def calculate_swath(
    lon_nadir: NDArray1DFloat64,
    lat_nadir: NDArray1DFloat64,
    delta_ac: float,
    half_gap: float,
    half_swath: int,
    spheroid: Spheroid | None = None,
) -> tuple[NDArray2DFloat64, NDArray2DFloat64]: ...
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

class CrossoverResult:
    @property
    def index1(self) -> int: ...
    @property
    def index2(self) -> int: ...
    @property
    def point(self) -> Point: ...
