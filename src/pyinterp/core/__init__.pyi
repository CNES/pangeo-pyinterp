from typing import (
    Any,
    ClassVar,
    Iterator,
    List,
    Optional,
    Self,
    Tuple,
    TypeAlias,
    overload,
)

import numpy
import numpy.typing

from . import fill, geodetic

__all__ = [
    "Axis",
    "AxisBoundary",
    "AxisInt64",
    "Bilinear2D",
    "Bilinear3D",
    "Binning1DFloat32",
    "Binning1DFloat64",
    "Binning2DFloat32",
    "Binning2DFloat64",
    "CovarianceFunction",
    "DescriptiveStatisticsFloat32",
    "DescriptiveStatisticsFloat64",
    "GeoHash",
    "Grid2DFloat32",
    "Grid2DFloat64",
    "Grid2DInt8",
    "Grid2DUInt8",
    "Grid3DFloat32",
    "Grid3DFloat64",
    "Grid3DInt8",
    "Grid3DUInt8",
    "Grid4DFloat32",
    "Grid4DFloat64",
    "Grid4DInt8",
    "Grid4DUInt8",
    "Histogram2DFloat32",
    "Histogram2DFloat64",
    "fill",
    "geodetic",
]

from .array import (
    Array1DBool,
    Array1DFloat32,
    Array1DFloat64,
    Array1DInt64,
    Array2DFloat32,
    Array2DFloat64,
    Array2DInt8,
    Array2DUInt8,
    Array2DUInt64,
    Array3DFloat32,
    Array3DFloat64,
    Array3DInt8,
    Array3DUInt8,
    Array4DFloat32,
    Array4DFloat64,
    Array4DInt8,
    Array4DUInt8,
    ArrayFloat32,
    ArrayFloat64,
    ArrayInt64,
    ArrayUInt64,
)

class Axis:

    def __init__(self,
                 values: Array1DFloat64,
                 epsilon: float = ...,
                 is_circle: bool = ...) -> None:
        ...

    def back(self) -> float:
        ...

    def find_index(self,
                   coordinates: Array1DFloat64,
                   bounded: bool = ...) -> Array1DInt64:
        ...

    def find_indexes(self, coordinates: Array1DFloat64) -> Array1DInt64:
        ...

    def flip(self, inplace: bool = ...) -> Axis:
        ...

    def front(self) -> float:
        ...

    def increment(self) -> float:
        ...

    def is_ascending(self) -> bool:
        ...

    def is_regular(self) -> bool:
        ...

    def max_value(self) -> float:
        ...

    def min_value(self) -> float:
        ...

    def __copy__(self) -> Axis:
        ...

    def __eq__(self, other: Axis) -> bool:  # type: ignore[override]
        ...

    @overload
    def __getitem__(self, index: int) -> float:
        ...

    @overload
    def __getitem__(self, indices: slice) -> Array1DFloat64:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iter__(self) -> Iterator[float]:
        ...

    def __len__(self) -> int:
        ...

    def __ne__(self, other: Axis) -> bool:  # type: ignore[override]
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def is_circle(self) -> bool:
        ...


class AxisBoundary:
    __members__: ClassVar[dict] = ...  # read-only
    Expand: ClassVar[AxisBoundary] = ...
    Sym: ClassVar[AxisBoundary] = ...
    Undef: ClassVar[AxisBoundary] = ...
    Wrap: ClassVar[AxisBoundary] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, value: int) -> None:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __getstate__(self) -> int:
        ...

    def __hash__(self) -> int:
        ...

    def __index__(self) -> int:
        ...

    def __int__(self) -> int:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __setstate__(self, state: int) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def value(self) -> int:
        ...


class AxisInt64:

    def __init__(self, values: Array1DInt64) -> None:
        ...

    def back(self) -> int:
        ...

    def find_index(self,
                   coordinates: Array1DInt64,
                   bounded: bool = ...) -> Array1DInt64:
        ...

    def find_indexes(self, coordinates: Array1DInt64) -> Array1DInt64:
        ...

    def flip(self, inplace: bool = ...) -> AxisInt64:
        ...

    def front(self) -> int:
        ...

    def increment(self) -> int:
        ...

    def is_ascending(self) -> bool:
        ...

    def is_regular(self) -> bool:
        ...

    def max_value(self) -> int:
        ...

    def min_value(self) -> int:
        ...

    def __copy__(self) -> AxisInt64:
        ...

    def __eq__(self, other: AxisInt64) -> bool:  # type: ignore[override]
        ...

    @overload
    def __getitem__(self, index: int) -> int:
        ...

    @overload
    def __getitem__(self, indices: slice) -> Array1DInt64:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iter__(self) -> Iterator[int]:
        ...

    def __len__(self) -> int:
        ...

    def __ne__(self, other: AxisInt64) -> bool:  # type: ignore[override]
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Bilinear2D(BivariateInterpolator2D):

    def __init__(self) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Bilinear3D(BivariateInterpolator3D):

    def __init__(self) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Binning1DFloat32(Binning2DFloat32):

    def __init__(self,
                 x: Axis,
                 range: Optional[Tuple[float, float]] = None) -> None:
        ...

    def push(  # type: ignore[override]
        self,
        x: Array1DFloat32,
        z: Array1DFloat32,
        weights: Array1DFloat32 | None = ...,
    ) -> None:
        ...

    def range(self) -> Tuple[float, float]:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Binning1DFloat64(Binning2DFloat64):

    def __init__(self,
                 x: Axis,
                 range: Optional[Tuple[float, float]] = None) -> None:
        ...

    def push(  # type: ignore[override]
        self,
        x: Array1DFloat64,
        z: Array1DFloat64,
        weights: Array1DFloat64 | None = ...,
    ) -> None:
        ...

    def range(self) -> Tuple[float, float]:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Binning2DFloat32:

    def __init__(self,
                 x: Axis,
                 y: Axis,
                 wgs: Optional[geodetic.Spheroid] = ...) -> None:
        ...

    def clear(self) -> None:
        ...

    def count(self) -> Array2DUInt64:
        ...

    def kurtosis(self) -> Array2DFloat32:
        ...

    def max(self) -> Array2DFloat32:
        ...

    def mean(self) -> Array2DFloat32:
        ...

    def min(self) -> Array2DFloat32:
        ...

    def push(self,
             x: Array1DFloat32,
             y: Array1DFloat32,
             z: Array1DFloat32,
             simple: bool = ...) -> None:
        ...

    def skewness(self) -> Array2DFloat32:
        ...

    def sum(self) -> Array2DFloat32:
        ...

    def sum_of_weights(self) -> Array2DFloat32:
        ...

    def variance(self, ddof: int = ...) -> Array2DFloat32:
        ...

    def __copy__(self) -> Binning2DFloat32:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iadd__(
        self,
        other: Binning2DFloat32,
    ) -> Self:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def wgs(self) -> Optional[geodetic.Spheroid]:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...


class Binning2DFloat64:

    def __init__(self,
                 x: Axis,
                 y: Axis,
                 wgs: Optional[geodetic.Spheroid] = ...) -> None:
        ...

    def clear(self) -> None:
        ...

    def count(self) -> Array2DUInt64:
        ...

    def kurtosis(self) -> Array2DFloat64:
        ...

    def max(self) -> Array2DFloat64:
        ...

    def mean(self) -> Array2DFloat64:
        ...

    def min(self) -> Array2DFloat64:
        ...

    def push(self,
             x: Array1DFloat64,
             y: Array1DFloat64,
             z: Array1DFloat64,
             simple: bool = ...) -> None:
        ...

    def skewness(self) -> Array2DFloat64:
        ...

    def sum(self) -> Array2DFloat64:
        ...

    def sum_of_weights(self) -> Array2DFloat64:
        ...

    def variance(self, ddof: int = ...) -> Array2DFloat64:
        ...

    def __copy__(self) -> Binning2DFloat64:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iadd__(self, other: Binning2DFloat64) -> Self:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def wgs(self) -> Optional[geodetic.Spheroid]:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...


class BivariateInterpolator2D:

    def __init__(self, *args, **kwargs) -> None:
        ...


class BivariateInterpolator3D:

    def __init__(self, *args, **kwargs) -> None:
        ...


class CovarianceFunction:
    __members__: ClassVar[dict] = ...  # read-only
    Cauchy: ClassVar[CovarianceFunction] = ...
    Exponential: ClassVar[CovarianceFunction] = ...
    Gaussian: ClassVar[CovarianceFunction] = ...
    Linear: ClassVar[CovarianceFunction] = ...
    Matern_12: ClassVar[CovarianceFunction] = ...
    Matern_32: ClassVar[CovarianceFunction] = ...
    Matern_52: ClassVar[CovarianceFunction] = ...
    Spherical: ClassVar[CovarianceFunction] = ...
    WhittleMatern: ClassVar[CovarianceFunction] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, value: int) -> None:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __getstate__(self) -> int:
        ...

    def __hash__(self) -> int:
        ...

    def __index__(self) -> int:
        ...

    def __int__(self) -> int:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __setstate__(self, state: int) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def value(self) -> int:
        ...


class DescriptiveStatisticsFloat32:

    def __init__(self,
                 values: ArrayFloat32,
                 weights: Optional[ArrayFloat32] = ...,
                 axis: Optional[List[int]] = ...) -> None:
        ...

    def count(self) -> ArrayInt64:
        ...

    def kurtosis(self) -> ArrayFloat32:
        ...

    def max(self) -> ArrayFloat32:
        ...

    def mean(self) -> ArrayFloat32:
        ...

    def min(self) -> ArrayFloat32:
        ...

    def skewness(self) -> ArrayFloat32:
        ...

    def sum(self) -> ArrayFloat32:
        ...

    def sum_of_weights(self) -> ArrayFloat32:
        ...

    def variance(self, ddof: int = ...) -> ArrayFloat32:
        ...

    def __add__(
            self, other: DescriptiveStatisticsFloat32
    ) -> DescriptiveStatisticsFloat32:
        ...

    def __copy__(self) -> DescriptiveStatisticsFloat32:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iadd__(self, other: DescriptiveStatisticsFloat32) -> Self:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class DescriptiveStatisticsFloat64:

    def __init__(self,
                 values: ArrayFloat64,
                 weights: Optional[ArrayFloat64] = ...,
                 axis: Optional[List[int]] = ...) -> None:
        ...

    def count(self) -> ArrayInt64:
        ...

    def kurtosis(self) -> ArrayFloat64:
        ...

    def max(self) -> ArrayFloat64:
        ...

    def mean(self) -> ArrayFloat64:
        ...

    def min(self) -> ArrayFloat64:
        ...

    def skewness(self) -> ArrayFloat64:
        ...

    def sum(self) -> ArrayFloat64:
        ...

    def sum_of_weights(self) -> ArrayFloat64:
        ...

    def variance(self, ddof: int = ...) -> ArrayFloat64:
        ...

    def __add__(
            self, other: DescriptiveStatisticsFloat64
    ) -> DescriptiveStatisticsFloat64:
        ...

    def __copy__(self) -> DescriptiveStatisticsFloat64:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iadd__(self, other: DescriptiveStatisticsFloat64) -> Self:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class GeoHash:

    def __init__(self,
                 longitude: float,
                 latitude: float,
                 precision: int = ...) -> None:
        ...

    def area(self, wgs: Optional[geodetic.Spheroid] = ...) -> float:
        ...

    def bounding_box(self) -> geodetic.Box:
        ...

    def center(self) -> geodetic.Point:
        ...

    @staticmethod
    def error_with_precision(precision: int = 1) -> Tuple[float, float]:
        ...

    @staticmethod
    def from_string(code: str, round: bool = False) -> GeoHash:
        ...

    @staticmethod
    def grid_properties(box: geodetic.Box,
                        precision: int = 1) -> Tuple[GeoHash, int, int]:
        ...

    def neighbors(self) -> List[GeoHash]:
        ...

    def number_of_bits(self) -> int:
        ...

    def precision(self) -> int:
        ...

    def reduce(self) -> Tuple[float, float, int]:
        ...


class Grid2DFloat32:

    def __init__(self, x: Axis, y: Axis, array: Array2DFloat32) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def array(self) -> Array2DFloat32:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...


class Grid2DFloat64:

    def __init__(self, x: Axis, y: Axis, array: Array2DFloat64) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def array(self) -> Array2DFloat64:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...


class Grid2DInt8:

    def __init__(self, x: Axis, y: Axis, array: Array2DInt8) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def array(self) -> Array2DInt8:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...


class Grid2DUInt8:

    def __init__(self, x: Axis, y: Axis, array: Array2DUInt8) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def array(self) -> Array2DUInt8:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...


class Grid3DFloat32:

    def __init__(self, x: Axis, y: Axis, z: Axis,
                 array: Array3DFloat32) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def array(self) -> Array3DFloat32:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...

    @property
    def z(self) -> Axis:
        ...


class Grid3DFloat64:

    def __init__(self, x: Axis, y: Axis, z: Axis,
                 array: Array3DFloat64) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def array(self) -> Array3DFloat64:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...

    @property
    def z(self) -> Axis:
        ...


class Grid3DInt8:

    def __init__(self, x: Axis, y: Axis, z: Axis, array: Array3DInt8) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def array(self) -> Array3DInt8:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...

    @property
    def z(self) -> Axis:
        ...


class Grid3DUInt8:

    def __init__(self, x: Axis, y: Axis, z: Axis, array: Array3DUInt8) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def array(self) -> Array3DInt8:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...

    @property
    def z(self) -> Axis:
        ...


class Grid4DFloat32:

    def __init__(self, x: Axis, y: Axis, z: Axis, u: Axis,
                 array: Array4DFloat32) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def array(self) -> Array4DFloat32:
        ...

    @property
    def u(self) -> Axis:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...

    @property
    def z(self) -> Axis:
        ...


class Grid4DFloat64:

    def __init__(self, x: Axis, y: Axis, z: Axis, u: Axis,
                 array: Array4DFloat64) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def array(self) -> Array4DFloat64:
        ...

    @property
    def u(self) -> Axis:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...

    @property
    def z(self) -> Axis:
        ...


class Grid4DInt8:

    def __init__(self, x: Axis, y: Axis, z: Axis, u: Axis,
                 array: Array4DInt8) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def array(self) -> Array4DInt8:
        ...

    @property
    def u(self) -> Axis:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...

    @property
    def z(self) -> Axis:
        ...


class Grid4DUInt8:

    def __init__(self, x: Axis, y: Axis, z: Axis, u: Axis,
                 array: Array4DUInt8) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def array(self) -> Array4DUInt8:
        ...

    @property
    def u(self) -> Axis:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...

    @property
    def z(self) -> Axis:
        ...


class Histogram2DFloat32:

    def __init__(self, x: Axis, y: Axis, bins: Optional[int] = ...) -> None:
        ...

    def clear(self) -> None:
        ...

    def count(self) -> Array2DUInt64:
        ...

    def histograms(self, *args, **kwargs) -> Any:
        ...

    def kurtosis(self) -> Array2DFloat32:
        ...

    def max(self) -> Array2DFloat32:
        ...

    def mean(self) -> Array2DFloat32:
        ...

    def min(self) -> Array2DFloat32:
        ...

    def push(self, x: Array1DFloat32, y: Array1DFloat32,
             z: Array1DFloat32) -> None:
        ...

    def quantile(self, q: float = ...) -> Array2DFloat32:
        ...

    def skewness(self) -> Array2DFloat32:
        ...

    def sum_of_weights(self) -> Array2DFloat32:
        ...

    def variance(self) -> Array2DFloat32:
        ...

    def __copy__(self) -> Histogram2DFloat32:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iadd__(self, other: Histogram2DFloat32) -> Self:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...


class Histogram2DFloat64:

    def __init__(self, x: Axis, y: Axis, bins: Optional[int] = ...) -> None:
        ...

    def clear(self) -> None:
        ...

    def count(self) -> Array2DUInt64:
        ...

    def histograms(self, *args, **kwargs) -> Any:
        ...

    def kurtosis(self) -> Array2DFloat64:
        ...

    def max(self) -> Array2DFloat64:
        ...

    def mean(self) -> Array2DFloat64:
        ...

    def min(self) -> Array2DFloat64:
        ...

    def push(self, x: Array1DFloat64, y: Array1DFloat64,
             z: Array1DFloat64) -> None:
        ...

    def quantile(self, q: float = ...) -> Array2DFloat64:
        ...

    def skewness(self) -> Array2DFloat64:
        ...

    def sum_of_weights(self) -> Array2DFloat64:
        ...

    def variance(self) -> Array2DFloat64:
        ...

    def __copy__(self) -> Histogram2DFloat64:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iadd__(self, other: Histogram2DFloat64) -> Self:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...


class InverseDistanceWeighting2D(BivariateInterpolator2D):

    def __init__(self, p: int = ...) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class InverseDistanceWeighting3D(BivariateInterpolator3D):

    def __init__(self, p: int = ...) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Nearest2D(BivariateInterpolator2D):

    def __init__(self) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Nearest3D(BivariateInterpolator3D):

    def __init__(self) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Period:

    def __init__(self, begin: int, last: int, within: bool = ...) -> None:
        ...

    @overload
    def contains(self, point: int) -> bool:
        ...

    @overload
    def contains(self, other: Period) -> bool:
        ...

    def duration(self) -> int:
        ...

    def end(self) -> int:
        ...

    def intersection(self, other: Period) -> Period:
        ...

    def intersects(self, other: Period) -> bool:
        ...

    def is_adjacent(self, other: Period) -> bool:
        ...

    def is_after(self, point: int) -> bool:
        ...

    def is_before(self, point: int) -> bool:
        ...

    def is_null(self) -> bool:
        ...

    def merge(self, merge: Period) -> Period:
        ...

    def __eq__(self, other: Period) -> bool:  # type: ignore[override]
        ...

    def __len__(self) -> int:
        ...

    def __lt__(self, other: Period) -> bool:
        ...

    def __ne__(self, other: Period) -> bool:  # type: ignore[override]
        ...

    @property
    def begin(self) -> int:
        ...

    @property
    def last(self) -> int:
        ...


Array1DPeriod: TypeAlias = numpy.ndarray[
    tuple[int],
    numpy.dtype[Period]  # type: ignore[type-var]
]


class PeriodList:

    def __init__(self,
                 periods: Array1DPeriod) -> None:  # type: ignore[type-var]
        ...

    def are_periods_sorted_and_disjointed(self) -> bool:
        ...

    def belong_to_a_period(self, dates: Array1DInt64) -> Array1DBool:
        ...

    def cross_a_period(self, dates: Array1DInt64) -> Array1DBool:
        ...

    def filter(self, min_duration: int) -> PeriodList:
        ...

    def intersection(self, period: Period) -> PeriodList:
        ...

    def is_it_close(self, period: int, epsilon: int) -> bool:
        ...

    def join_adjacent_periods(self, epsilon: int) -> PeriodList:
        ...

    def merge(self, other: PeriodList) -> None:
        ...

    def sort(self) -> None:
        ...

    def within(self, period: Period) -> PeriodList:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __len__(self) -> int:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def periods(self) -> Array1DPeriod:
        ...


class RTree3DFloat32:

    def __init__(self,
                 spheroid: Optional[geodetic.Spheroid] = ...,
                 ecef: bool = ...) -> None:
        ...

    def bounds(self) -> tuple:
        ...

    def clear(self) -> None:
        ...

    def insert(self, coordinates: ArrayFloat32, values: ArrayFloat32) -> None:
        ...

    def inverse_distance_weighting(self,
                                   coordinates: ArrayFloat32,
                                   radius: Optional[float] = ...,
                                   k: int = ...,
                                   p: int = ...,
                                   within: bool = ...,
                                   num_threads: int = ...) -> tuple:
        ...

    def packing(self, coordinates: ArrayFloat32, values: ArrayFloat32) -> None:
        ...

    def query(self,
              coordinates: ArrayFloat32,
              k: int = ...,
              within: bool = ...,
              num_threads: int = ...) -> tuple:
        ...

    def radial_basis_function(self,
                              coordinates: ArrayFloat32,
                              radius: Optional[float] = ...,
                              k: int = ...,
                              rbf: RadialBasisFunction = ...,
                              epsilon: Optional[float] = ...,
                              smooth: float = ...,
                              within: bool = ...,
                              num_threads: int = ...) -> tuple:
        ...

    def universal_kriging(self,
                          coordinates: ArrayFloat32,
                          radius: Optional[float] = ...,
                          k: int = ...,
                          covariance: CovarianceFunction = ...,
                          sigma: float = ...,
                          alpha: float = ...,
                          within: bool = ...,
                          num_threads: int = ...) -> tuple:
        ...

    def value(self,
              coordinates: ArrayFloat32,
              radius: Optional[float] = ...,
              k: int = ...,
              within: bool = ...,
              num_threads: int = ...) -> tuple:
        ...

    def window_function(self,
                        coordinates: ArrayFloat32,
                        radius: Optional[float] = ...,
                        k: int = ...,
                        wf: WindowFunction = ...,
                        arg: Optional[float] = ...,
                        within: bool = ...,
                        num_threads: int = ...) -> tuple:
        ...

    def __bool__(self) -> bool:
        ...

    def __copy__(self) -> RTree3DFloat32:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __len__(self) -> int:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class RTree3DFloat64:

    def __init__(self,
                 spheroid: Optional[geodetic.Spheroid] = ...,
                 ecef: bool = ...) -> None:
        ...

    def bounds(self) -> tuple:
        ...

    def clear(self) -> None:
        ...

    def insert(self, coordinates: ArrayFloat64, values: ArrayFloat64) -> None:
        ...

    def inverse_distance_weighting(self,
                                   coordinates: ArrayFloat64,
                                   radius: Optional[float] = ...,
                                   k: int = ...,
                                   p: int = ...,
                                   within: bool = ...,
                                   num_threads: int = ...) -> tuple:
        ...

    def packing(self, coordinates: ArrayFloat64, values: ArrayFloat64) -> None:
        ...

    def query(self,
              coordinates: ArrayFloat64,
              k: int = ...,
              within: bool = ...,
              num_threads: int = ...) -> tuple:
        ...

    def radial_basis_function(self,
                              coordinates: ArrayFloat64,
                              radius: Optional[float] = ...,
                              k: int = ...,
                              rbf: RadialBasisFunction = ...,
                              epsilon: Optional[float] = ...,
                              smooth: float = ...,
                              within: bool = ...,
                              num_threads: int = ...) -> tuple:
        ...

    def universal_kriging(self,
                          coordinates: ArrayFloat64,
                          radius: Optional[float] = ...,
                          k: int = ...,
                          covariance: CovarianceFunction = ...,
                          sigma: float = ...,
                          alpha: float = ...,
                          within: bool = ...,
                          num_threads: int = ...) -> tuple:
        ...

    def value(self,
              coordinates: ArrayFloat64,
              radius: Optional[float] = ...,
              k: int = ...,
              within: bool = ...,
              num_threads: int = ...) -> tuple:
        ...

    def window_function(self,
                        coordinates: ArrayFloat64,
                        radius: Optional[float] = ...,
                        k: int = ...,
                        wf: WindowFunction = ...,
                        arg: Optional[float] = ...,
                        within: bool = ...,
                        num_threads: int = ...) -> tuple:
        ...

    def __bool__(self) -> bool:
        ...

    def __copy__(self) -> RTree3DFloat64:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __len__(self) -> int:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class RadialBasisFunction:
    __members__: ClassVar[dict] = ...  # read-only
    Cubic: ClassVar[RadialBasisFunction] = ...
    Gaussian: ClassVar[RadialBasisFunction] = ...
    InverseMultiquadric: ClassVar[RadialBasisFunction] = ...
    Linear: ClassVar[RadialBasisFunction] = ...
    Multiquadric: ClassVar[RadialBasisFunction] = ...
    ThinPlate: ClassVar[RadialBasisFunction] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, value: int) -> None:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __getstate__(self) -> int:
        ...

    def __hash__(self) -> int:
        ...

    def __index__(self) -> int:
        ...

    def __int__(self) -> int:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __setstate__(self, state: int) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def value(self) -> int:
        ...


class StreamingHistogramFloat32:

    def __init__(self,
                 values: ArrayFloat32,
                 weights: Optional[ArrayFloat32] = ...,
                 axis: Optional[List[int]] = ...,
                 bin_count: Optional[int] = ...) -> None:
        ...

    def bins(self, *args, **kwargs) -> Any:
        ...

    def count(self) -> ArrayUInt64:
        ...

    def kurtosis(self) -> ArrayFloat32:
        ...

    def max(self) -> ArrayFloat32:
        ...

    def mean(self) -> ArrayFloat32:
        ...

    def min(self) -> ArrayFloat32:
        ...

    def quantile(self, q: float = ...) -> ArrayFloat32:
        ...

    def resize(self, arg0: int) -> None:
        ...

    def size(self) -> ArrayUInt64:
        ...

    def skewness(self) -> ArrayFloat32:
        ...

    def sum_of_weights(self) -> ArrayFloat32:
        ...

    def variance(self) -> ArrayFloat32:
        ...

    def __add__(self,
                other: StreamingHistogramFloat32) -> StreamingHistogramFloat32:
        ...

    def __copy__(self) -> StreamingHistogramFloat32:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iadd__(self, other: StreamingHistogramFloat32) -> Self:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class StreamingHistogramFloat64:

    def __init__(self,
                 values: ArrayFloat64,
                 weights: Optional[ArrayFloat64] = ...,
                 axis: Optional[List[int]] = ...,
                 bin_count: Optional[int] = ...) -> None:
        ...

    def bins(self, *args, **kwargs) -> Any:
        ...

    def count(self) -> ArrayUInt64:
        ...

    def kurtosis(self) -> ArrayFloat64:
        ...

    def max(self) -> ArrayFloat64:
        ...

    def mean(self) -> ArrayFloat64:
        ...

    def min(self) -> ArrayFloat64:
        ...

    def quantile(self, q: float = ...) -> ArrayFloat64:
        ...

    def resize(self, arg0: int) -> None:
        ...

    def size(self) -> ArrayUInt64:
        ...

    def skewness(self) -> ArrayFloat64:
        ...

    def sum_of_weights(self) -> ArrayFloat64:
        ...

    def variance(self) -> ArrayFloat64:
        ...

    def __add__(self,
                other: StreamingHistogramFloat64) -> StreamingHistogramFloat64:
        ...

    def __copy__(self) -> StreamingHistogramFloat64:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iadd__(self, other: StreamingHistogramFloat64) -> Self:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class TemporalAxis(AxisInt64):

    def __init__(self, values: numpy.ndarray) -> None:
        ...

    def back(self) -> numpy.ndarray:  # type: ignore[override]
        ...

    def dtype(self) -> numpy.dtype:
        ...

    def find_index(self,
                   coordinates: numpy.ndarray,
                   bounded: bool = ...) -> Array1DInt64:
        ...

    def find_indexes(self, coordinates: numpy.ndarray) -> Array1DInt64:
        ...

    def flip(self, inplace: bool = ...) -> TemporalAxis:
        ...

    def front(self) -> numpy.ndarray:  # type: ignore[override]
        ...

    def increment(self) -> numpy.ndarray:  # type: ignore[override]
        ...

    def is_ascending(self) -> bool:
        ...

    def is_regular(self) -> bool:
        ...

    def max_value(self) -> numpy.ndarray:  # type: ignore[override]
        ...

    def min_value(self) -> numpy.ndarray:  # type: ignore[override]
        ...

    def safe_cast(self, values: numpy.ndarray) -> numpy.ndarray:
        ...

    def __copy__(self) -> TemporalAxis:
        ...

    def __eq__(self, other: TemporalAxis) -> bool:  # type: ignore[override]
        ...

    @overload  # type: ignore[override]
    def __getitem__(self, index: int) -> numpy.ndarray:
        ...

    @overload  # type: ignore[override]
    def __getitem__(self, indices: slice) -> numpy.ndarray:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iter__(self) -> Iterator[numpy.ndarray]:  # type: ignore[override]
        ...

    def __len__(self) -> int:
        ...

    def __ne__(self, other: TemporalAxis) -> bool:  # type: ignore[override]
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class TemporalBilinear3D(TemporalBivariateInterpolator3D):

    def __init__(self) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class TemporalBivariateInterpolator3D:

    def __init__(self, *args, **kwargs) -> None:
        ...


class TemporalGrid3DFloat32:

    def __init__(self, x: Axis, y: Axis, z: AxisInt64,
                 array: Array3DFloat64) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def array(self) -> Array3DFloat64:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...

    @property
    def z(self) -> AxisInt64:
        ...


class TemporalGrid3DFloat64:

    def __init__(self, x: Axis, y: Axis, z: AxisInt64,
                 array: Array3DFloat64) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def array(self) -> Array3DFloat64:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...

    @property
    def z(self) -> AxisInt64:
        ...


class TemporalGrid3DInt8:

    def __init__(self, x: Axis, y: Axis, z: AxisInt64,
                 array: Array3DInt8) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def array(self) -> Array3DInt8:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...

    @property
    def z(self) -> AxisInt64:
        ...


class TemporalGrid4DFloat32:

    def __init__(self, x: Axis, y: Axis, z: AxisInt64, u: Axis,
                 array: Array4DFloat32) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def array(self) -> Array4DFloat32:
        ...

    @property
    def u(self) -> Axis:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...

    @property
    def z(self) -> AxisInt64:
        ...


class TemporalGrid4DFloat64:

    def __init__(self, x: Axis, y: Axis, z: AxisInt64, u: Axis,
                 array: Array4DFloat64) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def array(self) -> Array4DFloat64:
        ...

    @property
    def u(self) -> Axis:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...

    @property
    def z(self) -> AxisInt64:
        ...


class TemporalGrid4DInt8:

    def __init__(self, x: Axis, y: Axis, z: AxisInt64, u: Axis,
                 array: Array4DInt8) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...

    @property
    def array(self) -> Array4DInt8:
        ...

    @property
    def u(self) -> Axis:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...

    @property
    def z(self) -> AxisInt64:
        ...


class TemporalInverseDistanceWeighting3D(TemporalBivariateInterpolator3D):

    def __init__(self, p: int = ...) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class TemporalNearest3D(TemporalBivariateInterpolator3D):

    def __init__(self) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class WindowFunction:
    __members__: ClassVar[dict] = ...  # read-only
    Blackman: ClassVar[WindowFunction] = ...
    BlackmanHarris: ClassVar[WindowFunction] = ...
    Boxcar: ClassVar[WindowFunction] = ...
    FlatTop: ClassVar[WindowFunction] = ...
    Gaussian: ClassVar[WindowFunction] = ...
    Hamming: ClassVar[WindowFunction] = ...
    Lanczos: ClassVar[WindowFunction] = ...
    Nuttall: ClassVar[WindowFunction] = ...
    Parzen: ClassVar[WindowFunction] = ...
    ParzenSWOT: ClassVar[WindowFunction] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, value: int) -> None:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __getstate__(self) -> int:
        ...

    def __hash__(self) -> int:
        ...

    def __index__(self) -> int:
        ...

    def __int__(self) -> int:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __setstate__(self, state: int) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def value(self) -> int:
        ...


@overload
def bicubic_float32(grid: Grid2DFloat32,
                    x: Array1DFloat64,
                    y: Array1DFloat64,
                    nx: int = ...,
                    ny: int = ...,
                    fitting_model: str = ...,
                    boundary: str = ...,
                    bounds_error: bool = ...,
                    num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def bicubic_float32(grid: Grid3DFloat32,
                    x: Array1DFloat64,
                    y: Array1DFloat64,
                    z: Array1DFloat64,
                    nx: int = ...,
                    ny: int = ...,
                    fitting_model: str = ...,
                    boundary: str = ...,
                    bounds_error: bool = ...,
                    num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def bicubic_float32(grid: TemporalGrid3DFloat32,
                    x: Array1DFloat64,
                    y: Array1DFloat64,
                    z: Array1DInt64,
                    nx: int = ...,
                    ny: int = ...,
                    fitting_model: str = ...,
                    boundary: str = ...,
                    bounds_error: bool = ...,
                    num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def bicubic_float32(grid: Grid4DFloat32,
                    x: Array1DFloat64,
                    y: Array1DFloat64,
                    z: Array1DFloat64,
                    u: Array1DFloat64,
                    nx: int = ...,
                    ny: int = ...,
                    fitting_model: str = ...,
                    boundary: str = ...,
                    bounds_error: bool = ...,
                    num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def bicubic_float32(grid: TemporalGrid4DFloat32,
                    x: Array1DFloat64,
                    y: Array1DFloat64,
                    z: Array1DInt64,
                    u: Array1DFloat64,
                    nx: int = ...,
                    ny: int = ...,
                    fitting_model: str = ...,
                    boundary: str = ...,
                    bounds_error: bool = ...,
                    num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def bicubic_float64(grid: Grid2DFloat64,
                    x: Array1DFloat64,
                    y: Array1DFloat64,
                    nx: int = ...,
                    ny: int = ...,
                    fitting_model: str = ...,
                    boundary: str = ...,
                    bounds_error: bool = ...,
                    num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def bicubic_float64(grid: Grid3DFloat64,
                    x: Array1DFloat64,
                    y: Array1DFloat64,
                    z: Array1DFloat64,
                    nx: int = ...,
                    ny: int = ...,
                    fitting_model: str = ...,
                    boundary: str = ...,
                    bounds_error: bool = ...,
                    num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def bicubic_float64(grid: TemporalGrid3DFloat64,
                    x: Array1DFloat64,
                    y: Array1DFloat64,
                    z: Array1DInt64,
                    nx: int = ...,
                    ny: int = ...,
                    fitting_model: str = ...,
                    boundary: str = ...,
                    bounds_error: bool = ...,
                    num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def bicubic_float64(grid: Grid4DFloat64,
                    x: Array1DFloat64,
                    y: Array1DFloat64,
                    z: Array1DFloat64,
                    u: Array1DFloat64,
                    nx: int = ...,
                    ny: int = ...,
                    fitting_model: str = ...,
                    boundary: str = ...,
                    bounds_error: bool = ...,
                    num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def bicubic_float64(grid: TemporalGrid4DFloat64,
                    x: Array1DFloat64,
                    y: Array1DFloat64,
                    z: Array1DInt64,
                    u: Array1DFloat64,
                    nx: int = ...,
                    ny: int = ...,
                    fitting_model: str = ...,
                    boundary: str = ...,
                    bounds_error: bool = ...,
                    num_threads: int = ...) -> Array1DFloat64:
    ...


def bivariate_float32(grid: Grid2DFloat32,
                      x: Array1DFloat64,
                      y: Array1DFloat64,
                      interpolator: BivariateInterpolator2D,
                      bounds_error: bool = ...,
                      num_threads: int = ...) -> Array1DFloat64:
    ...


def bivariate_float64(grid: Grid2DFloat64,
                      x: Array1DFloat64,
                      y: Array1DFloat64,
                      interpolator: BivariateInterpolator2D,
                      bounds_error: bool = ...,
                      num_threads: int = ...) -> Array1DFloat64:
    ...


def bivariate_int8(grid: Grid2DInt8,
                   x: Array1DFloat64,
                   y: Array1DFloat64,
                   interpolator: BivariateInterpolator2D,
                   bounds_error: bool = ...,
                   num_threads: int = ...) -> Array1DFloat64:
    ...


def bivariate_uint8(grid: Grid2DUInt8,
                    x: Array1DFloat64,
                    y: Array1DFloat64,
                    interpolator: BivariateInterpolator2D,
                    bounds_error: bool = ...,
                    num_threads: int = ...) -> Array1DFloat64:
    ...


def interpolate1d(x: Axis,
                  y: Array1DFloat64,
                  xi: Array1DFloat64,
                  half_window_size: int = ...,
                  bounds_error: bool = ...,
                  kind: str = ...) -> Array1DFloat64:
    ...


@overload
def quadrivariate_float32(grid: TemporalGrid4DFloat32,
                          x: Array1DFloat64,
                          y: Array1DFloat64,
                          z: Array1DInt64,
                          u: Array1DFloat64,
                          interpolator: TemporalBivariateInterpolator3D,
                          z_method: Optional[str] = ...,
                          u_method: Optional[str] = ...,
                          bounds_error: bool = ...,
                          num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def quadrivariate_float32(grid: Grid4DFloat32,
                          x: Array1DFloat64,
                          y: Array1DFloat64,
                          z: Array1DFloat64,
                          u: Array1DFloat64,
                          interpolator: BivariateInterpolator3D,
                          z_method: Optional[str] = ...,
                          u_method: Optional[str] = ...,
                          bounds_error: bool = ...,
                          num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def quadrivariate_float64(grid: TemporalGrid4DFloat64,
                          x: Array1DFloat64,
                          y: Array1DFloat64,
                          z: Array1DInt64,
                          u: Array1DFloat64,
                          interpolator: TemporalBivariateInterpolator3D,
                          z_method: Optional[str] = ...,
                          u_method: Optional[str] = ...,
                          bounds_error: bool = ...,
                          num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def quadrivariate_float64(grid: Grid4DFloat64,
                          x: Array1DFloat64,
                          y: Array1DFloat64,
                          z: Array1DFloat64,
                          u: Array1DFloat64,
                          interpolator: BivariateInterpolator3D,
                          z_method: Optional[str] = ...,
                          u_method: Optional[str] = ...,
                          bounds_error: bool = ...,
                          num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def spline_float32(grid: Grid2DFloat32,
                   x: Array1DFloat64,
                   y: Array1DFloat64,
                   nx: int = ...,
                   ny: int = ...,
                   fitting_model: str = ...,
                   boundary: str = ...,
                   bounds_error: bool = ...,
                   num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def spline_float32(grid: Grid3DFloat32,
                   x: Array1DFloat64,
                   y: Array1DFloat64,
                   z: Array1DFloat64,
                   nx: int = ...,
                   ny: int = ...,
                   fitting_model: str = ...,
                   boundary: str = ...,
                   bounds_error: bool = ...,
                   num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def spline_float32(grid: TemporalGrid3DFloat32,
                   x: Array1DFloat64,
                   y: Array1DFloat64,
                   z: Array1DInt64,
                   nx: int = ...,
                   ny: int = ...,
                   fitting_model: str = ...,
                   boundary: str = ...,
                   bounds_error: bool = ...,
                   num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def spline_float32(grid: Grid4DFloat32,
                   x: Array1DFloat64,
                   y: Array1DFloat64,
                   z: Array1DFloat64,
                   u: Array1DFloat64,
                   nx: int = ...,
                   ny: int = ...,
                   fitting_model: str = ...,
                   boundary: str = ...,
                   bounds_error: bool = ...,
                   num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def spline_float32(grid: TemporalGrid4DFloat32,
                   x: Array1DFloat64,
                   y: Array1DFloat64,
                   z: Array1DInt64,
                   u: Array1DFloat64,
                   nx: int = ...,
                   ny: int = ...,
                   fitting_model: str = ...,
                   boundary: str = ...,
                   bounds_error: bool = ...,
                   num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def spline_float64(grid: Grid2DFloat64,
                   x: Array1DFloat64,
                   y: Array1DFloat64,
                   nx: int = ...,
                   ny: int = ...,
                   fitting_model: str = ...,
                   boundary: str = ...,
                   bounds_error: bool = ...,
                   num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def spline_float64(grid: Grid3DFloat64,
                   x: Array1DFloat64,
                   y: Array1DFloat64,
                   z: Array1DFloat64,
                   nx: int = ...,
                   ny: int = ...,
                   fitting_model: str = ...,
                   boundary: str = ...,
                   bounds_error: bool = ...,
                   num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def spline_float64(grid: TemporalGrid3DFloat64,
                   x: Array1DFloat64,
                   y: Array1DFloat64,
                   z: Array1DInt64,
                   nx: int = ...,
                   ny: int = ...,
                   fitting_model: str = ...,
                   boundary: str = ...,
                   bounds_error: bool = ...,
                   num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def spline_float64(grid: Grid4DFloat64,
                   x: Array1DFloat64,
                   y: Array1DFloat64,
                   z: Array1DFloat64,
                   u: Array1DFloat64,
                   nx: int = ...,
                   ny: int = ...,
                   fitting_model: str = ...,
                   boundary: str = ...,
                   bounds_error: bool = ...,
                   num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def spline_float64(grid: TemporalGrid4DFloat64,
                   x: Array1DFloat64,
                   y: Array1DFloat64,
                   z: Array1DInt64,
                   u: Array1DFloat64,
                   nx: int = ...,
                   ny: int = ...,
                   fitting_model: str = ...,
                   boundary: str = ...,
                   bounds_error: bool = ...,
                   num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def trivariate_float32(grid: Grid3DFloat32,
                       x: Array1DFloat64,
                       y: Array1DFloat64,
                       z: Array1DFloat64,
                       interpolator: BivariateInterpolator3D,
                       z_method: Optional[str] = ...,
                       bounds_error: bool = ...,
                       num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def trivariate_float32(grid: TemporalGrid3DFloat32,
                       x: Array1DFloat64,
                       y: Array1DFloat64,
                       z: Array1DInt64,
                       interpolator: TemporalBivariateInterpolator3D,
                       z_method: Optional[str] = ...,
                       bounds_error: bool = ...,
                       num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def trivariate_float64(grid: Grid3DFloat64,
                       x: Array1DFloat64,
                       y: Array1DFloat64,
                       z: Array1DFloat64,
                       interpolator: BivariateInterpolator3D,
                       z_method: Optional[str] = ...,
                       bounds_error: bool = ...,
                       num_threads: int = ...) -> Array1DFloat64:
    ...


@overload
def trivariate_float64(grid: TemporalGrid3DFloat64,
                       x: Array1DFloat64,
                       y: Array1DFloat64,
                       z: Array1DInt64,
                       interpolator: TemporalBivariateInterpolator3D,
                       z_method: Optional[str] = ...,
                       bounds_error: bool = ...,
                       num_threads: int = ...) -> Array1DFloat64:
    ...
