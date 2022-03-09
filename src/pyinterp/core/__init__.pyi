from typing import Any, ClassVar, List, Optional, Tuple, overload

import numpy

from . import dateutils, fill, geodetic, geohash


class Axis:
    __hash__: ClassVar[None] = ...

    def __init__(self,
                 values: numpy.ndarray[numpy.float64],
                 epsilon: float = ...,
                 is_circle: bool = ...) -> None:
        ...

    def back(self) -> float:
        ...

    def find_index(self,
                   coordinates: numpy.ndarray[numpy.float64],
                   bounded: bool = ...) -> numpy.ndarray[numpy.int64]:
        ...

    def find_indexes(
        self, coordinates: numpy.ndarray[numpy.float64]
    ) -> numpy.ndarray[numpy.int64]:
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

    def __eq__(self, other: Axis) -> bool:
        ...

    @overload
    def __getitem__(self, index: int) -> float:
        ...

    @overload
    def __getitem__(self, indices: slice) -> numpy.ndarray[numpy.float64]:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __len__(self) -> int:
        ...

    def __ne__(self, other: Axis) -> bool:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def is_circle(self) -> bool:
        ...


class AxisBoundary:
    __doc__: ClassVar[str] = ...  # read-only
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
    def __init__(self, *args, **kwargs) -> None:
        ...


class Bilinear2D(BivariateInterpolator2D):
    def __init__(self) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...


class Bilinear3D(BivariateInterpolator3D):
    def __init__(self) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...


class Binning1DFloat32(Binning2DFloat32):
    def __init__(self, x: Axis) -> None:
        ...

    def push(self,
             x: numpy.ndarray[numpy.float32],
             z: numpy.ndarray[numpy.float32],
             weights: Optional[numpy.ndarray[numpy.float32]] = ...) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...


class Binning1DFloat64(Binning2DFloat64):
    def __init__(self, x: Axis) -> None:
        ...

    def push(self,
             x: numpy.ndarray[numpy.float64],
             z: numpy.ndarray[numpy.float64],
             weights: Optional[numpy.ndarray[numpy.float64]] = ...) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...


class Binning2DFloat32:
    def __init__(self,
                 x: Axis,
                 y: Axis,
                 wgs: Optional[geodetic.System] = ...) -> None:
        ...

    def clear(self) -> None:
        ...

    def count(self) -> numpy.ndarray[numpy.uint64]:
        ...

    def kurtosis(self) -> numpy.ndarray[numpy.float32]:
        ...

    def max(self) -> numpy.ndarray[numpy.float32]:
        ...

    def mean(self) -> numpy.ndarray[numpy.float32]:
        ...

    def min(self) -> numpy.ndarray[numpy.float32]:
        ...

    def push(self,
             x: numpy.ndarray[numpy.float32],
             y: numpy.ndarray[numpy.float32],
             z: numpy.ndarray[numpy.float32],
             simple: bool = ...) -> None:
        ...

    def skewness(self) -> numpy.ndarray[numpy.float32]:
        ...

    def sum(self) -> numpy.ndarray[numpy.float32]:
        ...

    def sum_of_weights(self) -> numpy.ndarray[numpy.float32]:
        ...

    def variance(self, ddof: int = ...) -> numpy.ndarray[numpy.float32]:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iadd__(self, other: Binning2DFloat32) -> Binning2DFloat32:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def wgs(self) -> Optional[geodetic.System]:
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
                 wgs: Optional[geodetic.System] = ...) -> None:
        ...

    def clear(self) -> None:
        ...

    def count(self) -> numpy.ndarray[numpy.uint64]:
        ...

    def kurtosis(self) -> numpy.ndarray[numpy.float64]:
        ...

    def max(self) -> numpy.ndarray[numpy.float64]:
        ...

    def mean(self) -> numpy.ndarray[numpy.float64]:
        ...

    def min(self) -> numpy.ndarray[numpy.float64]:
        ...

    def push(self,
             x: numpy.ndarray[numpy.float64],
             y: numpy.ndarray[numpy.float64],
             z: numpy.ndarray[numpy.float64],
             simple: bool = ...) -> None:
        ...

    def skewness(self) -> numpy.ndarray[numpy.float64]:
        ...

    def sum(self) -> numpy.ndarray[numpy.float64]:
        ...

    def sum_of_weights(self) -> numpy.ndarray[numpy.float64]:
        ...

    def variance(self, ddof: int = ...) -> numpy.ndarray[numpy.float64]:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iadd__(self, other: Binning2DFloat64) -> Binning2DFloat64:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def wgs(self) -> Optional[geodetic.System]:
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


class DescriptiveStatisticsFloat32:
    def __init__(self,
                 values: numpy.ndarray[numpy.float32],
                 weights: Optional[numpy.ndarray[numpy.float32]] = ...,
                 axis: Optional[List[int]] = ...) -> None:
        ...

    def copy(self) -> DescriptiveStatisticsFloat32:
        ...

    def count(self) -> numpy.ndarray[numpy.uint64]:
        ...

    def kurtosis(self) -> numpy.ndarray[numpy.float32]:
        ...

    def max(self) -> numpy.ndarray[numpy.float32]:
        ...

    def mean(self) -> numpy.ndarray[numpy.float32]:
        ...

    def min(self) -> numpy.ndarray[numpy.float32]:
        ...

    def skewness(self) -> numpy.ndarray[numpy.float32]:
        ...

    def sum(self) -> numpy.ndarray[numpy.float32]:
        ...

    def sum_of_weights(self) -> numpy.ndarray[numpy.float32]:
        ...

    def variance(self, ddof: int = ...) -> numpy.ndarray[numpy.float32]:
        ...

    def __add__(
            self, other: DescriptiveStatisticsFloat32
    ) -> DescriptiveStatisticsFloat32:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iadd__(
            self, other: DescriptiveStatisticsFloat32
    ) -> DescriptiveStatisticsFloat32:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...


class DescriptiveStatisticsFloat64:
    def __init__(self,
                 values: numpy.ndarray[numpy.float64],
                 weights: Optional[numpy.ndarray[numpy.float64]] = ...,
                 axis: Optional[List[int]] = ...) -> None:
        ...

    def copy(self) -> DescriptiveStatisticsFloat64:
        ...

    def count(self) -> numpy.ndarray[numpy.uint64]:
        ...

    def kurtosis(self) -> numpy.ndarray[numpy.float64]:
        ...

    def max(self) -> numpy.ndarray[numpy.float64]:
        ...

    def mean(self) -> numpy.ndarray[numpy.float64]:
        ...

    def min(self) -> numpy.ndarray[numpy.float64]:
        ...

    def skewness(self) -> numpy.ndarray[numpy.float64]:
        ...

    def sum(self) -> numpy.ndarray[numpy.float64]:
        ...

    def sum_of_weights(self) -> numpy.ndarray[numpy.float64]:
        ...

    def variance(self, ddof: int = ...) -> numpy.ndarray[numpy.float64]:
        ...

    def __add__(
            self, other: DescriptiveStatisticsFloat64
    ) -> DescriptiveStatisticsFloat64:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iadd__(
            self, other: DescriptiveStatisticsFloat64
    ) -> DescriptiveStatisticsFloat64:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...


class GeoHash:
    def __init__(self,
                 longitude: float,
                 latitude: float,
                 precision: int = ...) -> None:
        ...

    def area(self, wgs: Optional[geodetic.System] = ...) -> float:
        ...

    def bounding_box(self) -> geodetic.Box:
        ...

    def center(self) -> geodetic.Point:
        ...

    @staticmethod
    def error_with_precision(precision: int) -> Tuple[float, float]:
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
    def __init__(self, x: Axis, y: Axis,
                 array: numpy.ndarray[numpy.float32]) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def array(self) -> numpy.ndarray[numpy.float32]:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...


class Grid2DFloat64:
    def __init__(self, x: Axis, y: Axis,
                 array: numpy.ndarray[numpy.float64]) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def array(self) -> numpy.ndarray[numpy.float64]:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...


class Grid2DInt8:
    def __init__(self, x: Axis, y: Axis,
                 array: numpy.ndarray[numpy.int8]) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def array(self) -> numpy.ndarray[numpy.int8]:
        ...

    @property
    def x(self) -> Axis:
        ...

    @property
    def y(self) -> Axis:
        ...


class Grid3DFloat32:
    def __init__(self, x: Axis, y: Axis, z: Axis,
                 array: numpy.ndarray[numpy.float32]) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def array(self) -> numpy.ndarray[numpy.float32]:
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
                 array: numpy.ndarray[numpy.float64]) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def array(self) -> numpy.ndarray[numpy.float64]:
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
    def __init__(self, x: Axis, y: Axis, z: Axis,
                 array: numpy.ndarray[numpy.int8]) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def array(self) -> numpy.ndarray[numpy.int8]:
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
                 array: numpy.ndarray[numpy.float32]) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def array(self) -> numpy.ndarray[numpy.float32]:
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
                 array: numpy.ndarray[numpy.float64]) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def array(self) -> numpy.ndarray[numpy.float64]:
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
                 array: numpy.ndarray[numpy.int8]) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def array(self) -> numpy.ndarray[numpy.int8]:
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

    def count(self) -> numpy.ndarray[numpy.uint64]:
        ...

    def histograms(self, *args, **kwargs) -> Any:
        ...

    def kurtosis(self) -> numpy.ndarray[numpy.float32]:
        ...

    def max(self) -> numpy.ndarray[numpy.float32]:
        ...

    def mean(self) -> numpy.ndarray[numpy.float32]:
        ...

    def min(self) -> numpy.ndarray[numpy.float32]:
        ...

    def push(self, x: numpy.ndarray[numpy.float32],
             y: numpy.ndarray[numpy.float32],
             z: numpy.ndarray[numpy.float32]) -> None:
        ...

    def quantile(self, q: float = ...) -> numpy.ndarray[numpy.float32]:
        ...

    def skewness(self) -> numpy.ndarray[numpy.float32]:
        ...

    def sum_of_weights(self) -> numpy.ndarray[numpy.float32]:
        ...

    def variance(self) -> numpy.ndarray[numpy.float32]:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iadd__(self, other: Histogram2DFloat32) -> Histogram2DFloat32:
        ...

    def __setstate__(self, arg0: tuple) -> None:
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

    def count(self) -> numpy.ndarray[numpy.uint64]:
        ...

    def histograms(self, *args, **kwargs) -> Any:
        ...

    def kurtosis(self) -> numpy.ndarray[numpy.float64]:
        ...

    def max(self) -> numpy.ndarray[numpy.float64]:
        ...

    def mean(self) -> numpy.ndarray[numpy.float64]:
        ...

    def min(self) -> numpy.ndarray[numpy.float64]:
        ...

    def push(self, x: numpy.ndarray[numpy.float64],
             y: numpy.ndarray[numpy.float64],
             z: numpy.ndarray[numpy.float64]) -> None:
        ...

    def quantile(self, q: float = ...) -> numpy.ndarray[numpy.float64]:
        ...

    def skewness(self) -> numpy.ndarray[numpy.float64]:
        ...

    def sum_of_weights(self) -> numpy.ndarray[numpy.float64]:
        ...

    def variance(self) -> numpy.ndarray[numpy.float64]:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iadd__(self, other: Histogram2DFloat64) -> Histogram2DFloat64:
        ...

    def __setstate__(self, arg0: tuple) -> None:
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

    def __setstate__(self, arg0: tuple) -> None:
        ...


class InverseDistanceWeighting3D(BivariateInterpolator3D):
    def __init__(self, p: int = ...) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...


class Nearest2D(BivariateInterpolator2D):
    def __init__(self) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...


class Nearest3D(BivariateInterpolator3D):
    def __init__(self) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...


class RTree3DFloat32:
    def __init__(self, system: Optional[geodetic.System] = ...) -> None:
        ...

    def bounds(self) -> tuple:
        ...

    def clear(self) -> None:
        ...

    def insert(self, coordinates: numpy.ndarray[numpy.float32],
               values: numpy.ndarray[numpy.float32]) -> None:
        ...

    def inverse_distance_weighting(self,
                                   coordinates: numpy.ndarray[numpy.float32],
                                   radius: Optional[float],
                                   k: int = ...,
                                   p: int = ...,
                                   within: bool = ...,
                                   num_threads: int = ...) -> tuple:
        ...

    def packing(self, coordinates: numpy.ndarray[numpy.float32],
                values: numpy.ndarray[numpy.float32]) -> None:
        ...

    def query(self,
              coordinates: numpy.ndarray[numpy.float32],
              k: int = ...,
              within: bool = ...,
              num_threads: int = ...) -> tuple:
        ...

    def radial_basis_function(self,
                              coordinates: numpy.ndarray[numpy.float32],
                              radius: Optional[float],
                              k: int = ...,
                              rbf: RadialBasisFunction = ...,
                              epsilon: Optional[float] = ...,
                              smooth: float = ...,
                              within: bool = ...,
                              num_threads: int = ...) -> tuple:
        ...

    def window_function(self,
                        coordinates: numpy.ndarray[numpy.float32],
                        radius: float,
                        k: int = ...,
                        wf: WindowFunction = ...,
                        arg: Optional[float] = ...,
                        within: bool = ...,
                        num_threads: int = ...) -> tuple:
        ...

    def __bool__(self) -> bool:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __len__(self) -> int:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...


class RTree3DFloat64:
    def __init__(self, system: Optional[geodetic.System] = ...) -> None:
        ...

    def bounds(self) -> tuple:
        ...

    def clear(self) -> None:
        ...

    def insert(self, coordinates: numpy.ndarray[numpy.float64],
               values: numpy.ndarray[numpy.float64]) -> None:
        ...

    def inverse_distance_weighting(self,
                                   coordinates: numpy.ndarray[numpy.float64],
                                   radius: Optional[float],
                                   k: int = ...,
                                   p: int = ...,
                                   within: bool = ...,
                                   num_threads: int = ...) -> tuple:
        ...

    def packing(self, coordinates: numpy.ndarray[numpy.float64],
                values: numpy.ndarray[numpy.float64]) -> None:
        ...

    def query(self,
              coordinates: numpy.ndarray[numpy.float64],
              k: int = ...,
              within: bool = ...,
              num_threads: int = ...) -> tuple:
        ...

    def radial_basis_function(self,
                              coordinates: numpy.ndarray[numpy.float64],
                              radius: Optional[float],
                              k: int = ...,
                              rbf: RadialBasisFunction = ...,
                              epsilon: Optional[float] = ...,
                              smooth: float = ...,
                              within: bool = ...,
                              num_threads: int = ...) -> tuple:
        ...

    def window_function(self,
                        coordinates: numpy.ndarray[numpy.float64],
                        radius: float,
                        k: int = ...,
                        wf: WindowFunction = ...,
                        arg: Optional[float] = ...,
                        within: bool = ...,
                        num_threads: int = ...) -> tuple:
        ...

    def __bool__(self) -> bool:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __len__(self) -> int:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...


class RadialBasisFunction:
    __doc__: ClassVar[str] = ...  # read-only
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
                 values: numpy.ndarray[numpy.float32],
                 weights: Optional[numpy.ndarray[numpy.float32]] = ...,
                 axis: Optional[List[int]] = ...,
                 bin_count: Optional[int] = ...) -> None:
        ...

    def bins(self, *args, **kwargs) -> Any:
        ...

    def count(self) -> numpy.ndarray[numpy.uint64]:
        ...

    def kurtosis(self) -> numpy.ndarray[numpy.float32]:
        ...

    def max(self) -> numpy.ndarray[numpy.float32]:
        ...

    def mean(self) -> numpy.ndarray[numpy.float32]:
        ...

    def min(self) -> numpy.ndarray[numpy.float32]:
        ...

    def quantile(self, q: float = ...) -> numpy.ndarray[numpy.float32]:
        ...

    def resize(self, arg0: int) -> None:
        ...

    def size(self) -> numpy.ndarray[numpy.uint64]:
        ...

    def skewness(self) -> numpy.ndarray[numpy.float32]:
        ...

    def sum_of_weights(self) -> numpy.ndarray[numpy.float32]:
        ...

    def variance(self) -> numpy.ndarray[numpy.float32]:
        ...

    def __add__(self,
                other: StreamingHistogramFloat32) -> StreamingHistogramFloat32:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iadd__(
            self,
            other: StreamingHistogramFloat32) -> StreamingHistogramFloat32:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...


class StreamingHistogramFloat64:
    def __init__(self,
                 values: numpy.ndarray[numpy.float64],
                 weights: Optional[numpy.ndarray[numpy.float64]] = ...,
                 axis: Optional[List[int]] = ...,
                 bin_count: Optional[int] = ...) -> None:
        ...

    def bins(self, *args, **kwargs) -> Any:
        ...

    def count(self) -> numpy.ndarray[numpy.uint64]:
        ...

    def kurtosis(self) -> numpy.ndarray[numpy.float64]:
        ...

    def max(self) -> numpy.ndarray[numpy.float64]:
        ...

    def mean(self) -> numpy.ndarray[numpy.float64]:
        ...

    def min(self) -> numpy.ndarray[numpy.float64]:
        ...

    def quantile(self, q: float = ...) -> numpy.ndarray[numpy.float64]:
        ...

    def resize(self, arg0: int) -> None:
        ...

    def size(self) -> numpy.ndarray[numpy.uint64]:
        ...

    def skewness(self) -> numpy.ndarray[numpy.float64]:
        ...

    def sum_of_weights(self) -> numpy.ndarray[numpy.float64]:
        ...

    def variance(self) -> numpy.ndarray[numpy.float64]:
        ...

    def __add__(self,
                other: StreamingHistogramFloat64) -> StreamingHistogramFloat64:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __iadd__(
            self,
            other: StreamingHistogramFloat64) -> StreamingHistogramFloat64:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...


class TemporalAxis(AxisInt64):
    __hash__: ClassVar[None] = ...

    def __init__(self, values: numpy.ndarray) -> None:
        ...

    def back(self) -> numpy.ndarray:
        ...

    def dtype(self) -> numpy.dtype:
        ...

    def find_index(self,
                   coordinates: numpy.ndarray,
                   bounded: bool = ...) -> numpy.ndarray[numpy.int64]:
        ...

    def find_indexes(self,
                     coordinates: numpy.ndarray) -> numpy.ndarray[numpy.int64]:
        ...

    def flip(self, inplace: bool = ...) -> TemporalAxis:
        ...

    def front(self) -> numpy.ndarray:
        ...

    def increment(self) -> numpy.ndarray:
        ...

    def is_ascending(self) -> bool:
        ...

    def is_regular(self) -> bool:
        ...

    def max_value(self) -> numpy.ndarray:
        ...

    def min_value(self) -> numpy.ndarray:
        ...

    def safe_cast(self, values: numpy.ndarray) -> numpy.ndarray:
        ...

    def __eq__(self, other: TemporalAxis) -> bool:
        ...

    @overload
    def __getitem__(self, index: int) -> numpy.ndarray:
        ...

    @overload
    def __getitem__(self, indices: slice) -> numpy.ndarray:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __len__(self) -> int:
        ...

    def __ne__(self, other: TemporalAxis) -> bool:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...


class TemporalBilinear3D(TemporalBivariateInterpolator3D):
    def __init__(self) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...


class TemporalBivariateInterpolator3D:
    def __init__(self, *args, **kwargs) -> None:
        ...


class TemporalGrid3DFloat32:
    def __init__(self, x: Axis, y: Axis, z: AxisInt64,
                 array: numpy.ndarray[numpy.float32]) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def array(self) -> numpy.ndarray[numpy.float32]:
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
                 array: numpy.ndarray[numpy.float64]) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def array(self) -> numpy.ndarray[numpy.float64]:
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
                 array: numpy.ndarray[numpy.int8]) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def array(self) -> numpy.ndarray[numpy.int8]:
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
                 array: numpy.ndarray[numpy.float32]) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def array(self) -> numpy.ndarray[numpy.float32]:
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
                 array: numpy.ndarray[numpy.float64]) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def array(self) -> numpy.ndarray[numpy.float64]:
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
                 array: numpy.ndarray[numpy.int8]) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def array(self) -> numpy.ndarray[numpy.int8]:
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

    def __setstate__(self, arg0: tuple) -> None:
        ...


class TemporalNearest3D(TemporalBivariateInterpolator3D):
    def __init__(self) -> None:
        ...

    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...


class WindowFunction:
    __doc__: ClassVar[str] = ...  # read-only
    __members__: ClassVar[dict] = ...  # read-only
    Blackman: ClassVar[WindowFunction] = ...
    BlackmanHarris: ClassVar[WindowFunction] = ...
    Boxcar: ClassVar[WindowFunction] = ...
    FlatTop: ClassVar[WindowFunction] = ...
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
                    x: numpy.ndarray[numpy.float64],
                    y: numpy.ndarray[numpy.float64],
                    nx: int = ...,
                    ny: int = ...,
                    fitting_model: str = ...,
                    boundary: str = ...,
                    bounds_error: bool = ...,
                    num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def bicubic_float32(grid: Grid3DFloat32,
                    x: numpy.ndarray[numpy.float64],
                    y: numpy.ndarray[numpy.float64],
                    z: numpy.ndarray[numpy.float64],
                    nx: int = ...,
                    ny: int = ...,
                    fitting_model: str = ...,
                    boundary: str = ...,
                    bounds_error: bool = ...,
                    num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def bicubic_float32(grid: TemporalGrid3DFloat32,
                    x: numpy.ndarray[numpy.float64],
                    y: numpy.ndarray[numpy.float64],
                    z: numpy.ndarray[numpy.int64],
                    nx: int = ...,
                    ny: int = ...,
                    fitting_model: str = ...,
                    boundary: str = ...,
                    bounds_error: bool = ...,
                    num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def bicubic_float32(grid: Grid4DFloat32,
                    x: numpy.ndarray[numpy.float64],
                    y: numpy.ndarray[numpy.float64],
                    z: numpy.ndarray[numpy.float64],
                    u: numpy.ndarray[numpy.float64],
                    nx: int = ...,
                    ny: int = ...,
                    fitting_model: str = ...,
                    boundary: str = ...,
                    bounds_error: bool = ...,
                    num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def bicubic_float32(grid: TemporalGrid4DFloat32,
                    x: numpy.ndarray[numpy.float64],
                    y: numpy.ndarray[numpy.float64],
                    z: numpy.ndarray[numpy.int64],
                    u: numpy.ndarray[numpy.float64],
                    nx: int = ...,
                    ny: int = ...,
                    fitting_model: str = ...,
                    boundary: str = ...,
                    bounds_error: bool = ...,
                    num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def bicubic_float64(grid: Grid2DFloat64,
                    x: numpy.ndarray[numpy.float64],
                    y: numpy.ndarray[numpy.float64],
                    nx: int = ...,
                    ny: int = ...,
                    fitting_model: str = ...,
                    boundary: str = ...,
                    bounds_error: bool = ...,
                    num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def bicubic_float64(grid: Grid3DFloat64,
                    x: numpy.ndarray[numpy.float64],
                    y: numpy.ndarray[numpy.float64],
                    z: numpy.ndarray[numpy.float64],
                    nx: int = ...,
                    ny: int = ...,
                    fitting_model: str = ...,
                    boundary: str = ...,
                    bounds_error: bool = ...,
                    num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def bicubic_float64(grid: TemporalGrid3DFloat64,
                    x: numpy.ndarray[numpy.float64],
                    y: numpy.ndarray[numpy.float64],
                    z: numpy.ndarray[numpy.int64],
                    nx: int = ...,
                    ny: int = ...,
                    fitting_model: str = ...,
                    boundary: str = ...,
                    bounds_error: bool = ...,
                    num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def bicubic_float64(grid: Grid4DFloat64,
                    x: numpy.ndarray[numpy.float64],
                    y: numpy.ndarray[numpy.float64],
                    z: numpy.ndarray[numpy.float64],
                    u: numpy.ndarray[numpy.float64],
                    nx: int = ...,
                    ny: int = ...,
                    fitting_model: str = ...,
                    boundary: str = ...,
                    bounds_error: bool = ...,
                    num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def bicubic_float64(grid: TemporalGrid4DFloat64,
                    x: numpy.ndarray[numpy.float64],
                    y: numpy.ndarray[numpy.float64],
                    z: numpy.ndarray[numpy.int64],
                    u: numpy.ndarray[numpy.float64],
                    nx: int = ...,
                    ny: int = ...,
                    fitting_model: str = ...,
                    boundary: str = ...,
                    bounds_error: bool = ...,
                    num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


def bivariate_float32(grid: Grid2DFloat32,
                      x: numpy.ndarray[numpy.float64],
                      y: numpy.ndarray[numpy.float64],
                      interpolator: BivariateInterpolator2D,
                      bounds_error: bool = ...,
                      num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


def bivariate_float64(grid: Grid2DFloat64,
                      x: numpy.ndarray[numpy.float64],
                      y: numpy.ndarray[numpy.float64],
                      interpolator: BivariateInterpolator2D,
                      bounds_error: bool = ...,
                      num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


def bivariate_int8(grid: Grid2DInt8,
                   x: numpy.ndarray[numpy.float64],
                   y: numpy.ndarray[numpy.float64],
                   interpolator: BivariateInterpolator2D,
                   bounds_error: bool = ...,
                   num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def quadrivariate_float32(
        grid: TemporalGrid4DFloat32,
        x: numpy.ndarray[numpy.float64],
        y: numpy.ndarray[numpy.float64],
        z: numpy.ndarray[numpy.int64],
        u: numpy.ndarray[numpy.float64],
        interpolator: TemporalBivariateInterpolator3D,
        z_method: Optional[str] = ...,
        u_method: Optional[str] = ...,
        bounds_error: bool = ...,
        num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def quadrivariate_float64(
        grid: Grid4DFloat64,
        x: numpy.ndarray[numpy.float64],
        y: numpy.ndarray[numpy.float64],
        z: numpy.ndarray[numpy.float64],
        u: numpy.ndarray[numpy.float64],
        interpolator: BivariateInterpolator3D,
        z_method: Optional[str] = ...,
        u_method: Optional[str] = ...,
        bounds_error: bool = ...,
        num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def spline_float32(grid: Grid2DFloat32,
                   x: numpy.ndarray[numpy.float64],
                   y: numpy.ndarray[numpy.float64],
                   nx: int = ...,
                   ny: int = ...,
                   fitting_model: str = ...,
                   boundary: str = ...,
                   bounds_error: bool = ...,
                   num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def spline_float32(grid: Grid3DFloat32,
                   x: numpy.ndarray[numpy.float64],
                   y: numpy.ndarray[numpy.float64],
                   z: numpy.ndarray[numpy.float64],
                   nx: int = ...,
                   ny: int = ...,
                   fitting_model: str = ...,
                   boundary: str = ...,
                   bounds_error: bool = ...,
                   num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def spline_float32(grid: TemporalGrid3DFloat32,
                   x: numpy.ndarray[numpy.float64],
                   y: numpy.ndarray[numpy.float64],
                   z: numpy.ndarray[numpy.int64],
                   nx: int = ...,
                   ny: int = ...,
                   fitting_model: str = ...,
                   boundary: str = ...,
                   bounds_error: bool = ...,
                   num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def spline_float32(grid: Grid4DFloat32,
                   x: numpy.ndarray[numpy.float64],
                   y: numpy.ndarray[numpy.float64],
                   z: numpy.ndarray[numpy.float64],
                   u: numpy.ndarray[numpy.float64],
                   nx: int = ...,
                   ny: int = ...,
                   fitting_model: str = ...,
                   boundary: str = ...,
                   bounds_error: bool = ...,
                   num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def spline_float32(grid: TemporalGrid4DFloat32,
                   x: numpy.ndarray[numpy.float64],
                   y: numpy.ndarray[numpy.float64],
                   z: numpy.ndarray[numpy.int64],
                   u: numpy.ndarray[numpy.float64],
                   nx: int = ...,
                   ny: int = ...,
                   fitting_model: str = ...,
                   boundary: str = ...,
                   bounds_error: bool = ...,
                   num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def spline_float64(grid: Grid2DFloat64,
                   x: numpy.ndarray[numpy.float64],
                   y: numpy.ndarray[numpy.float64],
                   nx: int = ...,
                   ny: int = ...,
                   fitting_model: str = ...,
                   boundary: str = ...,
                   bounds_error: bool = ...,
                   num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def spline_float64(grid: Grid3DFloat64,
                   x: numpy.ndarray[numpy.float64],
                   y: numpy.ndarray[numpy.float64],
                   z: numpy.ndarray[numpy.float64],
                   nx: int = ...,
                   ny: int = ...,
                   fitting_model: str = ...,
                   boundary: str = ...,
                   bounds_error: bool = ...,
                   num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def spline_float64(grid: TemporalGrid3DFloat64,
                   x: numpy.ndarray[numpy.float64],
                   y: numpy.ndarray[numpy.float64],
                   z: numpy.ndarray[numpy.int64],
                   nx: int = ...,
                   ny: int = ...,
                   fitting_model: str = ...,
                   boundary: str = ...,
                   bounds_error: bool = ...,
                   num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def spline_float64(grid: Grid4DFloat64,
                   x: numpy.ndarray[numpy.float64],
                   y: numpy.ndarray[numpy.float64],
                   z: numpy.ndarray[numpy.float64],
                   u: numpy.ndarray[numpy.float64],
                   nx: int = ...,
                   ny: int = ...,
                   fitting_model: str = ...,
                   boundary: str = ...,
                   bounds_error: bool = ...,
                   num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def spline_float64(grid: TemporalGrid4DFloat64,
                   x: numpy.ndarray[numpy.float64],
                   y: numpy.ndarray[numpy.float64],
                   z: numpy.ndarray[numpy.int64],
                   u: numpy.ndarray[numpy.float64],
                   nx: int = ...,
                   ny: int = ...,
                   fitting_model: str = ...,
                   boundary: str = ...,
                   bounds_error: bool = ...,
                   num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def trivariate_float32(grid: Grid3DFloat32,
                       x: numpy.ndarray[numpy.float64],
                       y: numpy.ndarray[numpy.float64],
                       z: numpy.ndarray[numpy.float64],
                       interpolator: BivariateInterpolator3D,
                       z_method: Optional[str] = ...,
                       bounds_error: bool = ...,
                       num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def trivariate_float32(grid: TemporalGrid3DFloat32,
                       x: numpy.ndarray[numpy.float64],
                       y: numpy.ndarray[numpy.float64],
                       z: numpy.ndarray[numpy.int64],
                       interpolator: TemporalBivariateInterpolator3D,
                       z_method: Optional[str] = ...,
                       bounds_error: bool = ...,
                       num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def trivariate_float64(grid: Grid3DFloat64,
                       x: numpy.ndarray[numpy.float64],
                       y: numpy.ndarray[numpy.float64],
                       z: numpy.ndarray[numpy.float64],
                       interpolator: BivariateInterpolator3D,
                       z_method: Optional[str] = ...,
                       bounds_error: bool = ...,
                       num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...


@overload
def trivariate_float64(grid: TemporalGrid3DFloat64,
                       x: numpy.ndarray[numpy.float64],
                       y: numpy.ndarray[numpy.float64],
                       z: numpy.ndarray[numpy.int64],
                       interpolator: TemporalBivariateInterpolator3D,
                       z_method: Optional[str] = ...,
                       bounds_error: bool = ...,
                       num_threads: int = ...) -> numpy.ndarray[numpy.float64]:
    ...
