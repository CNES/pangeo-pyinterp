import typing
from typing import (
    Any,
    Generic,
    Iterator,
    Literal,
    Self,
    TypeAlias,
    TypeVar,
    overload,
)

import numpy as np
from numpy.typing import NDArray

from ..type_hints import (
    FourDims,
    NDArray1DNumeric,
    NDArray1DDateTime64,
    NDArray1DFloat32,
    NDArray1DFloat64,
    NDArray1DInt64,
    NDArray1DNumericWithTime,
    NDArray1DTimeDelta64,
    NDArray1DUInt32,
    NDArray2DInt64,
    NDArray2DUInt64,
    OneDim,
    ThreeDims,
    TwoDims,
)
from . import fill, geometry, period
from .config import geometric, rtree, windowed

__all__ = [
    "Axis",
    "Binning1D",
    "Binning1DFloat32",
    "Binning1DFloat64",
    "Binning2D",
    "Binning2DFloat32",
    "Binning2DFloat64",
    "DescriptiveStatistics",
    "DescriptiveStatisticsFloat32",
    "DescriptiveStatisticsFloat64",
    "Grid",
    "Grid1D",
    "Grid2D",
    "Grid3D",
    "Grid4D",
    "Histogram2D",
    "Histogram2DFloat32",
    "Histogram2DFloat64",
    "RTree3D",
    "RTree3DFloat32",
    "RTree3DFloat64",
    "TDigest",
    "TDigestFloat32",
    "TDigestFloat64",
    "bivariate",
    "fill",
    "geometry",
    "period",
    "quadrivariate",
    "trivariate",
]

_DType = TypeVar("_DType", bound=np.generic)

# Type alias for temporal coordinate arrays
TemporalArray: TypeAlias = NDArray1DDateTime64 | NDArray1DTimeDelta64

def univariate(
    grid: GridHolder, x: NDArray1DNumeric, config: windowed.Univariate
) -> NDArray1DFloat64 | NDArray1DFloat32: ...
def univariate_derivative(
    grid: GridHolder, x: NDArray1DNumeric, config: windowed.Univariate
) -> NDArray1DFloat64 | NDArray1DFloat32: ...
def bivariate(
    grid: GridHolder,
    x: NDArray1DNumeric,
    y: NDArray1DNumeric,
    config: geometric.Bivariate | windowed.Bivariate,
) -> NDArray1DFloat32 | NDArray1DFloat64: ...
def trivariate(
    grid: GridHolder,
    x: NDArray1DNumeric,
    y: NDArray1DNumeric,
    z: NDArray1DNumericWithTime,
    config: geometric.Trivariate | windowed.Trivariate,
) -> NDArray1DFloat32 | NDArray1DFloat64: ...
def quadrivariate(
    grid: GridHolder,
    x: NDArray1DNumeric,
    y: NDArray1DNumeric,
    z: NDArray1DNumericWithTime,
    u: NDArray1DNumeric,
    config: geometric.Quadrivariate | windowed.Quadrivariate,
) -> NDArray1DFloat32 | NDArray1DFloat64: ...

class Axis:
    def __init__(
        self,
        values: NDArray1DNumeric,
        epsilon: float = ...,
        period: float | None = ...,
    ) -> None: ...
    def back(self) -> float: ...
    def find_index(
        self, coordinates: NDArray1DFloat64, bounded: bool = ...
    ) -> NDArray1DInt64: ...
    def find_indexes(
        self, coordinates: NDArray1DNumeric
    ) -> NDArray2DInt64: ...
    def flip(self, inplace: bool = ...) -> Axis: ...
    def front(self) -> float: ...
    def increment(self) -> float: ...
    def is_ascending(self) -> bool: ...
    def is_regular(self) -> bool: ...
    def max_value(self) -> float: ...
    def min_value(self) -> float: ...
    def __copy__(self) -> Axis: ...
    def __eq__(self, other: object) -> bool: ...
    @overload
    def __getitem__(self, index: int) -> float: ...
    @overload
    def __getitem__(self, axis_slice: slice) -> NDArray1DFloat64: ...
    def __iter__(self) -> Iterator[float]: ...
    def __len__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    @property
    def is_periodic(self) -> bool: ...
    @property
    def period(self) -> float | None: ...

_FloatDType = TypeVar("_FloatDType", np.float32, np.float64)

class Binning2DHolder(Generic[_FloatDType]):
    def __init__(
        self,
        x: Axis,
        y: Axis,
        spheroid: geometry.geographic.Spheroid | None = ...,
    ) -> None: ...
    def clear(self) -> None: ...
    def count(self) -> NDArray2DUInt64: ...
    def kurtosis(self) -> np.ndarray[TwoDims, np.dtype[_FloatDType]]: ...
    def max(self) -> np.ndarray[TwoDims, np.dtype[_FloatDType]]: ...
    def mean(self) -> np.ndarray[TwoDims, np.dtype[_FloatDType]]: ...
    def min(self) -> np.ndarray[TwoDims, np.dtype[_FloatDType]]: ...
    def push(
        self,
        x: np.ndarray[OneDim, np.dtype[np.floating[Any]]],
        y: np.ndarray[OneDim, np.dtype[np.floating[Any]]],
        z: np.ndarray[OneDim, np.dtype[np.floating[Any]]],
        simple: bool = ...,
    ) -> None: ...
    def skewness(self) -> np.ndarray[TwoDims, np.dtype[_FloatDType]]: ...
    def sum(self) -> np.ndarray[TwoDims, np.dtype[_FloatDType]]: ...
    def sum_of_weights(self) -> np.ndarray[TwoDims, np.dtype[_FloatDType]]: ...
    def variance(
        self, ddof: int = ...
    ) -> np.ndarray[TwoDims, np.dtype[_FloatDType]]: ...
    def __copy__(self) -> Binning2DHolder[_FloatDType]: ...
    def __iadd__(self, other: Binning2DHolder[Any]) -> Self: ...
    @property
    def spheroid(self) -> geometry.geographic.Spheroid | None: ...
    @property
    def x(self) -> Axis: ...
    @property
    def y(self) -> Axis: ...

class Binning1DHolder(Generic[_FloatDType]):
    def __init__(
        self, x: Axis, range: tuple[float, float] | None = ...
    ) -> None: ...
    def clear(self) -> None: ...
    def push(
        self,
        x: np.ndarray[OneDim, np.dtype[np.floating[Any]]],
        z: np.ndarray[OneDim, np.dtype[np.floating[Any]]],
        weights: np.ndarray[OneDim, np.dtype[np.floating[Any]]] | None = ...,
    ) -> None: ...
    def count(self) -> np.ndarray[OneDim, np.dtype[np.uint64]]: ...
    def kurtosis(self) -> np.ndarray[OneDim, np.dtype[_FloatDType]]: ...
    def max(self) -> np.ndarray[OneDim, np.dtype[_FloatDType]]: ...
    def mean(self) -> np.ndarray[OneDim, np.dtype[_FloatDType]]: ...
    def min(self) -> np.ndarray[OneDim, np.dtype[_FloatDType]]: ...
    def skewness(self) -> np.ndarray[OneDim, np.dtype[_FloatDType]]: ...
    def sum(self) -> np.ndarray[OneDim, np.dtype[_FloatDType]]: ...
    def sum_of_weights(self) -> np.ndarray[OneDim, np.dtype[_FloatDType]]: ...
    def variance(
        self, ddof: int = ...
    ) -> np.ndarray[OneDim, np.dtype[_FloatDType]]: ...
    def range(self) -> tuple[float, float]: ...
    def __copy__(self) -> Binning1DHolder[_FloatDType]: ...
    def __iadd__(self, other: Binning1DHolder[Any]) -> Self: ...
    @property
    def x(self) -> Axis: ...
    @property
    def y(self) -> Axis: ...
    @property
    def spheroid(self) -> geometry.geographic.Spheroid | None: ...

@overload
def Binning2D(
    x: Axis,
    y: Axis,
    spheroid: geometry.geographic.Spheroid | None = None,
    *,
    dtype: None = None,
) -> Binning2DHolder[np.float64]: ...
@overload
def Binning2D(
    x: Axis,
    y: Axis,
    spheroid: geometry.geographic.Spheroid | None = None,
    *,
    dtype: Literal["float32"],
) -> Binning2DHolder[np.float32]: ...
@overload
def Binning2D(
    x: Axis,
    y: Axis,
    spheroid: geometry.geographic.Spheroid | None = None,
    *,
    dtype: Literal["float64"],
) -> Binning2DHolder[np.float64]: ...
@overload
def Binning2D(
    x: Axis,
    y: Axis,
    spheroid: geometry.geographic.Spheroid | None = None,
    *,
    dtype: type[_FloatDType] | np.dtype[_FloatDType],
) -> Binning2DHolder[_FloatDType]: ...
@overload
def Binning2D(
    x: Axis,
    y: Axis,
    spheroid: geometry.geographic.Spheroid | None = None,
    *,
    dtype: str,
) -> Binning2DHolder[np.float32] | Binning2DHolder[np.float64]: ...
@overload
def Binning1D(
    x: Axis,
    range: tuple[float, float] | None = None,
    *,
    dtype: None = None,
) -> Binning1DHolder[np.float64]: ...
@overload
def Binning1D(
    x: Axis,
    range: tuple[float, float] | None = None,
    *,
    dtype: Literal["float32"],
) -> Binning1DHolder[np.float32]: ...
@overload
def Binning1D(
    x: Axis,
    range: tuple[float, float] | None = None,
    *,
    dtype: Literal["float64"],
) -> Binning1DHolder[np.float64]: ...
@overload
def Binning1D(
    x: Axis,
    range: tuple[float, float] | None = None,
    *,
    dtype: type[_FloatDType] | np.dtype[_FloatDType],
) -> Binning1DHolder[_FloatDType]: ...
@overload
def Binning1D(
    x: Axis,
    range: tuple[float, float] | None = None,
    *,
    dtype: str,
) -> Binning1DHolder[np.float32] | Binning1DHolder[np.float64]: ...

Binning2DFloat32: TypeAlias = Binning2DHolder[np.float32]
Binning2DFloat64: TypeAlias = Binning2DHolder[np.float64]
Binning1DFloat32: TypeAlias = Binning1DHolder[np.float32]
Binning1DFloat64: TypeAlias = Binning1DHolder[np.float64]

class DescriptiveStatisticsHolder(Generic[_FloatDType]):
    def __init__(
        self,
        values: np.ndarray[Any, np.dtype[np.floating[Any]]],
        weights: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = ...,
        axis: list[int] | None = ...,
    ) -> None: ...
    def count(self) -> np.ndarray[Any, np.dtype[np.uint64]]: ...
    def min(self) -> np.ndarray[Any, np.dtype[_FloatDType]]: ...
    def max(self) -> np.ndarray[Any, np.dtype[_FloatDType]]: ...
    def mean(self) -> np.ndarray[Any, np.dtype[_FloatDType]]: ...
    def variance(
        self,
        ddof: int = ...,
    ) -> np.ndarray[Any, np.dtype[_FloatDType]]: ...
    def skewness(self) -> np.ndarray[Any, np.dtype[_FloatDType]]: ...
    def kurtosis(self) -> np.ndarray[Any, np.dtype[_FloatDType]]: ...
    def sum(self) -> np.ndarray[Any, np.dtype[_FloatDType]]: ...
    def sum_of_weights(self) -> np.ndarray[Any, np.dtype[_FloatDType]]: ...
    def __copy__(self) -> DescriptiveStatisticsHolder[_FloatDType]: ...
    def __iadd__(self, other: DescriptiveStatisticsHolder[Any]) -> Self: ...

@overload
def DescriptiveStatistics(
    values: np.ndarray[Any, np.dtype[np.floating[Any]]],
    weights: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
    axis: list[int] | None = None,
    *,
    dtype: None = None,
) -> DescriptiveStatisticsHolder[np.float64]: ...
@overload
def DescriptiveStatistics(
    values: np.ndarray[Any, np.dtype[np.floating[Any]]],
    weights: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
    axis: list[int] | None = None,
    *,
    dtype: Literal["float32"],
) -> DescriptiveStatisticsHolder[np.float32]: ...
@overload
def DescriptiveStatistics(
    values: np.ndarray[Any, np.dtype[np.floating[Any]]],
    weights: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
    axis: list[int] | None = None,
    *,
    dtype: Literal["float64"],
) -> DescriptiveStatisticsHolder[np.float64]: ...
@overload
def DescriptiveStatistics(
    values: np.ndarray[Any, np.dtype[np.floating[Any]]],
    weights: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
    axis: list[int] | None = None,
    *,
    dtype: type[_FloatDType] | np.dtype[_FloatDType],
) -> DescriptiveStatisticsHolder[_FloatDType]: ...
@overload
def DescriptiveStatistics(
    values: np.ndarray[Any, np.dtype[np.floating[Any]]],
    weights: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
    axis: list[int] | None = None,
    *,
    dtype: str,
) -> (
    DescriptiveStatisticsHolder[np.float32]
    | DescriptiveStatisticsHolder[np.float64]
): ...

DescriptiveStatisticsFloat32: TypeAlias = DescriptiveStatisticsHolder[
    np.float32
]
DescriptiveStatisticsFloat64: TypeAlias = DescriptiveStatisticsHolder[
    np.float64
]

class TDigestHolder(Generic[_FloatDType]):
    def __init__(
        self,
        values: np.ndarray[Any, np.dtype[np.floating[Any]]],
        weights: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = ...,
        axis: list[int] | None = ...,
        compression: int = ...,
    ) -> None: ...
    def count(self) -> np.ndarray[Any, np.dtype[np.uint64]]: ...
    def min(self) -> np.ndarray[Any, np.dtype[_FloatDType]]: ...
    def max(self) -> np.ndarray[Any, np.dtype[_FloatDType]]: ...
    def mean(self) -> np.ndarray[Any, np.dtype[_FloatDType]]: ...
    def sum_of_weights(self) -> np.ndarray[Any, np.dtype[_FloatDType]]: ...
    @overload
    def quantile(self, q: float) -> np.ndarray[Any, np.dtype[_FloatDType]]: ...
    @overload
    def quantile(
        self, quantiles: np.ndarray[OneDim, np.dtype[np.floating[Any]]]
    ) -> np.ndarray[Any, np.dtype[_FloatDType]]: ...
    def __copy__(self) -> TDigestHolder[_FloatDType]: ...
    def __iadd__(self, other: TDigestHolder[Any]) -> Self: ...

@overload
def TDigest(
    values: np.ndarray[Any, np.dtype[np.floating[Any]]],
    weights: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
    axis: list[int] | None = None,
    compression: int = 100,
    *,
    dtype: None = None,
) -> TDigestHolder[np.float64]: ...
@overload
def TDigest(
    values: np.ndarray[Any, np.dtype[np.floating[Any]]],
    weights: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
    axis: list[int] | None = None,
    compression: int = 100,
    *,
    dtype: Literal["float32"],
) -> TDigestHolder[np.float32]: ...
@overload
def TDigest(
    values: np.ndarray[Any, np.dtype[np.floating[Any]]],
    weights: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
    axis: list[int] | None = None,
    compression: int = 100,
    *,
    dtype: Literal["float64"],
) -> TDigestHolder[np.float64]: ...
@overload
def TDigest(
    values: np.ndarray[Any, np.dtype[np.floating[Any]]],
    weights: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
    axis: list[int] | None = None,
    compression: int = 100,
    *,
    dtype: type[_FloatDType] | np.dtype[_FloatDType],
) -> TDigestHolder[_FloatDType]: ...
@overload
def TDigest(
    values: np.ndarray[Any, np.dtype[np.floating[Any]]],
    weights: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
    axis: list[int] | None = None,
    compression: int = 100,
    *,
    dtype: str,
) -> TDigestHolder[np.float32] | TDigestHolder[np.float64]: ...

TDigestFloat32: TypeAlias = TDigestHolder[np.float32]
TDigestFloat64: TypeAlias = TDigestHolder[np.float64]

class GridHolder(Generic[_DType]):
    def __getstate__(self) -> tuple: ...
    @property
    def array(self) -> NDArray[_DType]: ...
    @property
    def dtype(self) -> np.dtype[_DType]: ...
    @property
    def has_temporal_axis(self) -> bool: ...
    @property
    def ndim(self) -> int: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def temporal_axis_index(self) -> int: ...
    @property
    def x(self) -> Axis: ...
    @property
    def y(self) -> Axis | None: ...
    @property
    def z(self) -> Axis | TemporalAxis | None: ...
    @property
    def u(self) -> Axis | None: ...

class Grid1D(GridHolder[_DType]):
    @property
    def array(self) -> np.ndarray[OneDim, np.dtype[_DType]]: ...
    @property
    def shape(self) -> OneDim: ...
    @property
    def x(self) -> Axis: ...
    @property
    def y(self) -> None: ...
    @property
    def z(self) -> None: ...
    @property
    def u(self) -> None: ...

class Grid2D(GridHolder[_DType]):
    @property
    def array(self) -> np.ndarray[TwoDims, np.dtype[_DType]]: ...
    @property
    def shape(self) -> TwoDims: ...
    @property
    def x(self) -> Axis: ...
    @property
    def y(self) -> Axis: ...
    @property
    def z(self) -> None: ...
    @property
    def u(self) -> None: ...

class Grid3D(GridHolder[_DType]):
    @property
    def array(self) -> np.ndarray[ThreeDims, np.dtype[_DType]]: ...
    @property
    def shape(self) -> ThreeDims: ...
    @property
    def x(self) -> Axis: ...
    @property
    def y(self) -> Axis: ...
    @property
    def z(self) -> Axis: ...
    @property
    def u(self) -> None: ...

class TemporalGrid3D(GridHolder[_DType]):
    @property
    def array(self) -> np.ndarray[ThreeDims, np.dtype[_DType]]: ...
    @property
    def shape(self) -> ThreeDims: ...
    @property
    def x(self) -> Axis: ...
    @property
    def y(self) -> Axis: ...
    @property
    def z(self) -> TemporalAxis: ...
    @property
    def u(self) -> None: ...

class Grid4D(GridHolder[_DType]):
    @property
    def array(self) -> np.ndarray[FourDims, np.dtype[_DType]]: ...
    @property
    def shape(self) -> FourDims: ...
    @property
    def x(self) -> Axis: ...
    @property
    def y(self) -> Axis: ...
    @property
    def z(self) -> Axis: ...
    @property
    def u(self) -> Axis: ...

class Histogram2DHolder(Generic[_FloatDType]):
    def __init__(
        self,
        x: Axis,
        y: Axis,
        compression: int | None = ...,
    ) -> None: ...
    def clear(self) -> None: ...
    def count(self) -> NDArray2DInt64: ...
    def max(self) -> np.ndarray[TwoDims, np.dtype[_FloatDType]]: ...
    def mean(self) -> np.ndarray[TwoDims, np.dtype[_FloatDType]]: ...
    def min(self) -> np.ndarray[TwoDims, np.dtype[_FloatDType]]: ...
    def push(
        self,
        x: np.ndarray[OneDim, np.dtype[np.floating[Any]]],
        y: np.ndarray[OneDim, np.dtype[np.floating[Any]]],
        z: np.ndarray[OneDim, np.dtype[np.floating[Any]]],
    ) -> None: ...
    def quantile(self, q: float) -> np.ndarray[TwoDims, np.dtype[_FloatDType]]: ...
    def sum_of_weights(self) -> np.ndarray[TwoDims, np.dtype[_FloatDType]]: ...
    def __copy__(self) -> Histogram2DHolder[_FloatDType]: ...
    def __iadd__(
        self, other: Histogram2DHolder[_FloatDType]
    ) -> Histogram2DHolder[_FloatDType]: ...
    @property
    def x(self) -> Axis: ...
    @property
    def y(self) -> Axis: ...

@overload
def Histogram2D(
    x: Axis,
    y: Axis,
    compression: int | None = None,
    *,
    dtype: None = None,
) -> Histogram2DHolder[np.float64]: ...
@overload
def Histogram2D(
    x: Axis,
    y: Axis,
    compression: int | None = None,
    *,
    dtype: Literal["float32"],
) -> Histogram2DHolder[np.float32]: ...
@overload
def Histogram2D(
    x: Axis,
    y: Axis,
    compression: int | None = None,
    *,
    dtype: Literal["float64"],
) -> Histogram2DHolder[np.float64]: ...
@overload
def Histogram2D(
    x: Axis,
    y: Axis,
    compression: int | None = None,
    *,
    dtype: type[_FloatDType] | np.dtype[_FloatDType],
) -> Histogram2DHolder[_FloatDType]: ...
@overload
def Histogram2D(
    x: Axis,
    y: Axis,
    compression: int | None = None,
    *,
    dtype: str,
) -> Histogram2DHolder[np.float32] | Histogram2DHolder[np.float64]: ...

Histogram2DFloat32: TypeAlias = Histogram2DHolder[np.float32]
Histogram2DFloat64: TypeAlias = Histogram2DHolder[np.float64]

class TemporalGrid4D(GridHolder[_DType]):
    @property
    def array(self) -> np.ndarray[FourDims, np.dtype[_DType]]: ...
    @property
    def shape(self) -> FourDims: ...
    @property
    def x(self) -> Axis: ...
    @property
    def y(self) -> Axis: ...
    @property
    def z(self) -> TemporalAxis: ...
    @property
    def u(self) -> Axis: ...

class RTree3DHolder(Generic[_FloatDType]):
    def __init__(
        self, spheroid: geometry.geographic.Spheroid | None = ...
    ) -> None: ...
    def bounds(
        self,
    ) -> (
        tuple[
            np.ndarray[tuple[Literal[3]], np.dtype[Any]],
            np.ndarray[tuple[Literal[3]], np.dtype[Any]],
        ]
        | None
    ): ...
    def clear(self) -> None: ...
    def empty(self) -> bool: ...
    def insert(
        self,
        coordinates: np.ndarray[TwoDims, np.dtype[Any]],
        values: np.ndarray[OneDim, np.dtype[Any]],
    ) -> None: ...
    def inverse_distance_weighting(
        self,
        coordinates: np.ndarray[TwoDims, np.dtype[Any]],
        config: rtree.InverseDistanceWeighting | None = ...,
    ) -> tuple[np.ndarray[OneDim, np.dtype[_FloatDType]], NDArray1DUInt32]: ...
    def kriging(
        self,
        coordinates: np.ndarray[TwoDims, np.dtype[Any]],
        config: rtree.Kriging | None = ...,
    ) -> tuple[np.ndarray[OneDim, np.dtype[_FloatDType]], NDArray1DUInt32]: ...
    def packing(
        self,
        coordinates: np.ndarray[TwoDims, np.dtype[Any]],
        values: np.ndarray[OneDim, np.dtype[Any]],
    ) -> None: ...
    def query(
        self,
        coordinates: np.ndarray[TwoDims, np.dtype[Any]],
        config: rtree.Query | None = ...,
    ) -> tuple[
        np.ndarray[TwoDims, np.dtype[_FloatDType]],
        np.ndarray[TwoDims, np.dtype[_FloatDType]],
    ]: ...
    def radial_basis_function(
        self,
        coordinates: np.ndarray[TwoDims, np.dtype[Any]],
        config: rtree.RadialBasisFunction | None = ...,
    ) -> tuple[np.ndarray[OneDim, np.dtype[_FloatDType]], NDArray1DUInt32]: ...
    def size(self) -> int: ...
    def window_function(
        self,
        coordinates: np.ndarray[TwoDims, np.dtype[Any]],
        config: rtree.InterpolationWindow | None = ...,
    ) -> tuple[np.ndarray[OneDim, np.dtype[_FloatDType]], NDArray1DUInt32]: ...
    @property
    def spheroid(self) -> geometry.geographic.Spheroid | None: ...

@overload
def RTree3D(
    spheroid: geometry.geographic.Spheroid | None = None,
    *,
    dtype: None = None,
) -> RTree3DHolder[np.float64]: ...
@overload
def RTree3D(
    spheroid: geometry.geographic.Spheroid | None = None,
    *,
    dtype: Literal["float32"],
) -> RTree3DHolder[np.float32]: ...
@overload
def RTree3D(
    spheroid: geometry.geographic.Spheroid | None = None,
    *,
    dtype: Literal["float64"],
) -> RTree3DHolder[np.float64]: ...
@overload
def RTree3D(
    spheroid: geometry.geographic.Spheroid | None = None,
    *,
    dtype: type[_FloatDType] | np.dtype[_FloatDType],
) -> RTree3DHolder[_FloatDType]: ...
@overload
def RTree3D(
    spheroid: geometry.geographic.Spheroid | None = None,
    *,
    dtype: str,
) -> RTree3DHolder[np.float32] | RTree3DHolder[np.float64]: ...

RTree3DFloat64: TypeAlias = RTree3DHolder[np.float64]
RTree3DFloat32: TypeAlias = RTree3DHolder[np.float32]

_TemporalScalar = TypeVar("_TemporalScalar", np.datetime64, np.timedelta64)
_TemporalArray = TypeVar(
    "_TemporalArray", NDArray1DDateTime64, NDArray1DTimeDelta64
)

class TemporalAxis(Generic[_TemporalScalar, _TemporalArray]):
    @overload
    def __init__(
        self: TemporalAxis[np.datetime64, NDArray1DDateTime64],
        points: NDArray1DDateTime64,
        epsilon: np.timedelta64 | None = ...,
        period: np.timedelta64 | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: TemporalAxis[np.timedelta64, NDArray1DTimeDelta64],
        points: NDArray1DTimeDelta64,
        epsilon: np.timedelta64 | None = ...,
        period: np.timedelta64 | None = ...,
    ) -> None: ...
    def back(self) -> _TemporalScalar: ...
    def cast_to_temporal_axis(
        self, array: _TemporalArray
    ) -> _TemporalArray: ...
    def find_index(
        self, array: _TemporalArray, bounded: bool = ...
    ) -> NDArray1DInt64: ...
    def find_indexes(self, array: _TemporalArray) -> NDArray2DInt64: ...
    def flip(
        self, inplace: bool = ...
    ) -> TemporalAxis[_TemporalScalar, _TemporalArray]: ...
    def front(self) -> _TemporalScalar: ...
    def increment(self, step: int = ...) -> np.timedelta64: ...
    def is_ascending(self) -> bool: ...
    def is_regular(self) -> bool: ...
    def max_value(self) -> _TemporalScalar: ...
    def min_value(self) -> _TemporalScalar: ...
    def __copy__(self) -> TemporalAxis[_TemporalScalar, _TemporalArray]: ...
    def __eq__(self, other: object) -> bool: ...
    @overload
    def __getitem__(self, index: int) -> _TemporalScalar: ...
    @overload
    def __getitem__(self, axis_slice: slice) -> _TemporalArray: ...
    def __iter__(self) -> typing.Iterator[_TemporalScalar]: ...
    def __len__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    @property
    def dtype(self) -> np.dtype: ...
    @property
    def is_periodic(self) -> bool: ...
    @property
    def period(self) -> np.timedelta64 | None: ...

@overload
def Grid(
    x: Axis,
    array: np.ndarray[OneDim, np.dtype[_DType]],
) -> Grid1D[_DType]: ...
@overload
def Grid(
    x: Axis,
    y: Axis,
    array: np.ndarray[TwoDims, np.dtype[_DType]],
) -> Grid2D[_DType]: ...
@overload
def Grid(
    x: Axis,
    y: Axis,
    z: Axis,
    array: np.ndarray[ThreeDims, np.dtype[_DType]],
) -> Grid3D[_DType]: ...
@overload
def Grid(
    x: Axis,
    y: Axis,
    z: TemporalAxis,
    array: np.ndarray[ThreeDims, np.dtype[_DType]],
) -> TemporalGrid3D[_DType]: ...
@overload
def Grid(
    x: Axis,
    y: Axis,
    z: Axis,
    u: Axis,
    array: np.ndarray[FourDims, np.dtype[_DType]],
) -> Grid4D[_DType]: ...
@overload
def Grid(
    x: Axis,
    y: Axis,
    z: TemporalAxis,
    u: Axis,
    array: np.ndarray[FourDims, np.dtype[_DType]],
) -> TemporalGrid4D[_DType]: ...
