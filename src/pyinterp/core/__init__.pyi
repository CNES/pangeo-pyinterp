from typing import Any, Optional, Tuple, Union
import numpy
from . import geodetic
from . import fill


class AxisBoundary:
    Expand: 'AxisBoundary'
    Sym: 'AxisBoundary'
    Undef: 'AxisBoundary'
    Wrap: 'AxisBoundary'


class Axis:
    is_circle: bool

    def __init__(self,
                 values: numpy.ndarray[numpy.float64],
                 epsilon: float = 1e-6,
                 is_circle: bool = False) -> None:
        ...

    def __eq__(self, other: 'Axis') -> bool:
        ...

    def __getstate__(self) -> Tuple:
        ...

    def __ne__(self, other: 'Axis') -> bool:
        ...

    def __repr__(self) -> str:
        ...

    def __setstate__(self, state: Tuple) -> None:
        ...

    def __getitem__(
            self,
            arg0: Union[int,
                        slice]) -> Union[float, numpy.ndarray[numpy.float64]]:
        ...

    def back(self) -> float:
        ...

    def front(self) -> float:
        ...

    def max_value(self) -> float:
        ...

    def min_value(self) -> float:
        ...

    def flip(self, inplace: bool = False) -> 'Axis':
        ...

    def is_ascending(self) -> bool:
        ...

    def is_regular(self) -> bool:
        ...

    def find_index(self,
                   coordinates: numpy.ndarray[numpy.float64],
                   bounded: bool = False) -> numpy.ndarray[numpy.float64]:
        ...

    def find_indexes(
        self, coordinates: numpy.ndarray[numpy.float64]
    ) -> numpy.ndarray[numpy.float64]:
        ...

    def increment(self) -> float:
        ...

    def __len__(self) -> int:
        ...


class TemporalAxis:
    is_circle: bool

    def __init__(self,
                 values: numpy.ndarray[numpy.int64],
                 epsilon: int = 0,
                 is_circle: bool = False) -> None:
        ...

    def back(self) -> int:
        ...

    def front(self) -> int:
        ...

    def max_value(self) -> int:
        ...

    def min_value(self) -> int:
        ...

    def flip(self, inplace: bool = False) -> 'Axis':
        ...

    def is_ascending(self) -> bool:
        ...

    def is_regular(self) -> bool:
        ...

    def find_index(self,
                   coordinates: numpy.ndarray[numpy.int64],
                   bounded: bool = False) -> numpy.ndarray[numpy.int64]:
        ...

    def increment(self) -> int:
        ...

    def __setstate__(self, state: Tuple) -> None:
        ...

    def __eq__(self, other: 'Axis') -> bool:
        ...

    def __ne__(self, other: 'Axis') -> bool:
        ...

    def __getitem__(
            self,
            arg0: Union[int, slice]) -> Union[int, numpy.ndarray[numpy.int64]]:
        ...

    def __getstate__(self) -> Tuple:
        ...

    def __len__(self) -> int:
        ...

    def __repr__(self) -> str:
        ...


class Binning2DFloat64:
    x: Axis
    y: Axis
    wgs: geodetic.System

    def __init__(self,
                 x: Axis,
                 y: Axis,
                 wgs: Optional[geodetic.System] = None) -> None:
        ...

    def clear(self) -> None:
        ...

    def push(self,
             x: numpy.ndarray[numpy.float64],
             y: numpy.ndarray[numpy.float64],
             z: numpy.ndarray[numpy.float64],
             simple: Optional[bool] = True) -> None:
        ...

    def count(self) -> numpy.ndarray[numpy.float64]:
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

    def variance(self) -> numpy.ndarray[numpy.float64]:
        ...

    def __iadd__(self, other: "Binning2DFloat64") -> "Binning2DFloat64":
        ...

    def __getstate__(self) -> Tuple:
        ...

    def __setstate__(self, state: Tuple) -> "Binning2DFloat64":
        ...


class Binning2DFloat32:
    x: Axis
    y: Axis

    def __init__(self,
                 x: Axis,
                 y: Axis,
                 wgs: Optional[geodetic.System] = None) -> None:
        ...

    def clear(self) -> None:
        ...

    def push(self,
             x: numpy.ndarray[numpy.float32],
             y: numpy.ndarray[numpy.float32],
             z: numpy.ndarray[numpy.float32],
             simple: Optional[bool] = True) -> None:
        ...

    def count(self) -> numpy.ndarray[numpy.float32]:
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

    def variance(self) -> numpy.ndarray[numpy.float32]:
        ...

    def __iadd__(self, other: "Binning2DFloat32") -> "Binning2DFloat32":
        ...

    def __getstate__(self) -> Tuple:
        ...

    def __setstate__(self, state: Tuple) -> "Binning2DFloat32":
        ...


class Grid2DFloat64:
    array: numpy.ndarray[numpy.float64]
    x: Axis
    y: Axis

    def __getstate__(self) -> tuple:
        ...

    def __init__(self, x: Axis, y: Axis,
                 array: numpy.ndarray[numpy.float64]) -> None:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Grid2DFloat32:
    array: numpy.ndarray[numpy.float32]
    x: Axis
    y: Axis

    def __getstate__(self) -> tuple:
        ...

    def __init__(self, x: Axis, y: Axis,
                 array: numpy.ndarray[numpy.float32]) -> None:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Grid3DFloat64:
    array: numpy.ndarray[numpy.float64]
    x: Axis
    y: Axis
    z: Axis

    def __getstate__(self) -> tuple:
        ...

    def __init__(self, x: Axis, y: Axis, z: Axis,
                 array: numpy.ndarray[numpy.float64]) -> None:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Grid3DFloat32:
    array: numpy.ndarray[numpy.float32]
    x: Axis
    y: Axis
    z: Axis

    def __getstate__(self) -> tuple:
        ...

    def __init__(self, x: Axis, y: Axis, z: Axis,
                 array: numpy.ndarray[numpy.float32]) -> None:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Grid4DFloat64:
    array: numpy.ndarray[numpy.float64]
    x: Axis
    y: Axis
    z: Axis
    u: Axis

    def __getstate__(self) -> tuple:
        ...

    def __init__(self, x: Axis, y: Axis, z: Axis, u: Axis,
                 array: numpy.ndarray[numpy.float64]) -> None:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class Grid4DFloat32:
    array: numpy.ndarray[numpy.float32]
    x: Axis
    y: Axis
    z: Axis
    u: Axis

    def __getstate__(self) -> tuple:
        ...

    def __init__(self, x: Axis, y: Axis, z: Axis, u: Axis,
                 array: numpy.ndarray[numpy.float32]) -> None:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class TemporalGrid3DFloat64:
    array: numpy.ndarray[numpy.float64]
    x: Axis
    y: Axis
    z: TemporalAxis

    def __getstate__(self) -> tuple:
        ...

    def __init__(self, x: Axis, y: Axis, z: TemporalAxis,
                 array: numpy.ndarray[numpy.float64]) -> None:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class TemporalGrid3DFloat32:
    array: numpy.ndarray[numpy.float32]
    x: Axis
    y: Axis
    z: TemporalAxis

    def __getstate__(self) -> tuple:
        ...

    def __init__(self, x: Axis, y: Axis, z: TemporalAxis,
                 array: numpy.ndarray[numpy.float32]) -> None:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class TemporalGrid4DFloat64:
    array: numpy.ndarray[numpy.float64]
    x: Axis
    y: Axis
    z: TemporalAxis
    u: Axis

    def __getstate__(self) -> tuple:
        ...

    def __init__(self, x: Axis, y: Axis, z: TemporalAxis, u: Axis,
                 array: numpy.ndarray[numpy.float64]) -> None:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class TemporalGrid4DFloat32:
    array: numpy.ndarray[numpy.float32]
    x: Axis
    y: Axis
    z: TemporalAxis
    u: Axis

    def __getstate__(self) -> tuple:
        ...

    def __init__(self, x: Axis, y: Axis, z: TemporalAxis, u: Axis,
                 array: numpy.ndarray[numpy.float32]) -> None:
        ...

    def __setstate__(self, state: tuple) -> None:
        ...


class RadialBasisFunction:
    Cubic: 'RadialBasisFunction'
    Gaussian: 'RadialBasisFunction'
    InverseMultiquadric: 'RadialBasisFunction'
    Linear: 'RadialBasisFunction'
    Multiquadric: 'RadialBasisFunction'
    ThinPlate: 'RadialBasisFunction'


class RTree3DFloat64:
    def __init__(self, system: Optional[geodetic.System]) -> None:
        ...

    def bounds(
            self
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        ...

    def clear(self) -> None:
        ...

    def insert(self, coordinates: numpy.ndarray[numpy.float64],
               values: numpy.ndarray[numpy.float64]) -> None:
        ...

    def inverse_distance_weighting(
        self,
        coordinates: numpy.ndarray[numpy.float64],
        radius: Optional[float],
        k: int = 9,
        p: int = 2,
        within: bool = True,
        num_threads: int = 0
    ) -> Tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
        ...

    def radial_basis_function(
        self,
        coordinates: numpy.ndarray[numpy.float64],
        radius: Optional[float],
        k: int = 9,
        rbf: RadialBasisFunction = RadialBasisFunction.Multiquadric,
        epsilon: Optional[float] = None,
        smooth: float = 0,
        within: bool = True,
        num_threads: int = 0
    ) -> Tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
        ...

    def query(
        self,
        coordinates: numpy.ndarray[numpy.float64],
        k: int = 4,
        within: bool = False,
        num_threads: int = 0
    ) -> Tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
        ...

    def packing(self, coordinates: numpy.ndarray[numpy.float64],
                values: numpy.ndarray[numpy.float64]) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __bool__(self) -> bool:
        ...


class RTree3DFloat32:
    def __init__(self, system: Optional[geodetic.System]) -> None:
        ...

    def bounds(
            self
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        ...

    def clear(self) -> None:
        ...

    def insert(self, coordinates: numpy.ndarray[numpy.float32],
               values: numpy.ndarray[numpy.float32]) -> None:
        ...

    def inverse_distance_weighting(
        self,
        coordinates: numpy.ndarray[numpy.float32],
        radius: Optional[float],
        k: int = 9,
        p: int = 2,
        within: bool = True,
        num_threads: int = 0
    ) -> Tuple[numpy.ndarray[numpy.float32], numpy.ndarray[numpy.float32]]:
        ...

    def radial_basis_function(
        self,
        coordinates: numpy.ndarray[numpy.float32],
        radius: Optional[float],
        k: int = 9,
        rbf: RadialBasisFunction = RadialBasisFunction.Multiquadric,
        epsilon: Optional[float] = None,
        smooth: float = 0,
        within: bool = True,
        num_threads: int = 0
    ) -> Tuple[numpy.ndarray[numpy.float32], numpy.ndarray[numpy.float32]]:
        ...

    def query(
        self,
        coordinates: numpy.ndarray[numpy.float32],
        k: int = 4,
        within: bool = False,
        num_threads: int = 0
    ) -> Tuple[numpy.ndarray[numpy.float32], numpy.ndarray[numpy.float32]]:
        ...

    def packing(self, coordinates: numpy.ndarray[numpy.float32],
                values: numpy.ndarray[numpy.float32]) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __bool__(self) -> bool:
        ...


class FittingModel:
    Akima: 'FittingModel'
    AkimaPeriodic: 'FittingModel'
    CSpline: 'FittingModel'
    CSplinePeriodic: 'FittingModel'
    Linear: 'FittingModel'
    Polynomial: 'FittingModel'
    Steffen: 'FittingModel'


def bicubic_float64(grid: Union[Grid2DFloat64, Grid3DFloat64,
                                TemporalGrid3DFloat64, Grid4DFloat64,
                                TemporalGrid4DFloat64],
                    x: numpy.ndarray[numpy.float64],
                    y: numpy.ndarray[numpy.float64],
                    z: Optional[Union[numpy.ndarray[numpy.float64],
                                      numpy.ndarray[numpy.int64]]] = None,
                    u: Optional[numpy.ndarray[numpy.float64]] = None,
                    nx: int = 3,
                    ny: int = 3,
                    fitting_model: FittingModel = FittingModel.CSpline,
                    boundary: AxisBoundary = AxisBoundary.Undef,
                    bounds_error: bool = False,
                    num_threads: int = 0) -> numpy.ndarray[numpy.float64]:
    ...


def bicubic_float32(grid: Union[Grid2DFloat32, Grid3DFloat32,
                                TemporalGrid3DFloat32, Grid4DFloat32,
                                TemporalGrid4DFloat32],
                    x: numpy.ndarray[numpy.float64],
                    y: numpy.ndarray[numpy.float64],
                    z: Optional[Union[numpy.ndarray[numpy.float64],
                                      numpy.ndarray[numpy.int64]]] = None,
                    u: Optional[numpy.ndarray[numpy.float64]] = None,
                    nx: int = 3,
                    ny: int = 3,
                    fitting_model: FittingModel = FittingModel.CSpline,
                    boundary: AxisBoundary = AxisBoundary.Undef,
                    bounds_error: bool = False,
                    num_threads: int = 0) -> numpy.ndarray[numpy.float64]:
    ...


class BivariateInterpolator2D:
    ...


class Bilinear2D(BivariateInterpolator2D):
    ...


class InverseDistanceWeighting2D(BivariateInterpolator2D):
    ...


class Nearest2D(BivariateInterpolator2D):
    ...


def bivariate_float64(grid: Grid2DFloat64,
                      x: numpy.ndarray[numpy.float64],
                      y: numpy.ndarray[numpy.float64],
                      interpolator: BivariateInterpolator2D,
                      bounds_error: bool = False,
                      num_threads: int = 0) -> numpy.ndarray[numpy.float64]:
    ...


def bivariate_float32(grid: Grid2DFloat32,
                      x: numpy.ndarray[numpy.float32],
                      y: numpy.ndarray[numpy.float32],
                      interpolator: BivariateInterpolator2D,
                      bounds_error: bool = False,
                      num_threads: int = 0) -> numpy.ndarray[numpy.float32]:
    ...


class BivariateInterpolator3D:
    ...


class Bilinear3D(BivariateInterpolator3D):
    ...


class InverseDistanceWeighting3D(BivariateInterpolator3D):
    ...


class Nearest3D(BivariateInterpolator3D):
    ...


class TemporalBivariateInterpolator3D:
    ...


class TemporalBilinear3D(TemporalBivariateInterpolator3D):
    ...


class TemporalInverseDistanceWeighting3D(TemporalBivariateInterpolator3D):
    ...


class TemporalNearest3D(TemporalBivariateInterpolator3D):
    ...


def trivariate_float64(grid: Union[Grid3DFloat64, TemporalGrid3DFloat64],
                       x: numpy.ndarray[numpy.float64],
                       y: numpy.ndarray[numpy.float64],
                       z: numpy.ndarray[numpy.float64],
                       interpolator: Union[BivariateInterpolator3D,
                                           TemporalBivariateInterpolator3D],
                       z_method: Optional[str] = None,
                       bounds_error: bool = False,
                       num_threads: int = 0) -> numpy.ndarray[numpy.float64]:
    ...


def trivariate_float32(grid: Union[Grid3DFloat32, TemporalGrid3DFloat32],
                       x: numpy.ndarray[numpy.float64],
                       y: numpy.ndarray[numpy.float64],
                       z: numpy.ndarray[numpy.float64],
                       interpolator: Union[BivariateInterpolator3D,
                                           TemporalBivariateInterpolator3D],
                       z_method: Optional[str] = None,
                       bounds_error: bool = False,
                       num_threads: int = 0) -> numpy.ndarray[numpy.float64]:
    ...


def quadrivariate_float64(
        grid: Union[Grid4DFloat64, TemporalGrid4DFloat64],
        x: numpy.ndarray[numpy.float64],
        y: numpy.ndarray[numpy.float64],
        z: numpy.ndarray[numpy.float64],
        u: numpy.ndarray[numpy.float64],
        interpolator: Union[BivariateInterpolator3D,
                            TemporalBivariateInterpolator3D],
        z_method: Optional[str] = None,
        u_method: Optional[str] = None,
        bounds_error: bool = False,
        num_threads: int = 0) -> numpy.ndarray[numpy.float64]:
    ...


def quadrivariate_float32(
        grid: Union[Grid4DFloat64, TemporalGrid4DFloat64],
        x: numpy.ndarray[numpy.float64],
        y: numpy.ndarray[numpy.float64],
        z: numpy.ndarray[numpy.float64],
        u: numpy.ndarray[numpy.float64],
        interpolator: Union[BivariateInterpolator3D,
                            TemporalBivariateInterpolator3D],
        z_method: Optional[str] = None,
        u_method: Optional[str] = None,
        bounds_error: bool = False,
        num_threads: int = 0) -> numpy.ndarray[numpy.float64]:
    ...
