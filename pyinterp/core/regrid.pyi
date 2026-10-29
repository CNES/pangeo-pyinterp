from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from .geometry.geographic import Spheroid

_FloatArrayT = TypeVar(
    "_FloatArrayT", NDArray[np.float32], NDArray[np.float64]
)

def conservative(
    data: _FloatArrayT,
    source_x_bounds: NDArray[np.float64],
    source_y_bounds: NDArray[np.float64],
    target_x_bounds: NDArray[np.float64],
    target_y_bounds: NDArray[np.float64],
    *,
    geodetic: bool = ...,
    spheroid: Spheroid | None = ...,
    min_valid_fraction: float = ...,
    normalize: bool = ...,
    num_threads: int = ...,
) -> _FloatArrayT: ...
def latitude_measure(
    bounds: NDArray[np.float64], spheroid: Spheroid | None = ...
) -> NDArray[np.float64]: ...
def overlap_matrix(
    source_bounds: NDArray[np.float64],
    target_bounds: NDArray[np.float64],
    period: float = ...,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]: ...
