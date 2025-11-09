"""Replace undefined values."""
from __future__ import annotations

from typing import TYPE_CHECKING, overload
import concurrent.futures

import numpy

from . import core, grid, interface
from .grid import NUM_DIMS_2

if TYPE_CHECKING:
    from .typing import (
        NDArray1D,
        NDArray2D,
        NDArray2DFloat32,
        NDArray2DFloat64,
        NDArray3D,
        NDArray4D,
    )


@overload
def loess(  # type: ignore[overload-overlap]
    mesh: grid.Grid4D,
    nx: int = 3,
    ny: int = 3,
    value_type: str | None = None,
    num_threads: int = 0,
) -> NDArray4D:
    ...


@overload
def loess(  # type: ignore[overload-overlap]
    mesh: grid.Grid3D,
    nx: int = 3,
    ny: int = 3,
    value_type: str | None = None,
    num_threads: int = 0,
) -> NDArray3D:
    ...


@overload
def loess(
    mesh: grid.Grid2D,
    nx: int = 3,
    ny: int = 3,
    value_type: str | None = None,
    num_threads: int = 0,
) -> NDArray2D:
    ...


def loess(mesh: grid.Grid2D | grid.Grid3D | grid.Grid4D,
          nx: int = 3,
          ny: int = 3,
          value_type: str | None = None,
          num_threads: int = 0) -> NDArray2D | NDArray3D | NDArray4D:
    """Filter values using a locally weighted regression function (LOESS).

    Apply LOESS filtering to fill or smooth grid values using the tri-cube
    weight function: :math:`w(x)=(1-|d|^3)^3`.

    Args:
        mesh: Grid function on a uniform 2-dimensional grid to be filled.
        nx: Number of points of the half-window to be taken into account along
            the X-axis. Defaults to ``3``.
        ny: Number of points of the half-window to be taken into account along
            the Y-axis. Defaults to ``3``.
        value_type: Type of values processed by the filter. Supported are
            ``undefined``, ``defined``, ``all``. Default to ``undefined``.
        num_threads: The number of threads to use for the computation. If 0
            all CPUs are used. If 1 is given, no parallel computing code is used
            at all, which is useful for debugging. Defaults to ``0``.

    Returns:
        The grid will have NaN filled with extrapolated values.

    """
    value_type = value_type or 'undefined'
    instance = mesh._instance
    function = interface._core_function('loess', instance)

    if value_type not in ['undefined', 'defined', 'all']:
        raise ValueError(f'value type {value_type!r} is not defined')

    return getattr(core.fill, function)(instance, nx, ny,
                                        getattr(core.fill.ValueType,
                                                value_type.capitalize()),
                                        num_threads)


@overload
def gauss_seidel(  # type: ignore[overload-overlap]
    mesh: grid.Grid3D,
    first_guess: str = 'zonal_average',
    max_iteration: int | None = None,
    epsilon: float = 1e-4,
    relaxation: float | None = None,
    num_threads: int = 0,
) -> tuple[bool, NDArray3D]:
    ...


@overload
def gauss_seidel(
    mesh: grid.Grid2D,
    first_guess: str = 'zonal_average',
    max_iteration: int | None = None,
    epsilon: float = 1e-4,
    relaxation: float | None = None,
    num_threads: int = 0,
) -> tuple[bool, NDArray2D]:
    ...


def gauss_seidel(
    mesh: grid.Grid2D | grid.Grid3D,
    first_guess: str = 'zonal_average',
    max_iteration: int | None = None,
    epsilon: float = 1e-4,
    relaxation: float | None = None,
    num_threads: int = 0,
) -> tuple[bool, NDArray2D | NDArray3D]:
    r"""Replace all undefined values (NaN) in a grid using Gauss-Seidel method.

    Apply the Gauss-Seidel method by relaxation to fill all undefined values
    (NaN) in a grid.

    Args:
        mesh: Grid function on a uniform 2/3-dimensional grid to be filled.
        first_guess: Specifies the type of first guess grid.
            Supported values are:

            * ``zero`` means use ``0.0`` as an initial guess;
            * ``zonal_average`` means that zonal averages (i.e, averages in
              the X-axis direction) will be used.

            Defaults to ``zonal_average``.

        max_iteration: Maximum number of iterations to be used by relaxation.
            The default value is equal to the product of the grid dimensions.
        epsilon: Tolerance for ending relaxation before the maximum number of
            iterations limit. Defaults to ``1e-4``.
        relaxation: Relaxation constant.
            If ``0 < relaxation < 1``, the new value is an average weighted by
            the old and the one given by the Gauss-Seidel scheme. In this case,
            convergence is slowed down (under-relaxation). Over-relaxation
            consists in choosing a value of ``relaxation`` strictly greater
            than 1. For the method to converge, it is necessary that:
            ``1 < relaxation < 2``.
            If this parameter is not set, the method will choose the optimal
            value that allows the convergence criterion to be achieved in
            :math:`O(N)` iterations, for a grid of size :math:`N_x=N_y=N`,
            ``relaxation`` = :math:`{2\over{1+{\pi\over{N}}}}`; if the grid
            is of size :math:`Nx \times Ny`,
            :math:`N = N_{x}N_{y}\sqrt{2\over{N_{x}^2+N_{y}^2}}`
        num_threads: The number of threads to use for the computation. If 0 all
            CPUs are used. If 1 is given, no parallel computing code is used at
            all, which is useful for debugging. Defaults to ``0``.

    Returns:
        A boolean indicating if the calculation has converged, i. e. if
        the value of the residues is lower than the ``epsilon`` limit set, and
        the the grid will have the all NaN filled with extrapolated values.

    """
    if first_guess not in ['zero', 'zonal_average']:
        raise ValueError(f'first_guess type {first_guess!r} is not defined')

    ny = len(mesh.y)
    nx = len(mesh.x)
    nz = len(mesh.z) if isinstance(mesh, grid.Grid3D) else 0

    if relaxation is None:
        if nx == ny:
            n = nx
        else:
            n = nx * ny * numpy.sqrt(2 / (nx**2 + ny**2))
        relaxation = 2 / (1 + numpy.pi / n)

    if max_iteration is None:
        max_iteration = nx * ny

    first_guess = getattr(
        core.fill.FirstGuess,
        ''.join(item.capitalize() for item in first_guess.split('_')))

    instance = mesh._instance
    function = interface._core_function('gauss_seidel', instance)
    filled = numpy.copy(mesh.array)
    if nz == 0:
        _iterations, residual = getattr(core.fill,
                                        function)(filled, first_guess,
                                                  mesh.x.is_circle,
                                                  max_iteration, epsilon,
                                                  relaxation, num_threads)
    else:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_threads if num_threads else None) as executor:
            futures = [
                executor.submit(getattr(core.fill, function), filled[:, :, iz],
                                first_guess, mesh.x.is_circle, max_iteration,
                                epsilon, relaxation, 1) for iz in range(nz)
            ]
            residuals = []
            for future in concurrent.futures.as_completed(futures):
                _, residual = future.result()
                residuals.append(residual)
            residual = max(residuals)
    return residual <= epsilon, filled


def multi_grid(
    grid: NDArray2DFloat32 | NDArray2DFloat64,
    first_guess: str = 'zonal_average',
    is_circle: bool = False,
    max_iterations: int | None = None,
    epsilon: float = 1e-4,
    pre_smooth: int = 2,
    post_smooth: int = 2,
    num_threads: int = 0,
) -> tuple[int, float]:
    """Replace undefined values (NaN) in a 2-D grid using the multigrid V-cycle.

    Apply the multigrid V-cycle method to fill NaN values in a 2-D array.

    Args:
        grid: The grid to be processed.
        first_guess: Method to use for the first guess. Supported values are
            ``zero`` (use ``0.0``) and ``zonal_average`` (use zonal averages
            along the X axis). Defaults to ``zonal_average``.
        is_circle: True if the X axis of the grid defines a circle.
        max_iterations: Maximum number of iterations to be used by the
            multigrid method. If None, defaults to the product of the grid
            dimensions.
        epsilon: Tolerance for ending the multigrid method before reaching the
            maximum number of iterations. Defaults to ``1e-4``.
        pre_smooth: Number of smoothing iterations to perform before
            restriction. Defaults to ``2``.
        post_smooth: Number of smoothing iterations to perform after
            prolongation. Defaults to ``2``.
        num_threads: Number of threads to use for the computation. If 0 all
            CPUs are used. If 1 is given, no parallel computing code is used,
            which is useful for debugging. Defaults to ``0``.

    Returns:
        A tuple with the number of iterations performed and the final residual
        value.

    """
    if first_guess not in ['zero', 'zonal_average']:
        raise ValueError(f'first_guess type {first_guess!r} is not defined')

    if max_iterations is None:
        max_iterations = grid.shape[0] * grid.shape[1]

    first_guess_enum = getattr(
        core.fill.FirstGuess,
        ''.join(item.capitalize() for item in first_guess.split('_')))

    if grid.dtype == numpy.float32:
        return core.fill.multigrid_float32(
            grid,  # type: ignore[arg-type]
            first_guess_enum,
            is_circle,
            max_iterations,
            epsilon,
            pre_smooth,
            post_smooth,
            num_threads,
        )
    return core.fill.multigrid_float64(
        grid,  # type: ignore[arg-type]
        first_guess_enum,
        is_circle,
        max_iterations,
        epsilon,
        pre_smooth,
        post_smooth,
        num_threads,
    )


def fft_inpaint(
    grid: NDArray2DFloat32 | NDArray2DFloat64,
    first_guess: str = 'zonal_average',
    is_circle: bool = False,
    max_iterations: int | None = None,
    epsilon: float = 1e-4,
    sigma: float = 10.0,
    num_threads: int = 0,
) -> tuple[int, float]:
    """Fill undefined values in a grid using spectral in-painting.

    Replace NaN values in a 2D array using a spectral in-painting approach.
    Uses a Fast Fourier Transform (FFT) for periodic boundaries (is_circle=True)
    or a Discrete Cosine Transform (DCT) for reflective boundaries
    (is_circle=False). A Gaussian low-pass filter (sigma) controls the
    smoothness of the fill.

    Args:
        grid: The grid to be processed
        first_guess: Method to use for the first guess.
        is_circle: If true, uses a Fast Fourier Transform (FFT) assuming
            periodic boundaries. If false, uses a Discrete Cosine Transform
            (DCT) assuming reflective boundaries. Defaults to ``False``.
        max_iterations: Maximum number of iterations. Defaults to ``500``.
        epsilon: Tolerance for ending relaxation. Defaults to ``1e-4``.
        sigma: Standard deviation of the Gaussian low-pass filter in pixel
            units. Controls the smoothness of the fill. Defaults to ``10.0``.
        num_threads: The number of threads to use for the computation.
            Defaults to ``0``.

    Returns:
        A tuple containing the number of iterations performed and the maximum
        residual value.

    """
    if first_guess not in ['zero', 'zonal_average']:
        raise ValueError(f'first_guess type {first_guess!r} is not defined')

    if max_iterations is None:
        max_iterations = grid.shape[0] * grid.shape[1]

    first_guess_enum = getattr(
        core.fill.FirstGuess,
        ''.join(item.capitalize() for item in first_guess.split('_')))

    if grid.dtype == numpy.float32:
        return core.fill.fft_inpaint_float32(
            grid,  # type: ignore[arg-type]
            first_guess_enum,
            is_circle,
            max_iterations,
            epsilon,
            sigma,
            num_threads,
        )
    return core.fill.fft_inpaint_float64(
        grid,  # type: ignore[arg-type]
        first_guess_enum,
        is_circle,
        max_iterations,
        epsilon,
        sigma,
        num_threads,
    )


def matrix(x: NDArray2DFloat32 | NDArray2DFloat64,
           fill_value: float | None = None,
           in_place: bool = True) -> NDArray2DFloat32 | NDArray2DFloat64:
    """Fill in the gaps between defined values in a 2-dimensional array.

    Args:
        x: data to be filled.
        fill_value: Value used to fill undefined values. Should be compatible
            with the dtype of x (float32 or float64).
        in_place: If true, the data is filled in place. Defaults to ``True``.

    Returns:
        The data filled.

    """
    fill_value = fill_value or numpy.nan
    if len(x.shape) != NUM_DIMS_2:
        raise ValueError('x must be a 2-dimensional array')
    dtype = x.dtype
    if not in_place:
        x = numpy.copy(x)  # type: ignore[assignment]
    if dtype == numpy.float32:
        core.fill.matrix_float32(
            x,  # type: ignore[arg-type]
            fill_value,
        )
    core.fill.matrix_float64(
        x,  # type: ignore[arg-type]
        fill_value,
    )
    return x


def vector(
    x: NDArray1D,
    fill_value: numpy.generic | None = None,
    in_place: bool = True,
) -> NDArray1D:
    """Fill in the gaps between defined values in a 1-dimensional array.

    Args:
        x: data to be filled.
        fill_value: Value used to fill undefined values. The type should be
            compatible with the dtype of x (float for float32/float64, int for
            int64, datetime64 for datetime64, timedelta64 for timedelta64).
        in_place: If true, the data is filled in place. Defaults to ``True``.

    Returns:
        The data filled.

    """
    if not isinstance(x, numpy.ndarray):
        raise ValueError('x must be a numpy.ndarray')
    if len(x.shape) != 1:
        raise ValueError('x must be a 1-dimensional array')
    dtype = x.dtype
    if not in_place:
        x = numpy.copy(x)
    if dtype == numpy.float32:
        core.fill.vector_float32(
            x,  # type: ignore[arg-type]
            float('NaN') if fill_value is None else float(fill_value),
        )
    elif dtype == numpy.float64:
        core.fill.vector_float64(
            x,  # type: ignore[arg-type]
            float('NaN') if fill_value is None else float(fill_value),
        )
    elif dtype == numpy.int64:
        core.fill.vector_int64(
            x,  # type: ignore[arg-type]
            2**63 - 1 if fill_value is None else int(fill_value),
        )
    elif numpy.issubdtype(dtype, numpy.datetime64) or numpy.issubdtype(
            dtype, numpy.timedelta64):
        core.fill.vector_int64(
            x.view(int),  # type: ignore[arg-type]
            numpy.datetime64(fill_value).view(int),  # type: ignore[type-var]
        )
    else:
        raise ValueError(f'unsupported data type {dtype}')
    return x
