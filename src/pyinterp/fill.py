"""
Replace undefined values
------------------------
"""
from typing import Optional, Union
import concurrent.futures
import numpy as np
from . import core
from . import grid
from . import interface


def loess(mesh: Union[grid.Grid2D, grid.Grid3D],
          nx: Optional[int] = 3,
          ny: Optional[int] = 3,
          num_threads: Optional[int] = 0):
    """Fills undefined values using a locally weighted regression function or
    LOESS. The weight function used for LOESS is the tri-cube weight function,
    :math:`w(x)=(1-|d|^3)^3`

    Args:
        mesh (pyinterp.grid.Grid2D, pyinterp.grid.Grid3D): Grid function on
            a uniform 2-dimensional grid to be filled.
        nx (int, optional): Number of points of the half-window to be taken
            into account along the X-axis. Defaults to ``3``.
        ny (int, optional): Number of points of the half-window to be taken
            into account along the Y-axis. Defaults to ``3``.
        num_threads (int, optional): The number of threads to use for the
            computation. If 0 all CPUs are used. If 1 is given, no parallel
            computing code is used at all, which is useful for debugging.
            Defaults to ``0``.

    Return:
        numpy.ndarray: the grid will have NaN filled with extrapolated values.
    """
    instance = mesh._instance
    function = f"loess_{interface._core_function_suffix(instance)}"
    nz = len(mesh.z) if isinstance(mesh, grid.Grid3D) else 0

    if nz == 0:
        return getattr(core.fill, function)(instance, nx, ny, num_threads)

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_threads if num_threads else None) as executor:
        futures = dict()
        result = np.empty_like(mesh.array)
        for iz in range(nz):
            grid2d = grid.Grid2D(mesh.x, mesh.y, mesh.array[:, :, iz])
            futures[executor.submit(getattr(core.fill,
                                            function), grid2d._instance, nx,
                                    ny, num_threads)] = iz
        for future in concurrent.futures.as_completed(futures):
            iz = futures[future]
            result[:, :, iz] = future.result()
        return result


def gauss_seidel(mesh: Union[grid.Grid2D, grid.Grid3D],
                 first_guess: Optional[str] = "zonal_average",
                 max_iteration: Optional[int] = None,
                 epsilon: Optional[float] = 1e-4,
                 relaxation: Optional[float] = None,
                 num_threads: Optional[int] = 0):
    """
    Replaces all undefined values (NaN) in a grid using the Gauss-Seidel
    method by relaxation.

    Args:
        mesh (pyinterp.grid.Grid2D, pyinterp.grid.Grid3D): Grid function on
            a uniform 2/3-dimensional grid to be filled.
        first_guess (str, optional): Specifies the type of first guess grid.
            Supported values are:

                * ``zero`` means use ``0.0`` as an initial guess;
                * ``zonal_average`` means that zonal averages (i.e, averages in
                  the X-axis direction) will be used.

            Defaults to ``zonal_average``.

        max_iterations (int, optional): Maximum number of iterations to be used
            by relaxation. The default value is equal to the product of the
            grid dimensions.
        epsilon (float, optional): Tolerance for ending relaxation before the
            maximum number of iterations limit. Defaults to ``1e-4``.
        relaxation (float, opional): Relaxation constant.
            If ``0 < relaxation < 1``, the new value is an average weighted by
            the old and the one given by the Gauss-Seidel scheme. In this case,
            convergence is slowed down (under-relaxation). Over-relaxation
            consists in choosing a value of ``relaxation`` strictly greater
            than 1. For the method to converge, it is necessary that:
            ``1 < relaxation < 2``.
            If this parameter is not set, the method will choose the optimal
            value that allows the convergence criterion to be achieved in
            :math:`O(N)` iterations, for a grid of size :math:`N_x=N_y=N`,
            ``relaxation`` = :math:`{2\\over{1+{\\pi\\over{N}}}}`; if the grid
            is of size :math:`Nx \\times Ny`,
            :math:`N = N_{x}N_{y}\\sqrt{2\\over{N_{x}^2+N_{y}^2}}`
        num_threads (int, optional): The number of threads to use for the
            computation. If 0 all CPUs are used. If 1 is given, no parallel
            computing code is used at all, which is useful for debugging.
            Defaults to ``0``.

    Return:
        tuple: a boolean indicating if the calculation has converged, i. e. if
        the value of the residues is lower than the ``epsilon`` limit set, and
        the the grid will have the all NaN filled with extrapolated values.
    """
    if first_guess not in ['zero', 'zonal_average']:
        raise ValueError(f"first_guess type {first_guess!r} is not defined")

    ny = len(mesh.y)
    nx = len(mesh.x)
    nz = len(mesh.z) if isinstance(mesh, grid.Grid3D) else 0

    if relaxation is None:
        if nx == ny:
            N = nx
        else:
            N = nx * ny * np.sqrt(2 / (nx**2 + ny**2))
        relaxation = 2 / (1 + np.pi / N)

    if max_iteration is None:
        max_iteration = nx * ny

    first_guess = getattr(
        getattr(core.fill, "FirstGuess"),
        "".join(item.capitalize() for item in first_guess.split("_")))

    instance = mesh._instance
    function = f"gauss_seidel_{interface._core_function_suffix(instance)}"
    filled = np.copy(mesh.array)
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
