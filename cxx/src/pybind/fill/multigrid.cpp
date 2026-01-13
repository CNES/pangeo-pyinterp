// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/pybind/fill/multigrid.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

#include "pyinterp/eigen.hpp"
#include "pyinterp/fill/multi_grid.hpp"

namespace nb = nanobind;

namespace pyinterp::fill::pybind {

constexpr const char* const kMultigridDocstring = R"(
Fill undefined values using the Multigrid V-cycle method.

The Multigrid method solves the homogeneous Laplace equation at masked
(NaN) points using a geometric multigrid V-cycle with Gauss-Seidel smoothing.
This method is efficient for large grids and provides smooth interpolation
of boundary values into unknown regions.

Args:
    grid: Input 2D array with data to fill. NaN values mark undefined points.
    config: Multigrid configuration with:
        - first_guess: Initial guess strategy (zonal_average or zero)
        - is_periodic: Whether the x-axis is periodic (e.g., longitude)
        - max_iterations: Maximum number of V-cycles
        - epsilon: Convergence threshold for residual
        - pre_smooth: Number of pre-smoothing iterations per level
        - post_smooth: Number of post-smoothing iterations per level
        - num_threads: Number of threads (0 = use all available CPUs)

Returns:
    Tuple of (iterations, max_residual):
        - iterations: Number of V-cycles performed
        - max_residual: Maximum residual value at convergence
)";

/// Bind Multigrid function to Python module.
/// @tparam T Scalar type
/// @param m Python module
template <std::floating_point T>
void bind_multigrid(nanobind::module_& m) {
  m.def(
      "multigrid",
      [](const EigenDRef<RowMajorMatrix<T>>& grid,
         const config::fill::Multigrid& config) -> std::tuple<size_t, T> {
        return multigrid(grid, config);
      },
      nanobind::arg("grid"), nanobind::arg("config"), kMultigridDocstring,
      nanobind::call_guard<nanobind::gil_scoped_release>());
}

void bind_multigrid(nanobind::module_& m) {
  bind_multigrid<double>(m);
  bind_multigrid<float>(m);
}

}  // namespace pyinterp::fill::pybind
