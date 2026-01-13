// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/pybind/fill/gauss_seidel.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

#include "pyinterp/eigen.hpp"
#include "pyinterp/fill/gauss_seidel.hpp"

namespace nb = nanobind;

namespace pyinterp::fill::pybind {

constexpr const char* const kGaussSeidelDocstring = R"(
Fill undefined values using the Gauss-Seidel iterative method.

The Gauss-Seidel method solves the discrete Laplace equation at masked
(NaN) points using red-black ordering for efficient parallel computation.
The solution smoothly interpolates known boundary values into unknown regions.

Args:
    grid: Input 2D array with data to fill. NaN values mark undefined points.
    config: Gauss-Seidel configuration with:
        - first_guess: Initial guess strategy (zonal_average or zero)
        - is_periodic: Whether the x-axis is periodic (e.g., longitude)
        - max_iterations: Maximum number of iterations
        - epsilon: Convergence threshold for residual
        - relaxation: SOR relaxation parameter (1.0 = standard Gauss-Seidel)
        - num_threads: Number of threads (0 = use all available CPUs)

Returns:
    Tuple of (iterations, max_residual):
        - iterations: Number of iterations performed
        - max_residual: Maximum residual value at convergence
)";

/// Bind Gauss-Seidel function to Python module.
/// @tparam T Scalar type
/// @param m Python module
template <std::floating_point T>
void bind_gauss_seidel(nanobind::module_& m) {
  m.def(
      "gauss_seidel",
      [](const EigenDRef<RowMajorMatrix<T>>& grid,
         const config::fill::GaussSeidel& config) -> std::tuple<size_t, T> {
        return gauss_seidel(grid, config);
      },
      nanobind::arg("grid"), nanobind::arg("config"), kGaussSeidelDocstring,
      nanobind::call_guard<nanobind::gil_scoped_release>());
}

void bind_gauss_seidel(nanobind::module_& m) {
  bind_gauss_seidel<double>(m);
  bind_gauss_seidel<float>(m);
}

}  // namespace pyinterp::fill::pybind
