// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/pybind/fill/loess.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include "pyinterp/eigen.hpp"
#include "pyinterp/fill/loess.hpp"

namespace nb = nanobind;

namespace pyinterp::fill::pybind {

constexpr const char* const kLoessDocstring = R"(
Fill undefined values using locally weighted regression (LOESS).

The LOESS algorithm uses a tri-cube weight function to compute weighted
averages within a local window around each point. The weight function is:

.. math::
    w(d) = \begin{cases}
      (1 - |d|^3)^3 & |d| \le 1 \\
      0 & \text{otherwise}
    \end{cases}

Args:
    data: Input 2D array containing data to fill/smooth.
    config: LOESS configuration with:

        - nx: Half-window size along the x-axis (rows).
        - ny: Half-window size along the y-axis (columns).
        - value_type: Which values to process:

            * LoessValueType.kUndefined: Fill only NaN values (iterative)
            * LoessValueType.kDefined: Smooth only non-NaN values
            * LoessValueType.kAll: Both fill and smooth

        - first_guess: Initial guess strategy (zonal_average or zero)
        - is_periodic: Whether the x-axis is periodic (e.g., longitude)
        - max_iterations: Maximum iterations for convergence
        - epsilon: Convergence threshold
        - num_threads: Number of threads (0 = use all available CPUs)

Returns:
    New array with processed values, same shape as input.
)";

/// Bind LOESS function to Python module.
/// @tparam T Scalar type
/// @param m Python module
template <std::floating_point T>
void bind_loess(nanobind::module_& m) {
  m.def(
      "loess",
      [](const EigenDRef<const RowMajorMatrix<T>>& data,
         const config::fill::Loess& config) -> RowMajorMatrix<T> {
        return loess(data, config);
      },
      nanobind::arg("data"), nanobind::arg("config"), kLoessDocstring,
      nanobind::call_guard<nanobind::gil_scoped_release>());
}

void bind_loess(nanobind::module_& m) {
  bind_loess<double>(m);
  bind_loess<float>(m);
}

}  // namespace pyinterp::fill::pybind
