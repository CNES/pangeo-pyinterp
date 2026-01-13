// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <nanobind/nanobind.h>

#include "pyinterp/pybind/grid.hpp"
#include "pyinterp/pybind/grid_dispatch.hpp"
#include "pyinterp/pybind/windowed/bivariate.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::windowed::pybind {

constexpr const char* const kBivariateDocstring = R"doc(
    Perform bivariate interpolation on a 2D grid.

Args:
    grid: 2D grid containing data to interpolate.
    x: X-coordinates for interpolation.
    y: Y-coordinates for interpolation.
    config: Configuration parameters for interpolation.

Returns:
    Vector of interpolated values.

Raises:
    ValueError: If input arrays have mismatched shapes or if interpolation
                cannot be performed due to boundary conditions.
)doc";

// Dummy point type for dispatcher (windowed doesn't use Point template)
template <typename T>
struct DummyPoint {};

// Define GridHolder alias for convenience
using GridHolder = pyinterp::pybind::GridHolder;

// Define GridDispatcher alias for convenience
template <template <class> class PointType>
using GridDispatcher = pyinterp::pybind::GridDispatcher<PointType>;

namespace {

/// @brief Functor for windowed bivariate interpolation dispatch
struct BivariateInterpolator {
  /// @brief Call operator for 2D grids
  template <typename DataType, typename ResultType, typename GridType>
  auto operator()(const GridType& grid,
                  const Eigen::Ref<const Eigen::VectorXd>& x,
                  const Eigen::Ref<const Eigen::VectorXd>& y,
                  const config::windowed::Bivariate& config) const
      -> Vector<ResultType> {
    nb::gil_scoped_release release;
    return bivariate<DataType, ResultType>(grid, x, y, config);
  }
};

}  // namespace

auto init_bivariate(nb::module_& m) -> void {
  m.def(
      "bivariate",
      [](const GridHolder& grid, const Eigen::Ref<const Eigen::VectorXd>& x,
         const Eigen::Ref<const Eigen::VectorXd>& y,
         const config::windowed::Bivariate& config) -> nb::object {
        return GridDispatcher<DummyPoint>::dispatch_bivariate(
            grid, x, y, config, BivariateInterpolator{});
      },
      "grid"_a, "x"_a, "y"_a, "config"_a, kBivariateDocstring);
}

}  // namespace pyinterp::windowed::pybind
