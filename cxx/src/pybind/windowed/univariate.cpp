// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/pybind/windowed/univariate.hpp"

#include <nanobind/nanobind.h>

#include "pyinterp/pybind/grid.hpp"
#include "pyinterp/pybind/grid_dispatch.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::windowed::pybind {

constexpr const char* const kUnivariateDocstring = R"doc(
    Perform univariate interpolation on a 1D grid.

Args:
    grid: 1D grid containing data to interpolate.
    x: X-coordinates for interpolation.
    config: Configuration parameters for interpolation.

Returns:
    Vector of interpolated values.

Raises:
    ValueError: If input arrays have mismatched shapes or if interpolation
                cannot be performed due to boundary conditions.
)doc";

constexpr const char* const kUnivariateDerivativeDocstring = R"doc(
    Calculate derivatives on a 1D grid.

Args:
    grid: 1D grid containing data to interpolate.
    x: X-coordinates for derivative calculation.
    config: Configuration parameters for interpolation.

Returns:
    Vector of derivative values.

Raises:
    ValueError: If input arrays have mismatched shapes or if derivative
                calculation cannot be performed due to boundary conditions.
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

/// @brief Functor for windowed univariate interpolation dispatch
struct UnivariateInterpolator {
  /// @brief Call operator for 1D grids
  template <typename DataType, typename ResultType, typename GridType>
  auto operator()(const GridType& grid,
                  const Eigen::Ref<const Eigen::VectorXd>& x,
                  const config::windowed::Univariate& config) const
      -> Vector<ResultType> {
    nb::gil_scoped_release release;
    return univariate<DataType, ResultType>(grid, x, config);
  }
};

/// @brief Functor for windowed univariate derivative dispatch
struct UnivariateDerivativeCalculator {
  /// @brief Call operator for 1D grids
  template <typename DataType, typename ResultType, typename GridType>
  auto operator()(const GridType& grid,
                  const Eigen::Ref<const Eigen::VectorXd>& x,
                  const config::windowed::Univariate& config) const
      -> Vector<ResultType> {
    nb::gil_scoped_release release;
    return univariate_derivative<DataType, ResultType>(grid, x, config);
  }
};

}  // namespace

auto init_univariate(nb::module_& m) -> void {
  m.def(
      "univariate",
      [](const GridHolder& grid, const Eigen::Ref<const Eigen::VectorXd>& x,
         const config::windowed::Univariate& config) -> nb::object {
        return GridDispatcher<DummyPoint>::dispatch_univariate(
            grid, x, config, UnivariateInterpolator{});
      },
      "grid"_a, "x"_a, "config"_a, kUnivariateDocstring);

  m.def(
      "univariate_derivative",
      [](const GridHolder& grid, const Eigen::Ref<const Eigen::VectorXd>& x,
         const config::windowed::Univariate& config) -> nb::object {
        return GridDispatcher<DummyPoint>::dispatch_univariate(
            grid, x, config, UnivariateDerivativeCalculator{});
      },
      "grid"_a, "x"_a, "config"_a, kUnivariateDerivativeDocstring);
}

}  // namespace pyinterp::windowed::pybind
