// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <nanobind/nanobind.h>

#include "pyinterp/pybind/grid.hpp"
#include "pyinterp/pybind/grid_dispatch.hpp"
#include "pyinterp/pybind/windowed/trivariate.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::windowed::pybind {

constexpr const char* const kTrivariateDocstring = R"doc(
Perform trivariate interpolation on a 3D grid using windowed approach.

Args:
    grid: 3D grid containing data to interpolate.
    x: X-coordinates for interpolation.
    y: Y-coordinates for interpolation.
    z: Z-coordinates (third axis) for interpolation.
    config: Configuration parameters for interpolation.

Returns:
    Vector of interpolated values.

Raises:
    ValueError: If a point is out of the grid bounds
                and `config.common.bounds_error` is set to `True`.
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

/// @brief Functor for windowed trivariate interpolation dispatch
struct TrivariateInterpolator {
  /// @brief Call operator for 3D grids
  template <typename DataType, typename ResultType, typename ZType,
            typename GridType>
  auto operator()(const GridType& grid,
                  const Eigen::Ref<const Eigen::VectorXd>& x,
                  const Eigen::Ref<const Eigen::VectorXd>& y,
                  const Eigen::Ref<const Vector<ZType>>& z,
                  const config::windowed::Trivariate& config) const
      -> Vector<ResultType> {
    nb::gil_scoped_release release;
    return trivariate<GridType, ResultType, ZType>(grid, x, y, z, config);
  }
};

}  // namespace

auto init_trivariate(nb::module_& m) -> void {
  m.def(
      "trivariate",
      [](const GridHolder& grid, const Eigen::Ref<const Eigen::VectorXd>& x,
         const Eigen::Ref<const Eigen::VectorXd>& y, const nb::object& z,
         const config::windowed::Trivariate& config) -> nb::object {
        return GridDispatcher<DummyPoint>::dispatch_trivariate(
            grid, x, y, z, config, TrivariateInterpolator{});
      },
      "grid"_a, "x"_a, "y"_a, "z"_a, "config"_a, kTrivariateDocstring);
}

}  // namespace pyinterp::windowed::pybind
