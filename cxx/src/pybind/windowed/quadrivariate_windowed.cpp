// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <nanobind/nanobind.h>

#include "pyinterp/pybind/grid.hpp"
#include "pyinterp/pybind/grid_dispatch.hpp"
#include "pyinterp/pybind/windowed/quadrivariate.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::windowed::pybind {

constexpr const char* const kQuadrivariateDocstring = R"doc(
Perform quadrivariate interpolation on a 4D grid using windowed approach.

Args:
    grid: 4D grid containing data to interpolate.
    x: X-coordinates for interpolation.
    y: Y-coordinates for interpolation.
    z: Z-coordinates (third axis) for interpolation.
    u: U-coordinates (fourth axis) for interpolation.
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

/// @brief Functor for windowed quadrivariate interpolation dispatch
struct QuadrivariateInterpolator {
  /// @brief Call operator for 4D grids
  template <typename DataType, typename ResultType, typename ZType,
            typename GridType>
  auto operator()(const GridType& grid,
                  const Eigen::Ref<const Vector<double>>& x,
                  const Eigen::Ref<const Vector<double>>& y,
                  const Eigen::Ref<const Vector<ZType>>& z,
                  const Eigen::Ref<const Vector<double>>& u,
                  const config::windowed::Quadrivariate& config) const
      -> Vector<ResultType> {
    nb::gil_scoped_release release;
    return quadrivariate<GridType, ResultType, ZType>(grid, x, y, z, u, config);
  }
};

}  // namespace

auto init_quadrivariate(nb::module_& m) -> void {
  m.def(
      "quadrivariate",
      [](const GridHolder& grid, const Eigen::Ref<const Vector<double>>& x,
         const Eigen::Ref<const Vector<double>>& y, const nb::object& z,
         const Eigen::Ref<const Vector<double>>& u,
         const config::windowed::Quadrivariate& config) -> nb::object {
        return GridDispatcher<DummyPoint>::dispatch_quadrivariate(
            grid, x, y, z, u, config, QuadrivariateInterpolator{});
      },
      "grid"_a, "x"_a, "y"_a, "z"_a, "u"_a, "config"_a,
      kQuadrivariateDocstring);
}

}  // namespace pyinterp::windowed::pybind
