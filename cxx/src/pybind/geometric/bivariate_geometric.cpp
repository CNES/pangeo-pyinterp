// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <nanobind/nanobind.h>

#include "pyinterp/geometry/point.hpp"
#include "pyinterp/pybind/geometric/bivariate.hpp"
#include "pyinterp/pybind/grid.hpp"
#include "pyinterp/pybind/grid_dispatch.hpp"

namespace pyinterp::geometric::pybind {

// Define Point alias for convenience
template <typename T>
using Point = geometry::SphericalPoint<T>;

// Define GridHolder alias for convenience
using GridHolder = pyinterp::pybind::GridHolder;

// Define GridDispatcher alias for convenience
template <template <class> class PointType>
using GridDispatcher = pyinterp::pybind::GridDispatcher<PointType>;

namespace {

/// @brief Functor for bivariate interpolation dispatch
struct BivariateInterpolator {
  /// @brief Call operator for 2D grids
  template <typename DataType, typename ResultType, typename GridType>
  auto operator()(const GridType& grid,
                  const Eigen::Ref<const Eigen::VectorXd>& x,
                  const Eigen::Ref<const Eigen::VectorXd>& y,
                  const config::geometric::Bivariate& config) const
      -> Vector<ResultType> {
    nanobind::gil_scoped_release release;
    return bivariate<Point, DataType, ResultType>(grid, x, y, config);
  }
};

}  // namespace

auto init_bivariate(nanobind::module_& m) -> void {
  namespace nb = nanobind;

  m.def(
      "bivariate",
      [](const GridHolder& grid, const Eigen::Ref<const Eigen::VectorXd>& x,
         const Eigen::Ref<const Eigen::VectorXd>& y,
         const config::geometric::Bivariate& config) -> nb::object {
        return GridDispatcher<Point>::dispatch_bivariate(
            grid, x, y, config, BivariateInterpolator{});
      },
      nb::arg("grid"), nb::arg("x"), nb::arg("y"), nb::arg("config"),
      detail::kBivariateDocstring);
}

}  // namespace pyinterp::geometric::pybind
