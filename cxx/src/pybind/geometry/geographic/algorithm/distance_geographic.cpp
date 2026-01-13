// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "pyinterp/geometry/geographic/algorithms/distance.hpp"
#include "pyinterp/geometry/geographic/linestring.hpp"
#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/geometry/geographic/ring.hpp"
#include "pyinterp/geometry/geographic/segment.hpp"
#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::geometry::geographic::pybind {

constexpr auto kDistanceDoc = R"doc(
Calculate the distance between two geometric objects.

The distance is calculated on the surface of a spheroid (default: WGS84).
Different geodetic calculation strategies are available for accuracy/performance
trade-offs.

Args:
    geometry1: First geometric object.
    geometry2: Second geometric object.
    spheroid: Optional spheroid for geodetic calculations. If not provided, uses
        WGS84 ellipsoid.
    strategy: Calculation strategy.
Returns:
    Distance in meters.
)doc";

constexpr auto kCurvilinearDistanceDoc = R"doc(
Calculate the curvilinear distance along the geometric object.

The distance is calculated on the surface of a spheroid (default: WGS84).
Different geodetic calculation strategies are available for accuracy/performance
trade-offs.

Args:
    geometry: Geometric object.
    spheroid: Optional spheroid for geodetic calculations. If not provided, uses
        WGS84 ellipsoid.
    strategy: Calculation strategy.
Returns:
    Curvilinear distance in meters.
)doc";

auto init_distance(nb::module_& m) -> void {
  auto distance_impl = [](const auto& geometry1, const auto& geometry2,
                          const std::optional<Spheroid>& wgs,
                          const StrategyMethod strategy) -> double {
    nb::gil_scoped_release release;
    return distance(geometry1, geometry2, wgs, strategy);
  };

  auto curvilinear_distance_impl =
      [](const auto& geometry, const std::optional<Spheroid>& wgs,
         const StrategyMethod strategy) -> Eigen::VectorXd {
    nb::gil_scoped_release release;
    return curvilinear_distance(geometry, wgs, strategy);
  };

  geometry::pybind::define_binary_predicate_with_strategy<
      decltype(distance_impl), Spheroid, StrategyMethod,
      GEOMETRY_PAIRS(geographic)>(m, "distance", kDistanceDoc,
                                  std::move(distance_impl));

  geometry::pybind::define_unary_predicate_with_strategy<
      decltype(curvilinear_distance_impl), Spheroid, StrategyMethod,
      geographic::LineString, geographic::Ring>(
      m, "curvilinear_distance", kCurvilinearDistanceDoc,
      std::move(curvilinear_distance_impl));
}

}  // namespace pyinterp::geometry::geographic::pybind
