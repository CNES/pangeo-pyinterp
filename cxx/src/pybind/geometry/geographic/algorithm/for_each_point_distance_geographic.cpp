// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "pyinterp/geometry/geographic/algorithms/for_each_point_distance.hpp"
#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::geometry::geographic::pybind {

constexpr auto kForEachPointDistanceDoc = R"doc(
Calculate the distance from each point in a source geometry to a target geometry.

The distance is calculated on the surface of a spheroid (default: WGS84).
Different geodetic calculation strategies are available for accuracy/performance
trade-offs.

Args:
    source: Source geometry containing points (MultiPoint, LineString, or Ring).
    target: Target geometry to measure distance to.
    spheroid: Optional spheroid for geodetic calculations. If not provided, uses
        WGS84 ellipsoid.
    strategy: Calculation strategy.

Returns:
    Array of distances in meters.
)doc";

auto init_for_each_point_distance(nb::module_& m) -> void {
  auto distance_impl = [](const auto& source, const auto& target,
                          const std::optional<Spheroid>& spheroid,
                          const StrategyMethod strategy) -> Eigen::VectorXd {
    nb::gil_scoped_release release;
    return for_each_point_distance(source, target, spheroid, strategy);
  };

  // Bind for MultiPoint
  geometry::pybind::define_for_each_point_single_source_with_strategy<
      decltype(distance_impl), MultiPoint, Spheroid, StrategyMethod,
      CONTAINER_TYPES(geographic)>(m, "for_each_point_distance",
                                   kForEachPointDistanceDoc, distance_impl);

  // Bind for LineString
  geometry::pybind::define_for_each_point_single_source_with_strategy<
      decltype(distance_impl), LineString, Spheroid, StrategyMethod,
      CONTAINER_TYPES(geographic)>(m, "for_each_point_distance",
                                   kForEachPointDistanceDoc, distance_impl);

  // Bind for Ring
  geometry::pybind::define_for_each_point_single_source_with_strategy<
      decltype(distance_impl), Ring, Spheroid, StrategyMethod,
      CONTAINER_TYPES(geographic)>(m, "for_each_point_distance",
                                   kForEachPointDistanceDoc, distance_impl);
}

}  // namespace pyinterp::geometry::geographic::pybind
