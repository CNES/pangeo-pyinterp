// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::geometry::cartesian::pybind {

constexpr auto kForEachPointDistanceDoc = R"doc(
Calculate the distance from each point in a source geometry to a target geometry.

Args:
    source: Source geometry containing points (MultiPoint, LineString, or Ring).
    target: Target geometry to measure distance to.

Returns:
    Array of distances in coordinate units.
)doc";

// Calculate distance from each point in source to target geometry
template <typename SourceGeometry, typename TargetGeometry>
[[nodiscard]] inline auto for_each_point_distance(const SourceGeometry& source,
                                                  const TargetGeometry& target)
    -> Eigen::VectorXd {
  Eigen::VectorXd result(source.size());
  for (auto [distance, item] : std::ranges::views::zip(result, source)) {
    distance = boost::geometry::distance(item, target);
  }
  return result;
}

auto init_for_each_point_distance(nb::module_& m) -> void {
  auto distance_impl = [](const auto& source,
                          const auto& target) -> Eigen::VectorXd {
    nb::gil_scoped_release release;
    return for_each_point_distance(source, target);
  };

  // Bind for MultiPoint
  geometry::pybind::define_for_each_point_single_source<
      decltype(distance_impl), MultiPoint, CONTAINER_TYPES(cartesian)>(
      m, "for_each_point_distance", kForEachPointDistanceDoc, distance_impl);

  // Bind for LineString
  geometry::pybind::define_for_each_point_single_source<
      decltype(distance_impl), LineString, CONTAINER_TYPES(cartesian)>(
      m, "for_each_point_distance", kForEachPointDistanceDoc, distance_impl);

  // Bind for Ring
  geometry::pybind::define_for_each_point_single_source<
      decltype(distance_impl), Ring, CONTAINER_TYPES(cartesian)>(
      m, "for_each_point_distance", kForEachPointDistanceDoc, distance_impl);
}

}  // namespace pyinterp::geometry::cartesian::pybind
