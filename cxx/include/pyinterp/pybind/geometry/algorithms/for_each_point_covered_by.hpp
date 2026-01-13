// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace pyinterp::geometry::pybind {

constexpr auto kForEachPointCoveredByDoc = R"doc(
Test if each point in a source geometry is covered by a container geometry.

A point is covered by a container if it lies in the interior or on the
boundary of the container.

Args:
    source: Source geometry containing points (MultiPoint, LineString, or Ring).
    container: Container geometry (Box, Polygon, Ring, or MultiPolygon).

Returns:
    Boolean array indicating which points are covered by the container.

See Also:
    for_each_point_within: Exclude points on the boundary.
)doc";

/// @brief Initialize the for_each_point_covered_by algorithm
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
template <GeometryNamespace NS>
inline auto init_for_each_point_covered_by(nanobind::module_& m) -> void {
  auto covered_by_impl =
      [](const auto& source,
         const auto& container) -> Eigen::Matrix<bool, Eigen::Dynamic, 1> {
    nanobind::gil_scoped_release release;
    Eigen::Matrix<bool, Eigen::Dynamic, 1> result(source.size());
    for (std::size_t i = 0; i < source.size(); ++i) {
      result(i) = boost::geometry::covered_by(source[i], container);
    }
    return result;
  };

  if constexpr (NS == GeometryNamespace::kCartesian) {
    // Bind for MultiPoint
    geometry::pybind::define_for_each_point_single_source<
        decltype(covered_by_impl), cartesian::MultiPoint,
        CONTAINER_TYPES(cartesian)>(m, "for_each_point_covered_by",
                                    kForEachPointCoveredByDoc, covered_by_impl);
    // Bind for LineString
    geometry::pybind::define_for_each_point_single_source<
        decltype(covered_by_impl), cartesian::LineString,
        CONTAINER_TYPES(cartesian)>(m, "for_each_point_covered_by",
                                    kForEachPointCoveredByDoc, covered_by_impl);
    // Bind for Ring
    geometry::pybind::define_for_each_point_single_source<
        decltype(covered_by_impl), cartesian::Ring, CONTAINER_TYPES(cartesian)>(
        m, "for_each_point_covered_by", kForEachPointCoveredByDoc,
        covered_by_impl);
  } else {
    // Bind for MultiPoint
    geometry::pybind::define_for_each_point_single_source<
        decltype(covered_by_impl), geographic::MultiPoint,
        CONTAINER_TYPES(geographic)>(m, "for_each_point_covered_by",
                                     kForEachPointCoveredByDoc,
                                     covered_by_impl);
    // Bind for LineString
    geometry::pybind::define_for_each_point_single_source<
        decltype(covered_by_impl), geographic::LineString,
        CONTAINER_TYPES(geographic)>(m, "for_each_point_covered_by",
                                     kForEachPointCoveredByDoc,
                                     covered_by_impl);
    // Bind for Ring
    geometry::pybind::define_for_each_point_single_source<
        decltype(covered_by_impl), geographic::Ring,
        CONTAINER_TYPES(geographic)>(m, "for_each_point_covered_by",
                                     kForEachPointCoveredByDoc,
                                     covered_by_impl);
  }
}

}  // namespace pyinterp::geometry::pybind
