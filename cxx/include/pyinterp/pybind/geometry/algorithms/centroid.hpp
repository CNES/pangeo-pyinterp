// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>
#include <boost/geometry/algorithms/centroid.hpp>

#include "pyinterp/geometry/cartesian/point.hpp"
#include "pyinterp/geometry/geographic/box.hpp"
#include "pyinterp/geometry/geographic/segment.hpp"
#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace pyinterp::geometry::pybind {

constexpr auto kCentroidDoc = R"doc(
Calculate the centroid of a geometry.

Args:
    geometry: Geometric object to calculate the centroid of.

Returns:
    The centroid of the geometry.
)doc";

template <typename Geometry, typename Point>
auto centroid(const Geometry& g) -> Point {
  nanobind::gil_scoped_release release;
  Point pt;
  boost::geometry::centroid(g, pt);
  return pt;
};

/// @brief Initialize the centroid algorithm in the given module
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
template <GeometryNamespace NS>
inline auto init_centroid(nanobind::module_& m) -> void {
  if constexpr (NS == GeometryNamespace::kCartesian) {
    auto centroid_impl = [](auto&& geometry) {
      using GeometryType = std::decay_t<decltype(geometry)>;
      return centroid<GeometryType, cartesian::Point>(geometry);
    };
    geometry::pybind::define_unary_predicate<decltype(centroid_impl),
                                             GEOMETRY_TYPES(cartesian)>(
        m, "centroid", kCentroidDoc, std::move(centroid_impl));
  } else {
    m.def("centroid", &centroid<geographic::Box, geographic::Point>,
          kCentroidDoc);
    m.def("centroid", &centroid<geographic::Segment, geographic::Point>,
          kCentroidDoc);
  }
}

}  // namespace pyinterp::geometry::pybind
