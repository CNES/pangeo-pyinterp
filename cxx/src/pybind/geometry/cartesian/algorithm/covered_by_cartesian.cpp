// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace nb = nanobind;

namespace pyinterp::geometry::cartesian::pybind {

constexpr auto kCoveredByDoc = R"doc(
Checks if the first geometry is covered by the second geometry.

Args:
    geometry1: First geometric object.
    geometry2: Second geometric object.

Returns:
    True if geometry1 is covered by geometry2, False otherwise.
)doc";

auto init_covered_by(nb::module_& m) -> void {
  auto covered_by_impl = [](const auto& geometry1,
                            const auto& geometry2) -> bool {
    nb::gil_scoped_release release;
    return boost::geometry::covered_by(geometry1, geometry2);
  };

// Define all valid pairs of geometry types for covered_by calculation
#define PAIRS(NS)                                                              \
  std::pair<NS::Point, NS::Point>, std::pair<NS::Point, NS::Segment>,          \
      std::pair<NS::Point, NS::Box>, std::pair<NS::Point, NS::LineString>,     \
      std::pair<NS::Point, NS::Ring>, std::pair<NS::Point, NS::Polygon>,       \
      std::pair<NS::Point, NS::MultiPoint>,                                    \
      std::pair<NS::Point, NS::MultiLineString>,                               \
      std::pair<NS::Point, NS::MultiPolygon>, std::pair<NS::Box, NS::Box>,     \
      std::pair<NS::Box, NS::Ring>, std::pair<NS::Box, NS::Polygon>,           \
      std::pair<NS::Box, NS::MultiPolygon>,                                    \
      std::pair<NS::LineString, NS::Box>,                                      \
      std::pair<NS::LineString, NS::LineString>,                               \
      std::pair<NS::LineString, NS::Ring>,                                     \
      std::pair<NS::LineString, NS::Polygon>,                                  \
      std::pair<NS::LineString, NS::MultiLineString>,                          \
      std::pair<NS::LineString, NS::MultiPolygon>,                             \
      std::pair<NS::Ring, NS::Box>, std::pair<NS::Ring, NS::Ring>,             \
      std::pair<NS::Ring, NS::Polygon>, std::pair<NS::Ring, NS::MultiPolygon>, \
      std::pair<NS::Polygon, NS::Box>, std::pair<NS::Polygon, NS::Ring>,       \
      std::pair<NS::Polygon, NS::Polygon>,                                     \
      std::pair<NS::Polygon, NS::MultiPolygon>,                                \
      std::pair<NS::MultiPoint, NS::Segment>,                                  \
      std::pair<NS::MultiPoint, NS::Box>,                                      \
      std::pair<NS::MultiPoint, NS::LineString>,                               \
      std::pair<NS::MultiPoint, NS::Ring>,                                     \
      std::pair<NS::MultiPoint, NS::Polygon>,                                  \
      std::pair<NS::MultiPoint, NS::MultiPoint>,                               \
      std::pair<NS::MultiPoint, NS::MultiLineString>,                          \
      std::pair<NS::MultiPoint, NS::MultiPolygon>,                             \
      std::pair<NS::MultiLineString, NS::Box>,                                 \
      std::pair<NS::MultiLineString, NS::LineString>,                          \
      std::pair<NS::MultiLineString, NS::Ring>,                                \
      std::pair<NS::MultiLineString, NS::Polygon>,                             \
      std::pair<NS::MultiLineString, NS::MultiLineString>,                     \
      std::pair<NS::MultiLineString, NS::MultiPolygon>,                        \
      std::pair<NS::MultiPolygon, NS::Box>,                                    \
      std::pair<NS::MultiPolygon, NS::Ring>,                                   \
      std::pair<NS::MultiPolygon, NS::Polygon>,                                \
      std::pair<NS::MultiPolygon, NS::MultiPolygon>

  geometry::pybind::define_binary_predicate<decltype(covered_by_impl),
                                            PAIRS(cartesian)>(
      m, "covered_by", kCoveredByDoc, std::move(covered_by_impl));
}

}  // namespace pyinterp::geometry::cartesian::pybind
