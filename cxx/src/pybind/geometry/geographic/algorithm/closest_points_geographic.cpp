// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <boost/geometry.hpp>
#include <optional>

#include "pyinterp/geometry/geographic/algorithms/closest_points.hpp"
#include "pyinterp/geometry/geographic/algorithms/strategy.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"
#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace nb = nanobind;

namespace pyinterp::geometry::geographic::pybind {

constexpr auto kClosestPointsDoc = R"doc(
Calculate the closest points between two geometries.

The closest points are the pair of points, one on each geometry, that are
closest to each other.

Args:
    geometry1: First geometry.
    geometry2: Second geometry.
    spheroid: Optional spheroid for geodetic calculations. If not provided,
        uses WGS84 ellipsoid.
    strategy: Calculation strategy.

Returns:
    Closest points as a Segment.
)doc";

// Macro to create geometry pairs for the closest_points algorithm
#define PAIRS(NS)                                          \
  std::pair<NS::LineString, NS::LineString>,               \
      std::pair<NS::LineString, NS::MultiLineString>,      \
      std::pair<NS::LineString, NS::MultiPoint>,           \
      std::pair<NS::LineString, NS::MultiPolygon>,         \
      std::pair<NS::LineString, NS::Polygon>,              \
      std::pair<NS::MultiPoint, NS::LineString>,           \
      std::pair<NS::MultiPoint, NS::MultiLineString>,      \
      std::pair<NS::MultiPoint, NS::MultiPoint>,           \
      std::pair<NS::MultiPoint, NS::MultiPolygon>,         \
      std::pair<NS::MultiPoint, NS::Polygon>,              \
      std::pair<NS::MultiLineString, NS::LineString>,      \
      std::pair<NS::MultiLineString, NS::MultiLineString>, \
      std::pair<NS::MultiLineString, NS::MultiPoint>,      \
      std::pair<NS::MultiLineString, NS::MultiPolygon>,    \
      std::pair<NS::MultiLineString, NS::Polygon>,         \
      std::pair<NS::MultiPolygon, NS::LineString>,         \
      std::pair<NS::MultiPolygon, NS::MultiLineString>,    \
      std::pair<NS::MultiPolygon, NS::MultiPoint>,         \
      std::pair<NS::MultiPolygon, NS::MultiPolygon>,       \
      std::pair<NS::MultiPolygon, NS::Polygon>,            \
      std::pair<NS::Polygon, NS::LineString>,              \
      std::pair<NS::Polygon, NS::MultiLineString>,         \
      std::pair<NS::Polygon, NS::MultiPoint>,              \
      std::pair<NS::Polygon, NS::MultiPolygon>,            \
      std::pair<NS::Polygon, NS::Polygon>

auto init_closest_points(nb::module_& m) -> void {
  auto closest_points_impl = [](const auto& geometry1, const auto& geometry2,
                                const std::optional<Spheroid>& spheroid,
                                StrategyMethod strategy) -> Segment {
    nb::gil_scoped_release release;
    return closest_points(geometry1, geometry2, spheroid, strategy);
  };

  geometry::pybind::define_binary_predicate_with_strategy<
      decltype(closest_points_impl), Spheroid, StrategyMethod,
      PAIRS(geographic)>(m, "closest_points", kClosestPointsDoc,
                         std::move(closest_points_impl));
}

}  // namespace pyinterp::geometry::geographic::pybind
