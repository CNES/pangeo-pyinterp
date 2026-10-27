// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <concepts>
#include <optional>
#include <tuple>
#include <vector>

#include "pyinterp/geometry/geographic/algorithms/difference.hpp"
#include "pyinterp/geometry/geographic/algorithms/intersection.hpp"
#include "pyinterp/geometry/geographic/algorithms/union.hpp"
#include "pyinterp/pybind/geometry/geographic/overlay.hpp"

// Definition of the overlay operations declared in overlay.hpp. Only the
// translation units instantiating a strategy include this header.

namespace pyinterp::geometry::geographic::pybind {

template <Overlay Operation, typename Result, StrategyMethod Method,
          typename Geometry1, typename Geometry2>
auto overlay_strategy(const Geometry1 &geometry1, const Geometry2 &geometry2,
                      const std::optional<Spheroid> &spheroid)
    -> std::vector<Result> {
  if constexpr (Operation == Overlay::kDifference) {
    static_assert(std::same_as<Result, Polygon>);
    return difference<Geometry1, Geometry2, Method>(geometry1, geometry2,
                                                    spheroid);
  } else if constexpr (Operation == Overlay::kIntersection) {
    if constexpr (std::same_as<Result, Polygon>) {
      return intersection_polygon<Geometry1, Geometry2, Method>(
          geometry1, geometry2, spheroid);
    } else if constexpr (std::same_as<Result, LineString>) {
      return intersection_linestring<Geometry1, Geometry2, Method>(
          geometry1, geometry2, spheroid);
    } else {
      static_assert(std::same_as<Result, Point>);
      return intersection_point<Geometry1, Geometry2, Method>(
          geometry1, geometry2, spheroid);
    }
  } else {
    static_assert(Operation == Overlay::kUnion);
    if constexpr (std::same_as<Result, Polygon>) {
      return union_polygon<Geometry1, Geometry2, Method>(geometry1, geometry2,
                                                         spheroid);
    } else {
      static_assert(std::same_as<Result, LineString>);
      return union_linestring<Geometry1, Geometry2, Method>(
          geometry1, geometry2, spheroid);
    }
  }
}

/// @brief Instantiates an overlay operation for a strategy and a list of
/// geometry pairs
///
/// An explicit instantiation of this class emits addresses(), whose body takes
/// the address of each specialization and therefore forces its definition to
/// be emitted in the same translation unit.
///
/// @tparam Operation Overlay operation to instantiate
/// @tparam Result Type of the geometries returned by the operation
/// @tparam Method Calculation strategy
/// @tparam GeometryPairs std::pair<G1, G2> for each combination
template <Overlay Operation, typename Result, StrategyMethod Method,
          typename... GeometryPairs>
struct OverlayInstances {
  static auto addresses() {
    return std::tuple{
        &overlay_strategy<Operation, Result, Method,
                          typename GeometryPairs::first_type,
                          typename GeometryPairs::second_type>...};
  }
};

}  // namespace pyinterp::geometry::geographic::pybind
