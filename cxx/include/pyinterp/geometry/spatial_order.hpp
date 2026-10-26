// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "pyinterp/geometry/point_traits.hpp"

namespace pyinterp::geometry {

/// @brief Order in which to process points so that consecutive points are
/// close in space.
///
/// Sorts the points along a Morton (Z-order) curve of their bounding box.
/// Searching an R-tree for points in this order reuses the nodes already in
/// cache, where an arbitrary order (random query points) reloads them for
/// every search.
///
/// @tparam Point Point type
/// @param[in] points The points to order
/// @return The indexes of the points, in processing order
template <typename Point>
[[nodiscard]] auto spatial_order(const std::vector<Point>& points)
    -> std::vector<int64_t> {
  constexpr auto kDim = point::dimension_v<Point>;
  // Bits per coordinate so that the interleaved key fits in 64 bits
  constexpr auto kBits = 64 / kDim;
  constexpr auto kCells = static_cast<double>((uint64_t{1} << kBits) - 1);

  const auto size = points.size();
  auto order = std::vector<int64_t>(size);
  std::iota(order.begin(), order.end(), int64_t{0});
  if (size < 2) {
    return order;
  }

  // Bounding box of the finite coordinates
  auto lower = std::vector<double>(kDim, std::numeric_limits<double>::max());
  auto upper = std::vector<double>(kDim, std::numeric_limits<double>::lowest());
  for (const auto& point : points) {
    for (size_t dim = 0; dim < kDim; ++dim) {
      const auto value = static_cast<double>(point::get(point, dim));
      if (std::isfinite(value)) {
        lower[dim] = std::min(lower[dim], value);
        upper[dim] = std::max(upper[dim], value);
      }
    }
  }

  auto keys = std::vector<uint64_t>(size);
  for (size_t ix = 0; ix < size; ++ix) {
    uint64_t key = 0;
    for (size_t dim = 0; dim < kDim; ++dim) {
      const auto extent = upper[dim] - lower[dim];
      const auto value = static_cast<double>(point::get(points[ix], dim));
      // Undefined coordinates are sorted with the lowest ones: the searches
      // will reject them anyway.
      const auto scaled = extent > 0 && std::isfinite(value)
                              ? (value - lower[dim]) / extent * kCells
                              : 0.0;
      const auto cell = static_cast<uint64_t>(std::clamp(scaled, 0.0, kCells));
      for (size_t bit = 0; bit < kBits; ++bit) {
        key |= ((cell >> bit) & 1U) << (bit * kDim + dim);
      }
    }
    keys[ix] = key;
  }

  std::ranges::sort(order, [&keys](const int64_t a, const int64_t b) {
    return keys[a] < keys[b];
  });
  return order;
}

}  // namespace pyinterp::geometry
