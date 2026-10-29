// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
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
  std::ranges::iota(order, int64_t{0});
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

  // Spreads the bits of a byte kDim positions apart.
  constexpr auto kSpread = [] {
    auto table = std::array<uint64_t, 256>{};
    for (uint64_t byte = 0; byte < 256; ++byte) {
      for (size_t bit = 0; bit < 8; ++bit) {
        table[byte] |= ((byte >> bit) & 1U) << (bit * kDim);
      }
    }
    return table;
  }();

  // (key, index) pairs, sorted by key below
  auto items = std::vector<std::pair<uint64_t, int64_t>>(size);
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
      for (size_t byte = 0; byte * 8 < kBits; ++byte) {
        key |= kSpread[(cell >> (byte * 8)) & 0xFFU] << (byte * 8 * kDim + dim);
      }
    }
    items[ix] = {key, static_cast<int64_t>(ix)};
  }

  // Least significant digit radix sort, one byte of the keys at a time.
  // Bytes equal for all the keys are skipped.
  auto histogram = std::vector<std::array<size_t, 256>>(8);
  for (const auto& item : items) {
    for (size_t byte = 0; byte < 8; ++byte) {
      ++histogram[byte][(item.first >> (byte * 8)) & 0xFFU];
    }
  }
  auto buffer = std::vector<std::pair<uint64_t, int64_t>>(size);
  for (size_t byte = 0; byte < 8; ++byte) {
    auto& offsets = histogram[byte];
    if (std::ranges::find(offsets, size) != offsets.end()) {
      continue;
    }
    std::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin(),
                        size_t{0});
    for (const auto& item : items) {
      buffer[offsets[(item.first >> (byte * 8)) & 0xFFU]++] = item;
    }
    std::swap(items, buffer);
  }

  for (size_t ix = 0; ix < size; ++ix) {
    order[ix] = items[ix].second;
  }
  return order;
}

}  // namespace pyinterp::geometry
