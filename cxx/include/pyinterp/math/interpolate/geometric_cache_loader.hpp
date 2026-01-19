// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "pyinterp/math/interpolate/cache_fwd.hpp"
#include "pyinterp/math/interpolate/geometric_cache.hpp"

namespace pyinterp::math::interpolate::geometric {

namespace detail {

/// Load the geometric cache from the grid
/// @tparam DataType Type of data stored in the cache
/// @tparam GridType Type of the grid
/// @tparam AxisTypes Types of the axes
/// @param[out] cache The interpolation cache to update
/// @param[in] grid The data grid to load from
/// @param[in] query_coords The query coordinates
/// @param[in] bounds_error Whether to raise an error on out-of-bounds access
/// @param[out] error_out Optional output for error description
/// @return True if the cache was loaded successfully
template <typename DataType, typename GridType, typename... AxisTypes>
auto load_cache(InterpolationCache<DataType, AxisTypes...>& cache,
                const GridType& grid,
                const std::tuple<AxisTypes...>& query_coords, bool bounds_error,
                std::optional<std::string>& error_out) -> bool {
  static constexpr size_t kNDim = sizeof...(AxisTypes);

  // Arrays to store found indices
  std::array<std::pair<size_t, size_t>, kNDim> indices;
  bool success = true;

  // Find indices for all axes
  auto find_indices = [&]<size_t I>(std::integral_constant<size_t, I>) {
    if (!success) {
      return;  // Short-circuit if previous dimension failed
    }

    const auto& val = std::get<I>(query_coords);

    auto found = grid.template find_indexes<I>(val, bounds_error);
    if (!found) {
      success = false;
      if (bounds_error) {
        error_out = grid.template construct_bounds_error_description<I>(val);
      }
      return;
    }
    indices[I] = *found;
  };

  // Apply find_indices to all dimensions
  [&]<size_t... Is>(std::index_sequence<Is...>) {
    (find_indices(std::integral_constant<size_t, Is>{}), ...);
  }(std::make_index_sequence<kNDim>{});

  if (!success) {
    cache.invalidate();
    return false;
  }

  // Load coordinates into cache
  auto load_coords = [&]<size_t I>(std::integral_constant<size_t, I>) {
    const auto& ax = grid.template axis<I>();
    const auto [i0, i1] = indices[I];
    cache.template set_coords<I>(ax.coordinate_value(i0),
                                 ax.coordinate_value(i1));
  };

  [&]<size_t... Is>(std::index_sequence<Is...>) {
    (load_coords(std::integral_constant<size_t, Is>{}), ...);
  }(std::make_index_sequence<kNDim>{});

  // Load values into cache using recursive approach
  // For 2D: 4 values, 3D: 8 values, 4D: 16 values
  constexpr size_t num_values =
      InterpolationCache<DataType, AxisTypes...>::kNumValues;

  // Helper to get grid index for a given corner
  auto get_grid_index = [&indices](size_t corner_idx, size_t dim) -> size_t {
    // Extract bit at position (kNDim - 1 - dim) from corner_idx
    size_t bit = (corner_idx >> (kNDim - 1 - dim)) & 1;
    return bit == 0 ? indices[dim].first : indices[dim].second;
  };

  // Load all corner values
  for (size_t corner = 0; corner < num_values; ++corner) {
    // Build index tuple for this corner
    auto get_value = [&]<size_t... Is>(std::index_sequence<Is...>) {
      return static_cast<DataType>(grid.value(get_grid_index(corner, Is)...));
    };
    cache.value(corner) = get_value(std::make_index_sequence<kNDim>{});
  }

  cache.finalize();
  return cache.is_valid();
}

}  // namespace detail

/// @brief Update the geometric interpolation cache if the query coordinates
/// are outside the currently loaded cell.
///
/// This function implements a smart caching strategy:
/// 1. Fast path: If the query point is within the cached cell, return
/// immediately
/// 2. Slow path: If outside, reload the cache from the grid
///
/// @tparam DataType Type of data stored in the cache
/// @tparam GridType Type of the grid
/// @tparam AxisTypes Types of the axes
/// @param[in,out] cache The interpolation cache to update
/// @param[in] grid The data grid to load from
/// @param[in] coords The query coordinates
/// @param[in] bounds_error Whether to raise an error on out-of-bounds access
/// @return Result of the cache loading operation
template <typename DataType, typename GridType, typename... AxisTypes>
[[nodiscard]] auto update_cache_if_needed(
    InterpolationCache<DataType, AxisTypes...>& cache, const GridType& grid,
    const std::tuple<AxisTypes...>& coords, bool bounds_error = false)
    -> CacheLoadResult {
  // Fast path: check if point is in cached cell
  bool in_domain = std::apply(
      [&](const auto&... args) {
        return cache.has_domain() && cache.contains(args...);
      },
      coords);

  if (in_domain) {
    return {
        .success = true, .was_updated = false, .error_message = std::nullopt};
  }

  // Slow path: reload cache
  auto result = CacheLoadResult{.was_updated = true};
  result.success = detail::load_cache(cache, grid, coords, bounds_error,
                                      result.error_message);
  return result;
}

}  // namespace pyinterp::math::interpolate::geometric
