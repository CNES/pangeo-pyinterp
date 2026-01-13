// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "pyinterp/math.hpp"
#include "pyinterp/math/axis.hpp"
#include "pyinterp/math/interpolate/cache.hpp"

namespace pyinterp::math::interpolate {

/// @brief Result of a cache loading operation
struct CacheLoadResult {
  /// @brief True if the cache was loaded successfully
  bool success{false};
  /// @brief True if the cache was updated
  bool was_updated{false};
  /// @brief Error message if loading failed
  std::optional<std::string> error_message;
};

namespace detail {

/// Generic cache loader for N-dimensions
/// @tparam DataType Type of data stored in the grid and cache
/// @tparam GridType Type of the grid
/// @tparam AxisTypes Types of the axes
/// @param[in] grid The data grid to load from
/// @param[in] query_coords The query coordinates
/// @param[in] boundary Boundary handling mode (allowed modes: kUndef or kShrink
/// only; other modes have undefined behavior)
/// @param[in] bounds_error Whether to raise an error on out-of-bounds access
/// @param[out] cache The interpolation cache to update
/// @param[out] error_out Optional output for error description
/// @return True if the cache was loaded; false otherwise. On failure, error_out
/// may contain an error description when bounds_error is true and the requested
/// coordinates are out of bounds; otherwise the failure occurred because the
/// interpolation window extends beyond the grid bounds.
template <typename DataType, typename GridType, typename... AxisTypes>
auto load_cache_generic(const GridType& grid,
                        const std::tuple<AxisTypes...>& query_coords,
                        axis::Boundary boundary, bool bounds_error,
                        InterpolationCache<DataType, AxisTypes...>& cache,
                        std::optional<std::string>& error_out) -> bool {
  static constexpr size_t kNDim = GridType::kNDim;

  // 1. Find indices for all axes
  std::array<std::vector<int64_t>, kNDim> grid_indices;
  std::array<std::pair<int64_t, int64_t>, kNDim> bracketing_indices;
  std::array<size_t, kNDim> points_per_dim;
  bool success = true;

  auto find_and_store_indices = [&]<size_t I>(
                                    std::integral_constant<size_t, I>) {
    if (!success) return;  // Short-circuit if previous dimension failed

    const auto& val = std::get<I>(query_coords);
    const auto& ax = grid.template axis<I>();

    auto indices =
        ax.find_indexes(val, cache.template half_window<I>(), boundary);

    if (!indices) {
      // Indices may be not found for two reasons:
      // - The query coordinate is outside the axis domain: set success = false;
      //   if bounds_error is true, set error_out.
      // - The interpolation window extends beyond the axis bounds:
      //   set success = false and leave error_out unset.
      success = false;
      if (bounds_error && !ax.contains(val)) {
        error_out = grid.template construct_bounds_error_description<I>(val);
      }
      return;
    }
    grid_indices[I] = std::move(indices->first);
    bracketing_indices[I] = std::move(indices->second);
    points_per_dim[I] = grid_indices[I].size();
  };

  // Iterate 0..N-1
  [&]<size_t... Is>(std::index_sequence<Is...>) {
    (find_and_store_indices(std::integral_constant<size_t, Is>{}), ...);
  }(std::make_index_sequence<kNDim>{});

  if (!success) return false;

  // Resize the cache if necessary
  bool resized = false;
  for (size_t i = 0; i < kNDim; ++i) {
    if (points_per_dim[i] != cache.points_per_dim(i)) {
      resized = true;
      break;
    }
  }
  if (resized) {
    cache.resize(std::move(points_per_dim));
  }

  // 2. Load Coordinates into Cache
  auto load_coords = [&]<size_t I>(std::integral_constant<size_t, I>) {
    const auto& ax = grid.template axis<I>();
    const auto& indices = grid_indices[I];

    // Handle Periodicity (Only X-axis/Index 0 is supported as periodic)
    std::pair<double, double> periodicity{0.0, 0.0};

    if constexpr (I == 0) {
      if (ax.is_periodic()) {
        // Use the query coordinate as reference to ensure cached coordinates
        // are in the same period representation as the query coordinate
        periodicity.first = ax.period().value();
        periodicity.second = std::get<I>(query_coords);
      }
    }

    for (size_t k = 0; k < indices.size(); ++k) {
      auto raw_val = ax(indices[k]);
      if constexpr (I == 0) {
        if (periodicity.first != 0.0) {
          // Normalize raw_val to be within [x_query - period/2, x_query +
          // period/2)
          raw_val = math::normalize_period_half(raw_val, periodicity.second,
                                                periodicity.first);
        }
      }
      cache.template set_coord<I>(k, raw_val);
    }
  };

  [&]<size_t... Is>(std::index_sequence<Is...>) {
    (load_coords(std::integral_constant<size_t, Is>{}), ...);
  }(std::make_index_sequence<kNDim>{});

  // 3. Load Values (Generic N-Loop)
  auto& flat_values = cache.values_flat();
  size_t flat_idx = 0;

  // Use array to collect indices, then std::apply to call grid.value
  std::array<int64_t, kNDim> current_indices;

  auto recursive_copy = [&](this auto&& self, size_t dim) -> void {
    if (dim == kNDim) {
      // Base case: we have all indices, call grid.value
      // Convert array to tuple and apply to grid.value
      flat_values[flat_idx++] = std::apply(
          [&](auto... indices) {
            return static_cast<DataType>(grid.value(indices...));
          },
          [&]<size_t... Is>(std::index_sequence<Is...>) {
            return std::make_tuple(current_indices[Is]...);
          }(std::make_index_sequence<kNDim>{}));
      return;
    }

    // Recursively iterate through this dimension
    for (int64_t grid_idx : grid_indices[dim]) {
      current_indices[dim] = grid_idx;
      self(dim + 1);
    }
  };

  recursive_copy(0);

  cache.finalize(bracketing_indices);
  return cache.is_valid();
}

}  // namespace detail

/// @brief Update the interpolation cache if the query coordinates are outside
/// the currently loaded domain.
/// @tparam DataType Type of data stored in the grid and cache
/// @tparam GridType Type of the grid
/// @tparam AxisTypes Types of the axes
/// @param[in,out] cache The interpolation cache to update
/// @param[in] grid The data grid to load from
/// @param[in] coords The query coordinates
/// @param[in] boundary Boundary handling mode
/// @param[in] bounds_error Whether to raise an error on out-of-bounds access
/// @return Result of the cache loading operation
template <typename DataType, typename GridType, typename... AxisTypes>
[[nodiscard]] auto update_cache_if_needed(
    InterpolationCache<DataType, AxisTypes...>& cache, const GridType& grid,
    const std::tuple<AxisTypes...>& coords,
    axis::Boundary boundary = axis::Boundary::kUndef, bool bounds_error = false)
    -> CacheLoadResult {
  // 1. Check if update is needed (Fast path)
  bool in_domain = std::apply(
      [&](const auto&... args) {
        return cache.has_domain() && cache.contains(args...);
      },
      coords);

  if (in_domain) {
    return {
        .success = true, .was_updated = false, .error_message = std::nullopt};
  }

  // 2. Reload
  auto result = CacheLoadResult{.was_updated = true};

  // Pass bounds_error and reference to result.error_message
  result.success = detail::load_cache_generic(
      grid, coords, boundary, bounds_error, cache, result.error_message);
  return result;
}

}  // namespace pyinterp::math::interpolate
