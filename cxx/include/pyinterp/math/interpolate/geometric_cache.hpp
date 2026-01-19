// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "pyinterp/math/interpolate/cache_fwd.hpp"

namespace pyinterp::math::interpolate::geometric {

/// Concept for numeric types suitable for interpolation (cache-specific)
/// This is compatible with but distinct from the main Numeric concept
/// to avoid redefinition issues when both headers are included.
template <typename T>
concept CacheNumeric = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

/// Geometric Interpolation Cache
///
/// A lightweight cache specialized for geometric interpolation which always
/// uses exactly 2 points per axis. This cache stores:
/// - 2 coordinate values per axis
/// - 2^N corner values (4 for 2D, 8 for 3D, 16 for 4D)
/// - Domain bounds for fast containment checking
///
/// @tparam DataType Data type of grid values
/// @tparam AxisTypes Types of each axis (e.g., double, double for 2D)
template <CacheNumeric DataType, typename... AxisTypes>
class InterpolationCache {
 public:
  /// Number of dimensions
  static constexpr size_t kNDim = sizeof...(AxisTypes);
  /// Number of corner values (2^N)
  static constexpr size_t kNumValues = (1 << kNDim);

  static_assert(kNDim >= 1 && kNDim <= 4, "Cache supports 1-4 dimensions");

  /// Default constructor
  InterpolationCache() = default;

  /// Check if the query point lies within the cached cell
  /// @tparam Coords Coordinate types
  /// @param[in] query_coords The query coordinates
  /// @return True if the point is within the cached cell
  template <typename... Coords>
    requires(sizeof...(Coords) == kNDim)
  [[nodiscard]] auto contains(Coords... query_coords) const noexcept -> bool {
    return contains_impl(std::index_sequence_for<AxisTypes...>{},
                         query_coords...);
  }

  /// Check if the cache has a valid domain
  /// @return True if the cache has a valid domain
  [[nodiscard]] auto has_domain() const noexcept -> bool {
    return std::apply([](const auto&... args) { return (args.valid && ...); },
                      domains_);
  }

  /// Check if all values are valid (non-NaN for floating-point types)
  /// @return True if all values are valid
  [[nodiscard]] auto is_valid() const noexcept -> bool {
    if constexpr (std::floating_point<DataType>) {
      return std::none_of(values_.begin(), values_.end(),
                          [](auto v) { return std::isnan(v); });
    }
    return true;
  }

  /// Get the lower coordinate value for dimension I
  /// @tparam I Dimension index
  /// @return The lower coordinate value
  template <size_t I>
    requires(I < kNDim)
  [[nodiscard]] constexpr auto coord_lower() const noexcept {
    return std::get<I>(coords_)[0];
  }

  /// Get the upper coordinate value for dimension I
  /// @tparam I Dimension index
  /// @return The upper coordinate value
  template <size_t I>
    requires(I < kNDim)
  [[nodiscard]] constexpr auto coord_upper() const noexcept {
    return std::get<I>(coords_)[1];
  }

  /// Get coordinate array for dimension I
  /// @tparam I Dimension index
  /// @return Reference to the coordinate array [lower, upper]
  template <size_t I>
    requires(I < kNDim)
  [[nodiscard]] constexpr auto coords() const noexcept -> const
      std::array<std::tuple_element_t<I, std::tuple<AxisTypes...>>, 2>& {
    return std::get<I>(coords_);
  }

  /// Get mutable coordinate array for dimension I
  /// @tparam I Dimension index
  /// @return Reference to the coordinate array [lower, upper]
  template <size_t I>
    requires(I < kNDim)
  [[nodiscard]] constexpr auto coords() noexcept
      -> std::array<std::tuple_element_t<I, std::tuple<AxisTypes...>>, 2>& {
    return std::get<I>(coords_);
  }

  /// Set coordinates for dimension I
  /// @tparam I Dimension index
  /// @param[in] lower The lower coordinate value
  /// @param[in] upper The upper coordinate value
  template <size_t I>
    requires(I < kNDim)
  void set_coords(std::tuple_element_t<I, std::tuple<AxisTypes...>> lower,
                  std::tuple_element_t<I, std::tuple<AxisTypes...>> upper) {
    std::get<I>(coords_)[0] = lower;
    std::get<I>(coords_)[1] = upper;
  }

  /// Get corner value by linear index
  ///
  /// For 2D (kNDim=2), indices map as:
  ///   0 = (0,0), 1 = (0,1), 2 = (1,0), 3 = (1,1)
  ///
  /// For 3D (kNDim=3), indices map as:
  ///   0 = (0,0,0), 1 = (0,0,1), 2 = (0,1,0), 3 = (0,1,1),
  ///   4 = (1,0,0), 5 = (1,0,1), 6 = (1,1,0), 7 = (1,1,1)
  ///
  /// @param[in] index Linear corner index (0 to 2^N - 1)
  /// @return The corner value
  [[nodiscard]] constexpr auto value(size_t index) const noexcept -> DataType {
    return values_[index];
  }

  /// Get mutable reference to corner value
  /// @param[in] index Linear corner index
  /// @return Reference to the corner value
  [[nodiscard]] constexpr auto value(size_t index) noexcept -> DataType& {
    return values_[index];
  }

  /// Get corner value by multi-dimensional indices
  /// @tparam Indices Index types (each 0 or 1)
  /// @param[in] indices Indices for each dimension (each 0 or 1)
  /// @return The corner value
  template <typename... Indices>
    requires(sizeof...(Indices) == kNDim)
  [[nodiscard]] constexpr auto value_at(Indices... indices) const noexcept
      -> DataType {
    return values_[flatten_index(indices...)];
  }

  /// Set corner value by multi-dimensional indices
  /// @tparam Indices Index types (each 0 or 1)
  /// @param[in] val The value to set
  /// @param[in] indices Indices for each dimension (each 0 or 1)
  template <typename... Indices>
    requires(sizeof...(Indices) == kNDim)
  void set_value_at(DataType val, Indices... indices) {
    values_[flatten_index(indices...)] = val;
  }

  /// Direct access to values array
  /// @return Reference to the values array
  [[nodiscard]] constexpr auto values() noexcept
      -> std::array<DataType, kNumValues>& {
    return values_;
  }

  /// @copydoc values()
  [[nodiscard]] constexpr auto values() const noexcept
      -> const std::array<DataType, kNumValues>& {
    return values_;
  }

  /// Finalize the cache by updating domain bounds
  void finalize() { finalize_impl(std::index_sequence_for<AxisTypes...>{}); }

  /// Invalidate the cache (clear domain)
  void invalidate() noexcept {
    std::apply([](auto&... args) { ((args.valid = false), ...); }, domains_);
  }

 private:
  /// Coordinate values for each dimension [lower, upper]
  std::tuple<std::array<AxisTypes, 2>...> coords_{};
  /// Corner values (2^N values)
  std::array<DataType, kNumValues> values_{};
  /// Domain bounds for each dimension
  std::tuple<DomainBounds<AxisTypes>...> domains_{};

  /// Flatten N-dimensional indices (each 0 or 1) to linear index
  template <typename... Indices>
  [[nodiscard]] static constexpr auto flatten_index(Indices... indices)
      -> size_t {
    size_t result = 0;
    size_t dim = 0;
    // Row-major order: first index varies slowest
    ((result |= (static_cast<size_t>(indices) << (kNDim - 1 - dim++))), ...);
    return result;
  }

  /// Implementation of contains using index sequence
  template <size_t... Is, typename... Coords>
  [[nodiscard]] auto contains_impl(std::index_sequence<Is...>,
                                   Coords... coords) const noexcept -> bool {
    return ((std::get<Is>(domains_).contains(coords)) && ...);
  }

  /// Implementation of finalize using index sequence
  template <size_t... Is>
  void finalize_impl(std::index_sequence<Is...>) {
    (update_domain<Is>(), ...);
  }

  /// Update domain bounds for dimension I
  template <size_t I>
  void update_domain() {
    const auto& coord_arr = std::get<I>(coords_);
    auto& domain = std::get<I>(domains_);
    domain.min = coord_arr[0];
    domain.max = coord_arr[1];
    domain.valid = true;
  }
};

/// Type alias for 2D geometric cache
/// @tparam DataType Data type of grid values
template <CacheNumeric DataType>
using Cache2D = InterpolationCache<DataType, double, double>;

/// Type alias for 3D geometric cache
/// @tparam DataType Data type of grid values
/// @tparam ZType Type of the third axis (e.g., double or int64_t for temporal)
template <CacheNumeric DataType, typename ZType = double>
using Cache3D = InterpolationCache<DataType, double, double, ZType>;

/// Type alias for 4D geometric cache
/// @tparam DataType Data type of grid values
/// @tparam ZType Type of the third axis (e.g., double or int64_t for temporal)
template <CacheNumeric DataType, typename ZType = double>
using Cache4D = InterpolationCache<DataType, double, double, ZType, double>;

}  // namespace pyinterp::math::interpolate::geometric
