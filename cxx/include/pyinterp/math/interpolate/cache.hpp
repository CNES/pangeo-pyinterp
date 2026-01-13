// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

namespace pyinterp::math::interpolate {

/// Concept for numeric types suitable for interpolation
template <typename T>
concept Numeric = std::floating_point<T> || std::integral<T>;

/// Domain bounds for a single dimension
struct DomainBounds {
  double min{};       /// Minimum bound
  double max{};       /// Maximum bound
  bool valid{false};  /// Indicates if bounds are valid

  /// Check if a value is within bounds (inclusive)
  /// @param[in] value The value to check
  /// @return True if value is within bounds
  [[nodiscard]] constexpr auto contains(double value) const noexcept -> bool {
    return valid && value >= min && value <= max;
  }
};

/// Unified N-Dimensional Interpolation Cache
/// Stores a local window of the grid (coordinates and values)
/// X and Y axes can have independent window sizes
/// Additional axes (Z, U, etc.) always use window_size=2 (4 points, 2 on each
/// side)
/// @tparam DataType Data type loaded in the cache
/// @tparam AxisTypes Types of each axis
template <Numeric DataType, typename... AxisTypes>
class InterpolationCache {
 public:
  /// Number of dimensions
  static constexpr size_t kNDim = sizeof...(AxisTypes);

  // Defines a dynamic stride: OuterStride is the step for X, InnerStride for Y.
  using StrideType = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

  /// A read-only Map view of a vector (1D)
  using VectorMap =
      Eigen::Map<const Eigen::Matrix<DataType, Eigen::Dynamic, 1>>;

  // A read-only Map view of the spatial matrix (X, Y)
  using MatrixMap =
      Eigen::Map<const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>,
                 0, StrideType>;

  /// Default constructor (Disabled to enforce window size)
  InterpolationCache() = delete;

  /// Constructor with independent X and Y window sizes
  /// Additional dimensions (if any) always use half_window_size=1 (2 points)
  /// @param[in] half_x_window_size The half window size for X axis
  /// @param[in] half_y_window_size The half window size for Y axis
  explicit InterpolationCache(size_t half_x_window_size,
                              size_t half_y_window_size);

  /// Access value at coordinates [i, j, k...]
  /// @tparam Indices Index types
  /// @param[in] indices Indices for each dimension
  /// @return The value at the specified indices
  template <typename... Indices>
    requires(sizeof...(Indices) == kNDim)
  [[nodiscard]] constexpr auto operator[](Indices... indices) const
      -> DataType {
    return values_[flatten_index(indices...)];
  }

  /// Mutable access at coordinates [i, j, k...]
  /// @tparam Indices Index types
  /// @param[in] indices Indices for each dimension
  /// @return Reference to the value at the specified indices
  template <typename... Indices>
    requires(sizeof...(Indices) == kNDim)
  constexpr auto operator[](Indices... indices) -> DataType& {
    return values_[flatten_index(indices...)];
  }

  /// Direct access to flat buffer
  [[nodiscard]] auto values_flat() -> std::vector<DataType>& { return values_; }

  /// @copydoc values_flat()
  [[nodiscard]] auto values_flat() const -> const std::vector<DataType>& {
    return values_;
  }

  /// Get generic coordinate vector for dimension I
  /// @tparam I Dimension index
  /// @return Reference to the coordinate vector
  template <size_t I>
  [[nodiscard]] constexpr auto coords()
      -> std::vector<std::tuple_element_t<I, std::tuple<AxisTypes...>>>& {
    return std::get<I>(coords_);
  }

  /// Get generic coordinate vector for dimension I as an Eigen vector
  /// @tparam I Dimension index
  /// @return Eigen vector mapping the coordinate data
  template <size_t I>
  [[nodiscard]] auto coords_as_eigen() -> VectorMap {
    auto& vec = std::get<I>(coords_);
    return VectorMap(vec.data(), vec.size());
  }

  /// Get specific coordinate value
  /// @tparam I Dimension index
  /// @param[in] idx Index within the coordinate vector
  /// @return The coordinate value at the specified index
  template <size_t I>
  [[nodiscard]] auto coord(size_t idx) const {
    return std::get<I>(coords_)[idx];
  }

  /// Set specific coordinate value
  /// @tparam I Dimension index
  /// @param[in] idx Index within the coordinate vector
  /// @param[in] val The coordinate value to set
  template <size_t I, typename T>
  auto set_coord(size_t idx, T val) {
    std::get<I>(coords_)[idx] =
        static_cast<std::tuple_element_t<I, std::tuple<AxisTypes...>>>(val);
  }

  /// Check if the query point lies within the loaded domain
  /// @tparam Coords Coordinate types
  /// @param[in] query_coords The query coordinates
  /// @return True if the point is within the domain
  template <typename... Coords>
    requires(sizeof...(Coords) == kNDim)
  [[nodiscard]] auto contains(Coords... query_coords) const noexcept -> bool {
    return contains_impl(std::index_sequence_for<AxisTypes...>{},
                         query_coords...);
  }

  /// Check if the cache has a valid domain
  /// @return True if the cache has a valid domain
  [[nodiscard]] auto has_domain() const noexcept -> bool {
    return std::ranges::all_of(domains_, [](const auto& d) { return d.valid; });
  }

  /// Finalize the cache by updating domain bounds
  void finalize(const std::array<std::pair<int64_t, int64_t>, kNDim>&
                    bracketing_indices) {
    update_domains(bracketing_indices, std::index_sequence_for<AxisTypes...>{});
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

  /// Get half-window size for specified dimension
  /// @param[in] dim Dimension index
  /// @return Half-window size for the specified dimension
  template <size_t dim>
  [[nodiscard]] constexpr auto half_window() const noexcept -> size_t {
    if constexpr (dim == 0) {
      return x_half_window_size_;
    } else if constexpr (dim == 1) {
      return y_half_window_size_;
    } else {
      return 1;
    }
  }

  /// Get half-window size for X axis
  /// @return Half-window size for X axis
  [[nodiscard]] constexpr auto x_half_window() const noexcept -> size_t {
    return half_window<0>();
  }

  /// Get half-window size for Y axis
  /// @return Half-window size for Y axis
  [[nodiscard]] constexpr auto y_half_window() const noexcept -> size_t {
    return half_window<1>();
  }

  // Get number of points along X axis
  /// @return Number of points along X axis
  [[nodiscard]] auto x_points() const noexcept -> size_t { return x_points_; }

  // Get number of points along Y axis
  /// @return Number of points along Y axis
  [[nodiscard]] auto y_points() const noexcept -> size_t { return y_points_; }

  /// Get number of points along specified dimension
  /// @param[in] dim Dimension index
  /// @return Number of points along the specified dimension
  [[nodiscard]] auto points_per_dim(size_t dim) const noexcept -> size_t {
    return points_per_dim_[dim];
  }

  /// Get a VectorMap view for 1D cache
  /// @return VectorMap of the values
  [[nodiscard]] auto vector() const -> VectorMap {
    static_assert(kNDim == 1, "Vector access requires exactly 1 dimension");
    return VectorMap(values_.data(), points_per_dim_[0]);
  }

  /// Resize the cache to the specified dimensions
  /// @param[in] points_per_dim Number of points for each dimension
  void resize(std::array<size_t, kNDim>&& points_per_dim);

  /// Get a MatrixMap view of the X-Y plane at specified higher-dim indices
  /// @tparam RestIndices Types for higher-dimension indices
  /// @param[in] indices Indices for higher dimensions (if any)
  /// @return MatrixMap of the X-Y plane at the specified indices
  template <typename... RestIndices>
    requires(sizeof...(RestIndices) == (kNDim > 2 ? kNDim - 2 : 0))
  [[nodiscard]] auto matrix(RestIndices... indices) const -> MatrixMap {
    static_assert(kNDim >= 2,
                  "Matrix access requires at least 2 dimensions (x, y)");

    // Calculate the offset in the flat buffer
    size_t offset = calculate_offset<2>(indices...);

    // Create the Map with actual X and Y dimensions
    return MatrixMap(values_.data() + offset,
                     points_per_dim_[0],  // X points (rows)
                     points_per_dim_[1],  // Y points (cols)
                     StrideType(strides_[0], strides_[1]));
  }

 private:
  /// X half-window size
  size_t x_half_window_size_;
  /// Y half-window size
  size_t y_half_window_size_;
  /// Number of points along X axis
  size_t x_points_;
  /// Number of points along Y axis
  size_t y_points_;
  /// Number of points per dimension
  std::array<size_t, kNDim> points_per_dim_;

  /// Coordinate vectors for each dimension
  std::tuple<std::vector<AxisTypes>...> coords_;
  /// Domain bounds for each dimension
  std::array<DomainBounds, kNDim> domains_;
  /// Flat buffer of values
  std::vector<DataType> values_;
  /// Pre-calculated strides for each dimension
  std::array<size_t, kNDim> strides_;

  // Helper to resize all coordinate vectors using index-based access
  template <size_t... Is>
  void resize_coords(std::index_sequence<Is...>) {
    (std::get<Is>(coords_).resize(points_per_dim_[Is]), ...);
  }

  // Flatten N-dimensional indices to 1D index using pre-calculated strides
  template <typename... Indices>
  [[nodiscard]] constexpr auto flatten_index(Indices... indices) const
      -> size_t {
    size_t offset = 0;
    size_t dim = 0;
    ((offset += indices * strides_[dim++]), ...);
    return offset;
  }

  // Implementation of contains using index sequence
  template <size_t... Is, typename... Coords>
  [[nodiscard]] auto contains_impl(std::index_sequence<Is...>,
                                   Coords... coords) const -> bool {
    // Fold expression to check all domains
    return ((domains_[Is].contains(static_cast<double>(coords))) && ...);
  }

  // Update domain bounds for all dimensions
  template <size_t... Is>
  void update_domains(
      const std::array<std::pair<int64_t, int64_t>, kNDim>& bracketing_indices,
      std::index_sequence<Is...>) {
    (update_domain_for_axis<Is>(bracketing_indices[Is]), ...);
  }

  // Update domain bounds for a specific dimension I
  template <size_t I>
  void update_domain_for_axis(
      const std::pair<int64_t, int64_t>& bracketing_indices);

  // Helper to calculate offset starting from Dimension 'StartDim'
  template <size_t StartDim, typename... Indices>
  [[nodiscard]] constexpr auto calculate_offset(Indices... indices) const
      -> size_t {
    size_t offset = 0;
    size_t dim = StartDim;
    // For each index provided, multiply by the stride of that dimension
    ((offset += indices * strides_[dim++]), ...);
    return offset;
  }
};

// ============================================================================
// Implementation
// ============================================================================
template <Numeric DataType, typename... AxisTypes>
InterpolationCache<DataType, AxisTypes...>::InterpolationCache(
    size_t half_x_window_size, size_t half_y_window_size)
    : x_half_window_size_(half_x_window_size),
      y_half_window_size_(half_y_window_size),
      x_points_(half_x_window_size * 2),
      y_points_(half_y_window_size * 2) {
  static_assert(kNDim >= 1, "Cache must have at least 1 dimension");
  if (half_x_window_size < 1 || half_y_window_size < 1) {
    throw std::invalid_argument("Half window sizes must be at least 1");
  }

  // Set up points per dimension array
  if constexpr (kNDim >= 1) {
    points_per_dim_[0] = x_points_;
  }
  if constexpr (kNDim >= 2) {
    points_per_dim_[1] = y_points_;
  }
  // Additional dimensions always use window_size=2 (4 points)
  if constexpr (kNDim > 2) {
    for (size_t i = 2; i < kNDim; ++i) {
      points_per_dim_[i] = 2;  // 1 points on each side, total 2 points
    }
  }

  // Resize coordinate vectors
  resize_coords(std::index_sequence_for<AxisTypes...>{});

  // Calculate total elements
  size_t total_elements = 1;
  for (size_t i = 0; i < kNDim; ++i) {
    total_elements *= points_per_dim_[i];
  }

  // Allocate flat value buffer
  values_.resize(total_elements);

  // Initialize strides for C-style (Row-Major) layout
  size_t stride = 1;
  for (int i = kNDim - 1; i >= 0; --i) {
    strides_[i] = stride;
    stride *= points_per_dim_[i];
  }
}

// ============================================================================

template <Numeric DataType, typename... AxisTypes>
template <size_t I>
void InterpolationCache<DataType, AxisTypes...>::update_domain_for_axis(
    const std::pair<int64_t, int64_t>& bracketing_indices) {
  const auto& vec = std::get<I>(coords_);
  const auto window_size = vec.size();
  assert(window_size >= 2);

  const auto& [left_idx, right_idx] = bracketing_indices;

  domains_[I].valid = left_idx >= 0 && right_idx >= 0 &&
                      std::cmp_less(left_idx, window_size) &&
                      std::cmp_less(right_idx, window_size);
  if (!domains_[I].valid) {
    return;
  }

  domains_[I].min = static_cast<double>(vec[left_idx]);
  domains_[I].max = static_cast<double>(vec[right_idx]);
}

// ============================================================================

template <Numeric DataType, typename... AxisTypes>
void InterpolationCache<DataType, AxisTypes...>::resize(
    std::array<size_t, kNDim>&& points_per_dim) {
  points_per_dim_ = std::move(points_per_dim);

  if constexpr (kNDim >= 1) {
    x_points_ = points_per_dim_[0];
  }
  if constexpr (kNDim >= 2) {
    y_points_ = points_per_dim_[1];
  }

  // Resize coordinate vectors
  resize_coords(std::index_sequence_for<AxisTypes...>{});

  // Calculate total elements
  size_t total_elements = 1;
  for (size_t i = 0; i < kNDim; ++i) {
    total_elements *= points_per_dim_[i];
  }

  // Allocate flat value buffer
  values_.resize(total_elements);

  // Initialize strides for C-style (Row-Major) layout
  size_t stride = 1;
  for (int i = kNDim - 1; i >= 0; --i) {
    strides_[i] = stride;
    stride *= points_per_dim_[i];
  }
}

}  // namespace pyinterp::math::interpolate
