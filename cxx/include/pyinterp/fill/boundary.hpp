// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <cstdint>

namespace pyinterp::fill {
namespace detail {

/// @brief Handles periodic boundary conditions for a grid index.
/// Used when the grid wraps around (e.g., longitude coordinates).
///
/// @param[in] index The index to normalize
/// @param[in] size The size of the dimension
/// @return The normalized index within [0, size)
///
/// Examples:
///   periodic_index(-1, 10) = 9
///   periodic_index(10, 10) = 0
///   periodic_index(11, 10) = 1
constexpr auto periodic_index(const int64_t index, const int64_t size)
    -> int64_t {
  return (index + size) % size;
}

/// @brief Handles reflective (Neumann) boundary conditions for a grid index.
/// Used when the grid has zero-derivative boundaries (values reflect at edges).
///
/// @param[in] index The index to clamp/reflect
/// @param[in] size The size of the dimension
/// @return The reflected index within [0, size)
///
/// Examples:
///   reflective_index(-1, 10) = 1     (reflects at boundary)
///   reflective_index(-2, 10) = 2
///   reflective_index(10, 10) = 8     (reflects at boundary)
///   reflective_index(11, 10) = 7
///   reflective_index(18, 10) = 0
///   reflective_index(19, 10) = 1
constexpr auto reflective_index(const int64_t index, const int64_t size)
    -> int64_t {
  if (size == 1) {
    return 0;
  }
  const int64_t period = 2 * (size - 1);
  auto normalized = index % period;
  if (normalized < 0) {
    normalized += period;
  }
  if (normalized < size) {
    return normalized;
  }
  return period - normalized;
}

}  // namespace detail

/// @brief Helper struct to compute neighbor indices with boundary conditions.
///
/// This encapsulates the logic for finding the four neighbors (left, right, up,
/// down) of a cell in a 2D grid, respecting boundary conditions.
///
/// Optimized for nested loops: create once, then call update_x(ix) in outer
/// loop and update_y(iy) in inner loop.
///
/// Usage:
///   Neighbors<true, false> nbr(x_size, y_size);
///   for (int64_t ix = 0; ix < x_size; ++ix) {
///     nbr.update_x(ix);  // Compute ix0/ix1 once per row
///     for (int64_t iy = 0; iy < y_size; ++iy) {
///       nbr.update_y(iy);  // Compute iy0/iy1 for each cell
///       grid_value = 0.25 * (grid(nbr.ix0, iy) + grid(nbr.ix1, iy) +
///                            grid(ix, nbr.iy0) + grid(ix, nbr.iy1));
///     }
///   }
template <bool XPeriodic, bool YPeriodic>
struct Neighbors {
  int64_t ix0{};  ///< Index of left neighbor (ix - 1)
  int64_t ix1{};  ///< Index of right neighbor (ix + 1)
  int64_t iy0{};  ///< Index of down neighbor (iy - 1)
  int64_t iy1{};  ///< Index of up neighbor (iy + 1)

  /// @brief Constructs the helper with grid dimensions.
  ///
  /// @param[in] x_size Size of x dimension
  /// @param[in] y_size Size of y dimension
  constexpr Neighbors(const int64_t x_size, const int64_t y_size)
      : x_size_(x_size), y_size_(y_size) {}

  /// @brief Updates x-direction neighbor indices.
  ///
  /// @param[in] ix Current x index
  constexpr auto update_x(const int64_t ix) -> void {
    ix0 = XPeriodic ? detail::periodic_index(ix - 1, x_size_)
                    : detail::reflective_index(ix - 1, x_size_);
    ix1 = XPeriodic ? detail::periodic_index(ix + 1, x_size_)
                    : detail::reflective_index(ix + 1, x_size_);
  }

  /// @brief Updates y-direction neighbor indices.
  ///
  /// @param[in] iy Current y index
  constexpr auto update_y(const int64_t iy) -> void {
    iy0 = YPeriodic ? detail::periodic_index(iy - 1, y_size_)
                    : detail::reflective_index(iy - 1, y_size_);
    iy1 = YPeriodic ? detail::periodic_index(iy + 1, y_size_)
                    : detail::reflective_index(iy + 1, y_size_);
  }

 private:
  int64_t x_size_;  ///< Cached x dimension size
  int64_t y_size_;  ///< Cached y dimension size
};

/// @brief Runtime version of Neighbors for when boundary type isn't known at
/// compile time.
///
/// Optimized for nested loops: create once, then call update_x(ix) in outer
/// loop and update_y(iy) in inner loop.
///
/// Usage:
///   DynamicNeighbors nbr(x_size, y_size, is_circle, false);
///   for (int64_t ix = 0; ix < x_size; ++ix) {
///     nbr.update_x(ix);  // Compute ix0/ix1 once per row
///     for (int64_t iy = 0; iy < y_size; ++iy) {
///       nbr.update_y(iy);  // Compute iy0/iy1 for each cell
///       grid_value = 0.25 * (grid(nbr.ix0, iy) + grid(nbr.ix1, iy) +
///                            grid(ix, nbr.iy0) + grid(ix, nbr.iy1));
///     }
///   }
struct DynamicNeighbors {
  int64_t ix0{};  ///< Index of left neighbor (ix - 1)
  int64_t ix1{};  ///< Index of right neighbor (ix + 1)
  int64_t iy0{};  ///< Index of down neighbor (iy - 1)
  int64_t iy1{};  ///< Index of up neighbor (iy + 1)

  /// @brief Constructs the helper with grid dimensions and boundary types.
  ///
  /// @param[in] x_size Size of x dimension
  /// @param[in] y_size Size of y dimension
  /// @param[in] x_periodic True if x dimension uses periodic boundaries
  /// @param[in] y_periodic True if y dimension uses periodic boundaries
  constexpr DynamicNeighbors(const int64_t x_size, const int64_t y_size,
                             const bool x_periodic = false,
                             const bool y_periodic = false)
      : x_size_(x_size),
        y_size_(y_size),
        get_x_(x_periodic ? detail::periodic_index : detail::reflective_index),
        get_y_(y_periodic ? detail::periodic_index : detail::reflective_index) {
  }

  /// @brief Updates x-direction neighbor indices.
  ///
  /// @param[in] ix Current x index
  constexpr auto update_x(const int64_t ix) -> void {
    ix0 = get_x_(ix - 1, x_size_);
    ix1 = get_x_(ix + 1, x_size_);
  }

  /// @brief Updates y-direction neighbor indices.
  ///
  /// @param iy Current y index
  constexpr auto update_y(const int64_t iy) -> void {
    iy0 = get_y_(iy - 1, y_size_);
    iy1 = get_y_(iy + 1, y_size_);
  }

 private:
  int64_t x_size_;                      ///< Cached x dimension size
  int64_t y_size_;                      ///< Cached y dimension size
  int64_t (*get_x_)(int64_t, int64_t);  ///< X-direction boundary function
  int64_t (*get_y_)(int64_t, int64_t);  ///< Y-direction boundary function
};

}  // namespace pyinterp::fill
