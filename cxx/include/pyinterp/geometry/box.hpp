// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <boost/geometry.hpp>

namespace pyinterp::geometry {

/// @brief Type representing a bounding box in geographic coordinates
template <typename Point>
class Box {
 public:
  /// @brief Default constructor creating an empty box
  constexpr Box() noexcept = default;

  /// @brief Constructor from corner points
  /// @param[in] min_corner Minimum corner point (lon, lat)
  /// @param[in] max_corner Maximum corner point (lon, lat)
  constexpr Box(const Point& min_corner, const Point& max_corner) noexcept
      : min_corner_(min_corner), max_corner_(max_corner) {}

  /// @brief Get the min corner of this box (const)
  /// @return Minimum corner point (lon, lat)
  [[nodiscard]] constexpr auto min_corner() const noexcept -> Point {
    return min_corner_;
  }

  /// @brief Get the max corner of this box (const)
  /// @return Maximum corner point (lon, lat)
  [[nodiscard]] constexpr auto max_corner() const noexcept -> Point {
    return max_corner_;
  }

  /// @brief Get the min corner of this box (mutable)
  /// @return Reference to minimum corner point (lon, lat)
  [[nodiscard]] constexpr auto min_corner() noexcept -> Point& {
    return min_corner_;
  }

  /// @brief Get the max corner of this box (mutable)
  /// @return Reference to maximum corner point (lon, lat)
  [[nodiscard]] constexpr auto max_corner() noexcept -> Point& {
    return max_corner_;
  }

 private:
  Point min_corner_{};
  Point max_corner_{};
};

}  // namespace pyinterp::geometry
