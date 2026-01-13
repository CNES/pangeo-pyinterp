// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <boost/geometry.hpp>

#include "pyinterp/geometry/point.hpp"

namespace pyinterp::geometry::cartesian {

/// @brief Type representing a point in cartesian coordinates
class Point {
 public:
  /// @brief Build an undefined point
  constexpr Point() noexcept = default;

  /// @brief Constructs a point from x and y coordinates
  /// @param[in] x X coordinate
  /// @param[in] y Y coordinate
  constexpr Point(double x, double y) noexcept : x_{x}, y_{y} {}

  /// @brief Get the x coordinate
  /// @return X coordinate
  [[nodiscard]] constexpr auto x() const noexcept -> double { return x_; }

  /// @brief Get the y coordinate
  /// @return Y coordinate
  [[nodiscard]] constexpr auto y() const noexcept -> double { return y_; }

  /// @brief Get a mutable reference to the x coordinate
  /// @return Reference to x coordinate
  [[nodiscard]] constexpr auto x() noexcept -> double& { return x_; }

  /// @brief Get a mutable reference to the y coordinate
  /// @return Reference to y coordinate
  [[nodiscard]] constexpr auto y() noexcept -> double& { return y_; }
  /// @brief Templated access for Boost.Geometry traits
  /// @tparam I Index (0 for lon, 1 for lat)
  /// @return Value at index I
  template <std::size_t I>
  [[nodiscard]] constexpr auto get() const noexcept -> double {
    static_assert(I < 2, "Index out of bounds");
    if constexpr (I == 0) {
      return x_;
    }
    return y_;
  }

  /// @brief Templated setter for Boost.Geometry traits
  /// @tparam I Index (0 for lon, 1 for lat)
  /// @param[in] v Value to set at index I
  template <std::size_t I>
  constexpr void set(double v) noexcept {
    static_assert(I < 2, "Index out of bounds");
    I == 0 ? x_ = v : y_ = v;
  }

  /// @brief Equality operator
  /// @param[in] other Point to compare with
  /// @return True if both points are equal
  constexpr auto operator==(const Point& other) const noexcept -> bool {
    return x_ == other.x_ && y_ == other.y_;
  }

 private:
  /// @brief X coordinate
  double x_{};
  /// @brief Y coordinate
  double y_{};
};

}  // namespace pyinterp::geometry::cartesian

// Boost.Geometry traits
namespace boost::geometry::traits {

template <>
struct tag<pyinterp::geometry::cartesian::Point> {
  using type = point_tag;
};

template <>
struct coordinate_type<pyinterp::geometry::cartesian::Point> {
  using type = double;
};

template <>
struct coordinate_system<pyinterp::geometry::cartesian::Point> {
  using type = cs::cartesian;
};

template <>
struct dimension<pyinterp::geometry::cartesian::Point> : boost::mpl::int_<2> {};

template <std::size_t I>
struct access<pyinterp::geometry::cartesian::Point, I> {
  static auto get(const pyinterp::geometry::cartesian::Point& p) -> double {
    return p.get<I>();
  }
  static void set(pyinterp::geometry::cartesian::Point& p, double v) {
    p.set<I>(v);
  }
};

}  // namespace boost::geometry::traits

template <>
struct std::hash<pyinterp::geometry::cartesian::Point> {
  auto operator()(const pyinterp::geometry::cartesian::Point& point)
      const noexcept -> std::size_t {
    return pyinterp::geometry::hash(point);
  }
};
