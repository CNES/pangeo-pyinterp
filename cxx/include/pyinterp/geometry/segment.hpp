// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <boost/geometry.hpp>

#include "pyinterp/serialization_buffer.hpp"

namespace pyinterp::geometry {

/// @brief Type representing a segment (two endpoints) in geodetic coordinates
template <typename Point>
class Segment {
 public:
  /// @brief Build an undefined segment
  constexpr Segment() noexcept = default;

  /// @brief Construct a segment from two endpoints
  /// @param[in] a First endpoint
  /// @param[in] b Second endpoint
  constexpr Segment(const Point& a, const Point& b) noexcept : a_{a}, b_{b} {}

  /// @brief Get the first endpoint (const)
  [[nodiscard]] constexpr auto a() const noexcept -> const Point& { return a_; }
  /// @brief Get the second endpoint (const)
  [[nodiscard]] constexpr auto b() const noexcept -> const Point& { return b_; }

  /// @brief Get the first endpoint (mutable)
  [[nodiscard]] constexpr auto a() noexcept -> Point& { return a_; }
  /// @brief Get the second endpoint (mutable)
  [[nodiscard]] constexpr auto b() noexcept -> Point& { return b_; }

  /// @brief  Get the coordinate arrays of the segment endpoints
  /// @returns Pair of coordinate arrays (X, Y)
  [[nodiscard]] inline auto to_arrays() const
      -> std::pair<Eigen::Vector<double, 2>, Eigen::Vector<double, 2>> {
    Eigen::Vector<double, 2> xs;
    Eigen::Vector<double, 2> ys;
    xs(0) = boost::geometry::get<0>(a_);
    xs(1) = boost::geometry::get<0>(b_);
    ys(0) = boost::geometry::get<1>(a_);
    ys(1) = boost::geometry::get<1>(b_);
    return {xs, ys};
  }

  /// @brief Templated access for Boost.Geometry traits
  /// @tparam Index Endpoint index (0 for a, 1 for b)
  /// @tparam Dim Coordinate index (0 for lon, 1 for lat)
  /// @return Value at endpoint Index, coordinate Dim
  template <std::size_t Index, std::size_t Dim>
  [[nodiscard]] constexpr auto get() const noexcept -> double {
    static_assert(Index < 2, "Endpoint index out of bounds");
    static_assert(Dim < 2, "Coordinate index out of bounds");
    const Point& p = (Index == 0) ? a_ : b_;
    if constexpr (Dim == 0) {
      return boost::geometry::get<0>(p);
    } else {
      return boost::geometry::get<1>(p);
    }
  }

  /// @brief Templated setter for Boost.Geometry traits
  /// @tparam Index Endpoint index (0 for a, 1 for b)
  /// @tparam Dim Coordinate index (0 for lon, 1 for lat)
  /// @param[in] v Value to set at endpoint Index, coordinate Dim
  template <std::size_t Index, std::size_t Dim>
  constexpr void set(double v) noexcept {
    static_assert(Index < 2, "Endpoint index out of bounds");
    static_assert(Dim < 2, "Coordinate index out of bounds");
    Point& p = (Index == 0) ? a_ : b_;
    if constexpr (Dim == 0) {
      boost::geometry::set<0>(p, v);
    } else {
      boost::geometry::set<1>(p, v);
    }
  }

  /// @brief Serialize the segment state for storage or transmission.
  /// @return Serialized state as a vector of points.
  [[nodiscard]] constexpr auto pack() const -> serialization::Writer {
    serialization::Writer writer;
    writer.write(kMagicNumber);
    writer.write(a_);
    writer.write(b_);
    return writer;
  }

  /// @brief Reconstruct a segment from its serialized state.
  /// @param[in] reader Reader containing the serialized state.
  [[nodiscard]] static constexpr auto unpack(serialization::Reader& reader)
      -> Segment {
    const auto magic_number = reader.template read<uint32_t>();
    if (magic_number != kMagicNumber) {
      throw std::invalid_argument("Invalid serialized segment state.");
    }
    auto a = reader.template read<Point>();
    auto b = reader.template read<Point>();
    return Segment{a, b};
  }

 private:
  /// @brief Magic number to identify serialized segments
  static constexpr uint32_t kMagicNumber = 0x5345474d;
  /// @brief First endpoint
  Point a_{};
  /// @brief Second endpoint
  Point b_{};
};

}  // namespace pyinterp::geometry
