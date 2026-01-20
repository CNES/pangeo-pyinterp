// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <Eigen/Core>
#include <boost/geometry.hpp>
#include <ranges>
#include <stdexcept>
#include <vector>

#include "pyinterp/broadcast.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/serialization_buffer.hpp"

namespace pyinterp::geometry {

/// @brief Type representing a linestring in geodetic coordinates
template <typename Point>
class LineString {
 public:
  /// @brief Alias for the underlying container type
  using container_type = std::vector<Point>;

  /// @brief Alias for the point type
  using value_type = Point;

  /// @brief Alias for iterator types
  using iterator = container_type::iterator;

  /// @brief Alias for const iterator types
  using const_iterator = container_type::const_iterator;

  /// @brief Default constructor
  constexpr LineString() = default;

  /// @brief Constructor from a vector of points
  /// @param[in] points Vector of points defining the linestring
  explicit constexpr LineString(std::vector<Point> points)
      : points_{std::move(points)} {}

  /// @brief Constructor from separate coordinate arrays
  /// @param[in] xs X coordinate array
  /// @param[in] ys Y coordinate array
  inline LineString(const Eigen::Ref<const Vector<double>>& xs,
                    const Eigen::Ref<const Vector<double>>& ys) {
    broadcast::check_eigen_shape("xs", xs, "ys", ys);
    points_.reserve(xs.size());
    for (auto [x, y] : std::ranges::views::zip(xs, ys)) {
      points_.emplace_back(x, y);
    }
  }

  /// @brief Add a point to the linestring
  /// @param[in] pt Point to add
  constexpr void push_back(const Point& pt) { points_.push_back(pt); }

  /// @brief Clear all points from the linestring
  constexpr void clear() noexcept { points_.clear(); }

  /// @brief Resize the linestring to contain n points
  constexpr void resize(std::size_t n) { points_.resize(n); }

  /// @brief Get the number of points in the linestring
  /// @returns Number of points
  [[nodiscard]] constexpr auto size() const noexcept -> std::size_t {
    return points_.size();
  }

  /// @brief Check if the linestring is empty
  /// @returns True if the linestring contains no points
  [[nodiscard]] constexpr auto empty() const noexcept -> bool {
    return points_.empty();
  }

  /// @brief Access to a mutable point at given index
  /// @param[in] ix Index of the point
  /// @returns Reference to the point at index ix
  [[nodiscard]] constexpr auto operator[](std::size_t ix) -> Point& {
    return points_[ix];
  }

  /// @brief Access to a const point at given index
  /// @param[in] ix Index of the point
  /// @returns Const reference to the point at index ix
  [[nodiscard]] constexpr auto operator[](std::size_t ix) const
      -> const Point& {
    return points_[ix];
  }

  /// @brief Get coordinate arrays of the linestring points
  /// @returns Pair of coordinate arrays (X, Y)
  [[nodiscard]] inline auto to_arrays() const
      -> std::pair<Vector<double>, Vector<double>> {
    Vector<double> xs(static_cast<int64_t>(points_.size()));
    Vector<double> ys(static_cast<int64_t>(points_.size()));
    for (size_t ix = 0; ix < points_.size(); ++ix) {
      xs(static_cast<int64_t>(ix)) = boost::geometry::get<0>(points_[ix]);
      ys(static_cast<int64_t>(ix)) = boost::geometry::get<1>(points_[ix]);
    }
    return {xs, ys};
  }

  /// @brief Get iterator to the beginning (mutable)
  /// @returns Iterator to the first point
  [[nodiscard]] constexpr auto begin() noexcept -> iterator {
    return points_.begin();
  }

  /// @brief Get iterator to the end (mutable)
  /// @returns Iterator to one past the last point
  [[nodiscard]] constexpr auto end() noexcept -> iterator {
    return points_.end();
  }

  /// @brief Get iterator to the beginning (const)
  /// @returns Const iterator to the first point
  [[nodiscard]] constexpr auto begin() const noexcept -> const_iterator {
    return points_.begin();
  }
  /// @brief Get iterator to the end (const)
  /// @returns Const iterator to one past the last point
  [[nodiscard]] constexpr auto end() const noexcept -> const_iterator {
    return points_.end();
  }

  /// @brief Get a reference to the first point
  /// @returns Reference to the first point
  [[nodiscard]] constexpr auto front() noexcept -> Point& {
    return points_.front();
  }

  /// @brief Get a const reference to the first point
  /// @returns Const reference to the first point
  [[nodiscard]] constexpr auto front() const noexcept -> const Point& {
    return points_.front();
  }

  /// @brief Get a reference to the last point
  /// @returns Reference to the last point
  [[nodiscard]] constexpr auto back() noexcept -> Point& {
    return points_.back();
  }

  /// @brief Get a const reference to the last point
  /// @returns Const reference to the last point
  [[nodiscard]] constexpr auto back() const noexcept -> const Point& {
    return points_.back();
  }

  /// @brief Serialize the ring state for storage or transmission.
  /// @return Serialized state as a vector of points.
  [[nodiscard]] constexpr auto pack() const -> serialization::Writer {
    serialization::Writer writer;
    writer.write(kMagicNumber);
    writer.write(points_);
    return writer;
  }

  /// @brief Deserialize a ring from serialized state.
  /// @param[in] state Reference to serialization Reader containing encoded ring
  /// data.
  /// @return New Ring instance with restored points.
  /// @throw std::invalid_argument If the state is invalid or empty.
  [[nodiscard]] static auto unpack(serialization::Reader& state) -> LineString {
    if (state.size() == 0) {
      throw std::invalid_argument(
          "Cannot restore linestring from empty state.");
    }
    auto magic_number = state.read<uint32_t>();
    if (magic_number != kMagicNumber) {
      throw std::invalid_argument(
          "Invalid linestring state (bad magic number).");
    }
    auto points = state.read_vector<Point>();
    return LineString(std::move(points));
  }

 private:
  /// @brief Magic number for serialization validation
  static constexpr uint32_t kMagicNumber = 0x4c535452;  // "LSTR"
  /// @brief Underlying container of points
  container_type points_;
};

}  // namespace pyinterp::geometry
