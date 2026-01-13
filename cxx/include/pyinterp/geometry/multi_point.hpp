// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <ranges>
#include <stdexcept>
#include <vector>

#include "pyinterp/broadcast.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/serialization_buffer.hpp"

namespace pyinterp::geometry {

/// @brief MultiPoint: collection of points.
///
/// A `MultiPoint` represents a collection of points. It provides
/// basic container-like operations for constructing and iterating over
/// points.
template <typename Point>
class MultiPoint {
 public:
  /// @brief Underlying container type for points.
  using container_type = std::vector<Point>;

  /// @brief Iterator over points.
  using iterator = container_type::iterator;

  /// @brief Const iterator over points.
  using const_iterator = container_type::const_iterator;

  /// @brief Default constructor. Creates an empty collection.
  constexpr MultiPoint() = default;

  /// @brief Construct from a vector of points.
  /// @param[in] points Vector of points to move into the multipoint.
  explicit constexpr MultiPoint(std::vector<Point> points)
      : points_{std::move(points)} {}

  /// @brief Construct from two array of coordinates.
  /// @param[in] xs Array of x-coordinates.
  /// @param[in] ys Array of y-coordinates.
  constexpr MultiPoint(const Eigen::Ref<const Vector<double>>& xs,
                       const Eigen::Ref<const Vector<double>>& ys) {
    broadcast::check_eigen_shape("xs", xs, "ys", ys);
    points_.reserve(xs.size());
    for (auto [x, y] : std::ranges::views::zip(xs, ys)) {
      points_.emplace_back(x, y);
    }
  }

  /// @brief Append a point to the collection.
  /// @param[in] pt Point to append.
  constexpr void push_back(const Point& pt) { points_.push_back(pt); }

  /// @brief Remove all points from the collection.
  constexpr void clear() noexcept { points_.clear(); }

  /// @brief Resize the collection.
  /// @param[in] n New number of points.
  constexpr void resize(std::size_t n) { points_.resize(n); }

  /// @brief Number of points in the collection.
  [[nodiscard]] constexpr auto size() const noexcept -> std::size_t {
    return points_.size();
  }

  /// @brief Check whether the collection is empty.
  [[nodiscard]] constexpr auto empty() const noexcept -> bool {
    return points_.empty();
  }

  /// @brief Access the i-th point (mutable).
  [[nodiscard]] constexpr auto operator[](std::size_t ix) -> Point& {
    return points_[ix];
  }

  /// @brief Access the i-th point (const).
  [[nodiscard]] constexpr auto operator[](std::size_t ix) const
      -> const Point& {
    return points_[ix];
  }

  /// @brief Return iterator to first point (mutable).
  [[nodiscard]] constexpr auto begin() noexcept -> iterator {
    return points_.begin();
  }

  /// @brief Return iterator to past-the-end (mutable).
  [[nodiscard]] constexpr auto end() noexcept -> iterator {
    return points_.end();
  }

  /// @brief Return iterator to first point (const).
  [[nodiscard]] constexpr auto begin() const noexcept -> const_iterator {
    return points_.begin();
  }

  /// @brief Return iterator to past-the-end (const).
  [[nodiscard]] constexpr auto end() const noexcept -> const_iterator {
    return points_.end();
  }

  /// @brief Serialize the multipoint state for storage or transmission.
  /// @return Serialized state as a vector of points.
  [[nodiscard]] constexpr auto pack() const -> serialization::Writer {
    serialization::Writer writer;
    writer.write(kMagicNumber);
    writer.write(points_);
    return writer;
  }

  /// @brief Deserialize a multipoint from serialized state.
  /// @param[in] state Reference to serialization Reader containing encoded
  /// multipoint data.
  /// @return New MultiPoint instance with restored points.
  [[nodiscard]] static auto unpack(serialization::Reader& state) -> MultiPoint;

 private:
  /// @brief Magic number for validation
  static constexpr uint32_t kMagicNumber = 0x4d505420;  // "MPT "
  /// @brief Container of points.
  container_type points_;
};

// ============================================================================
// Implementation of templated methods
// ============================================================================

template <typename Point>
auto MultiPoint<Point>::unpack(serialization::Reader& state)
    -> MultiPoint<Point> {
  if (state.size() == 0) {
    throw std::invalid_argument("Cannot restore multipoint from empty state.");
  }
  auto magic_number = state.read<uint32_t>();
  if (magic_number != kMagicNumber) {
    throw std::invalid_argument("Invalid multipoint state (bad magic number).");
  }
  auto points = state.read_vector<Point>();
  return MultiPoint(std::move(points));
}

}  // namespace pyinterp::geometry
