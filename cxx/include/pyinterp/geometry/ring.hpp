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

/// @brief Ring: closed linestring (for polygon boundaries).
///
/// The `Ring` class represents a closed sequence of `Point` objects used
/// as polygon boundaries. It provides basic container-like operations and
/// iterator support.
template <typename Point>
class Ring {
 public:
  /// @brief Underlying container type for point storage.
  using container_type = std::vector<Point>;

  /// @brief Value type for the container
  using value_type = Point;

  /// @brief Iterator over points.
  using iterator = container_type::iterator;

  /// @brief Const iterator over points.
  using const_iterator = container_type::const_iterator;

  /// @brief Default constructor. Creates an empty ring.
  constexpr Ring() = default;

  /// @brief Construct a ring from a vector of points.
  /// @param[in] points Vector of points (copied/moved into the ring).
  constexpr explicit Ring(std::vector<Point> points)
      : points_{std::move(points)} {}

  /// @brief Construct a ring from two separate coordinate arrays.
  /// @param[in] xs X coordinate array.
  /// @param[in] ys Y coordinate array.
  inline Ring(const Eigen::Ref<const Vector<double>>& xs,
              const Eigen::Ref<const Vector<double>>& ys) {
    broadcast::check_eigen_shape("xs", xs, "ys", ys);
    points_.reserve(xs.size());
    for (auto [x, y] : std::ranges::views::zip(xs, ys)) {
      points_.emplace_back(x, y);
    }
  }

  /// @brief Append a point to the ring.
  /// @param[in] pt Point to append.
  constexpr void push_back(const Point& pt) { points_.push_back(pt); }

  /// @brief Remove all points from the ring.
  constexpr void clear() noexcept { points_.clear(); }

  /// @brief Resize the ring to contain `n` points.
  /// @param[in] n New size.
  constexpr void resize(std::size_t n) { points_.resize(n); }

  /// @brief Return the number of points in the ring.
  /// @returns Number of points.
  [[nodiscard]] constexpr auto size() const noexcept -> std::size_t {
    return points_.size();
  }

  /// @brief Check whether the ring is empty.
  /// @returns `true` if empty.
  [[nodiscard]] constexpr auto empty() const noexcept -> bool {
    return points_.empty();
  }

  /// @brief Get reference to the last point.
  [[nodiscard]] constexpr auto back() -> Point& { return points_.back(); }

  /// @brief Get const reference to the last point.
  [[nodiscard]] constexpr auto back() const -> const Point& {
    return points_.back();
  }

  /// @brief Access the i-th point (mutable).
  /// @param[in] ix Index of the point.
  /// @returns Reference to the point at index `ix`.
  [[nodiscard]] constexpr auto operator[](std::size_t ix) -> Point& {
    return points_[ix];
  }

  /// @brief Access the i-th point (const).
  /// @param[in] ix Index of the point.
  /// @returns Const reference to the point at index `ix`.
  [[nodiscard]] constexpr auto operator[](std::size_t ix) const
      -> const Point& {
    return points_[ix];
  }

  /// @brief Get the coordinate arrays of the ring points.
  /// @returns Pair of coordinate arrays (X, Y).
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

  /// @brief Return a mutable iterator to the first point.
  [[nodiscard]] constexpr auto begin() noexcept -> iterator {
    return points_.begin();
  }

  /// @brief Return a mutable iterator to past-the-end.
  [[nodiscard]] constexpr auto end() noexcept -> iterator {
    return points_.end();
  }

  /// @brief Return a const iterator to the first point.
  [[nodiscard]] constexpr auto begin() const noexcept -> const_iterator {
    return points_.begin();
  }

  /// @brief Return a const iterator to past-the-end.
  [[nodiscard]] constexpr auto end() const noexcept -> const_iterator {
    return points_.end();
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
  [[nodiscard]] static auto unpack(serialization::Reader& state) -> Ring {
    if (state.size() == 0) {
      throw std::invalid_argument("Cannot restore ring from empty state.");
    }
    auto magic_number = state.read<uint32_t>();
    if (magic_number != kMagicNumber) {
      throw std::invalid_argument("Invalid ring state (bad magic number).");
    }
    auto points = state.read_vector<Point>();
    return Ring(std::move(points));
  }

 private:
  /// @brief Magic number for validation
  static constexpr uint32_t kMagicNumber = 0x52494E47;  // "RING"
  /// @brief Container holding the points.
  container_type points_;
};

}  // namespace pyinterp::geometry

namespace std {

template <typename Ring, typename Point>
class back_insert_iterator_ring {
 public:
  using iterator_category = std::output_iterator_tag;
  using value_type = void;
  using difference_type = void;
  using pointer = void;
  using reference = void;
  using container_type = Ring;

  explicit back_insert_iterator_ring(Ring& ring) : ring_(&ring) {}

  constexpr auto operator=(const Point& point) -> back_insert_iterator_ring& {
    ring_->push_back(point);
    return *this;
  }

  constexpr auto operator=(Point&& point) -> back_insert_iterator_ring& {
    ring_->push_back(point);
    return *this;
  }

  constexpr auto operator*() -> back_insert_iterator_ring& { return *this; }
  constexpr auto operator++() -> back_insert_iterator_ring& { return *this; }
  constexpr auto operator++(int) -> back_insert_iterator_ring { return *this; }

 private:
  Ring* ring_;
};

}  // namespace std
