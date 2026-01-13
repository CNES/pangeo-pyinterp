// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <vector>

#include "pyinterp/geometry/polygon.hpp"
#include "pyinterp/serialization_buffer.hpp"

namespace pyinterp::geometry {

/// @brief MultiPolygon: collection of polygons.
///
/// A `MultiPolygon` represents a collection of polygons. It provides
/// basic container-like operations for constructing and iterating over
/// polygons.
template <typename Point>
class MultiPolygon {
 public:
  /// @brief Underlying container type for polygons.
  using container_type = std::vector<Polygon<Point>>;

  /// @brief Value type (polygon) for container operations.
  using value_type = Polygon<Point>;

  /// @brief Iterator over polygons.
  using iterator = container_type::iterator;

  /// @brief Const iterator over polygons.
  using const_iterator = container_type::const_iterator;

  /// @brief Default constructor. Creates an empty collection.
  constexpr MultiPolygon() = default;

  /// @brief Construct from a vector of polygons.
  /// @param[in] polygons Vector of polygons to move into the multipolygon.
  explicit constexpr MultiPolygon(std::vector<Polygon<Point>> polygons)
      : polygons_{std::move(polygons)} {}

  /// @brief Append a polygon to the collection.
  /// @param[in] pt Polygon to append.
  constexpr void push_back(const Polygon<Point>& pt) {
    polygons_.push_back(pt);
  }

  /// @brief Append a polygon to the collection (rvalue).
  /// @param[in] pt Polygon to append.
  constexpr void push_back(Polygon<Point>&& pt) {
    polygons_.push_back(std::move(pt));
  }

  /// @brief Get reference to the last polygon.
  [[nodiscard]] constexpr auto back() -> Polygon<Point>& {
    return polygons_.back();
  }

  /// @brief Get const reference to the last polygon.
  [[nodiscard]] constexpr auto back() const -> const Polygon<Point>& {
    return polygons_.back();
  }

  /// @brief Remove all polygons from the collection.
  constexpr void clear() noexcept { polygons_.clear(); }

  /// @brief Resize the collection.
  /// @param[in] n New number of polygons.
  constexpr void resize(std::size_t n) { polygons_.resize(n); }

  /// @brief Number of polygons in the collection.
  [[nodiscard]] constexpr auto size() const noexcept -> std::size_t {
    return polygons_.size();
  }

  /// @brief Check whether the collection is empty.
  [[nodiscard]] constexpr auto empty() const noexcept -> bool {
    return polygons_.empty();
  }

  /// @brief Access the i-th polygon (mutable).
  [[nodiscard]] constexpr auto operator[](std::size_t ix) -> Polygon<Point>& {
    return polygons_[ix];
  }

  /// @brief Access the i-th polygon (const).
  [[nodiscard]] constexpr auto operator[](std::size_t ix) const
      -> const Polygon<Point>& {
    return polygons_[ix];
  }

  /// @brief Return iterator to first polygon (mutable).
  [[nodiscard]] constexpr auto begin() noexcept -> iterator {
    return polygons_.begin();
  }

  /// @brief Return iterator to past-the-end (mutable).
  [[nodiscard]] constexpr auto end() noexcept -> iterator {
    return polygons_.end();
  }

  /// @brief Return iterator to first polygon (const).
  [[nodiscard]] constexpr auto begin() const noexcept -> const_iterator {
    return polygons_.begin();
  }

  /// @brief Return iterator to past-the-end (const).
  [[nodiscard]] constexpr auto end() const noexcept -> const_iterator {
    return polygons_.end();
  }

  /// @brief Serialize the multipolygon state for storage or transmission.
  /// @return Serialized state as a vector of points.
  [[nodiscard]] constexpr auto pack() const -> serialization::Writer {
    serialization::Writer writer;
    writer.write(kMagicNumber);
    writer.write(polygons_.size());
    for (const auto& polygon : polygons_) {
      writer.write(polygon.pack());
    }
    return writer;
  }

  /// @brief Deserialize a multipolygon from serialized state.
  /// @param[in] state Reference to serialization Reader containing encoded
  /// multipolygon data.
  /// @return New MultiPolygon instance with restored polygons.
  [[nodiscard]] static auto unpack(serialization::Reader& state)
      -> MultiPolygon;

 private:
  /// @brief Magic number for validation
  static constexpr uint32_t kMagicNumber = 0x4d554c54;  // "MULT"
  /// @brief Container of polygons.
  container_type polygons_;
};

// ============================================================================
// Implementation of templated methods
// ============================================================================

template <typename Point>
auto MultiPolygon<Point>::unpack(serialization::Reader& state)
    -> MultiPolygon<Point> {
  if (state.size() == 0) {
    throw std::invalid_argument(
        "Cannot restore multipolygon from empty state.");
  }
  auto magic_number = state.read<uint32_t>();
  if (magic_number != kMagicNumber) {
    throw std::invalid_argument(
        "Invalid multipolygon state (bad magic number).");
  }
  auto num_polygons = state.read<size_t>();
  std::vector<Polygon<Point>> polygons;
  polygons.reserve(num_polygons);
  for (size_t i = 0; i < num_polygons; ++i) {
    auto polygon_state = state.read_vector<std::byte>();
    auto reader = serialization::Reader(std::move(polygon_state));
    polygons.emplace_back(Polygon<Point>::unpack(reader));
  }
  return MultiPolygon(std::move(polygons));
}

}  // namespace pyinterp::geometry
