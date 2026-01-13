// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <vector>

#include "pyinterp/geometry/linestring.hpp"
#include "pyinterp/serialization_buffer.hpp"

namespace pyinterp::geometry {

/// @brief MultiLineString: collection of linestrings.
///
/// A `MultiLineString` represents a collection of `LineString` geometries.
/// It provides basic container-like operations for constructing and iterating
/// over linestrings.
template <typename Point>
class MultiLineString {
 public:
  /// @brief Underlying container type for linestrings.
  using container_type = std::vector<LineString<Point>>;
  /// @brief Iterator over linestrings.
  using iterator = container_type::iterator;

  /// @brief Const iterator over linestrings.
  using const_iterator = container_type::const_iterator;

  /// @brief Default constructor. Creates an empty collection.
  constexpr MultiLineString() = default;

  /// @brief Construct from a vector of linestrings.
  /// @param[in] lines Vector of linestrings to move into the multicollection.
  explicit constexpr MultiLineString(std::vector<LineString<Point>> lines)
      : lines_{std::move(lines)} {}

  /// @brief Append a linestring to the collection.
  /// @param[in] ls LineString to append.
  constexpr void push_back(const LineString<Point>& ls) {
    lines_.push_back(ls);
  }

  /// @brief Remove all linestrings from the collection.
  constexpr void clear() noexcept { lines_.clear(); }

  /// @brief Resize the collection.
  /// @param[in] n New number of linestrings.
  constexpr void resize(std::size_t n) { lines_.resize(n); }

  /// @brief Number of linestrings in the collection.
  [[nodiscard]] constexpr auto size() const noexcept -> std::size_t {
    return lines_.size();
  }

  /// @brief Check whether the collection is empty.
  [[nodiscard]] constexpr auto empty() const noexcept -> bool {
    return lines_.empty();
  }

  /// @brief Access the i-th linestring (mutable).
  [[nodiscard]] constexpr auto operator[](std::size_t ix)
      -> LineString<Point>& {
    return lines_[ix];
  }

  /// @brief Access the i-th linestring (const).
  [[nodiscard]] constexpr auto operator[](std::size_t ix) const
      -> const LineString<Point>& {
    return lines_[ix];
  }

  /// @brief Return iterator to first linestring (mutable).
  [[nodiscard]] constexpr auto begin() noexcept -> iterator {
    return lines_.begin();
  }

  /// @brief Return iterator to past-the-end (mutable).
  [[nodiscard]] constexpr auto end() noexcept -> iterator {
    return lines_.end();
  }

  /// @brief Return iterator to first linestring (const).
  [[nodiscard]] constexpr auto begin() const noexcept -> const_iterator {
    return lines_.begin();
  }

  /// @brief Return iterator to past-the-end (const).
  [[nodiscard]] constexpr auto end() const noexcept -> const_iterator {
    return lines_.end();
  }

  /// @brief Serialize the multilinestring state for storage or transmission.
  /// @return Serialized state as a vector of bytes.
  [[nodiscard]] constexpr auto pack() const -> serialization::Writer {
    serialization::Writer writer;
    writer.write(kMagicNumber);
    writer.write(lines_.size());
    for (const auto& ls : lines_) {
      writer.write(ls.pack());
    }
    return writer;
  }

  /// @brief Deserialize a multilinestring from serialized state.
  /// @param[in] state Reference to serialization Reader containing encoded
  /// multilinestring data.
  /// @return New MultiLineString instance with restored linestrings.
  [[nodiscard]] static auto unpack(serialization::Reader& state)
      -> MultiLineString;

 private:
  /// @brief Magic number for validation ("MLST").
  static constexpr uint32_t kMagicNumber = 0x4d4c5354;
  /// @brief Container of linestrings.
  container_type lines_;
};

// ============================================================================
// Implementation of templated methods
// ============================================================================

template <typename Point>
auto MultiLineString<Point>::unpack(serialization::Reader& state)
    -> MultiLineString<Point> {
  if (state.size() == 0) {
    throw std::invalid_argument(
        "Cannot restore multilinestring from empty state.");
  }
  auto magic_number = state.read<uint32_t>();
  if (magic_number != kMagicNumber) {
    throw std::invalid_argument(
        "Invalid multilinestring state (bad magic number).");
  }
  auto num_lines = state.read<size_t>();
  std::vector<LineString<Point>> lines;
  lines.reserve(num_lines);
  for (size_t i = 0; i < num_lines; ++i) {
    auto ls_state = state.read_vector<std::byte>();
    auto reader = serialization::Reader(std::move(ls_state));
    lines.emplace_back(LineString<Point>::unpack(reader));
  }
  return MultiLineString(std::move(lines));
}

}  // namespace pyinterp::geometry
