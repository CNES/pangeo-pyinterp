// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <cstdint>
#include <vector>

#include "pyinterp/geometry/ring.hpp"
#include "pyinterp/serialization_buffer.hpp"

namespace pyinterp::geometry {

/// @brief Polygon: exterior ring + optional interior rings (holes).
///
/// The `Polygon` class represents a polygon defined by an exterior ring and
/// optional interior rings (holes). It provides accessors for the exterior
/// ring and the interior rings.
template <typename Point>
class Polygon {
 public:
  /// @brief Alias for ring type.
  using ring_type = Ring<Point>;

  /// @brief Alias for container of interior rings.
  using inner_container_type = std::vector<Ring<Point>>;

  /// @brief Default constructor.
  constexpr Polygon() = default;

  /// @brief Construct a polygon with an exterior ring only.
  /// @param[in] exterior Exterior ring for the polygon.
  constexpr explicit Polygon(Ring<Point> exterior)
      : exterior_{std::move(exterior)} {}

  /// @brief Construct a polygon with exterior and interior rings.
  /// @param[in] exterior Exterior ring for the polygon.
  /// @param[in] interiors Interior rings (holes).
  constexpr Polygon(Ring<Point> exterior, std::vector<Ring<Point>> interiors)
      : exterior_{std::move(exterior)}, interiors_{std::move(interiors)} {}

  /// @brief Get the exterior ring (const).
  /// @returns Const reference to the exterior ring.
  [[nodiscard]] constexpr auto outer() const noexcept -> const Ring<Point>& {
    return exterior_;
  }

  /// @brief Get the exterior ring (mutable).
  /// @returns Reference to the exterior ring.
  [[nodiscard]] constexpr auto outer() noexcept -> Ring<Point>& {
    return exterior_;
  }

  /// @brief Get the interior rings (const).
  /// @returns Const reference to the container of interior rings.
  [[nodiscard]] constexpr auto inners() const noexcept
      -> const inner_container_type& {
    return interiors_;
  }

  /// @brief Get the interior rings (mutable).
  /// @returns Reference to the container of interior rings.
  [[nodiscard]] constexpr auto inners() noexcept -> inner_container_type& {
    return interiors_;
  }

  /// @brief Serialize the polygon state for storage or transmission.
  /// @return Serialized state as a vector of points.
  [[nodiscard]] constexpr auto pack() const -> serialization::Writer {
    serialization::Writer writer;
    writer.write(kMagicNumber);
    writer.write(exterior_.pack());
    writer.write(interiors_.size());
    for (const auto& ring : interiors_) {
      writer.write(ring.pack());
    }
    return writer;
  }

  /// @brief Deserialize a polygon from serialized state.
  /// @param[in] state Reference to serialization Reader containing encoded
  /// polygon data.
  /// @return New Polygon instance with restored rings.
  /// @throw std::invalid_argument If the state is invalid or empty.
  [[nodiscard]] static auto unpack(serialization::Reader& state) -> Polygon;

 private:
  /// @brief Magic number for validation
  static constexpr uint32_t kMagicNumber = 0x504f4c59;  // "POLY"
  /// @brief Exterior ring.
  Ring<Point> exterior_{};
  /// @brief Interior rings (holes).
  inner_container_type interiors_{};
};

// ============================================================================
// Implementation of templated methods
// ============================================================================

template <typename Point>
auto Polygon<Point>::unpack(serialization::Reader& state) -> Polygon<Point> {
  if (state.size() == 0) {
    throw std::invalid_argument("Cannot restore polygon from empty state.");
  }
  auto magic_number = state.read<uint32_t>();
  if (magic_number != kMagicNumber) {
    throw std::invalid_argument("Invalid polygon state (bad magic number).");
  }
  auto ring_state = state.read_vector<std::byte>();
  auto reader = serialization::Reader(std::move(ring_state));
  auto exterior = Ring<Point>::unpack(reader);
  auto num_interiors = state.read<size_t>();
  std::vector<Ring<Point>> interiors;
  interiors.reserve(num_interiors);
  for (size_t i = 0; i < num_interiors; ++i) {
    ring_state = state.read_vector<std::byte>();
    reader = serialization::Reader(std::move(ring_state));
    interiors.push_back(Ring<Point>::unpack(reader));
  }
  return {std::move(exterior), std::move(interiors)};
}

}  // namespace pyinterp::geometry
