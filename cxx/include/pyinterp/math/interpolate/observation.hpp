// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <concepts>
#include <cstdint>

#include "pyinterp/geometry/rtree_value_traits.hpp"
#include "pyinterp/serialization_buffer.hpp"

namespace pyinterp::math::interpolate {

/// @brief A scalar observation carrying its own measurement-error variance.
///
/// Used as the value type of the R-tree feeding Optimal Interpolation
/// (BLUE). The @c sigma2 field corresponds to the diagonal entry contributed
/// by this observation to the matrix @c R in
/// @f$ (C_{oo} + R)\,w = c_{og} @f$. Storing the variance alongside the value
/// avoids any external mapping between observations and their error model
/// (typically a per-mission noise level).
///
/// The struct is a plain POD (16 bytes for @c double) so that it can be
/// packed densely in the Boost R*-tree without any indirection.
template <std::floating_point T>
struct Observation {
  /// Observed value.
  T value;
  /// Measurement-error variance (`σ²_obs`).
  T sigma2;

  /// @brief Equality is defined for testability and serialization round-trips.
  [[nodiscard]] friend constexpr auto operator==(
      const Observation& a, const Observation& b) noexcept -> bool {
    return a.value == b.value && a.sigma2 == b.sigma2;
  }
};

}  // namespace pyinterp::math::interpolate

namespace pyinterp::geometry {

/// @brief Trait specialization for @c Observation: two scalars on the wire,
/// distinct serialization tag.
template <std::floating_point T>
struct value_traits<math::interpolate::Observation<T>> {
  using scalar_type = T;

  /// No useful promotion: composite values cannot be combined arithmetically
  /// with coordinates. Math methods that would require promotion are
  /// disabled via @c requires on @c RTree members.
  template <typename Coordinate>
  using promotion_type = math::interpolate::Observation<T>;

  /// Magic-number contribution unique to observation trees.
  /// Picked far from the default `0` so accidental collisions are impossible.
  static constexpr uint32_t serialization_tag = 0x4F000000U;  // 'O' << 24

  static auto write(serialization::Writer& buffer,
                    const math::interpolate::Observation<T>& value) -> void {
    buffer.write(value.value);
    buffer.write(value.sigma2);
  }

  [[nodiscard]] static auto read(serialization::Reader& state)
      -> math::interpolate::Observation<T> {
    const T value = state.read<T>();
    const T sigma2 = state.read<T>();
    return {.value = value, .sigma2 = sigma2};
  }
};

}  // namespace pyinterp::geometry
