// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <concepts>
#include <cstdint>
#include <span>

#include "pyinterp/math/interpolate/univariate/akima.hpp"

namespace pyinterp::math::interpolate::univariate {

/// @brief Akima periodic interpolation
/// Akima periodic interpolation ensures that the interpolation wraps around
/// seamlessly, making it ideal for cyclic data such as angles or time-of-day
/// values.
/// @tparam T Data type
template <std::floating_point T>
class AkimaPeriodic : public Akima<T> {
 public:
  using Akima<T>::Akima;

 private:
  /// @brief Compute the boundary conditions for periodic data.
  /// The m array is indexed as: [0..1] left boundary, [2..size+1] data slopes,
  /// [size+2..size+3] right boundary
  /// @param[in] m Pointer to slope array (with 2-element offset for boundary)
  /// @param[in] size Number of data points
  auto boundary_condition(std::span<T> m, const int64_t size) -> void final {
    // Wrap slopes from the end to fill left boundary
    m[0] = m[size - 1];  // Second-to-last slope
    m[1] = m[size];      // Last slope

    // Wrap slopes from the beginning to fill right boundary
    m[size + 1] = m[2];  // First slope
    m[size + 2] = m[3];  // Second slope
  }
};

}  // namespace pyinterp::math::interpolate::univariate
