// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <optional>

#include "pyinterp/math/fill.hpp"

namespace pyinterp::math::interpolate {

/// Result type for single-point interpolation
template <typename T>
struct InterpolationResult {
  std::optional<T> value;  ///< Interpolated value

  /// Returns true if the interpolation result contains a value
  [[nodiscard]] constexpr auto has_value() const noexcept -> bool {
    return value.has_value();
  }

  /// Returns the interpolated value or a default value if not present
  [[nodiscard]] constexpr auto value_or(T default_val) const -> T {
    return value.value_or(default_val);
  }

  /// Returns the interpolated value or fill value (e.g., NaN) if not present
  [[nodiscard]] constexpr auto value_or_fill() const -> T {
    return value.value_or(Fill<T>::value());
  }
};

}  // namespace pyinterp::math::interpolate
