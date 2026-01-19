// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <optional>
#include <string>

namespace pyinterp::math::interpolate {

/// Domain bounds for a single dimension
template <typename T>
struct DomainBounds {
  T min{};            /// Minimum bound
  T max{};            /// Maximum bound
  bool valid{false};  /// Indicates if bounds are valid

  /// Check if a value is within bounds (inclusive)
  /// @param[in] value The value to check
  /// @return True if value is within bounds
  [[nodiscard]] constexpr auto contains(T value) const noexcept -> bool {
    return valid && value >= min && value <= max;
  }
};

/// @brief Result of a cache loading operation
struct CacheLoadResult {
  /// @brief True if the cache was loaded successfully
  bool success{false};
  /// @brief True if the cache was updated (false if already contained point)
  bool was_updated{false};
  /// @brief Error message if loading failed due to out-of-bounds access
  std::optional<std::string> error_message;
};

}  // namespace pyinterp::math::interpolate
