// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <cmath>
#include <concepts>
#include <limits>

namespace pyinterp::math {

/// @brief Provides methods to handle fill values for different data types
/// @tparam T type of data
template <typename T>
struct Fill {
  /// @brief Get the fill value for the specified type T
  /// @return fill value
  static constexpr auto value() noexcept -> T
    requires std::floating_point<T>
  {
    return std::numeric_limits<T>::quiet_NaN();
  }

  /// @brief Get the fill value for the specified type T
  /// @return fill value
  static constexpr auto value() noexcept -> T
    requires std::integral<T>
  {
    return std::numeric_limits<T>::min();
  }

  /// @brief Check if the provided value is a fill value
  /// @param[in] x value to check
  /// @return true if the value is a fill value
  static constexpr auto is_fill_value(const T &x) noexcept -> bool
    requires std::floating_point<T>
  {
    return std::isnan(x);
  }

  /// @brief Check if the provided value is a fill value
  /// @param[in] x value to check
  /// @return true if the value is a fill value
  static constexpr auto is_fill_value(const T &x) noexcept -> bool
    requires std::integral<T>
  {
    return value() == x;
  }
};

}  // namespace pyinterp::math
