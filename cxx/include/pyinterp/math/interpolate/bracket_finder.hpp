// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

#include "pyinterp/eigen.hpp"

namespace pyinterp::math::interpolate {

/// @brief Base class for interval/bracket search over sorted coordinates.
/// @tparam T Type of the coordinates
template <typename T>
class BracketFinder {
 public:
  /// @brief Default constructor
  constexpr BracketFinder() = default;

  /// @brief Destructor
  virtual ~BracketFinder() = default;

  /// @brief Locate indices i,i+1 such that xa[i] <= x <= xa[i+1]
  /// @param[in] xa Sorted array of coordinates
  /// @param[in] x Coordinate to locate
  /// @return Pair of indices (i, i+1) if found, std::nullopt otherwise
  constexpr auto search(const Vector<T>& xa, const T& x) const noexcept
      -> std::optional<std::pair<int64_t, int64_t>>;
};

// ============================================================================
// Implementation
// ============================================================================

template <typename T>
constexpr auto BracketFinder<T>::search(const Vector<T>& xa,
                                        const T& x) const noexcept
    -> std::optional<std::pair<int64_t, int64_t>> {
  const auto begin = xa.begin();
  const auto end = xa.end();

  if (begin == end) {
    return std::nullopt;
  }
  const bool is_ascending = xa.size() == 1 || xa[0] <= xa[xa.size() - 1];
  const auto it = is_ascending
                      ? std::lower_bound(begin, end, x)
                      : std::lower_bound(begin, end, x, std::greater<T>());

  // Case 1: x is outside the coordinate range, no bracket found
  if (it == end) {
    return std::nullopt;
  }
  const auto idx = it - begin;

  // Case 2: x lands before or exactly at the first element
  if (idx == 0) {
    return (*it == x && xa.size() > 1) ? std::optional{std::make_pair(0, 1)}
                                       : std::nullopt;
  }

  // Now we know idx > 0, check if we indeed bracket x
  const auto i0 = idx - 1;
  const auto i1 = idx;

  const bool is_bracketed = is_ascending ? (xa[i0] <= x && x <= xa[i1])
                                         : (xa[i0] >= x && x >= xa[i1]);

  if (is_bracketed) {
    return {{i0, i1}};
  }

  return std::nullopt;
}

}  // namespace pyinterp::math::interpolate
