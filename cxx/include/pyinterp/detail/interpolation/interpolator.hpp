// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <optional>
#include <utility>

#include "pyinterp/eigen.hpp"

namespace pyinterp::detail::interpolation {

/// @brief Base class for all interpolators
template <typename T>
class Interpolator {
 public:
  /// Constructor.
  Interpolator() = default;

  /// Destructor.
  virtual ~Interpolator() = default;

  /// @brief Search for the index of the first element in xa that is greater
  /// than x.
  /// @param xa The array to search.
  /// @param x The value to search for.
  /// @return An optional pair of indices (i, j) such that xa[i] < x <= xa[j].
  constexpr auto search(const Vector<T> &xa, const T &x) const
      -> std::optional<std::pair<Eigen::Index, Eigen::Index>>;
};

template <typename T>
constexpr auto Interpolator<T>::search(const Vector<T> &xa, const T &x) const
    -> std::optional<std::pair<Eigen::Index, Eigen::Index>> {
  const auto begin = xa.array().begin();
  const auto end = xa.array().end();
  const auto it = std::lower_bound(begin, end, x);
  if (it == end) {
    return std::nullopt;
  }
  if (it == begin) {
    return *it == x
               ? std::optional<std::pair<Eigen::Index, Eigen::Index>>({0, 1})
               : std::nullopt;
  }
  auto i = static_cast<Eigen::Index>(std::distance(begin, it));
  if (xa(i - 1) < x && x <= xa(i)) {
    return std::make_pair(i - 1, i);
  }
  return {};
}

}  // namespace pyinterp::detail::interpolation
