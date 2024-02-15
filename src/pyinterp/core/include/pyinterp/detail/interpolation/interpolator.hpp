// Copyright (c) 2024 CNES
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

  /// @brief Search for the index of the first element in xa that is greater
  /// than x.
  /// @param xa The array to search.
  /// @param x The value to search for.
  /// @param i The index to start the search from.
  /// @return An optional pair of indices (i, j) such that xa[i] < x <= xa[j].
  constexpr auto search(const Vector<T> &xa, const T &x, Eigen::Index *i) const
      -> std::optional<std::pair<Eigen::Index, Eigen::Index>>;
};

template <typename T>
constexpr auto Interpolator<T>::search(const Vector<T> &xa, const T &x) const
    -> std::optional<std::pair<Eigen::Index, Eigen::Index>> {
  auto index = Eigen::Index{};
  return search(xa, x, &index);
}

template <typename T>
constexpr auto Interpolator<T>::search(const Vector<T> &xa, const T &x,
                                       Eigen::Index *index) const
    -> std::optional<std::pair<Eigen::Index, Eigen::Index>> {
  // If the index is within the bounds of the array, check if x is within the
  // last interval.
  if (*index >= 0 && *index < xa.size() - 1 && xa(*index) < x &&
      x <= xa(*index + 1)) {
    return std::make_pair(*index, *index + 1);
  }
  const auto begin = xa.array().begin();
  const auto end = xa.array().end();
  for (auto i : {std::max(Eigen::Index(0), (*index) - 1), Eigen::Index(0)}) {
    const auto it = std::lower_bound(begin + i, end, x);
    if (it == end) {
      return std::nullopt;
    }
    if (it == begin) {
      return *it == x
                 ? std::optional<std::pair<Eigen::Index, Eigen::Index>>({0, 1})
                 : std::nullopt;
    }
    i = static_cast<Eigen::Index>(std::distance(begin, it));
    if (xa(i - 1) < x && x <= xa(i)) {
      *index = i - 1;
      return std::make_pair(*index, i);
    }
  }
  return {};
}

}  // namespace pyinterp::detail::interpolation
