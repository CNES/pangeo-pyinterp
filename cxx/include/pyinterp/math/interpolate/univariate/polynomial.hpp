// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <ranges>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math/fill.hpp"
#include "pyinterp/math/interpolate/univariate.hpp"

namespace pyinterp::math::interpolate::univariate {

/// @brief Polynomial interpolation using Newton's divided differences
/// @tparam T Type of the data to interpolate
template <std::floating_point T>
class Polynomial : public Univariate<T> {
 public:
  using Univariate<T>::Univariate;
  using Univariate<T>::operator();
  using Univariate<T>::derivative;

  /// @brief The minimum size of the arrays to be interpolated.
  /// @return Minimum size of the arrays
  [[nodiscard]] constexpr auto min_size() const -> int64_t final { return 3; }

 private:
  /// @brief Coefficients of the interpolation (Newton's divided differences)
  Vector<T> work_{};

  /// @brief Compute the coefficients of the interpolation using divided
  /// differences
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @return True if coefficients were computed successfully
  [[nodiscard]] constexpr auto compute_coefficients(const Vector<T>& xa,
                                                    const Vector<T>& ya)
      -> bool final;

  /// @brief Compute Taylor series coefficients around point x
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] x The point around which to expand.
  /// @return Vector of Taylor coefficients
  [[nodiscard]] auto taylor(const Vector<T>& xa, T x) const -> Vector<T>;

  /// @brief Interpolation using Horner's method
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @param[in] x The point where the interpolation must be calculated.
  /// @return The interpolated value at the point x.
  [[nodiscard]] constexpr auto interpolate_(const Vector<T>& xa,
                                            const Vector<T>& ya, T x) const
      -> T final;

  /// @brief Returns the derivative of the interpolation function at point x.
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @param[in] x The point where the derivative must be calculated.
  /// @return The derivative of the interpolation function at the point x.
  [[nodiscard]] auto derivative_(const Vector<T>& xa, const Vector<T>& ya,
                                 T x) const -> T final;
};

// ============================================================================
// Implementation
// ============================================================================

template <std::floating_point T>
constexpr auto Polynomial<T>::compute_coefficients(const Vector<T>& xa,
                                                   const Vector<T>& ya)
    -> bool {
  if (!Univariate<T>::compute_coefficients(xa, ya)) [[unlikely]] {
    return false;
  }

  const auto size = xa.size();

  // Ensure work vector is properly sized
  if (work_.size() < size) {
    work_.resize(size);
  }

  // Initialize with y values
  work_(0) = ya(0);

  // First divided differences
  work_.segment(1, size - 1) =
      (ya.segment(1, size - 1) - ya.segment(0, size - 1)).array() /
      (xa.segment(1, size - 1) - xa.segment(0, size - 1)).array();

  // Higher order divided differences
  for (auto ix : std::views::iota(int64_t{2}, size)) {
    const auto n = size - ix;
    work_.segment(ix, n) =
        (work_.segment(ix, n) - work_.segment(ix - 1, n)).array() /
        (xa.segment(ix, n) - xa.segment(0, n)).array();
  }

  return true;
}

template <std::floating_point T>
auto Polynomial<T>::taylor(const Vector<T>& xa, const T x) const -> Vector<T> {
  const auto size = xa.size();
  Vector<T> c = Vector<T>::Zero(size);
  Vector<T> w = Vector<T>::Zero(size);

  // Initialize
  w(size - 1) = T{1};
  c(0) = work_(0);

  // Compute Taylor coefficients via transformation
  for (int64_t ix = size - 1; ix > 0; --ix) {
    const T dx = xa(size - 1 - ix) - x;

    // Update w coefficients
    w(ix - 1) = -w(ix) * dx;
    for (auto jx : std::views::iota(ix, size - 1)) {
      w(jx) -= w(jx + 1) * dx;
    }

    // Accumulate into Taylor coefficients
    for (auto jx : std::views::iota(ix - 1, size)) {
      c(jx - ix + 1) += w(jx) * work_(size - ix);
    }
  }

  return c;
}

template <std::floating_point T>
constexpr auto Polynomial<T>::interpolate_(const Vector<T>& xa,
                                           [[maybe_unused]] const Vector<T>& ya,
                                           const T x) const -> T {
  const auto search = this->search(xa, x);
  if (!search) [[unlikely]] {
    return Fill<T>::value();
  }

  const auto size = xa.size();

  // Horner's method for efficient polynomial evaluation
  T y = work_(size - 1);
  for (int64_t ix = size - 1; ix > 0; --ix) {
    y = std::fma(y, x - xa(ix - 1), work_(ix - 1));
  }

  return y;
}

template <std::floating_point T>
auto Polynomial<T>::derivative_(const Vector<T>& xa,
                                [[maybe_unused]] const Vector<T>& ya,
                                const T x) const -> T {
  const auto search = this->search(xa, x);
  if (!search) [[unlikely]] {
    return Fill<T>::value();
  }

  const auto coefficients = taylor(xa, x);
  return coefficients(1);
}

}  // namespace pyinterp::math::interpolate::univariate
