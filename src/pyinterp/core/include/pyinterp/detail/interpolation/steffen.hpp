// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include "pyinterp/detail/interpolation/interpolator_1d.hpp"

namespace pyinterp::detail::interpolation {

/// Steffen interpolation
template <typename T>
class Steffen : public Interpolator1D<T> {
 public:
  using Interpolator1D<T>::Interpolator1D;
  using Interpolator1D<T>::operator();
  using Interpolator1D<T>::derivative;

  /// Returns the minimum number of points required for the interpolation.
  auto min_size() const -> Eigen::Index override { return 3; }

 private:
  /// Interpolation
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  /// @param x The point where the interpolation must be calculated.
  /// @return The interpolated value at the point x.
  constexpr auto interpolate_(const Vector<T> &xa, const Vector<T> &ya,
                              const T &x) const -> T override;

  /// @brief Returns the derivative of the interpolation function at the point
  ///   x.
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  /// @param x The point where the derivative must be calculated.
  constexpr auto derivative_(const Vector<T> &xa, const Vector<T> &ya,
                             const T &x) const -> T override;

 private:
  /// Compute the coefficients of the interpolation
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  constexpr auto compute_coefficients(const Vector<T> &xa, const Vector<T> &ya)
      -> void override;

  /// Return the sign of x multiplied by the sign of y
  static constexpr auto copysign(const T &x, const T &y) -> T {
    return (x < T(0) && y > T(0)) || (x > T(0) && y < T(0)) ? -x : x;
  }
  /// The slopes of the interpolation
  Vector<T> y_prime_;
};

template <typename T>
constexpr auto Steffen<T>::compute_coefficients(const Vector<T> &xa,
                                                const Vector<T> &ya) -> void {
  Interpolator1D<T>::compute_coefficients(xa, ya);
  auto size = xa.size();
  if (y_prime_.size() < size) {
    y_prime_.resize(size);
  }
  // First assign the interval and slopes for the left boundary.
  // We use the "simplest possibility" method described in the paper
  // in section 2.2
  auto h0 = (xa[1] - xa[0]);
  auto s0 = (ya[1] - ya[0]) / h0;

  y_prime_[0] = s0;

  // Now we calculate all the necessary s, h, p, and y' variables  from 1 to
  // size-2 (0 to size - 2 inclusive)
  for (Eigen::Index i = 1; i < size - 1; ++i) {
    // Eq. 6
    auto hi = (xa[i + 1] - xa[i]);
    auto him1 = (xa[i] - xa[i - 1]);
    // Eq. 7
    auto si = (ya[i + 1] - ya[i]) / hi;
    auto sim1 = (ya[i] - ya[i - 1]) / him1;
    // Eq. 8
    auto pi = (sim1 * hi + si * him1) / (him1 + hi);

    y_prime_[i] =
        (Steffen::copysign(T(1), sim1) + Steffen::copysign(T(1), si)) *
        std::min(std::fabs(sim1), std::min(std::fabs(si), 0.5 * std::fabs(pi)));
  }
  // We also need y' for the rightmost boundary; we use the
  // "simplest possibility" method described in the paper in
  // section 2.2
  y_prime_[size - 1] =
      (ya[size - 1] - ya[size - 2]) / (xa[size - 1] - xa[size - 2]);
}

template <typename T>
constexpr auto Steffen<T>::interpolate_(const Vector<T> &xa,
                                        const Vector<T> &ya, const T &x) const
    -> T {
  auto where = this->search(xa, x);
  if (!where) {
    return std::numeric_limits<T>::quiet_NaN();
  }
  auto [i0, i1] = *where;
  const auto h = x - xa[i0];
  const auto hi = (xa[i1] - xa[i0]);
  const auto si = (ya[i1] - ya[i0]) / hi;
  const auto a = (y_prime_[i0] + y_prime_[i1] - 2 * si) / hi / hi;
  const auto b = (3 * si - 2 * y_prime_[i0] - y_prime_[i1]) / hi;
  const auto c = y_prime_[i0];
  const auto d = ya[i0];
  return d + h * (c + h * (b + h * a));
}

template <typename T>
constexpr auto Steffen<T>::derivative_(const Vector<T> &xa, const Vector<T> &ya,
                                       const T &x) const -> T {
  auto where = this->search(xa, x);
  if (!where) {
    return std::numeric_limits<T>::quiet_NaN();
  }
  auto [i0, i1] = *where;
  const auto h = x - xa[i0];
  const auto hi = (xa[i1] - xa[i0]);
  const auto si = (ya[i1] - ya[i0]) / hi;
  const auto a = (y_prime_[i0] + y_prime_[i1] - 2 * si) / hi / hi;
  const auto b = (3 * si - 2 * y_prime_[i0] - y_prime_[i1]) / hi;
  const auto c = y_prime_[i0];
  return c + h * (2 * b + h * 3 * a);
}

}  // namespace pyinterp::detail::interpolation
