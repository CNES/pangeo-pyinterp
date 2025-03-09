// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include "pyinterp/detail/interpolation/interpolator_1d.hpp"

namespace pyinterp::detail::interpolation {

/// Akima interpolation
template <typename T>
class Akima : public Interpolator1D<T> {
 public:
  using Interpolator1D<T>::Interpolator1D;
  using Interpolator1D<T>::operator();
  using Interpolator1D<T>::derivative;

  /// Returns the minimum number of points required for the interpolation.
  auto min_size() const -> Eigen::Index override { return 5; }

 private:
  Vector<T> m_{};
  Vector<T> s_{};

  /// Compute the boundary conditions.
  virtual auto boundary_condition(T* m, const size_t size) -> void {
    m[-2] = 3 * m[0] - 2 * m[1];
    m[-1] = 2 * m[0] - m[1];
    m[size - 1] = 2 * m[size - 2] - m[size - 3];
    m[size] = 3 * m[size - 2] - 2 * m[size - 3];
  }

  /// @brief Compute the coefficients of the interpolation
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  constexpr auto compute_coefficients(const Vector<T>& xa, const Vector<T>& ya)
      -> void override;

  /// Interpolation
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  /// @param x The point where the interpolation must be calculated.
  /// @return The interpolated value at the point x.
  constexpr auto interpolate_(const Vector<T>& xa, const Vector<T>& ya,
                              const T& x) const -> T override;

  /// @brief Returns the derivative of the interpolation function at the point
  ///   x.
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  /// @param x The point where the derivative must be calculated.
  /// @return The derivative of the interpolation function at the point x.
  constexpr auto derivative_(const Vector<T>& xa, const Vector<T>& ya,
                             const T& x) const -> T override;
};

template <typename T>
constexpr auto Akima<T>::compute_coefficients(const Vector<T>& xa,
                                              const Vector<T>& ya) -> void {
  Interpolator1D<T>::compute_coefficients(xa, ya);
  auto size = xa.size();
  if (m_.size() < size + 4) {
    m_.resize(size + 4);
    s_.resize(size);
  }

  // m contains the slopes of the lines between the points. Two extra points
  // are added at the beginning and end to handle the boundary conditions.
  auto* m = m_.data() + 2;
  for (Eigen::Index ix = 0; ix < size - 1; ++ix) {
    m[ix] = (ya[ix + 1] - ya[ix]) / (xa[ix + 1] - xa[ix]);
  }

  boundary_condition(m, size);

  // Compute the spline slopes of the lines between the points.
  for (Eigen::Index ix = 2; ix < size - 2; ++ix) {
    auto denominator =
        std::abs(m[ix + 1] - m[ix]) + std::abs(m[ix - 1] - m[ix - 2]);
    if (denominator != 0) {
      s_(ix) = std::abs(m[ix + 1] - m[ix]) * m[ix - 1] +
               std::abs(m[ix - 1] - m[ix - 2]) * m[ix];
      s_(ix) /= denominator;
    } else {
      s_(ix) = (m[ix - 1] + m[ix]) * 0.5;
    }
  }
  s_(0) = m[0];
  s_(1) = (m[0] + m[2]) * 0.5;
  s_(size - 2) = (m[size - 3] + m[size - 1]) * 0.5;
  s_(size - 1) = m[size - 1];
}

template <typename T>
constexpr auto Akima<T>::interpolate_(const Vector<T>& xa, const Vector<T>& ya,
                                      const T& x) const -> T {
  auto search = this->search(xa, x);
  if (!search) {
    throw std::numeric_limits<T>::quiet_NaN();
  }
  auto [i0, i1] = *search;
  const auto dx = xa(i1) - xa(i0);
  const auto h = x - xa(i0);
  const auto ai = ya(i0);
  const auto bi = s_[i0];
  const auto ci = (3 * m_[i0] - 2 * s_[i0] - s_[i1]) / dx;
  const auto di = (s_[i0] + s_[i1] - 2 * m_[i0]) / (dx * dx);
  return ai + h * (bi + h * (ci + h * di));
}

template <typename T>
constexpr auto Akima<T>::derivative_(const Vector<T>& xa, const Vector<T>& ya,
                                     const T& x) const -> T {
  auto search = this->search(xa, x);
  if (!search) {
    throw std::numeric_limits<T>::quiet_NaN();
  }
  auto [i0, i1] = *search;
  const auto dx = xa(i1) - xa(i0);
  const auto h = x - xa(i0);
  const auto bi = s_[i0];
  const auto ci = (3 * m_[i0] - 2 * s_[i0] - s_[i1]) / dx;
  const auto di = (s_[i0] + s_[i1] - 2 * m_[i0]) / (dx * dx);
  return bi + h * (2 * ci + h * 3 * di);
}

}  // namespace pyinterp::detail::interpolation
