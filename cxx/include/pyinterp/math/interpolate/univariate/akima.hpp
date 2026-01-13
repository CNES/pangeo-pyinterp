// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <ranges>
#include <span>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math/fill.hpp"
#include "pyinterp/math/interpolate/univariate.hpp"

namespace pyinterp::math::interpolate::univariate {

/// @brief Akima interpolation
/// Akima interpolation provides a smooth, continuously differentiable spline
/// that minimizes overshooting near local extrema, ensuring stable and accurate
/// results.
/// @tparam T Data type
template <std::floating_point T>
class Akima : public Univariate<T> {
 public:
  using Univariate<T>::Univariate;
  using Univariate<T>::operator();
  using Univariate<T>::derivative;

  /// @brief Returns the minimum number of points required for the
  /// interpolation.
  /// @return Minimum number of points
  [[nodiscard]] constexpr auto min_size() const -> int64_t final { return 5; }

 private:
  /// Segment slopes (including extended boundary points)
  Vector<T> m_{};

  /// Akima spline slopes at data points
  Vector<T> s_{};

  /// @brief Compute the boundary conditions using extrapolation
  /// @param[in] m Pointer to slope array (with 2-element offset for boundary)
  /// @param[in] size Number of data points
  virtual auto boundary_condition(std::span<T> m, const int64_t size) -> void {
    // Extrapolate slopes before first point
    m[0] = T{3} * m[2] - T{2} * m[3];
    m[1] = T{2} * m[2] - m[3];

    // Extrapolate slopes after last point
    m[size + 1] = T{2} * m[size] - m[size - 1];
    m[size + 2] = T{3} * m[size] - T{2} * m[size - 1];
  }

  /// @brief Compute the coefficients of the interpolation
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @return True if coefficients computed successfully
  [[nodiscard]] constexpr auto compute_coefficients(const Vector<T>& xa,
                                                    const Vector<T>& ya)
      -> bool final;

  /// @brief Interpolation using Akima cubic polynomials
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @param[in] x The point where the interpolation must be calculated.
  /// @return The interpolated value at the point x.
  [[nodiscard]] constexpr auto interpolate_(const Vector<T>& xa,
                                            const Vector<T>& ya,
                                            const T x) const -> T final;

  /// @brief Returns the derivative of the interpolation function at point x.
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @param[in] x The point where the derivative must be calculated.
  /// @return The derivative of the interpolation function at the point x.
  [[nodiscard]] constexpr auto derivative_(const Vector<T>& xa,
                                           const Vector<T>& ya, const T x) const
      -> T final;
};

// ============================================================================
// Implementation
// ============================================================================

template <std::floating_point T>
constexpr auto Akima<T>::compute_coefficients(const Vector<T>& xa,
                                              const Vector<T>& ya) -> bool {
  if (!Univariate<T>::compute_coefficients(xa, ya)) [[unlikely]] {
    return false;
  }

  const auto size = xa.size();

  // Resize arrays if needed (extra 4 elements for boundary conditions)
  if (m_.size() < size + 4) {
    m_.resize(size + 4);
    s_.resize(size);
  }

  // Compute segment slopes with offset pointer for boundary handling
  // m[0..1] are for left boundary, m[2..size+1] are actual slopes,
  // m[size+2..size+3] are for right boundary
  auto m_span = std::span<T>(m_.data(), m_.size());
  auto m_interior = m_span.subspan(2, size - 1);

  for (auto ix : std::views::iota(int64_t{0}, size - 1)) {
    m_interior[ix] = (ya(ix + 1) - ya(ix)) / (xa(ix + 1) - xa(ix));
  }

  boundary_condition(m_span, size - 1);

  // Compute Akima spline slopes at interior points using weighted average
  for (auto ix : std::views::iota(int64_t{2}, size - 2)) {
    const T w1 = std::abs(m_span[ix + 3] - m_span[ix + 2]);
    const T w2 = std::abs(m_span[ix + 1] - m_span[ix]);
    const T denominator = w1 + w2;

    if (denominator != T{0}) {
      // Weighted average based on curvature
      s_(ix) = std::fma(w1, m_span[ix + 1], w2 * m_span[ix + 2]) / denominator;
    } else {
      // Equal weights when no curvature difference
      s_(ix) = (m_span[ix + 1] + m_span[ix + 2]) * T{0.5};
    }
  }

  // Handle boundary point slopes with simpler rules
  s_(0) = m_span[2];
  s_(1) = (m_span[2] + m_span[3]) * T{0.5};
  s_(size - 2) = (m_span[size - 1] + m_span[size]) * T{0.5};
  s_(size - 1) = m_span[size];

  return true;
}

template <std::floating_point T>
constexpr auto Akima<T>::interpolate_(const Vector<T>& xa, const Vector<T>& ya,
                                      const T x) const -> T {
  const auto search = this->search(xa, x);
  if (!search) [[unlikely]] {
    return Fill<T>::value();
  }

  const auto [i0, i1] = *search;

  const T dx = xa(i1) - xa(i0);
  const T h = x - xa(i0);

  // Cubic polynomial coefficients: y = ai + bi*h + ci*h² + di*h³
  const T ai = ya(i0);
  const T bi = s_(i0);
  const T mi = m_(i0 + 2);  // Account for 2-element offset in m_
  const T ci = (T{3} * mi - T{2} * s_(i0) - s_(i1)) / dx;
  const T di = (s_(i0) + s_(i1) - T{2} * mi) / (dx * dx);

  // Evaluate using Horner's method: ai + h*(bi + h*(ci + h*di))
  return ai + h * std::fma(h, std::fma(h, di, ci), bi);
}

template <std::floating_point T>
constexpr auto Akima<T>::derivative_(const Vector<T>& xa,
                                     [[maybe_unused]] const Vector<T>& ya,
                                     const T x) const -> T {
  const auto search = this->search(xa, x);
  if (!search) [[unlikely]] {
    return Fill<T>::value();
  }

  const auto [i0, i1] = *search;

  const T dx = xa(i1) - xa(i0);
  const T h = x - xa(i0);

  // Derivative coefficients: dy/dx = bi + 2*ci*h + 3*di*h²
  const T bi = s_(i0);
  const T mi = m_(i0 + 2);  // Account for 2-element offset in m_
  const T ci = (T{3} * mi - T{2} * s_(i0) - s_(i1)) / dx;
  const T di = (s_(i0) + s_(i1) - T{2} * mi) / (dx * dx);

  // Evaluate: bi + h*(2*ci + h*3*di)
  return bi + h * std::fma(T{3} * h, di, T{2} * ci);
}

}  // namespace pyinterp::math::interpolate::univariate
