// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <ranges>
#include <stdexcept>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math/fill.hpp"
#include "pyinterp/math/interpolate/bracket_finder.hpp"

namespace pyinterp::math::interpolate {

/// @brief Univariate interpolation base class
/// @tparam T type of data (must be arithmetic)
template <std::floating_point T>
class Univariate : public BracketFinder<T> {
 public:
  /// @brief The minimum size of the arrays to be interpolated.
  [[nodiscard]] virtual constexpr auto min_size() const -> int64_t = 0;

  /// @brief Interpolate the value of y at x.
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @param[in] x The point where the interpolation must be calculated.
  /// @return The interpolated value at the point x.
  [[nodiscard]] auto operator()(const Vector<T>& xa, const Vector<T>& ya,
                                const T x) -> T {
    return compute_coefficients(xa, ya) ? interpolate_(xa, ya, x)
                                        : Fill<T>::value();
  }

  /// @brief Interpolate the values of y at x.
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @param[in] x The points where the interpolation must be calculated.
  /// @return The interpolated values at the points x.
  [[nodiscard]] auto operator()(const Vector<T>& xa, const Vector<T>& ya,
                                const Vector<T>& x) -> Vector<T> {
    if (!compute_coefficients(xa, ya)) {
      return Vector<T>::Constant(x.size(), Fill<T>::value());
    }

    auto y = Vector<T>(x.size());
    auto indices = std::views::iota(int64_t{0}, x.size());

    std::ranges::for_each(indices,
                          [&](auto i) { y(i) = interpolate_(xa, ya, x(i)); });

    return y;
  }

  /// @brief Calculate the derivative of y at x.
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @param[in] x The point where the derivative must be calculated.
  /// @return The derivative of the interpolation function at the point x.
  [[nodiscard]] auto derivative(const Vector<T>& xa, const Vector<T>& ya,
                                const T x) -> T {
    return compute_coefficients(xa, ya) ? derivative_(xa, ya, x)
                                        : Fill<T>::value();
  }

  /// @brief Calculate the derivatives of y at x.
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @param[in] x The points where the derivative must be calculated.
  /// @return The derivatives of the interpolation function at the points x.
  [[nodiscard]] auto derivative(const Vector<T>& xa, const Vector<T>& ya,
                                const Vector<T>& x) -> Vector<T> {
    if (!compute_coefficients(xa, ya)) {
      return Vector<T>::Constant(x.size(), Fill<T>::value());
    }

    auto y = Vector<T>(x.size());
    auto indices = std::views::iota(int64_t{0}, x.size());

    std::ranges::for_each(indices,
                          [&](auto i) { y(i) = derivative_(xa, ya, x(i)); });

    return y;
  }

 protected:
  /// @brief Interpolate the value of y at x.
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @param[in] x The point where the interpolation must be calculated.
  /// @return The interpolated value at the point x.
  [[nodiscard]] virtual auto interpolate_(const Vector<T>& xa,
                                          const Vector<T>& ya, const T x) const
      -> T = 0;

  /// @brief Calculate the derivative of y at x.
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @param[in] x The point where the derivative must be calculated.
  /// @return The derivative of the interpolation function at the point x.
  [[nodiscard]] virtual auto derivative_(const Vector<T>& xa,
                                         const Vector<T>& ya, const T x) const
      -> T = 0;

  /// @brief Check if the arrays are valid.
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @return True if the coefficients were successfully computed, false
  /// otherwise.
  [[nodiscard]] virtual constexpr auto compute_coefficients(const Vector<T>& xa,
                                                            const Vector<T>& ya)
      -> bool {
    if (xa.size() != ya.size()) [[unlikely]] {
      throw std::invalid_argument("xa and ya must have the same size");
    }
    return xa.size() >= min_size();
  }
};

}  // namespace pyinterp::math::interpolate
