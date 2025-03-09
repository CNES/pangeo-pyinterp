// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include "pyinterp/detail/interpolation/interpolator.hpp"

namespace pyinterp::detail::interpolation {

/// Base class for all 2D interpolators
/// @tparam T type of the data
template <typename T>
class Interpolator2D : public Interpolator<T> {
 public:
  /// The minimum size of the arrays to be interpolated.
  virtual auto min_size() const -> Eigen::Index = 0;

  /// Interpolate the value of y at x.
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  /// @param za Z-values of the data points.
  /// @param x The point where the interpolation must be calculated.
  /// @param y The point where the interpolation must be calculated.
  /// @return The interpolated value at the point x.
  auto operator()(const Vector<T> &xa, const Vector<T> &ya, const Matrix<T> &za,
                  const T &x, const T &y) -> T {
    compute_coefficients(xa, ya, za);
    return interpolate_(xa, ya, za, x, y);
  }

  /// Interpolate the values of y at x.
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  /// @param za Z-values of the data points.
  /// @param x The point where the interpolation must be calculated.
  /// @param y The point where the interpolation must be calculated.
  /// @return The interpolated value at the point x.
  auto operator()(const Vector<T> &xa, const Vector<T> &ya, const Matrix<T> &za,
                  const Vector<T> &x, const Vector<T> &y) -> Vector<T> {
    compute_coefficients(xa, ya, za);
    auto z = Vector<T>(x.size());
    for (Eigen::Index i = 0; i < x.size(); ++i) {
      z(i) = interpolate_(xa, ya, za, x(i), y(i));
    }
    return z;
  }

 protected:
  /// Interpolate the value of y at x using the index of the last search.
  virtual auto interpolate_(const Vector<T> &xa, const Vector<T> &ya,
                            const Matrix<T> &za, const T &x, const T &y) const
      -> T = 0;

  /// Check if the arrays are valid.
  virtual auto compute_coefficients(const Vector<T> &xa, const Vector<T> &ya,
                                    const Matrix<T> &za) -> void {
    if (xa.size() != za.rows()) {
      throw std::invalid_argument(
          "xa and za must have the same number of rows");
    }
    if (ya.size() != za.cols()) {
      throw std::invalid_argument(
          "ya and za must have the same number of columns");
    }
    if (xa.size() < min_size()) {
      throw std::invalid_argument("xa must have at least " +
                                  std::to_string(min_size()) + " elements");
    }
    if (ya.size() < min_size()) {
      throw std::invalid_argument("ya must have at least " +
                                  std::to_string(min_size()) + " elements");
    }
  }
};

}  // namespace pyinterp::detail::interpolation
