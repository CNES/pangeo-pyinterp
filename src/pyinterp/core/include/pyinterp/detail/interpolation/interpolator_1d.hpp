// Copyright (c) 2024 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include "pyinterp/detail/interpolation/interpolator.hpp"

namespace pyinterp::detail::interpolation {

/// @brief One-dimensional interpolation (interpolation along the x-axis).
/// @tparam T type of data
template <typename T>
class Interpolator1D : public Interpolator<T> {
 public:
  /// The minimum size of the arrays to be interpolated.
  virtual auto min_size() const -> Eigen::Index = 0;

  /// Interpolate the value of y at x.
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  /// @param x The point where the interpolation must be calculated.
  /// @return The interpolated value at the point x.
  auto operator()(const Eigen::Ref<const Vector<T>> &xa,
                  const Eigen::Ref<const Vector<T>> &ya, const T &x) -> T {
    compute_coefficients(xa, ya);
    auto index = Eigen::Index{};
    return (*this)(xa, ya, x, &index);
  }

  /// Interpolate the values of y at x.
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  /// @param x The points where the interpolation must be calculated.
  /// @return The interpolated values at the points x.
  auto operator()(const Eigen::Ref<const Vector<T>> &xa,
                  const Eigen::Ref<const Vector<T>> &ya,
                  const Eigen::Ref<const Vector<T>> &x) -> Vector<T> {
    compute_coefficients(xa, ya);
    auto ix = Eigen::Index{};
    auto y = Vector<T>(x.size());
    for (Eigen::Index i = 0; i < x.size(); ++i) {
      y(i) = (*this)(xa, ya, x(i), &ix);
    }
    return y;
  }

  /// Calculate the derivative of y at x.
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  /// @param x The point where the derivative must be calculated.
  /// @return The derivative of the interpolation function at the point x.
  auto derivative(const Eigen::Ref<const Vector<T>> &xa,
                  const Eigen::Ref<const Vector<T>> &ya, const T &x) -> T {
    compute_coefficients(xa, ya);
    auto index = Eigen::Index{};
    return derivative(xa, ya, x, &index);
  }

  /// Calculate the derivatives of y at x.
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  /// @param x The points where the derivative must be calculated.
  /// @return The derivatives of the interpolation function at the points x.
  auto derivative(const Eigen::Ref<const Vector<T>> &xa,
                  const Eigen::Ref<const Vector<T>> &ya,
                  const Eigen::Ref<const Vector<T>> &x) -> Vector<T> {
    compute_coefficients(xa, ya);
    auto ix = Eigen::Index{};
    auto y = Vector<T>(x.size());
    for (Eigen::Index i = 0; i < x.size(); ++i) {
      y(i) = derivative(xa, ya, x(i), &ix);
    }
    return y;
  }

 protected:
  /// Interpolate the value of y at x using the index of the last search.
  virtual auto operator()(const Eigen::Ref<const Vector<T>> &xa,
                          const Eigen::Ref<const Vector<T>> &ya, const T &x,
                          Eigen::Index *index) const -> T = 0;

  /// Calculate the derivative of y at x using the index of the last search.
  virtual auto derivative(const Eigen::Ref<const Vector<T>> &xa,
                          const Eigen::Ref<const Vector<T>> &ya, const T &x,
                          Eigen::Index *index) const -> T = 0;

  /// Check if the arrays are valid.
  virtual auto compute_coefficients(const Eigen::Ref<const Vector<T>> &xa,
                                    const Eigen::Ref<const Vector<T>> &ya)
      -> void {
    if (xa.size() != ya.size()) {
      throw std::invalid_argument("xa and ya must have the same size");
    }
    if (xa.size() < min_size()) {
      throw std::invalid_argument("xa and ya must have at least " +
                                  std::to_string(min_size()) + " elements");
    }
  }
};

}  // namespace pyinterp::detail::interpolation
