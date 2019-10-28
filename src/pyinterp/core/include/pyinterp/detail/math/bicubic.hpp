// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Core>
#include "pyinterp/detail/gsl/interpolate1d.hpp"
#include "pyinterp/detail/math.hpp"

namespace pyinterp::detail::math {

/// Set of coordinates/values used for interpolation
///  * q11 = (x1, y1)
///  * q12 = (x1, y2)
///  * .../...
///  * q1n = (x1, yn)
///  * q21 = (x2, y1)
///  * q22 = (x2, y2).
///  * .../...
///  * q2n = (x2, yn)
///  * .../...
///  * qnn = (xn, yn)
///
/// @code
/// Array({{x1, x2, ..., xn}, {y1, y2, ..., yn}},
///        {q11, q12, ..., q21, q22, ...., qnn})
/// @endcode
class XArray {
 public:
  /// Default constructor
  XArray() = delete;

  /// Creates a new Array
  XArray(const size_t x_size, const size_t y_size) {
    auto nx = x_size << 1U;
    auto ny = y_size << 1U;
    x_.resize(nx);
    y_.resize(ny);
    q_.resize(nx, ny);
  }

  /// Default destructor
  virtual ~XArray() = default;

  /// Copy constructor
  ///
  /// @param rhs right value
  XArray(const XArray &rhs) = default;

  /// Move constructor
  ///
  /// @param rhs right value
  XArray(XArray &&rhs) noexcept = default;

  /// Copy assignment operator
  ///
  /// @param rhs right value
  auto operator=(const XArray &rhs) -> XArray & = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  auto operator=(XArray &&rhs) noexcept -> XArray & = default;

  /// Get the half size of the window in abscissa.
  [[nodiscard]] inline auto nx() const noexcept -> size_t {
    return static_cast<size_t>(x_.size()) >> 1U;
  }

  /// Get the half size of the window in ordinate.
  [[nodiscard]] inline auto ny() const noexcept -> size_t {
    return static_cast<size_t>(y_.size()) >> 1U;
  }

  /// Get x-coordinates
  inline auto x() noexcept -> Eigen::VectorXd & { return x_; }

  /// Get x-coordinates
  [[nodiscard]] inline auto x() const noexcept -> const Eigen::VectorXd & {
    return x_;
  }

  /// Get y-coordinates
  inline auto y() noexcept -> Eigen::VectorXd & { return y_; }

  /// Get y-coordinates
  [[nodiscard]] inline auto y() const noexcept -> const Eigen::VectorXd & {
    return y_;
  }

  /// Get the values from the array for all x and y coordinates.
  inline auto q() noexcept -> Eigen::MatrixXd & { return q_; }

  /// Get the values from the array for all x and y coordinates.
  [[nodiscard]] inline auto q() const noexcept -> const Eigen::MatrixXd & {
    return q_;
  }

  /// Get the ith x-axis.
  [[nodiscard]] inline auto x(const size_t ix) const -> double {
    return x_(ix);
  }

  /// Get the ith y-axis.
  [[nodiscard]] inline auto y(const size_t jx) const -> double {
    return y_(jx);
  }

  /// Get the value at coordinate (ix, jx).
  [[nodiscard]] inline auto z(const size_t ix, const size_t jx) const
      -> double {
    return q_(ix, jx);
  }

  /// Set the ith x-axis.
  inline auto x(const size_t ix) -> double & { return x_(ix); }

  /// Get the ith y-axis.
  inline auto y(const size_t jx) -> double & { return y_(jx); }

  /// Get the value at coordinate (ix, jx).
  inline auto z(const size_t ix, const size_t jx) -> double & {
    return q_(ix, jx);
  }

  /// Normalizes the angle with respect to the first value of the X axis of this
  /// array.
  [[nodiscard]] inline auto normalize_angle(const double xi) const -> double {
    return math::normalize_angle(xi, x(0), 360.0);
  }

  /// Returns true if this instance does not contains at least one Not A Number
  /// (NaN).
  [[nodiscard]] inline auto is_valid() const -> bool { return !q_.hasNaN(); }

 private:
  Eigen::VectorXd x_{};
  Eigen::VectorXd y_{};
  Eigen::MatrixXd q_{};
};

/// Extension of cubic interpolation for interpolating data points on a
/// two-dimensional regular grid. The interpolated surface is smoother than
/// corresponding surfaces obtained by bilinear interpolation or
/// nearest-neighbor interpolation.
class Bicubic {
 public:
  /// Default constructor
  ///
  /// @param xr Calculation window.
  /// @param type method of calculation
  explicit Bicubic(const XArray &xr, const gsl_interp_type *type)
      : column_(xr.x().size()),
        interpolator_(std::max(xr.x().size(), xr.y().size()), type,
                      gsl::Accelerator()) {}

  /// Return the interpolated value of y for a given point x
  auto interpolate(const double x, const double y, const XArray &xr) -> double {
    return evaluate(&gsl::Interpolate1D::interpolate, x, y, xr);
  }

  /// Return the derivative for a given point x
  auto derivative(const double x, const double y, const XArray &xr) -> double {
    return evaluate(&gsl::Interpolate1D::derivative, x, y, xr);
  }

  /// Return the second derivative for a given point x
  auto second_derivative(const double x, const double y, const XArray &xr)
      -> double {
    return evaluate(&gsl::Interpolate1D::second_derivative, x, y, xr);
  }

 private:
  using InterpolateFunction = double (gsl::Interpolate1D::*)(
      const Eigen::VectorXd &, const Eigen::VectorXd &, const double);
  /// Column of the interpolation window (interpolation according to Y
  /// coordinates)
  Eigen::VectorXd column_;
  /// GSL interpolator
  gsl::Interpolate1D interpolator_;

  /// Evaluation of the GSL function performing the calculation.
  auto evaluate(
      const std::function<double(gsl::Interpolate1D &, const Eigen::VectorXd &,
                                 const Eigen::VectorXd &, const double)>
          &function,
      const double x, const double y, const XArray &xr) -> double {
    // Spline interpolation as function of Y-coordinate
    for (Eigen::Index ix = 0; ix < xr.x().size(); ++ix) {
      column_(ix) = function(interpolator_, xr.y(), xr.q().row(ix), y);
    }
    return function(interpolator_, xr.x(), column_, x);
  }
};

}  // namespace pyinterp::detail::math
