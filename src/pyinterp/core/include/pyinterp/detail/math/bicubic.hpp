// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Core>
#include "pyinterp/detail/gsl/interpolate1d.hpp"
#include "pyinterp/detail/math.hpp"

namespace pyinterp::detail::math {

/// Set of coordinates used for interpolation
class CoordsXY {
 public:
  /// Default constructor
  CoordsXY() = delete;

  /// Creates a new instance
  CoordsXY(const size_t x_size, const size_t y_size)
      : x_(new Eigen::VectorXd), y_(new Eigen::VectorXd) {
    auto nx = x_size << 1U;
    auto ny = y_size << 1U;
    x_->resize(nx);
    y_->resize(ny);
  }

  /// Creates a new instance from existing coordinates
  CoordsXY(std::shared_ptr<Eigen::VectorXd> x,
           std::shared_ptr<Eigen::VectorXd> y)
      : x_(std::move(x)), y_(std::move(y)) {}

  /// Default destructor
  virtual ~CoordsXY() = default;

  /// Copy constructor
  ///
  /// @param rhs right value
  CoordsXY(const CoordsXY &rhs) = default;

  /// Move constructor
  ///
  /// @param rhs right value
  CoordsXY(CoordsXY &&rhs) noexcept = default;

  /// Copy assignment operator
  ///
  /// @param rhs right value
  auto operator=(const CoordsXY &rhs) -> CoordsXY & = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  auto operator=(CoordsXY &&rhs) noexcept -> CoordsXY & = default;

  /// Get the half size of the window in abscissa.
  [[nodiscard]] inline auto nx() const noexcept -> size_t {
    return static_cast<size_t>(x_->size()) >> 1U;
  }

  /// Get the half size of the window in ordinate.
  [[nodiscard]] inline auto ny() const noexcept -> size_t {
    return static_cast<size_t>(y_->size()) >> 1U;
  }

  /// Get x-coordinates
  inline auto x() noexcept -> std::shared_ptr<Eigen::VectorXd> & { return x_; }

  /// Get x-coordinates
  [[nodiscard]] inline auto x() const noexcept
      -> const std::shared_ptr<Eigen::VectorXd> & {
    return x_;
  }

  /// Get y-coordinates
  inline auto y() noexcept -> std::shared_ptr<Eigen::VectorXd> & { return y_; }

  /// Get y-coordinates
  [[nodiscard]] inline auto y() const noexcept
      -> const std::shared_ptr<Eigen::VectorXd> & {
    return y_;
  }

  /// Get the ith x-axis.
  [[nodiscard]] inline auto x(const size_t ix) const -> double {
    return (*x_)(ix);
  }

  /// Get the ith y-axis.
  [[nodiscard]] inline auto y(const size_t jx) const -> double {
    return (*y_)(jx);
  }

  /// Set the ith x-axis.
  inline auto x(const size_t ix) -> double & { return (*x_)(ix); }

  /// Get the ith y-axis.
  inline auto y(const size_t jx) -> double & { return (*y_)(jx); }

  /// Normalizes the angle with respect to the first value of the X axis of this
  /// array.
  [[nodiscard]] inline auto normalize_angle(const double xi) const -> double {
    return math::normalize_angle(xi, (*x_)(0), 360.0);
  }

 private:
  std::shared_ptr<Eigen::VectorXd> x_{};
  std::shared_ptr<Eigen::VectorXd> y_{};
};

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
class XArray : public CoordsXY {
 public:
  /// Default constructor
  XArray() = delete;

  /// Creates a new Array
  XArray(const size_t x_size, const size_t y_size)
      : CoordsXY(x_size, y_size), q_(new Eigen::MatrixXd) {
    q_->resize(x()->size(), y()->size());
  }

  /// Creates a new Array from existing coordinates/values
  XArray(std::shared_ptr<Eigen::VectorXd> x, std::shared_ptr<Eigen::VectorXd> y,
         std::shared_ptr<Eigen::MatrixXd> q)
      : CoordsXY(std::move(x), std::move(y)), q_(std::move(q)) {}

  /// Default destructor
  ~XArray() override = default;

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

  /// Get the values from the array for all x and y coordinates.
  inline auto q() noexcept -> std::shared_ptr<Eigen::MatrixXd> & { return q_; }

  /// Get the values from the array for all x and y coordinates.
  [[nodiscard]] inline auto q() const noexcept
      -> const std::shared_ptr<Eigen::MatrixXd> & {
    return q_;
  }

  /// Get the value at coordinate (ix, jx).
  [[nodiscard]] inline auto q(const size_t ix, const size_t jx) const
      -> double {
    return (*q_)(ix, jx);
  }

  /// Get the value at coordinate (ix, jx).
  inline auto q(const size_t ix, const size_t jx) -> double & {
    return (*q_)(ix, jx);
  }

  /// Returns true if this instance does not contains at least one Not A Number
  /// (NaN).
  [[nodiscard]] inline auto is_valid() const -> bool { return !q_->hasNaN(); }

 private:
  std::shared_ptr<Eigen::MatrixXd> q_{};
};


/// Set of coordinates/values used for 3D-interpolation
class XArrayStack : public CoordsXY {
 public:
  /// Default constructor
  XArrayStack() = delete;

  /// Creates a new instance
  XArrayStack(const size_t x_size, const size_t y_size, const size_t z_size)
      : CoordsXY(x_size, y_size), z_(), q_() {
    auto nz = z_size << 1U;
    z_.resize(nz);
    q_.resize(nz);

    for (auto iz = 0U; iz < nz; ++iz) {
      q_(iz) = std::shared_ptr<Eigen::MatrixXd>(
          new Eigen::MatrixXd(x()->size(), y()->size()));
    }
  }

  /// Get the set of coordinates/values for the ith z-layer
  [[nodiscard]] auto xarray(const Eigen::Index iz) const -> XArray {
    return XArray(x(), y(), q_(iz));
  }

  /// Default destructor
  ~XArrayStack() override = default;

  /// Copy constructor
  ///
  /// @param rhs right value
  XArrayStack(const XArrayStack &rhs) = default;

  /// Move constructor
  ///
  /// @param rhs right value
  XArrayStack(XArrayStack &&rhs) noexcept = default;

  /// Copy assignment operator
  ///
  /// @param rhs right value
  auto operator=(const XArrayStack &rhs) -> XArrayStack & = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  auto operator=(XArrayStack &&rhs) noexcept -> XArrayStack & = default;

  /// Get the half size of the window in z.
  [[nodiscard]] inline auto nz() const noexcept -> size_t {
    return static_cast<size_t>(z_.size()) >> 1U;
  }

  /// Get z-coordinates
  inline auto z() noexcept -> Eigen::VectorXd & { return z_; }

  /// Get z-coordinates
  [[nodiscard]] inline auto z() const noexcept -> const Eigen::VectorXd & {
    return z_;
  }

  /// Get the ith z-axis.
  [[nodiscard]] inline auto z(const size_t ix) const -> double {
    return z_(ix);
  }

  /// Set the ith z-axis.
  inline auto z(const size_t ix) -> double & { return z_(ix); }

  /// Get the value at coordinate (ix, jx, kx).
  inline auto q(const size_t ix, const size_t jx, const size_t kx) -> double & {
    return (*q_(kx))(ix, jx);
  }

  /// Returns true if this instance does not contains at least one Not A Number
  /// (NaN).
  [[nodiscard]] inline auto is_valid() const -> bool {
    for (Eigen::Index kx = 0; kx < q_.size(); ++kx) {
      if ((*q_(kx)).hasNaN()) {
        return false;
      }
    }
    return true;
  }

 private:
  Eigen::VectorXd z_;
  Eigen::Matrix<std::shared_ptr<Eigen::MatrixXd>, -1, 1> q_;
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
      : column_(xr.x()->size()),
        interpolator_(std::max(xr.x()->size(), xr.y()->size()), type,
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
    for (Eigen::Index ix = 0; ix < xr.x()->size(); ++ix) {
      column_(ix) = function(interpolator_, *(xr.y()), xr.q()->row(ix), y);
    }
    return function(interpolator_, *(xr.x()), column_, x);
  }
};

}  // namespace pyinterp::detail::math
