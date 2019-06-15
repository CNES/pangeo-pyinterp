#pragma once
#include "pyinterp/detail/gsl/interpolate1d.hpp"
#include "pyinterp/detail/math.hpp"
#include <Eigen/Core>

namespace pyinterp {
namespace detail {
namespace math {

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
  XArray &operator=(const XArray &rhs) = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  XArray &operator=(XArray &&rhs) noexcept = default;

  /// Get the half size of the window in abscissa.
  inline size_t nx() const noexcept { return x_.size(); }

  /// Get the half size of the window in ordinate.
  inline size_t ny() const noexcept { return y_.size(); }

  /// Get x-coordinates
  inline Eigen::VectorXd &x() noexcept { return x_; }

  /// Get x-coordinates
  inline const Eigen::VectorXd &x() const noexcept { return x_; }

  /// Get y-coordinates
  inline Eigen::VectorXd &y() noexcept { return y_; }

  /// Get y-coordinates
  inline const Eigen::VectorXd &y() const noexcept { return y_; }

  /// Get the values from the array for all x and y coordinates.
  inline Eigen::MatrixXd &q() noexcept { return q_; }

  /// Get the values from the array for all x and y coordinates.
  inline const Eigen::MatrixXd &q() const noexcept { return q_; }

  /// Get the ith x-axis.
  inline double x(const size_t ix) const { return x_(ix); }

  /// Get the ith y-axis.
  inline double y(const size_t jx) const { return y_(jx); }

  /// Get the value at coordinate (ix, jx).
  inline double z(const size_t ix, const size_t jx) const { return q_(ix, jx); }

  /// Set the ith x-axis.
  inline double &x(const size_t ix) { return x_(ix); }

  /// Get the ith y-axis.
  inline double &y(const size_t jx) { return y_(jx); }

  /// Get the value at coordinate (ix, jx).
  inline double &z(const size_t ix, const size_t jx) { return q_(ix, jx); }

  /// Normalizes the angle with respect to the first value of the X axis of this
  /// array.
  inline constexpr double normalize_angle(const double xi) const {
    return math::normalize_angle(xi, x(0));
  }

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
  /// @param type method of calculation
  explicit Bicubic(const gsl_interp_type *type = gsl_interp_cspline)
      : type_(type) {}

  /// Return the interpolated value of y for a given point x
  double interpolate(const double x, const double y, const XArray &xr,
                     gsl::Accelerator acc = gsl::Accelerator()) const {
    return evaluate(&gsl::Interpolate1D::interpolate, x, y, xr, std::move(acc));
  }

  /// Return the derivative for a given point x
  double derivative(const double x, const double y, const XArray &xr,
                    gsl::Accelerator acc = gsl::Accelerator()) const {
    return evaluate(&gsl::Interpolate1D::derivative, x, y, xr, std::move(acc));
  }

  /// Return the second derivative for a given point x
  double second_derivative(const double x, const double y, const XArray &xr,
                           gsl::Accelerator acc = gsl::Accelerator()) const {
    return evaluate(&gsl::Interpolate1D::second_derivative, x, y, xr,
                    std::move(acc));
  }

 private:
  using InterpolateFunction =
      double (gsl::Interpolate1D::*)(const double) const;
  const gsl_interp_type *type_;

  /// Evaluation of the GSL function performing the calculation.
  double evaluate(
      const std::function<double(const gsl::Interpolate1D &, double)> &function,
      const double x, const double y, const XArray &xr,
      gsl::Accelerator &&acc) const {
    Eigen::VectorXd fy(xr.nx());

    // Spline interpolation as function of Y-coordinate
    for (auto ix = 0; ix < xr.nx(); ++ix) {
      // The block containing the processed row must be copied into a new
      // memory block.
      Eigen::VectorXd row = xr.q().row(ix);
      auto interpolator = gsl::Interpolate1D(type_, xr.y(), row, acc);
      fy(ix) = function(interpolator, y);
    }
    auto interpolator = gsl::Interpolate1D(type_, xr.x(), fy, acc);
    return function(interpolator, x);
  }
};

}  // namespace math
}  // namespace detail
}  // namespace pyinterp