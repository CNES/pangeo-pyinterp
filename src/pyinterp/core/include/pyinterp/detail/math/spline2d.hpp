// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <gsl/gsl_interp.h>

#include <Eigen/Core>
#include <string>

#include "pyinterp/detail/gsl/interpolate1d.hpp"
#include "pyinterp/detail/math/frame.hpp"

namespace pyinterp::detail::math {

/// Spline gridded 2D interpolation
class Spline2D {
 public:
  /// Default constructor
  ///
  /// @param xr Calculation window.
  /// @param type method of calculation
  explicit Spline2D(const Frame2D &xr, const std::string &kind)
      : column_(xr.y()->size()),
        x_interpolator_(xr.x()->size(),
                        gsl::Interpolate1D::parse_interp_type(kind),
                        gsl::Accelerator()),
        y_interpolator_(xr.y()->size(),
                        gsl::Interpolate1D::parse_interp_type(kind),
                        gsl::Accelerator()) {}

  /// Return the interpolated value of y for a given point x
  auto interpolate(const double x, const double y, const Frame2D &xr)
      -> double {
    return evaluate(&gsl::Interpolate1D::interpolate, x, y, xr);
  }

  /// Return the derivative for a given point x
  auto derivative(const double x, const double y, const Frame2D &xr) -> double {
    return evaluate(&gsl::Interpolate1D::derivative, x, y, xr);
  }

  /// Return the second derivative for a given point x
  auto second_derivative(const double x, const double y, const Frame2D &xr)
      -> double {
    return evaluate(&gsl::Interpolate1D::second_derivative, x, y, xr);
  }

 private:
  using InterpolateFunction = double (gsl::Interpolate1D::*)(
      const Eigen::VectorXd &, const Eigen::VectorXd &, const double);
  /// Column of the interpolation window (interpolation according to Y
  /// coordinates)
  Eigen::VectorXd column_;

  /// GSL interpolators
  gsl::Interpolate1D x_interpolator_;
  gsl::Interpolate1D y_interpolator_;

  /// Evaluation of the GSL function performing the calculation.
  auto evaluate(
      const std::function<double(gsl::Interpolate1D &, const Eigen::VectorXd &,
                                 const Eigen::VectorXd &, const double)>
          &function,
      const double x, const double y, const Frame2D &xr) -> double {
    // Spline interpolation as function of X-coordinate
    for (Eigen::Index ix = 0; ix < xr.y()->size(); ++ix) {
      column_(ix) = function(x_interpolator_, *(xr.x()), xr.q()->col(ix), x);
    }
    return function(y_interpolator_, *(xr.y()), column_, y);
  }
};

}  // namespace pyinterp::detail::math
