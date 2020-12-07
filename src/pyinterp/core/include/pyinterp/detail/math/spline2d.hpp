// Copyright (c) 2020 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <gsl/gsl_interp.h>

#include <Eigen/Core>

#include "pyinterp/detail/gsl/interpolate1d.hpp"
#include "pyinterp/detail/math/xarray.hpp"

namespace pyinterp::detail::math {

/// Spline gridded 2D interpolation
class Spline2D {
 public:
  /// Default constructor
  ///
  /// @param xr Calculation window.
  /// @param type method of calculation
  explicit Spline2D(const XArray2D &xr, const gsl_interp_type *type)
      : column_(xr.x()->size()),
        interpolator_(std::max(xr.x()->size(), xr.y()->size()), type,
                      gsl::Accelerator()) {}

  /// Return the interpolated value of y for a given point x
  auto interpolate(const double x, const double y, const XArray2D &xr)
      -> double {
    return evaluate(&gsl::Interpolate1D::interpolate, x, y, xr);
  }

  /// Return the derivative for a given point x
  auto derivative(const double x, const double y, const XArray2D &xr)
      -> double {
    return evaluate(&gsl::Interpolate1D::derivative, x, y, xr);
  }

  /// Return the second derivative for a given point x
  auto second_derivative(const double x, const double y, const XArray2D &xr)
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
      const double x, const double y, const XArray2D &xr) -> double {
    // Spline interpolation as function of Y-coordinate
    for (Eigen::Index ix = 0; ix < xr.x()->size(); ++ix) {
      column_(ix) = function(interpolator_, *(xr.y()), xr.q()->row(ix), y);
    }
    return function(interpolator_, *(xr.x()), column_, x);
  }
};

}  // namespace pyinterp::detail::math
