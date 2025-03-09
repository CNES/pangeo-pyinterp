// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Core>
#include <string>

#include "pyinterp/detail/interpolation/factory_1d.hpp"
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
        x_interpolator_(interpolation::factory_1d<double>(kind)),
        y_interpolator_(interpolation::factory_1d<double>(kind)) {}

  /// Return the interpolated value of y for a given point x
  auto interpolate(const double x, const double y, const Frame2D &xr)
      -> double {
    // Spline interpolation as function of X-coordinate
    for (Eigen::Index ix = 0; ix < xr.y()->size(); ++ix) {
      column_(ix) = (*x_interpolator_)(*(xr.x()), xr.q()->col(ix), x);
    }
    return (*y_interpolator_)(*(xr.y()), column_, y);
  }

 private:
  /// Column of the interpolation window (interpolation according to Y
  /// coordinates)
  Eigen::VectorXd column_;

  /// Interpolators
  std::unique_ptr<interpolation::Interpolator1D<double>> x_interpolator_;
  std::unique_ptr<interpolation::Interpolator1D<double>> y_interpolator_;
};

}  // namespace pyinterp::detail::math
