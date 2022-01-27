// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <gsl/gsl_spline2d.h>

#include <Eigen/Core>
#include <string>

#include "pyinterp/detail/gsl/interpolate2d.hpp"
#include "pyinterp/detail/math/frame.hpp"

namespace pyinterp::detail::math {

/// Bicubic interpolation
class Bicubic {
 public:
  /// Default constructor
  ///
  /// @param xr Calculation window.
  /// @param kind method of calculation
  explicit Bicubic(const Frame2D &xr, const std::string &kind)
      : interpolator_(xr.x()->size(), xr.y()->size(),
                      Bicubic::parse_interp2d_type(kind), gsl::Accelerator(),
                      gsl::Accelerator()) {}

  /// Return the interpolated value of y for a given point x
  auto interpolate(const double x, const double y, const Frame2D &xr)
      -> double {
    return interpolator_.evaluate(*(xr.x()), *(xr.y()), *(xr.q()), x, y);
  }

 private:
  /// GSL interpolator
  gsl::Interpolate2D interpolator_;

  static inline auto parse_interp2d_type(const std::string &kind)
      -> const gsl_interp2d_type * {
    if (kind == "bilinear") {
      return gsl_interp2d_bilinear;
    }
    if (kind == "bicubic") {
      return gsl_interp2d_bicubic;
    }
    throw std::invalid_argument("Invalid bicubic type: " + kind);
  }
};

}  // namespace pyinterp::detail::math
