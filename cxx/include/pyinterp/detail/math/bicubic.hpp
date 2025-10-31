// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Core>
#include <string>

#include "pyinterp/detail/interpolation/factory_2d.hpp"
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
      : interpolator_(interpolation::factory_2d<double>(kind)) {}

  /// Return the interpolated value of y for a given point x
  auto interpolate(const double x, const double y, const Frame2D &xr)
      -> double {
    return (*interpolator_)(*(xr.x()), *(xr.y()), *(xr.q()), x, y);
  }

 private:
  /// Interpolator
  std::unique_ptr<interpolation::Interpolator2D<double>> interpolator_;
};

}  // namespace pyinterp::detail::math
