// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/detail/math/bicubic.hpp"
#include "pyinterp/detail/thread.hpp"
#include "pyinterp/grid.hpp"

namespace pyinterp {

/// Fitting model
enum FittingModel {
  kLinear,           //!< Linear interpolation
  kPolynomial,       //!< Polynomial interpolation
  kCSpline,          //!< Cubic spline with natural boundary conditions.
  kCSplinePeriodic,  //!< Cubic spline with periodic boundary conditions.
  kAkima,            //!< Non-rounded Akima spline with natural boundary
                     //!< conditions
  kAkimaPeriodic,    //!< Non-rounded Akima spline with periodic boundary
                     //!< conditions
  kSteffen           //!< Steffen’s method guarantees the monotonicity of
                     //!< the interpolating function between the given
                     //!< data points.
};

/// Extension of cubic interpolation for interpolating data points on a
/// two-dimensional regular grid. The interpolated surface is smoother than
/// corresponding surfaces obtained by bilinear interpolation or
/// nearest-neighbor interpolation.
///
/// @tparam Type The type of data used by the numerical grid.
template <typename Type>
auto bicubic(const Grid2D<Type>& grid, const pybind11::array_t<double>& x,
             const pybind11::array_t<double>& y, size_t nx, size_t ny,
             FittingModel fitting_model, Axis::Boundary boundary,
             bool bounds_error, size_t num_threads)
    -> pybind11::array_t<double>;
}  // namespace pyinterp
