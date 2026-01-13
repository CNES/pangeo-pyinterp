// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <concepts>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "pyinterp/math/interpolate/bivariate.hpp"
#include "pyinterp/math/interpolate/bivariate/bicubic.hpp"
#include "pyinterp/math/interpolate/bivariate/bilinear.hpp"
#include "pyinterp/math/interpolate/bivariate/spline.hpp"
#include "pyinterp/math/interpolate/univariate.hpp"
#include "pyinterp/math/interpolate/univariate/akima.hpp"
#include "pyinterp/math/interpolate/univariate/akima_periodic.hpp"
#include "pyinterp/math/interpolate/univariate/cspline.hpp"
#include "pyinterp/math/interpolate/univariate/cspline_not_a_knot.hpp"
#include "pyinterp/math/interpolate/univariate/cspline_periodic.hpp"
#include "pyinterp/math/interpolate/univariate/linear.hpp"
#include "pyinterp/math/interpolate/univariate/polynomial.hpp"
#include "pyinterp/math/interpolate/univariate/steffen.hpp"

namespace pyinterp::math::interpolate {
namespace univariate {

/// @brief Known univariate interpolation methods
enum class Method : int8_t {
  kAkima,            ///< Akima spline
  kAkimaPeriodic,    ///< Akima spline with periodic boundary conditions
  kCSpline,          ///< Cubic spline with natural boundary conditions
  kCSplineNotAKnot,  ///< Cubic spline with not-a-knot boundary conditions
  kCSplinePeriodic,  ///< Cubic spline with periodic boundary conditions
  kLinear,           ///< Linear
  kSteffen,          ///< Steffen
  kPolynomial        ///< Polynomial
};

/// @brief Factory method to create univariate interpolation objects
/// @tparam T type of data (must be floating point)
/// @param[in] method Interpolation method
/// @return Unique pointer to the interpolation object
template <std::floating_point T>
auto factory(const Method method) -> std::unique_ptr<Univariate<T>> {
  switch (method) {
    case univariate::Method::kAkima:
      return std::make_unique<Akima<T>>();
    case univariate::Method::kAkimaPeriodic:
      return std::make_unique<AkimaPeriodic<T>>();
    case univariate::Method::kCSpline:
      return std::make_unique<CSpline<T>>();
    case univariate::Method::kCSplineNotAKnot:
      return std::make_unique<CSplineNotAKnot<T>>();
    case univariate::Method::kCSplinePeriodic:
      return std::make_unique<CSplinePeriodic<T>>();
    case univariate::Method::kLinear:
      return std::make_unique<Linear<T>>();
    case univariate::Method::kSteffen:
      return std::make_unique<Steffen<T>>();
    case univariate::Method::kPolynomial:
      return std::make_unique<Polynomial<T>>();
    default:
      throw std::invalid_argument("Unknown interpolation method");
  }
}

}  // namespace univariate

namespace bivariate {

/// @brief Known bivariate interpolation methods
enum class Method : int8_t {
  kBilinear,  ///< Bilinear
  kBicubic    ///< Bicubic
};

}  // namespace bivariate

/// @brief Factory method to create univariate interpolation objects
/// @tparam T type of data (must be floating point)
/// @param[in] method Interpolation method
/// @return Unique pointer to the interpolation object
template <std::floating_point T>
auto factory(const univariate::Method method)
    -> std::unique_ptr<BivariateBase<T>> {
  return std::make_unique<bivariate::Spline<T>>(univariate::factory<T>(method));
}

/// @brief Factory method to create bivariate interpolation objects
/// @tparam T type of data (must be floating point)
/// @param[in] method Interpolation method
/// @return Unique pointer to the interpolation object
template <std::floating_point T>
auto factory(const bivariate::Method method)
    -> std::unique_ptr<BivariateBase<T>> {
  switch (method) {
    case bivariate::Method::kBilinear:
      return std::make_unique<bivariate::Bilinear<T>>();
    case bivariate::Method::kBicubic:
      return std::make_unique<bivariate::Bicubic<T>>();
    default:
      throw std::invalid_argument("Unknown interpolation method");
  }
}

}  // namespace pyinterp::math::interpolate
