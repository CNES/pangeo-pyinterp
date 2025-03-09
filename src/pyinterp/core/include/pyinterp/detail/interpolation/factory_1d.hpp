#pragma once
#include <memory>
#include <stdexcept>

#include "pyinterp/detail/interpolation/akima.hpp"
#include "pyinterp/detail/interpolation/akima_periodic.hpp"
#include "pyinterp/detail/interpolation/cspline.hpp"
#include "pyinterp/detail/interpolation/cspline_not_a_knot.hpp"
#include "pyinterp/detail/interpolation/cspline_periodic.hpp"
#include "pyinterp/detail/interpolation/linear.hpp"
#include "pyinterp/detail/interpolation/polynomial.hpp"
#include "pyinterp/detail/interpolation/steffen.hpp"

namespace pyinterp::detail::interpolation {

template <typename T>
static inline auto factory_1d(const std::string &kind)
    -> std::unique_ptr<Interpolator1D<T>> {
  if (kind == "linear") {
    return std::make_unique<Linear<T>>();
  }
  if (kind == "polynomial") {
    return std::make_unique<Polynomial<T>>();
  }
  if (kind == "c_spline_not_a_knot") {
    return std::make_unique<CSplineNotAKnot<T>>();
  }
  if (kind == "c_spline") {
    return std::make_unique<CSpline<T>>();
  }
  if (kind == "c_spline_periodic") {
    return std::make_unique<CSplinePeriodic<T>>();
  }
  if (kind == "akima") {
    return std::make_unique<Akima<T>>();
  }
  if (kind == "akima_periodic") {
    return std::make_unique<AkimaPeriodic<T>>();
  }
  if (kind == "steffen") {
    return std::make_unique<Steffen<T>>();
  }
  throw std::invalid_argument("Invalid interpolation type: " + kind);
}

}  // namespace pyinterp::detail::interpolation
