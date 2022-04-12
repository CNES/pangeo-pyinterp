// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <functional>
#include <stdexcept>
#include <string>

#include "pyinterp/detail/geometry/point.hpp"
#include "pyinterp/detail/math/bivariate.hpp"
#include "pyinterp/detail/math/linear.hpp"
#include "pyinterp/detail/math/nearest.hpp"

namespace pyinterp::detail::math {

// Interpolation method used for the Z axis
enum ZMethod { kLinear = 0x0, kNearest = 0x1 };

/// Function performing the interpolation between two points
template <typename T, typename U>
using z_method_t =
    std::function<U(const T &, const T &, const T &, const U &, const U &)>;

/// Get the function used to perform the interpolation on the Z-Axis
template <template <class> class Point = geometry::TemporalEquatorial2D,
          typename T>
constexpr auto get_z_method(
    const Bivariate<geometry::TemporalEquatorial2D, T> * /*unused*/,
    const ZMethod method) -> z_method_t<int64_t, T> {
  switch (method) {
    case kLinear:
      return &linear<int64_t, T>;
    case kNearest:
    default:
      return &nearest<int64_t, T>;
  }
}

/// Get the function used to perform the interpolation on the Z-Axis
template <template <class> class Point, typename T>
constexpr auto get_z_method(const Bivariate<Point, T> * /*unused*/,
                            const ZMethod method) -> z_method_t<T, T> {
  switch (method) {
    case kLinear:
      return &linear<T, T>;
    case kNearest:
    default:
      return &nearest<T, T>;
  }
}

/// Get the function used to perform the interpolation on the Z-Axis
template <template <class> class Point, typename T>
constexpr auto get_z_interpolation_method(
    const Bivariate<Point, T> *interpolator, const std::string &method) {
  if (method == "linear") {
    return pyinterp::detail::math::get_z_method<Point, T>(
        interpolator, pyinterp::detail::math::kLinear);
  }
  if (method == "nearest") {
    return pyinterp::detail::math::get_z_method<Point, T>(
        interpolator, pyinterp::detail::math::kNearest);
  }
  throw std::invalid_argument("unknown interpolation method: " + method);
}

/// Performs the interpolation
///
/// @param p Query point
/// @param p0 Point of coordinate (x0, y0, z0)
/// @param p1 Point of coordinate (x1, y1, z1)
/// @param q000 Point value for the coordinate (x0, y0, z0)
/// @param q010 Point value for the coordinate (x0, y1, z0)
/// @param q100 Point value for the coordinate (x1, y0, z0)
/// @param q110 Point value for the coordinate (x1, y1, z0)
/// @param q001 Point value for the coordinate (x0, y0, z1)
/// @param q011 Point value for the coordinate (x0, y1, z1)
/// @param q101 Point value for the coordinate (x1, y0, z1)
/// @param q111 Point value for the coordinate (x1, y1, z1)
/// @return interpolated value at coordinate (x, y, z)
template <template <class> class Point = geometry::TemporalEquatorial2D,
          typename T>
constexpr auto trivariate(
    const geometry::TemporalEquatorial2D<T> &p,
    const geometry::TemporalEquatorial2D<T> &p0,
    const geometry::TemporalEquatorial2D<T> &p1, const T &q000, const T &q010,
    const T &q100, const T &q110, const T &q001, const T &q011, const T &q101,
    const T &q111,
    const Bivariate<geometry::TemporalEquatorial2D, T> *bivariate,
    const z_method_t<int64_t, T> &interpolator = &linear<int64_t, T>) -> T {
  auto z0 = bivariate->evaluate(p, p0, p1, q000, q010, q100, q110);
  auto z1 = bivariate->evaluate(p, p0, p1, q001, q011, q101, q111);
  return interpolator(p.timestamp(), p0.timestamp(), p1.timestamp(), z0, z1);
}

/// Performs the interpolation
///
/// @param p Query point
/// @param p0 Point of coordinate (x0, y0, z0)
/// @param p1 Point of coordinate (x1, y1, z1)
/// @param q000 Point value for the coordinate (x0, y0, z0)
/// @param q010 Point value for the coordinate (x0, y1, z0)
/// @param q100 Point value for the coordinate (x1, y0, z0)
/// @param q110 Point value for the coordinate (x1, y1, z0)
/// @param q001 Point value for the coordinate (x0, y0, z1)
/// @param q011 Point value for the coordinate (x0, y1, z1)
/// @param q101 Point value for the coordinate (x1, y0, z1)
/// @param q111 Point value for the coordinate (x1, y1, z1)
/// @return interpolated value at coordinate (x, y, z)
template <template <class> class Point, typename T>
constexpr auto trivariate(const Point<T> &p, const Point<T> &p0,
                          const Point<T> &p1, const T &q000, const T &q010,
                          const T &q100, const T &q110, const T &q001,
                          const T &q011, const T &q101, const T &q111,
                          const Bivariate<Point, T> *bivariate,
                          const z_method_t<T, T> &interpolator = &linear<T, T>)
    -> T {
  auto z0 = bivariate->evaluate(p, p0, p1, q000, q010, q100, q110);
  auto z1 = bivariate->evaluate(p, p0, p1, q001, q011, q101, q111);
  return interpolator(boost::geometry::get<2>(p), boost::geometry::get<2>(p0),
                      boost::geometry::get<2>(p1), z0, z1);
}

}  // namespace pyinterp::detail::math
