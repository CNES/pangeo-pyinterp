#pragma once
#include "pyinterp/detail/math/bivariate.hpp"
#include "pyinterp/detail/math/linear.hpp"

namespace pyinterp {
namespace detail {
namespace math {

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
inline T trivariate(const Point<T>& p, const Point<T>& p0, const Point<T>& p1,
                    const T& q000, const T& q010, const T& q100, const T& q110,
                    const T& q001, const T& q011, const T& q101, const T& q111,
                    const Bivariate<Point, T>* bivariate) {
  auto z0 = bivariate->evaluate(p, p0, p1, q000, q010, q100, q110);
  auto z1 = bivariate->evaluate(p, p0, p1, q001, q011, q101, q111);
  return linear(boost::geometry::get<2>(p), boost::geometry::get<2>(p0),
                boost::geometry::get<2>(p1), z0, z1);
}

}  // namespace math
}  // namespace detail
}  // namespace pyinterp