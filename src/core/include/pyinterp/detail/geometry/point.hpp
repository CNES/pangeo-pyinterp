#pragma once
#include <boost/geometry.hpp>

namespace pyinterp {
namespace detail {
namespace geometry {

/// Points of a 3-dimensional Cartesian space.
///
/// @tparam T Storage class of the coordinates of the point.
template <typename T>
using Point3D =
    boost::geometry::model::point<T, 3, boost::geometry::cs::cartesian>;

/// Points of a N-dimensional Cartesian space.
///
/// @tparam T Storage class of the coordinates of the point.
/// @tparam N Number of dimensions of the Cartesian space.
template <typename T, size_t N>
using PointND =
    boost::geometry::model::point<T, N, boost::geometry::cs::cartesian>;

/// Points of a 3-dimensional spherical equatorial space.
///
/// @tparam T Storage class of the coordinates of the point.
template <typename T>
using EquatorialPoint3D = boost::geometry::model::point<
    T, 3, boost::geometry::cs::spherical_equatorial<boost::geometry::degree>>;

}  // namespace geometry
}  // namespace detail
}  // namespace pyinterp