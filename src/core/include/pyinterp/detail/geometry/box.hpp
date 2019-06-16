#pragma once
#include "pyinterp/detail/geometry/point.hpp"
#include <boost/geometry.hpp>

namespace pyinterp {
namespace detail {
namespace geometry {

/// Defines a box made of two describing points.
///
/// @tparam T Type of data handled by this box
/// @tparam N Number of dimensions in coordinate systems handled.
template <typename T, size_t N>
using BoxND = boost::geometry::model::box<PointND<T, N>>;

/// Defines a box made of two describing points in spherical equatorial space.
///
/// @tparam T Type of data handled by this box
/// @tparam N Number of dimensions in coordinate systems handled.
template <typename T>
using EquatorialBox3D = boost::geometry::model::box<EquatorialPoint3D<T>>;

}  // namespace geometry
}  // namespace detail
}  // namespace pyinterp