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

}  // namespace geometry
}  // namespace detail
}  // namespace pyinterp