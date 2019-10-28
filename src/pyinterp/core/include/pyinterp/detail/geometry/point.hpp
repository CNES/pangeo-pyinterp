// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <boost/geometry.hpp>

namespace pyinterp::detail::geometry {

/// Coordinate type
template <typename Point>
using coordinate_t = boost::geometry::coordinate_type<Point>;

namespace point {

/// Class allowing dynamic access to the axis of a point.
/// A recursive call is made as long as the requested axis is not reached.
///
/// @tparam Point Point handled
/// @tparam Axis The current axis handled by the class.
/// @tparam Count The number of dimension handled by the point
template <typename Point, size_t Axis, size_t Count>
struct GetDimension {
  /// Get accessor
  ///
  /// @param point Point handled.
  /// @param axis The axis of the point to be read.
  static auto get(const Point &point, size_t axis) ->
      typename coordinate_t<Point>::type {
    if (Axis == axis) {
      return boost::geometry::get<Axis>(point);
    }
    return GetDimension<Point, Axis + 1, Count>::get(point, axis);
  }

  /// Set accessor
  ///
  /// @param point Point handled.
  /// @param value Value to store in the point
  /// @param axis The axis of the point to be read.
  static void set(Point &point, const typename coordinate_t<Point>::type &value,
                  size_t axis) {
    if (Axis == axis) {
      return boost::geometry::set<Axis>(point, value);
    }
    return GetDimension<Point, Axis + 1, Count>::set(point, value, axis);
  }
};

/// Class allowing dynamic access to the axis of a point.
/// This class is used when recursive calls are completed.
///
/// @tparam Point Point handled
/// @tparam Count The number of dimension handled by the point
template <typename Point, size_t Count>
struct GetDimension<Point, Count, Count> {
  /// Get accessor
  static auto get(const Point & /*point*/, size_t /*axis*/) ->
      typename coordinate_t<Point>::type {
    return std::numeric_limits<typename coordinate_t<Point>::type>::max();
  }

  /// Set accessor
  static void set(Point & /*point*/,
                  const typename coordinate_t<Point>::type & /*value*/,
                  size_t /*axis*/) {}
};

/// Get a point value for a given axis
template <typename Point>
auto get(const Point &point, size_t axis) ->
    typename coordinate_t<Point>::type {
  return GetDimension<
      Point, 0, boost::geometry::dimension<Point>::type::value>::get(point,
                                                                     axis);
}

/// Set a point value for a given axis
template <typename Point>
void set(Point &point, const typename coordinate_t<Point>::type &value,
         size_t axis) {
  return GetDimension<
      Point, 0, boost::geometry::dimension<Point>::type::value>::set(point,
                                                                     value,
                                                                     axis);
}

}  // namespace point

/// Points of a 2-dimensional Cartesian space.
///
/// @tparam T Storage class of the coordinates of the point.
template <typename T>
using Point2D =
    boost::geometry::model::point<T, 2, boost::geometry::cs::cartesian>;

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

/// Points of a 2-dimensional spherical equatorial space.
///
/// @tparam T Storage class of the coordinates of the point.
template <typename T>
using EquatorialPoint2D = boost::geometry::model::point<
    T, 2, boost::geometry::cs::spherical_equatorial<boost::geometry::degree>>;

/// Points of a 3-dimensional spherical equatorial space.
///
/// @tparam T Storage class of the coordinates of the point.
template <typename T>
using EquatorialPoint3D = boost::geometry::model::point<
    T, 3, boost::geometry::cs::spherical_equatorial<boost::geometry::degree>>;

/// Points of a 2-dimensional spheroid space.
///
/// @tparam T Storage class of the coordinates of the point.
template <typename T>
using SpheriodPoint2D = boost::geometry::model::point<
    T, 2, boost::geometry::cs::geographic<boost::geometry::degree>>;

}  // namespace pyinterp::detail::geometry
