// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <boost/geometry.hpp>
#include <limits>
#include <set>

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

/// Points of a 2-dimensional spherical equatorial space.
///
/// @tparam T Storage class of the coordinates of the point.
template <typename T>
using EquatorialPoint2D = boost::geometry::model::point<
    T, 2, boost::geometry::cs::spherical_equatorial<boost::geometry::degree>>;

/// Points of a 2-dimensional spherical equatorial space and time
///
/// @tparam T Storage class of the coordinates of the point.
template <typename T>
class TemporalEquatorial2D : public EquatorialPoint2D<T> {
 public:
  /// Default constructor
  ///
  /// @param x X-coordinate of the point
  /// @param y Y-coordinate of the point
  /// @param timestamp Date associated with the point.
  TemporalEquatorial2D(const T &x, const T &y, const int64_t timestamp)
      : EquatorialPoint2D<T>(x, y), timestamp_(timestamp) {}

  /// Construction of a point associated with an indefinite time reference
  /// (NaT)
  ///
  /// @param x X-coordinate of the point
  /// @param y Y-coordinate of the point
  TemporalEquatorial2D(const T &x, const T &y)
      : EquatorialPoint2D<T>(x, y),
        timestamp_(std::numeric_limits<int64_t>::min()) {}

  /// Gets the time stamp associated with the point
  [[nodiscard]] constexpr auto timestamp() const -> int64_t {
    return timestamp_;
  }

 private:
  int64_t timestamp_;
};

/// Points of a 3-dimensional spherical equatorial space.
///
/// @tparam T Storage class of the coordinates of the point.
template <typename T>
using EquatorialPoint3D = boost::geometry::model::point<
    T, 3, boost::geometry::cs::spherical_equatorial<boost::geometry::degree>>;

/// Points of a 4-dimensional spherical equatorial space.
///
/// @tparam T Storage class of the coordinates of the point.
template <typename T>
using EquatorialPoint4D = boost::geometry::model::point<
    T, 4, boost::geometry::cs::spherical_equatorial<boost::geometry::degree>>;

/// Points of a 2-dimensional spheroid space.
///
/// @tparam T Storage class of the coordinates of the point.
template <typename T>
using GeographicPoint2D = boost::geometry::model::point<
    T, 2, boost::geometry::cs::geographic<boost::geometry::degree>>;

}  // namespace pyinterp::detail::geometry

namespace boost::geometry::traits {

/// Coordinate tag from TemporalEquatorial2D.
template <typename T>
struct tag<pyinterp::detail::geometry::TemporalEquatorial2D<T>> {
  /// Typedef for type
  using type = point_tag;
};

/// Coordinate type from TemporalEquatorial2D
template <typename T>
struct coordinate_type<pyinterp::detail::geometry::TemporalEquatorial2D<T>> {
  /// Typedef for type
  using type = T;
};

/// Coordinate system from TemporalEquatorial2D
template <typename T>
struct coordinate_system<pyinterp::detail::geometry::TemporalEquatorial2D<T>> {
  /// Typedef for type
  using type = cs::spherical_equatorial<degree>;
};

template <typename T>
struct dimension<pyinterp::detail::geometry::TemporalEquatorial2D<T>>
    : boost::mpl::int_<2> {};

/// access struct defining with TemporalEquatorial2D
template <typename T, size_t I>
struct access<pyinterp::detail::geometry::TemporalEquatorial2D<T>, I> {
  /// Pointer accessor
  static auto get(pyinterp::detail::geometry::TemporalEquatorial2D<T> const &p)
      -> T {
    return p.template get<I>();
  }

  /// Pointer setter
  static void set(pyinterp::detail::geometry::TemporalEquatorial2D<T> &p,
                  T const &v) {  // NOLINT
    p.template set<I>(v);
  }
};

}  // namespace boost::geometry::traits
