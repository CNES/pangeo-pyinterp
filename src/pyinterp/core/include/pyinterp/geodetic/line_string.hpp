// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <boost/geometry.hpp>
#include <utility>
#include <vector>

#include "pyinterp/detail/geometry/point.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/geodetic/point.hpp"

namespace pyinterp::geodetic {

/// A linestring (named so by OGC) is a collection of points.
class LineString {
 public:
  /// Build a new line string with the coordinates provided.
  ///
  /// @param lon Longitudes of the points in degrees.
  /// @param lat Latitudes of the points in degrees.
  LineString(const Eigen::Ref<const Vector<double>>& lon,
             const Eigen::Ref<const Vector<double>>& lat);

  /// Test if this linestring intersects with another linestring.
  ///
  /// @param rhs the linestring to test.
  /// @return true if the linestrings intersect.
  [[nodiscard]] auto intersects(const LineString& rhs) const -> bool {
    return boost::geometry::intersects(line_string_, rhs.line_string_);
  }

  /// Get the coordinate of the intersection between this linestring and
  /// another one.
  ///
  /// @param rhs the linestring to test.
  /// @return the intersection point or none if no intersection is found.
  [[nodiscard]] auto intersection(const LineString& rhs) const
      -> std::optional<Point>;

  /// Find the nearest index of a point in this linestring to a given
  /// point.
  ///
  /// @param point the point to search.
  /// @return the index of the nearest point or none if no intersection is
  ///         found.
  [[nodiscard]] inline auto nearest(const Point& point) const -> size_t {
    std::vector<std::pair<Point, size_t>> result;
    rtree_.query(
        boost::geometry::index::nearest(Point(point.lon(), point.lat()), 1),
        std::back_inserter(result));
    return result[0].second;
  }

  /// Get the length of the linestring
  ///
  /// @return the length of the line string
  [[nodiscard]] inline auto size() const -> size_t {
    return line_string_.size();
  }

  /// Get a point of the linestring at a given index
  ///
  /// @param index the index of the point
  /// @return the point at the given index
  [[nodiscard]] inline auto at(const size_t index) const -> Point {
    return line_string_.at(index);
  }

  /// Get a point of the linestring at a given index
  ///
  /// @param index the index of the point
  /// @return the point at the given index
  [[nodiscard]] inline auto operator[](const size_t index) const -> Point {
    return line_string_[index];
  }

  /// Returns a read-only (constant) iterator that points to the
  /// first point of the linestring.
  [[nodiscard]] auto begin() const
      -> boost::geometry::model::linestring<Point>::const_iterator {
    return line_string_.begin();
  }

  /// Returns a read-only (constant) iterator that points to the
  /// last point of the linestring.
  [[nodiscard]] auto end() const
      -> boost::geometry::model::linestring<Point>::const_iterator {
    return line_string_.end();
  }

  /// Get a tuple that fully encodes the state of this instance
  [[nodiscard]] auto getstate() const -> pybind11::tuple;

  /// Create a new instance from a registered state of an instance of this
  /// object.
  static auto setstate(const pybind11::tuple& state) -> LineString;

 private:
  boost::geometry::model::linestring<Point> line_string_;
  boost::geometry::index::rtree<std::pair<Point, size_t>,
                                boost::geometry::index::quadratic<16>>
      rtree_;
};

}  // namespace pyinterp::geodetic
