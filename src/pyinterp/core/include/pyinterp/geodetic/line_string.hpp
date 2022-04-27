// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <boost/geometry.hpp>
#include <string>
#include <utility>
#include <vector>

#include "pyinterp/detail/geometry/point.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/geodetic/point.hpp"
#include "pyinterp/geodetic/spheroid.hpp"

namespace pyinterp::geodetic {

/// A linestring (named so by OGC) is a collection of points.
class LineString : public boost::geometry::model::linestring<Point> {
 public:
  using Base = boost::geometry::model::linestring<Point>;

  /// Default constructor
  LineString() = default;

  /// Build a new line string from a list of points.
  explicit LineString(const pybind11::list& points) {
    for (const auto point : points) {
      push_back(point.cast<Point>());
    }
  }

  /// Build a new line string with the coordinates provided.
  ///
  /// @param lon Longitudes of the points in degrees.
  /// @param lat Latitudes of the points in degrees.
  LineString(const Eigen::Ref<const Vector<double>>& lon,
             const Eigen::Ref<const Vector<double>>& lat);

  /// Build a new line string from a GeoJSON line string.
  [[nodiscard]] static auto from_geojson(const pybind11::list& array)
      -> LineString;

  /// Returns a GeoJSON representation of this instance.
  [[nodiscard]] auto to_geojson() const -> pybind11::dict;

  /// Add a point to this linestring.
  auto append(const Point& point) -> void { Base::push_back(point); }

  /// Test if this linestring intersects with another linestring.
  ///
  /// @param rhs the linestring to test.
  /// @return true if the linestrings intersect.
  [[nodiscard]] auto intersects(const LineString& rhs,
                                const std::optional<Spheroid>& wgs) const
      -> bool;

  /// Compute the intersection of this linestring with another linestring.
  ///
  /// @param rhs the linestring to test.
  /// @return the intersection of the two linestrings.
  [[nodiscard]] auto intersection(const LineString& rhs,
                                  const std::optional<Spheroid>& wgs) const
      -> LineString;

  /// Get the length of the linestring
  ///
  /// @return the length of the line string
  [[nodiscard]] inline auto size() const -> size_t { return Base::size(); }

  /// Get a point of the linestring at a given index
  ///
  /// @param index the index of the point
  /// @return the point at the given index
  [[nodiscard]] inline auto operator()(const size_t index) const -> Point {
    if (index >= size()) {
      throw std::out_of_range("LineString index out of range");
    }
    return Base::operator[](index);
  }

  /// Get a point of the linestring at a given index
  ///
  /// @param index the index of the point
  /// @return the point at the given index
  [[nodiscard]] inline auto operator[](const size_t index) const -> Point {
    return Base::operator[](index);
  }

  /// Returns a read-only (constant) iterator that points to the
  /// first point of the linestring.
  [[nodiscard]] auto begin() const -> decltype(Base::begin()) {
    return Base::begin();
  }

  /// Returns an iterator that points to the first point of the linestring.
  [[nodiscard]] auto begin() -> decltype(Base::begin()) {
    return Base::begin();
  }

  /// Returns a read-only (constant) iterator that points to the
  /// last point of the linestring.
  [[nodiscard]] auto end() const -> decltype(Base::end()) {
    return Base::end();
  }

  /// Returns an iterator that points to the last point of the linestring.
  [[nodiscard]] auto end() -> decltype(Base::end()) { return Base::end(); }

  /// Get a tuple that fully encodes the state of this instance
  [[nodiscard]] auto getstate() const -> pybind11::tuple;

  /// Create a new instance from a registered state of an instance of this
  /// object.
  static auto setstate(const pybind11::tuple& state) -> LineString;

  /// Converts a Polygon into a string with the same meaning as that of this
  /// instance.
  [[nodiscard]] auto to_string() const -> std::string {
    std::stringstream ss;
    ss << boost::geometry::dsv(*this);
    return ss.str();
  }

  /// Returns the curvilinear distance along the linestring.
  [[nodiscard]] auto curvilinear_distance(
      DistanceStrategy strategy, const std::optional<Spheroid>& wgs) const
      -> Vector<double>;
};

}  // namespace pyinterp::geodetic

namespace boost::geometry::traits {
namespace pg = pyinterp::geodetic;

template <>
struct tag<pg::LineString> {
  using type = linestring_tag;
};

}  // namespace boost::geometry::traits
