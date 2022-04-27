// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <boost/geometry.hpp>
#include <string>
#include <vector>

#include "pyinterp/geodetic/polygon.hpp"

namespace pyinterp::geodetic {

/// Forward declaration
class Polygon;

/// A multipolygon is a collection of polygons.
class MultiPolygon : public boost::geometry::model::multi_polygon<Polygon> {
 public:
  using Base = boost::geometry::model::multi_polygon<Polygon>;

  /// Default constructor
  MultiPolygon() = default;

  /// Create a new instance from Python
  MultiPolygon(std::initializer_list<Polygon> polygons) : Base(polygons) {}

  /// Create a new instance from Python
  explicit MultiPolygon(const pybind11::list &polygons);

  /// Build a new box from a GeoJSON box.
  static auto from_geojson(const pybind11::list &data) -> MultiPolygon;

  /// Calculates the envelope of this polygon.
  [[nodiscard]] auto envelope() const -> Box;

  /// Add a polygon to this multipolygon
  auto append(Polygon &&polygon) -> void { Base::emplace_back(polygon); }

  /// Add a multi-polygon to this multipolygon
  auto operator+=(const MultiPolygon &other) -> MultiPolygon {
    Base::insert(Base::end(), other.begin(), other.end());
    return *this;
  }

  /// Calculate the area
  [[nodiscard]] auto area(const std::optional<Spheroid> &wgs) const -> double {
    return geodetic::area(*this, wgs);
  }

  /// Calculate the distance between this instance and an other multi-polygon
  [[nodiscard]] auto distance(const MultiPolygon &other) const -> double {
    return geodetic::distance(*this, other);
  }

  /// Calculate the distance between this instance and a polygon
  [[nodiscard]] auto distance(const Polygon &other) const -> double {
    return geodetic::distance(*this, other);
  }

  /// Calculate the distance between this instance and a point
  [[nodiscard]] auto distance(const Point &other) const -> double {
    return geodetic::distance(*this, other);
  }

  /// @brief Test if the given point is inside or on border of this instance
  ///
  /// @param pt Point to test
  //  @return True if the given point is inside or on border of this Polygon
  [[nodiscard]] auto covered_by(const Point &point) const -> bool {
    return boost::geometry::covered_by(point, *this);
  }

  /// @brief Test if the coordinates of the points provided are located inside
  /// or at the edge of this Polygon.
  ///
  /// @param lon Longitudes coordinates in degrees to check
  /// @param lat Latitude coordinates in degrees to check
  /// @return Returns a vector containing a flag equal to 1 if the coordinate is
  /// located in the Polygon or at the edge otherwise 0.
  [[nodiscard]] auto covered_by(const Eigen::Ref<const Eigen::VectorXd> &lon,
                                const Eigen::Ref<const Eigen::VectorXd> &lat,
                                const size_t num_threads) const
      -> pybind11::array_t<bool> {
    return geodetic::covered_by<Point, MultiPolygon>(*this, lon, lat,
                                                     num_threads);
  }

  /// Return true if this instance contains this polygon
  [[nodiscard]] auto contains(const Polygon &polygon) const -> bool {
    for (const auto &item : *this) {
      if (boost::geometry::equals(item, polygon)) {
        return true;
      }
    }
    return false;
  }

  /// Combines this instance with another multi-polygon
  [[nodiscard]] auto union_(const MultiPolygon &other) const -> MultiPolygon;

  /// Combines this instance with another polygon
  [[nodiscard]] auto union_(const Polygon &other) const -> MultiPolygon;

  /// Calculates the intersection between this instance and a polygon.
  [[nodiscard]] auto intersection(const Polygon &other) const -> MultiPolygon;

  /// Calculates the intersection between this instance and another one.
  [[nodiscard]] auto intersection(const MultiPolygon &other) const
      -> MultiPolygon;

  /// Checks if this multipolygon intersects with another one.
  [[nodiscard]] auto intersects(const MultiPolygon &other) const -> bool {
    return boost::geometry::intersects(*this, other);
  }

  /// Checks if this multipolygon intersects with a polygon.
  [[nodiscard]] auto intersects(const Polygon &other) const -> bool {
    return boost::geometry::intersects(*this, other);
  }

  /// Checks if this multipolygon touches with another one.
  [[nodiscard]] auto touches(const MultiPolygon &other) const -> bool {
    return boost::geometry::touches(*this, other);
  }

  /// Checks if this multipolygon touches with a polygon.
  [[nodiscard]] auto touches(const Polygon &other) const -> bool {
    return boost::geometry::touches(*this, other);
  }

  /// Returns the GEOJSon coordinates of this instance.
  [[nodiscard]] auto coordinates() const -> pybind11::list;

  /// Returns a GeoJSON representation of this instance.
  [[nodiscard]] auto to_geojson() const -> pybind11::dict;

  /// Converts a Polygon into a string with the same meaning as that of this
  /// instance.
  [[nodiscard]] auto to_string() const -> std::string {
    std::stringstream ss;
    ss << boost::geometry::dsv(*this);
    return ss.str();
  }

  [[nodiscard]] inline auto operator()(const size_t idx) const
      -> const Polygon & {
    if (idx >= Base::size()) {
      throw std::out_of_range("Multipolygon index out of range");
    }
    return Base::operator[](idx);
  }

  /// Return the number of polygons in this instance
  [[nodiscard]] inline auto size() const -> size_t { return Base::size(); }

  /// Returns the number of the interior rings of all polygons
  [[nodiscard]] inline auto num_interior_rings() const -> size_t {
    return boost::geometry::num_interior_rings(*this);
  }

  /// Returns an iterator that points to the first polygon of this instance.
  [[nodiscard]] inline auto begin() -> decltype(Base::begin()) {
    return Base::begin();
  }

  /// Returns an iterator that points to the last polygon of this instance.
  [[nodiscard]] inline auto end() -> decltype(Base::end()) {
    return Base::end();
  }

  /// Returns a read-only (constant) iterator that points to the
  /// first polygon of this instance.
  [[nodiscard]] inline auto begin() const -> decltype(Base::begin()) {
    return Base::begin();
  }

  /// Returns a read-only (constant) iterator that points to the
  /// last polygon of this instance.
  [[nodiscard]] inline auto end() const -> decltype(Base::end()) {
    return Base::end();
  }

  /// Get a tuple that fully encodes the state of this instance
  [[nodiscard]] auto getstate() const -> pybind11::tuple {
    return pybind11::make_tuple(coordinates());
  }

  /// Create a new instance from a registered state of an instance of this
  /// object.
  static auto setstate(const pybind11::tuple &state) -> MultiPolygon {
    if (state.size() != 1) {
      throw std::runtime_error("invalid state");
    }
    return MultiPolygon::from_geojson(state[0].cast<pybind11::list>());
  }
};

}  // namespace pyinterp::geodetic

namespace boost::geometry::traits {
namespace pg = pyinterp::geodetic;

template <>
struct tag<pg::MultiPolygon> {
  using type = multi_polygon_tag;
};

}  // namespace boost::geometry::traits
