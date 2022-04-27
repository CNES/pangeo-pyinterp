// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/numpy.h>

#include <Eigen/Core>
#include <list>
#include <set>
#include <string>
#include <tuple>

#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/detail/math.hpp"
#include "pyinterp/detail/thread.hpp"
#include "pyinterp/geodetic/algorithm.hpp"
#include "pyinterp/geodetic/point.hpp"
#include "pyinterp/geodetic/spheroid.hpp"

namespace pyinterp::geodetic {

/// Forward declaration
class Polygon;

// Defines a box made of two describing points.
class Box : public boost::geometry::model::box<Point> {
 public:
  /// @brief Default constructor
  Box() : boost::geometry::model::box<Point>() {}

  /// @brief Constructor taking the minimum corner point and the maximum corner
  /// point.
  /// @param min_corner the minimum corner point
  /// @param max_corner the maximum corner point
  Box(const Point &min_corner, const Point &max_corner)
      : boost::geometry::model::box<Point>(min_corner, max_corner) {}

  /// Build a new box from a GeoJSON box.
  static auto from_geojson(const pybind11::list &data) -> Box {
    if (data.size() != 4) {
      throw std::invalid_argument("Box must be a list of 4 elements");
    }
    return {Point(data[0].cast<double>(), data[1].cast<double>()),
            Point(data[2].cast<double>(), data[3].cast<double>())};
  }

  /// @brief Returns the box covering the whole earth.
  [[nodiscard]] static auto whole_earth() -> Box {
    return {{-180, -90}, {180, 90}};
  }

  /// @brief Returns the center of the box.
  [[nodiscard]] inline auto centroid() const -> Point {
    return boost::geometry::return_centroid<Point, Box>(*this);
  }

  /// @brief Returns the delta of the box in latitude and longitude.
  [[nodiscard]] inline auto delta(bool round) const
      -> std::tuple<double, double> {
    auto x = this->max_corner().lon() - this->min_corner().lon();
    auto y = this->max_corner().lat() - this->min_corner().lat();
    if (round) {
      x = Box::max_decimal_power(x);
      y = Box::max_decimal_power(y);
    }
    return std::make_tuple(x, y);
  }

  /// @brief Returns a point inside the box, making an effort to round to
  /// minimal precision.
  [[nodiscard]] inline auto round() const -> Point {
    const auto xy = delta(true);
    const auto x = std::get<0>(xy);
    const auto y = std::get<1>(xy);
    return {std::ceil(this->min_corner().lon() / x) * x,
            std::ceil(this->min_corner().lat() / y) * y};
  }

  /// @brief Test if the given point is inside or on border of this instance
  ///
  /// @param pt Point to test
  //  @return True if the given point is inside or on border of this Box
  [[nodiscard]] auto covered_by(const Point &point) const -> bool {
    return boost::geometry::covered_by(point, *this);
  }

  /// @brief Test if the coordinates of the points provided are located inside
  /// or at the edge of this box.
  ///
  /// @param lon Longitudes coordinates in degrees to check
  /// @param lat Latitude coordinates in degrees to check
  /// @return Returns a vector containing a flag equal to 1 if the coordinate is
  /// located in the box or at the edge otherwise 0.
  [[nodiscard]] auto covered_by(const Eigen::Ref<const Eigen::VectorXd> &lon,
                                const Eigen::Ref<const Eigen::VectorXd> &lat,
                                const size_t num_threads) const
      -> pybind11::array_t<bool> {
    return geodetic::covered_by<Point, Box>(*this, lon, lat, num_threads);
  }

  /// Converts a Box into a string with the same meaning as that of this
  /// instance.
  [[nodiscard]] auto to_string() const -> std::string {
    std::stringstream ss;
    ss << boost::geometry::dsv(*this);
    return ss.str();
  }

  /// Calculate the area
  [[nodiscard]] auto area(const std::optional<Spheroid> &wgs) const -> double;

  /// Calculate the distance between two boxes
  [[nodiscard]] auto distance(const Box &other) const -> double;

  /// Calculate the distance between this instance and a point
  [[nodiscard]] auto distance(const Point &other) const -> double;

  /// Get a tuple that fully encodes the state of this instance
  [[nodiscard]] auto getstate() const -> pybind11::tuple {
    return pybind11::make_tuple(this->min_corner().getstate(),
                                this->max_corner().getstate());
  }

  /// Create a new instance from a registered state of an instance of this
  /// object.
  static auto setstate(const pybind11::tuple &state) -> Box {
    if (state.size() != 2) {
      throw std::runtime_error("invalid state");
    }
    return {Point::setstate(state[0].cast<pybind11::tuple>()),
            Point::setstate(state[1].cast<pybind11::tuple>())};
  }

  auto operator==(const Box &other) const -> bool {
    return boost::geometry::equals(*this, other);
  }

  /// Converts this instance into a polygon
  explicit operator Polygon() const;

  /// Returns a GeoJSON representation of this instance.
  [[nodiscard]] auto to_geojson() const -> pybind11::dict;

 private:
  // Returns the maximum power of 10 from a number (x > 0)
  static auto max_decimal_power(const double x) -> double {
    auto m = static_cast<int32_t>(std::floor(std::log10(x)));
    return detail::math::power10(m);
  }
};

}  // namespace pyinterp::geodetic

// BOOST specialization to accept pyinterp::geodectic::Box as a geometry
// entity
namespace boost::geometry::traits {

namespace pg = pyinterp::geodetic;

/// Box tag
template <>
struct tag<pg::Box> {
  using type = box_tag;
};

/// Type of a point
template <>
struct point_type<pg::Box> {
  using type = pg::Point;
};

template <std::size_t Dimension>
struct indexed_access<pg::Box, min_corner, Dimension> {
  /// get corner of box
  static inline auto get(pg::Box const &box) -> double {
    return geometry::get<Dimension>(box.min_corner());
  }

  /// set corner of box
  static inline void set(pg::Box &box,  // NOLINT
                         double const &value) {
    geometry::set<Dimension>(box.min_corner(), value);
  }
};

template <std::size_t Dimension>
struct indexed_access<pg::Box, max_corner, Dimension> {
  /// get corner of box
  static inline auto get(pg::Box const &box) -> double {
    return geometry::get<Dimension>(box.max_corner());
  }

  /// set corner of box
  static inline void set(pg::Box &box,  // NOLINT
                         double const &value) {
    geometry::set<Dimension>(box.max_corner(), value);
  }
};

}  // namespace boost::geometry::traits
