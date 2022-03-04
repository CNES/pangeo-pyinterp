// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <Eigen/Core>
#include <boost/geometry.hpp>
#include <string>

#include "pyinterp/geodetic/algorithm.hpp"
#include "pyinterp/geodetic/point.hpp"

namespace pyinterp::geodetic {

/// Forward declaration
class Box;

class Polygon : public boost::geometry::model::polygon<Point> {
 public:
  using Base = boost::geometry::model::polygon<Point>;
  using Base::polygon;

  /// Create a new instance from Python
  Polygon(const pybind11::list &outer, const pybind11::list &inners);

  /// Returns the outer ring
  [[nodiscard]] auto outer() const -> pybind11::list {
    auto outer = pybind11::list();

    for (const auto &item : Base::outer()) {
      outer.append(item);
    }
    return outer;
  }

  /// Returns the inner rings
  [[nodiscard]] auto inners() const -> pybind11::list {
    auto inners = pybind11::list();

    for (const auto &inner : Base::inners()) {
      auto buffer = pybind11::list();
      for (const auto &item : inner) {
        buffer.append(item);
      }
      inners.append(buffer);
    }
    return inners;
  }

  /// Calculates the envelope of this polygon.
  [[nodiscard]] auto envelope() const -> Box;

  /// Get a tuple that fully encodes the state of this instance
  [[nodiscard]] auto getstate() const -> pybind11::tuple {
    return pybind11::make_tuple(this->outer(), this->inners());
  }

  /// Calculate the area
  [[nodiscard]] auto area(const std::optional<System> &wgs) const -> double {
    return geodetic::area(*this, wgs);
  }

  /// Calculate the distance between two polygons
  [[nodiscard]] auto distance(const Polygon &other) const -> double {
    return geodetic::distance(*this, other);
  }

  /// Calculate the distance between this instance and a point
  [[nodiscard]] auto distance(const Point &other) const -> double {
    return geodetic::distance(*this, other);
  }

  /// Create a new instance from a registered state of an instance of this
  /// object.
  static auto setstate(const pybind11::tuple &state) -> Polygon {
    if (state.size() != 2) {
      throw std::runtime_error("invalid state");
    }
    return {state[0].cast<pybind11::list>(), state[1].cast<pybind11::list>()};
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
      -> pybind11::array_t<int8_t> {
    return geodetic::covered_by<Point, Polygon>(*this, lon, lat, num_threads);
  }

  /// Converts a Polygon into a string with the same meaning as that of this
  /// instance.
  [[nodiscard]] auto to_string() const -> std::string {
    std::stringstream ss;
    ss << boost::geometry::dsv(*this);
    return ss.str();
  }
};

}  // namespace pyinterp::geodetic

namespace boost::geometry::traits {
namespace pg = pyinterp::geodetic;

template <>
struct tag<pg::Polygon> {
  using type = polygon_tag;
};
template <>
struct ring_const_type<pg::Polygon> {
  using type = const model::polygon<pg::Point>::ring_type &;
};
template <>
struct ring_mutable_type<pg::Polygon> {
  using type = model::polygon<pg::Point>::ring_type &;
};
template <>
struct interior_const_type<pg::Polygon> {
  using type = const model::polygon<pg::Point>::inner_container_type &;
};
template <>
struct interior_mutable_type<pg::Polygon> {
  using type = model::polygon<pg::Point>::inner_container_type &;
};

template <>
struct exterior_ring<pg::Polygon> {
  static auto get(model::polygon<pg::Point> &p)
      -> model::polygon<pg::Point>::ring_type & {
    return p.outer();
  }
  static auto get(model::polygon<pg::Point> const &p)
      -> model::polygon<pg::Point>::ring_type const & {
    return p.outer();
  }
};

template <>
struct interior_rings<pg::Polygon> {
  static auto get(model::polygon<pg::Point> &p)
      -> model::polygon<pg::Point>::inner_container_type & {
    return p.inners();
  }
  static auto get(model::polygon<pg::Point> const &p)
      -> model::polygon<pg::Point>::inner_container_type const & {
    return p.inners();
  }
};

}  // namespace boost::geometry::traits
