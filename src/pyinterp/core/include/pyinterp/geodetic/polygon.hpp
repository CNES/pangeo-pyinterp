// Copyright (c) 2021 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <boost/geometry.hpp>

#include "pyinterp/geodetic/algorithm.hpp"
#include "pyinterp/geodetic/point.hpp"

namespace pyinterp::geodetic {

class Polygon : public boost::geometry::model::polygon<Point> {
 public:
  using boost::geometry::model::polygon<Point>::polygon;

  /// Create a new instance from Python
  Polygon(const pybind11::list& outer, const pybind11::list& inners) {
    try {
      for (const auto& item : outer) {
        auto point = item.cast<geodetic::Point>();
        boost::geometry::append(this->outer(), point);
      }
    } catch (const pybind11::cast_error&) {
      throw std::invalid_argument(
          "outer must be a list of pyinterp.geodetic.Point");
    }
    if (!inners.empty()) {
      try {
        auto index = 0;
        this->inners().resize(inners.size());
        for (const auto& inner : inners) {
          auto points = inner.cast<pybind11::list>();
          for (const auto& item : points) {
            auto point = item.cast<geodetic::Point>();
            boost::geometry::append(this->inners()[index], point);
          }
          ++index;
        }
      } catch (const pybind11::cast_error&) {
        throw std::invalid_argument(
            "inners must be a list of "
            "list of pyinterp.geodetic.Point");
      }
    }
  }

  /// Get a tuple that fully encodes the state of this instance
  [[nodiscard]] auto getstate() const -> pybind11::tuple {
    auto inners = pybind11::list();
    auto outer = pybind11::list();

    for (const auto& item : this->outer()) {
      outer.append(item);
    }

    for (const auto& inner : this->inners()) {
      auto buffer = pybind11::list();
      for (const auto& item : inner) {
        buffer.append(item);
      }
      inners.append(buffer);
    }
    return pybind11::make_tuple(outer, inners);
  }

  /// Calculate the area
  [[nodiscard]] auto area(const std::optional<System>& wgs) const -> double {
    return geodetic::area(*this, wgs);
  }

  /// Create a new instance from a registered state of an instance of this
  /// object.
  static auto setstate(const pybind11::tuple& state) -> Polygon {
    if (state.size() != 2) {
      throw std::runtime_error("invalid state");
    }
    return Polygon(state[0].cast<pybind11::list>(),
                   state[1].cast<pybind11::list>());
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
  using type = const model::polygon<pg::Point>::ring_type&;
};
template <>
struct ring_mutable_type<pg::Polygon> {
  using type = model::polygon<pg::Point>::ring_type&;
};
template <>
struct interior_const_type<pg::Polygon> {
  using type = const model::polygon<pg::Point>::inner_container_type&;
};
template <>
struct interior_mutable_type<pg::Polygon> {
  using type = model::polygon<pg::Point>::inner_container_type&;
};

template <>
struct exterior_ring<pg::Polygon> {
  static auto get(model::polygon<pg::Point>& p)
      -> model::polygon<pg::Point>::ring_type& {
    return p.outer();
  }
  static auto get(model::polygon<pg::Point> const& p)
      -> model::polygon<pg::Point>::ring_type const& {
    return p.outer();
  }
};

template <>
struct interior_rings<pg::Polygon> {
  static auto get(model::polygon<pg::Point>& p)
      -> model::polygon<pg::Point>::inner_container_type& {
    return p.inners();
  }
  static auto get(model::polygon<pg::Point> const& p)
      -> model::polygon<pg::Point>::inner_container_type const& {
    return p.inners();
  }
};

}  // namespace boost::geometry::traits