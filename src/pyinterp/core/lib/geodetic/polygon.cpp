// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geodetic/polygon.hpp"

#include <pybind11/pybind11.h>

#include "pyinterp/geodetic/box.hpp"
#include "pyinterp/geodetic/multipolygon.hpp"

namespace pyinterp::geodetic {

Polygon::Polygon(const pybind11::list &outer, const pybind11::list &inners) {
  try {
    for (const auto item : outer) {
      auto point = item.cast<geodetic::Point>();
      boost::geometry::append(Base::outer(), point);
    }
  } catch (const pybind11::cast_error &) {
    throw std::invalid_argument(
        "outer must be a list of pyinterp.geodetic.Point");
  }
  if (!inners.empty()) {
    try {
      auto index = 0;
      Base::inners().resize(inners.size());
      for (const auto inner : inners) {
        auto points = inner.cast<pybind11::list>();
        for (const auto item : points) {
          auto point = item.cast<geodetic::Point>();
          boost::geometry::append(Base::inners()[index], point);
        }
        ++index;
      }
    } catch (const pybind11::cast_error &) {
      throw std::invalid_argument(
          "inners must be a list of "
          "list of pyinterp.geodetic.Point");
    }
  }
}

auto Polygon::from_geojson(const pybind11::list &data) -> Polygon {
  auto polygon = Polygon();
  if (data.empty()) {
    return polygon;
  }
  auto outer = data[0].cast<pybind11::list>();
  auto *base = dynamic_cast<boost::geometry::model::polygon<Point> *>(&polygon);
  for (const auto item : outer) {
    base->outer().push_back(Point::from_geojson(item.cast<pybind11::list>()));
  }
  for (size_t ix = 1; ix < data.size(); ++ix) {
    auto inner = data[ix].cast<pybind11::list>();
    if (inner.empty()) {
      continue;
    }
    auto &back = base->inners().emplace_back();
    for (const auto coordinates : inner) {
      back.push_back(Point::from_geojson(coordinates.cast<pybind11::list>()));
    }
  }
  return polygon;
}

/// Calculates the envelope of this polygon.
[[nodiscard]] auto Polygon::envelope() const -> Box {
  auto box = Box();
  boost::geometry::envelope(*this, box);
  return box;
}

auto Polygon::coordinates() const -> pybind11::list {
  auto coordinates = pybind11::list();
  auto ring = pybind11::list();

  for (const auto &item : Base::outer()) {
    ring.append(item.coordinates());
  }
  coordinates.append(ring);

  for (const auto &inner : Base::inners()) {
    ring = pybind11::list();
    for (const auto &item : inner) {
      ring.append(item.coordinates());
    }
    coordinates.append(ring);
  }
  return coordinates;
}

auto Polygon::to_geojson() const -> pybind11::dict {
  auto result = pybind11::dict();
  result["type"] = "Polygon";
  result["coordinates"] = coordinates();
  return result;
}

auto Polygon::union_(const Polygon &other) const -> MultiPolygon {
  auto result = MultiPolygon();
  boost::geometry::union_(*this, other, result);
  return result;
}

auto Polygon::intersection(const Polygon &other) const -> MultiPolygon {
  auto result = MultiPolygon();
  boost::geometry::intersection(*this, other, result);
  return result;
}

}  // namespace pyinterp::geodetic
