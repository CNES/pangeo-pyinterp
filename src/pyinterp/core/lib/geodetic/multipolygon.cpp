// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geodetic/multipolygon.hpp"

#include "pyinterp/geodetic/box.hpp"

namespace pyinterp::geodetic {

MultiPolygon::MultiPolygon(const pybind11::list &polygons) {
  try {
    if (polygons.empty()) {
      return;
    }
    resize(polygons.size());
    for (auto ix = size_t(0); ix < polygons.size(); ++ix) {
      auto polygon = polygons[ix].cast<geodetic::Polygon>();
      (*this)[ix] = polygon;
    }
  } catch (const pybind11::cast_error &) {
    throw std::invalid_argument(
        "polygons must be a list of pyinterp.geodetic.Polygon");
  }
}

auto MultiPolygon::from_geojson(const pybind11::list &data) -> MultiPolygon {
  auto multipolygon = MultiPolygon();
  for (auto item : data) {
    auto polygon = geodetic::Polygon::from_geojson(item.cast<pybind11::list>());
    multipolygon.push_back(polygon);
  }
  return multipolygon;
}

/// Calculates the envelope of this polygon.
[[nodiscard]] auto MultiPolygon::envelope() const -> Box {
  auto box = Box();
  boost::geometry::envelope(*this, box);
  return box;
}

}  // namespace pyinterp::geodetic
