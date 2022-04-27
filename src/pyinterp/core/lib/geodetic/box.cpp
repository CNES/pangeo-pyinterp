// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geodetic/box.hpp"

#include "pyinterp/geodetic/polygon.hpp"

namespace pyinterp::geodetic {

/// Calculate the area
auto Box::area(const std::optional<Spheroid> &wgs) const -> double {
  return static_cast<Polygon>(*this).area(wgs);
}

/// Calculate the distance between two boxes
auto Box::distance(const Box &other) const -> double {
  return static_cast<Polygon>(*this).distance(static_cast<Polygon>(other));
}

/// Calculate the distance between this instance and a point
[[nodiscard]] auto Box::distance(const Point &other) const -> double {
  return static_cast<Polygon>(*this).distance(other);
}

/// Converts this instance into a polygon
Box::operator Polygon() const {
  Polygon result;
  boost::geometry::convert(*this, result);
  return result;
}

auto Box::to_geojson() const -> pybind11::dict {
  return static_cast<Polygon>(*this).to_geojson();
}

}  // namespace pyinterp::geodetic
