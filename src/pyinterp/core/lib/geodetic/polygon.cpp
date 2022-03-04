// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geodetic/polygon.hpp"

#include <pybind11/pybind11.h>

#include "pyinterp/geodetic/box.hpp"

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

/// Calculates the envelope of this polygon.
[[nodiscard]] auto Polygon::envelope() const -> Box {
  auto box = Box();
  boost::geometry::envelope(*this, box);
  return box;
}

}  // namespace pyinterp::geodetic
