// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/geometry/multi_polygon.hpp"

namespace pyinterp::geometry::geographic {

/// @brief MultiPolygon: collection of polygons.
///
/// A `MultiPolygon` represents a collection of polygons. It provides
/// basic container-like operations for constructing and iterating over
/// polygons.
using MultiPolygon = pyinterp::geometry::MultiPolygon<Point>;

}  // namespace pyinterp::geometry::geographic

namespace boost::geometry::traits {

template <>
struct tag<pyinterp::geometry::geographic::MultiPolygon> {
  using type = multi_polygon_tag;
};

}  // namespace boost::geometry::traits
