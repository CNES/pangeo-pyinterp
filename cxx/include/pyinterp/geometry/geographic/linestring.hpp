// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/geometry/linestring.hpp"

namespace pyinterp::geometry::geographic {

/// @brief Type representing a linestring in geodetic coordinates
using LineString = pyinterp::geometry::LineString<Point>;

}  // namespace pyinterp::geometry::geographic

namespace boost::geometry::traits {

template <>
struct tag<pyinterp::geometry::geographic::LineString> {
  using type = linestring_tag;
};

template <>
struct point_type<pyinterp::geometry::geographic::LineString> {
  using type = pyinterp::geometry::geographic::Point;
};

}  // namespace boost::geometry::traits
