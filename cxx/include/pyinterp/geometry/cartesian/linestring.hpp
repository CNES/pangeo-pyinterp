// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include "pyinterp/geometry/cartesian/point.hpp"
#include "pyinterp/geometry/linestring.hpp"

namespace pyinterp::geometry::cartesian {

/// @brief Type representing a linestring in cartesian coordinates
using LineString = pyinterp::geometry::LineString<Point>;

}  // namespace pyinterp::geometry::cartesian

namespace boost::geometry::traits {

template <>
struct tag<pyinterp::geometry::cartesian::LineString> {
  using type = linestring_tag;
};

template <>
struct point_type<pyinterp::geometry::cartesian::LineString> {
  using type = pyinterp::geometry::cartesian::Point;
};

}  // namespace boost::geometry::traits
