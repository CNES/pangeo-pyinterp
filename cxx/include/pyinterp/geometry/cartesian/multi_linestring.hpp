// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include "pyinterp/geometry/cartesian/point.hpp"
#include "pyinterp/geometry/multi_linestring.hpp"

namespace pyinterp::geometry::cartesian {

/// @brief MultiLineString: collection of linestrings.
///
/// A `MultiLineString` represents a collection of `LineString` geometries.
/// It provides basic container-like operations for constructing and iterating
/// over linestrings.
using MultiLineString = pyinterp::geometry::MultiLineString<Point>;

}  // namespace pyinterp::geometry::cartesian

namespace boost::geometry::traits {

template <>
struct tag<pyinterp::geometry::cartesian::MultiLineString> {
  using type = multi_linestring_tag;
};

}  // namespace boost::geometry::traits
