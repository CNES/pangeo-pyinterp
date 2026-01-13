// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <boost/geometry.hpp>

#include "pyinterp/geometry/cartesian/point.hpp"
#include "pyinterp/geometry/segment.hpp"

namespace pyinterp::geometry::cartesian {

/// @brief Type representing a segment (two endpoints) in cartesian coordinates
using Segment = pyinterp::geometry::Segment<Point>;

}  // namespace pyinterp::geometry::cartesian

// Boost.Geometry traits
namespace boost::geometry::traits {

template <>
struct tag<pyinterp::geometry::cartesian::Segment> {
  using type = segment_tag;
};

template <>
struct point_type<pyinterp::geometry::cartesian::Segment> {
  using type = pyinterp::geometry::cartesian::Point;
};

/// @brief Indexed access to segment endpoints and coordinates
template <std::size_t Index, std::size_t Dim>
struct indexed_access<pyinterp::geometry::cartesian::Segment, Index, Dim> {
  static auto get(const pyinterp::geometry::cartesian::Segment& s) -> double {
    return s.template get<Index, Dim>();
  }
  static void set(pyinterp::geometry::cartesian::Segment& s, double v) {
    s.template set<Index, Dim>(v);
  }
};

}  // namespace boost::geometry::traits
