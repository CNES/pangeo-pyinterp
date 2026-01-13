// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <boost/geometry.hpp>

#include "pyinterp/geometry/box.hpp"
#include "pyinterp/geometry/cartesian/point.hpp"

namespace pyinterp::geometry::cartesian {

/// @brief Type representing a bounding box in cartesian coordinates
using Box = pyinterp::geometry::Box<Point>;

}  // namespace pyinterp::geometry::cartesian

// Boost.Geometry traits
namespace boost::geometry::traits {

template <>
struct tag<pyinterp::geometry::cartesian::Box> {
  using type = box_tag;
};

template <>
struct point_type<pyinterp::geometry::cartesian::Box> {
  using type = pyinterp::geometry::cartesian::Point;
};

template <std::size_t I, std::size_t D>
struct indexed_access<pyinterp::geometry::cartesian::Box, I, D> {
  static auto get(const pyinterp::geometry::cartesian::Box& b) -> double {
    if constexpr (I == min_corner) {
      return b.min_corner().get<D>();
    } else {
      return b.max_corner().get<D>();
    }
  }
  static void set(pyinterp::geometry::cartesian::Box& b, double v) {
    if constexpr (I == min_corner) {
      b.min_corner().set<D>(v);
    } else {
      b.max_corner().set<D>(v);
    }
  };
};

}  // namespace boost::geometry::traits
