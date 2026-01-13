// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <vector>

#include "pyinterp/geometry/cartesian/point.hpp"
#include "pyinterp/geometry/cartesian/ring.hpp"
#include "pyinterp/geometry/polygon.hpp"

namespace pyinterp::geometry::cartesian {
/// @brief Polygon: exterior ring + optional interior rings (holes).
///
/// The `Polygon` class represents a polygon defined by an exterior ring and
/// optional interior rings (holes). It provides accessors for the exterior
/// ring and the interior rings.
using Polygon = pyinterp::geometry::Polygon<Point>;

}  // namespace pyinterp::geometry::cartesian

namespace boost::geometry::traits {

template <>
struct tag<pyinterp::geometry::cartesian::Polygon> {
  using type = polygon_tag;
};

template <>
struct point_type<pyinterp::geometry::cartesian::Polygon> {
  using type = pyinterp::geometry::cartesian::Point;
};

template <>
struct ring_const_type<pyinterp::geometry::cartesian::Polygon> {
  using type = const pyinterp::geometry::cartesian::Ring&;
};

template <>
struct ring_mutable_type<pyinterp::geometry::cartesian::Polygon> {
  using type = pyinterp::geometry::cartesian::Ring&;
};

template <>
struct interior_const_type<pyinterp::geometry::cartesian::Polygon> {
  using type = const std::vector<pyinterp::geometry::cartesian::Ring>&;
};

template <>
struct interior_mutable_type<pyinterp::geometry::cartesian::Polygon> {
  using type = std::vector<pyinterp::geometry::cartesian::Ring>&;
};

template <>
struct exterior_ring<pyinterp::geometry::cartesian::Polygon> {
  static auto get(pyinterp::geometry::cartesian::Polygon& p)
      -> pyinterp::geometry::cartesian::Ring& {
    return p.outer();
  }
  static auto get(const pyinterp::geometry::cartesian::Polygon& p)
      -> const pyinterp::geometry::cartesian::Ring& {
    return p.outer();
  }
};

template <>
struct interior_rings<pyinterp::geometry::cartesian::Polygon> {
  static auto get(pyinterp::geometry::cartesian::Polygon& p)
      -> std::vector<pyinterp::geometry::cartesian::Ring>& {
    return p.inners();
  }
  static auto get(const pyinterp::geometry::cartesian::Polygon& p)
      -> const std::vector<pyinterp::geometry::cartesian::Ring>& {
    return p.inners();
  }
};

}  // namespace boost::geometry::traits
