// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <vector>

#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/geometry/geographic/ring.hpp"
#include "pyinterp/geometry/polygon.hpp"

namespace pyinterp::geometry::geographic {

/// @brief Polygon: exterior ring + optional interior rings (holes).
///
/// The `Polygon` class represents a polygon defined by an exterior ring and
/// optional interior rings (holes). It provides accessors for the exterior
/// ring and the interior rings.
using Polygon = pyinterp::geometry::Polygon<Point>;

}  // namespace pyinterp::geometry::geographic

namespace boost::geometry::traits {

template <>
struct tag<pyinterp::geometry::geographic::Polygon> {
  using type = polygon_tag;
};

template <>
struct point_type<pyinterp::geometry::geographic::Polygon> {
  using type = pyinterp::geometry::geographic::Point;
};

template <>
struct ring_const_type<pyinterp::geometry::geographic::Polygon> {
  using type = const pyinterp::geometry::geographic::Ring&;
};

template <>
struct ring_mutable_type<pyinterp::geometry::geographic::Polygon> {
  using type = pyinterp::geometry::geographic::Ring&;
};

template <>
struct interior_const_type<pyinterp::geometry::geographic::Polygon> {
  using type = const std::vector<pyinterp::geometry::geographic::Ring>&;
};

template <>
struct interior_mutable_type<pyinterp::geometry::geographic::Polygon> {
  using type = std::vector<pyinterp::geometry::geographic::Ring>&;
};

template <>
struct exterior_ring<pyinterp::geometry::geographic::Polygon> {
  static auto get(pyinterp::geometry::geographic::Polygon& p)
      -> pyinterp::geometry::geographic::Ring& {
    return p.outer();
  }
  static auto get(const pyinterp::geometry::geographic::Polygon& p)
      -> const pyinterp::geometry::geographic::Ring& {
    return p.outer();
  }
};

template <>
struct interior_rings<pyinterp::geometry::geographic::Polygon> {
  static auto get(pyinterp::geometry::geographic::Polygon& p)
      -> std::vector<pyinterp::geometry::geographic::Ring>& {
    return p.inners();
  }
  static auto get(const pyinterp::geometry::geographic::Polygon& p)
      -> const std::vector<pyinterp::geometry::geographic::Ring>& {
    return p.inners();
  }
};

}  // namespace boost::geometry::traits
