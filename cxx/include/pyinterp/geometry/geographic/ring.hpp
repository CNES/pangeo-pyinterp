// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <iterator>

#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/geometry/ring.hpp"

namespace pyinterp::geometry::geographic {

/// @brief Ring: closed linestring (for polygon boundaries).
///
/// The `Ring` class represents a closed sequence of `Point` objects used
/// as polygon boundaries. It provides basic container-like operations and
/// iterator support.
using Ring = pyinterp::geometry::Ring<Point>;

}  // namespace pyinterp::geometry::geographic

namespace boost::geometry::traits {

template <>
struct tag<pyinterp::geometry::geographic::Ring> {
  using type = ring_tag;
};

template <>
struct point_type<pyinterp::geometry::geographic::Ring> {
  using type = pyinterp::geometry::geographic::Point;
};

}  // namespace boost::geometry::traits

// Make Ring compatible with boost::geometry range utilities
namespace std {
template <>
class back_insert_iterator<pyinterp::geometry::geographic::Ring>
    : public back_insert_iterator_ring<pyinterp::geometry::geographic::Ring,
                                       pyinterp::geometry::geographic::Point> {
};

}  // namespace std
