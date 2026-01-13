// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/geometry/multi_point.hpp"

namespace pyinterp::geometry::geographic {

/// @brief MultiPoint: collection of points.
///
/// A `MultiPoint` represents a collection of points. It provides
/// basic container-like operations for constructing and iterating over
/// points.
using MultiPoint = pyinterp::geometry::MultiPoint<Point>;

}  // namespace pyinterp::geometry::geographic

namespace boost::geometry::traits {

template <>
struct tag<pyinterp::geometry::geographic::MultiPoint> {
  using type = multi_point_tag;
};

}  // namespace boost::geometry::traits
