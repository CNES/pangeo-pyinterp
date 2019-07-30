// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <boost/geometry.hpp>
#include "pyinterp/detail/geometry/point.hpp"

namespace pyinterp::detail::geometry {

/// Defines a box made of two describing points.
///
/// @tparam T Type of data handled by this box
/// @tparam N Number of dimensions in coordinate systems handled.
template <typename T, size_t N>
using BoxND = boost::geometry::model::box<PointND<T, N>>;

/// Defines a box made of two describing points in spherical equatorial space.
///
/// @tparam T Type of data handled by this box
template <typename T>
using EquatorialBox2D = boost::geometry::model::box<EquatorialPoint2D<T>>;

/// Defines a box made of three describing points in spherical equatorial
/// space.
///
/// @tparam T Type of data handled by this box
template <typename T>
using EquatorialBox3D = boost::geometry::model::box<EquatorialPoint3D<T>>;

}  // namespace pyinterp::detail::geometry
