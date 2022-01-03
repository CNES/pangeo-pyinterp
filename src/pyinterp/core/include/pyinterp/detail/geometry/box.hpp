// Copyright (c) 2022 CNES
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

}  // namespace pyinterp::detail::geometry
