// Copyright (c) 2020 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <boost/geometry.hpp>

#include "pyinterp/geodetic/point.hpp"

namespace pyinterp::geodetic {

using Polygon = boost::geometry::model::polygon<Point>;

}