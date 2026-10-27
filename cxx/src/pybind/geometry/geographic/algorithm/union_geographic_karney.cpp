// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

// Instantiates the geographic union for the Karney strategy (see
// pyinterp/pybind/geometry/geographic/overlay.hpp).
#include "pyinterp/pybind/geometry/geographic/overlay_instances.hpp"

namespace pyinterp::geometry::geographic::pybind {

template struct OverlayInstances<Overlay::kUnion, Polygon,
                                 StrategyMethod::kKarney,
                                 UNION_POLYGON_PAIRS(geographic)>;
template struct OverlayInstances<Overlay::kUnion, LineString,
                                 StrategyMethod::kKarney,
                                 UNION_LINESTRING_PAIRS(geographic)>;

}  // namespace pyinterp::geometry::geographic::pybind
