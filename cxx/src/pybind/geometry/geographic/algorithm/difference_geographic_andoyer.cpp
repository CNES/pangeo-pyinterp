// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

// Instantiates the geographic difference for the Andoyer strategy (see
// pyinterp/pybind/geometry/geographic/overlay.hpp).
#include "pyinterp/pybind/geometry/geographic/overlay_instances.hpp"

namespace pyinterp::geometry::geographic::pybind {

template struct OverlayInstances<Overlay::kDifference, Polygon,
                                 StrategyMethod::kAndoyer,
                                 DIFFERENCE_POLYGON_PAIRS(geographic)>;

}  // namespace pyinterp::geometry::geographic::pybind
