// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/pybind/geometry.hpp"

#include "pyinterp/pybind/geometry/cartesian.hpp"
#include "pyinterp/pybind/geometry/geographic.hpp"
#include "pyinterp/pybind/geometry/satellite.hpp"

namespace pyinterp::pybind {

void init_geometry(nanobind::module_& m) {
  auto geometry = m.def_submodule("geometry", "Geometry module");
  pyinterp::geometry::pybind::init_geographic(geometry);
  pyinterp::geometry::pybind::init_cartesian(geometry);
  pyinterp::geometry::pybind::init_satellite(geometry);
}

}  // namespace pyinterp::pybind
