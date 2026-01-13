// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/pybind/geometry/cartesian.hpp"

namespace pyinterp::geometry::pybind {

void init_cartesian(nanobind::module_& m) {
  auto cartesian_module =
      m.def_submodule("cartesian", "Cartesian calculations");
  auto algorithms_module =
      cartesian_module.def_submodule("algorithms", "Cartesian algorithms");
  geometry::cartesian::pybind::init_point(cartesian_module);
  geometry::cartesian::pybind::init_box(cartesian_module);
  geometry::cartesian::pybind::init_linestring(cartesian_module);
  geometry::cartesian::pybind::init_segment(cartesian_module);
  geometry::cartesian::pybind::init_multilinestring(cartesian_module);
  geometry::cartesian::pybind::init_ring(cartesian_module);
  geometry::cartesian::pybind::init_polygon(cartesian_module);
  geometry::cartesian::pybind::init_multipoint(cartesian_module);
  geometry::cartesian::pybind::init_multipolygon(cartesian_module);
  geometry::cartesian::pybind::init_algorithms(algorithms_module);
}

}  // namespace pyinterp::geometry::pybind
