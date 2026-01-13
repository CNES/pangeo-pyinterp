// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/pybind/geometry/geographic.hpp"

namespace pyinterp::geometry::pybind {

void init_geographic(nanobind::module_& m) {
  auto geographic_module =
      m.def_submodule("geographic", "Geographic calculations");
  auto algorithms_module =
      geographic_module.def_submodule("algorithms", "Geographic algorithms");
  geometry::geographic::pybind::init_point(geographic_module);
  geometry::geographic::pybind::init_box(geographic_module);
  geometry::geographic::pybind::init_linestring(geographic_module);
  geometry::geographic::pybind::init_segment(geographic_module);
  geometry::geographic::pybind::init_multilinestring(geographic_module);
  geometry::geographic::pybind::init_ring(geographic_module);
  geometry::geographic::pybind::init_polygon(geographic_module);
  geometry::geographic::pybind::init_multipoint(geographic_module);
  geometry::geographic::pybind::init_multipolygon(geographic_module);
  geometry::geographic::pybind::init_spheroid(geographic_module);
  geometry::geographic::pybind::init_coordinates(geographic_module);
  geometry::geographic::pybind::init_rtree(geographic_module);
  geometry::geographic::pybind::init_algorithms(algorithms_module);
}

}  // namespace pyinterp::geometry::pybind
