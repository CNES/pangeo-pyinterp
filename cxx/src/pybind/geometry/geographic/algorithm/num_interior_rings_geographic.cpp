// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"
#include "pyinterp/pybind/geometry/algorithms/num_interior_rings.hpp"

using pyinterp::geometry::pybind::GeometryNamespace;

namespace pyinterp::geometry::geographic::pybind {

auto init_num_interior_rings(nanobind::module_& m) -> void {
  geometry::pybind::init_num_interior_rings<GeometryNamespace::kGeographic>(m);
}

}  // namespace pyinterp::geometry::geographic::pybind
