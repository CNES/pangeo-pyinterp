// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"
#include "pyinterp/pybind/geometry/algorithms/reverse.hpp"

using pyinterp::geometry::pybind::GeometryNamespace;

namespace pyinterp::geometry::cartesian::pybind {

auto init_reverse(nanobind::module_& m) -> void {
  geometry::pybind::init_reverse<GeometryNamespace::kCartesian>(m);
}

}  // namespace pyinterp::geometry::cartesian::pybind
