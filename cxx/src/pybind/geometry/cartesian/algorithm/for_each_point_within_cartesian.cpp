// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"
#include "pyinterp/pybind/geometry/algorithms/for_each_point_within.hpp"

namespace pyinterp::geometry::cartesian::pybind {

auto init_for_each_point_within(nanobind::module_& m) -> void {
  using geometry::pybind::GeometryNamespace;
  using geometry::pybind::init_for_each_point_within;
  init_for_each_point_within<GeometryNamespace::kCartesian>(m);
}

}  // namespace pyinterp::geometry::cartesian::pybind
