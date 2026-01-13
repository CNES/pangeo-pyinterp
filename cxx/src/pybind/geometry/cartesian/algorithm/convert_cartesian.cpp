// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"
#include "pyinterp/pybind/geometry/algorithms/convert.hpp"

namespace pyinterp::geometry::cartesian::pybind {

auto init_convert(nanobind::module_& m) -> void {
  geometry::pybind::init_convert_cartesian(m);
}

}  // namespace pyinterp::geometry::cartesian::pybind
