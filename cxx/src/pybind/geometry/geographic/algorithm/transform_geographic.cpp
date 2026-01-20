// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithms/transform.hpp"

namespace nb = nanobind;

namespace pyinterp::geometry::geographic::pybind {

auto init_transform(nb::module_& m) -> void {
  geometry::pybind::init_transform_geographic(m);
}

}  // namespace pyinterp::geometry::geographic::pybind
