// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithms/convert.hpp"

namespace nb = nanobind;

namespace pyinterp::geometry::geographic::pybind {

auto init_convert(nb::module_& m) -> void {
  geometry::pybind::init_convert_geographic(m);
}

}  // namespace pyinterp::geometry::geographic::pybind
