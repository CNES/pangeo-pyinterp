// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/bivariate.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace geometry = pyinterp::detail::geometry;

void init_bivariate(py::module &m) {
  pyinterp::implement_bivariate<geometry::EquatorialPoint2D, double, double>(
      m, "Float64");
  pyinterp::implement_bivariate<geometry::EquatorialPoint2D, double, float>(
      m, "Float32");
  pyinterp::implement_bivariate<geometry::EquatorialPoint2D, double, int8_t>(
      m, "Int8");
}
