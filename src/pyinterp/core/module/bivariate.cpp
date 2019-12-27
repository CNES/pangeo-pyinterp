// Copyright (c) 2020 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <pybind11/pybind11.h>
#include "pyinterp/bivariate.hpp"

namespace py = pybind11;
namespace geometry = pyinterp::detail::geometry;

void init_bivariate(py::module& m) {
  pyinterp::implement_bivariate<geometry::EquatorialPoint2D, double, double>(
      m, "Float64");
  pyinterp::implement_bivariate<geometry::EquatorialPoint2D, double, float>(
      m, "Float32");
}
