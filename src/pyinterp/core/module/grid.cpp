// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <pybind11/pybind11.h>
#include "pyinterp/bivariate.hpp"
#include "pyinterp/trivariate.hpp"

namespace py = pybind11;
namespace geometry = pyinterp::detail::geometry;

void init_grid(py::module& m) {
  pyinterp::implement_bivariate_interpolator<geometry::EquatorialPoint2D,
                                             double>(m, "2D");
  pyinterp::implement_bivariate_interpolator<geometry::EquatorialPoint3D,
                                             double>(m, "3D");

  pyinterp::implement_grid<double>(m, "Float64");
  pyinterp::implement_grid<float>(m, "Float32");

  pyinterp::implement_bivariate<geometry::EquatorialPoint2D, double, double>(
      m, "Float64");
  pyinterp::implement_bivariate<geometry::EquatorialPoint2D, double, float>(
      m, "Float32");

  pyinterp::implement_trivariate<geometry::EquatorialPoint3D, double, double>(
      m, "Float64");
  pyinterp::implement_trivariate<geometry::EquatorialPoint3D, double, float>(
      m, "Float32");
}
