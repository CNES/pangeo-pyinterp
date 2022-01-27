// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <pybind11/pybind11.h>

#include "pyinterp/bivariate.hpp"

namespace py = pybind11;
namespace geometry = pyinterp::detail::geometry;

void init_bivariate_interpolator(py::module &m) {
  pyinterp::implement_bivariate_interpolator<geometry::EquatorialPoint2D,
                                             double>(m, "", "2D");
  pyinterp::implement_bivariate_interpolator<geometry::EquatorialPoint3D,
                                             double>(m, "", "3D");
  pyinterp::implement_bivariate_interpolator<geometry::TemporalEquatorial2D,
                                             double>(m, "Temporal", "3D");
}
