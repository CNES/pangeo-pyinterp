// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/quadrivariate.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace geometry = pyinterp::detail::geometry;

void init_quadrivariate(py::module &m) {
  pyinterp::implement_quadrivariate<geometry::EquatorialPoint3D, double, double,
                                    double>(m, "", "Float64");
  pyinterp::implement_quadrivariate<geometry::EquatorialPoint3D, double, double,
                                    float>(m, "", "Float32");
  pyinterp::implement_quadrivariate<geometry::TemporalEquatorial2D, double,
                                    int64_t, double>(m, "Temporal", "Float64");
  pyinterp::implement_quadrivariate<geometry::TemporalEquatorial2D, double,
                                    int64_t, float>(m, "Temporal", "Float32");
}
