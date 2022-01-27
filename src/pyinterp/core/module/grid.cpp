// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/grid.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_grid(py::module &m) {
  pyinterp::implement_grid<double>(m, "Float64");
  pyinterp::implement_grid<float>(m, "Float32");
  pyinterp::implement_grid<int8_t>(m, "Int8");
}
