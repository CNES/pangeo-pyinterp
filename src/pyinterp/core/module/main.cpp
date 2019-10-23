// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <pybind11/pybind11.h>
#include "pyinterp/detail/gsl/error_handler.hpp"

namespace py = pybind11;

extern void init_axis(py::module&);
extern void init_bicubic(py::module&);
extern void init_binning(py::module&);
extern void init_geodetic(py::module&);
extern void init_grid(py::module&);
extern void init_fill(py::module&);
extern void init_rtree(py::module&);

PYBIND11_MODULE(core, m) {
  m.doc() = R"__doc__(
Core module
-----------
)__doc__";

  auto geodetic = m.def_submodule("geodetic", R"__doc__(
Geographic coordinate system
----------------------------
)__doc__");

  auto fill = m.def_submodule("fill", R"__doc__(
Replace undefined values
------------------------
)__doc__");

  pyinterp::detail::gsl::set_error_handler();

  init_axis(m);
  init_binning(m);
  init_grid(m);
  init_bicubic(m);
  init_geodetic(geodetic);
  init_fill(fill);
  init_rtree(m);
}
