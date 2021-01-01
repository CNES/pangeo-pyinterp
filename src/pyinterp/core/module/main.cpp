// Copyright (c) 2021 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <pybind11/pybind11.h>

#include "pyinterp/detail/gsl/error_handler.hpp"

namespace py = pybind11;

extern void init_axis(py::module&);
extern void init_bicubic(py::module&);
extern void init_binning(py::module&);
extern void init_bivariate_interpolator(py::module&);
extern void init_bivariate(py::module&);
extern void init_fill(py::module&);
extern void init_geodetic(py::module&);
extern void init_geohash_int64(py::module&);
extern void init_geohash_string(py::module&);
extern void init_geohash_utility(py::module&);
extern void init_grid(py::module&);
extern void init_quadrivariate(py::module&);
extern void init_rtree(py::module&);
extern void init_spline(py::module&);
extern void init_storage_marshaller(py::module&);
extern void init_storage_unqlite(py::module&);
extern void init_trivariate(py::module&);

static void init_geohash(py::module& m) {
  auto int64 = m.def_submodule("int64", R"__doc__(
GeoHash encoded as integer 64 bits
----------------------------------
)__doc__");

  init_geohash_int64(int64);
  init_geohash_string(m);
  init_geohash_utility(m);
}

static void init_storage(py::module& m) {
  init_storage_marshaller(m);
  init_storage_unqlite(m);
}

PYBIND11_MODULE(core, m) {
  m.doc() = R"__doc__(
Core module
-----------
)__doc__";

  auto geodetic = m.def_submodule("geodetic", R"__doc__(
Geographic coordinate system
----------------------------
)__doc__");

  auto geohash = m.def_submodule("geohash", R"__doc__(
Geohash encoding/decoding
-------------------------
)__doc__");

  auto fill = m.def_submodule("fill", R"__doc__(
Replace undefined values
------------------------
)__doc__");

  auto storage = m.def_submodule("storage", R"__doc__(

Index storage support
---------------------
)__doc__");

  pyinterp::detail::gsl::set_error_handler();

  init_axis(m);
  init_binning(m);
  init_bivariate_interpolator(m);
  init_grid(m);
  init_bivariate(m);
  init_trivariate(m);
  init_quadrivariate(m);
  init_bicubic(m);
  init_geodetic(geodetic);
  init_fill(fill);
  init_rtree(m);

  // geohash
  init_geohash(geohash);
  init_storage(storage);
}
