// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <pybind11/pybind11.h>

#include "pyinterp/detail/gsl/error_handler.hpp"

namespace py = pybind11;

extern void init_axis(py::module &);
extern void init_bicubic(py::module &);
extern void init_binning(py::module &);
extern void init_bivariate_interpolator(py::module &);
extern void init_bivariate(py::module &);
extern void init_dateutils(py::module &);
extern void init_descriptive_statistics(py::module &);
extern void init_enum(py::module &, py::module &);
extern void init_interpolate1d(py::module &);
extern void init_fill(py::module &);
extern void init_geodetic(py::module &);
extern void init_geohash_class(py::module &);
extern void init_geohash_int64(py::module &);
extern void init_geohash_string(py::module &);
extern void init_grid(py::module &);
extern void init_histogram2d(py::module &);
extern void init_quadrivariate(py::module &);
extern void init_rtree(py::module &);
extern void init_spline(py::module &);
extern void init_streaming_histogram(py::module &);
extern void init_trivariate(py::module &);

static void init_geohash(py::module &m) {
  auto int64 = m.def_submodule("int64", R"__doc__(
GeoHash encoded as integer 64 bits
----------------------------------
)__doc__");

  init_geohash_int64(int64);
  init_geohash_string(m);
}

PYBIND11_MODULE(core, m) {  // NOLINT
  m.doc() = R"__doc__(
Core module
-----------
)__doc__";

  auto dateutils = m.def_submodule("dateutils", R"__doc__(
numpy datetime utilities
------------------------
)__doc__");

  auto geodetic = m.def_submodule("geodetic", R"__doc__(
Geographic coordinate system
----------------------------
)__doc__");

  auto geohash = m.def_submodule("geohash", R"__doc__(
GeoHash encoding/decoding
-------------------------
)__doc__");

  auto fill = m.def_submodule("fill", R"__doc__(
Replace undefined values
------------------------
)__doc__");

  pyinterp::detail::gsl::set_error_handler();

  init_enum(m, fill);

  init_dateutils(dateutils);
  init_geodetic(geodetic);

  init_axis(m);
  init_binning(m);
  init_histogram2d(m);
  init_bivariate_interpolator(m);
  init_descriptive_statistics(m);
  init_interpolate1d(m);
  init_grid(m);
  init_bivariate(m);
  init_trivariate(m);
  init_quadrivariate(m);
  init_bicubic(m);
  init_fill(fill);
  init_rtree(m);
  init_streaming_histogram(m);

  // geohash
  init_geohash(geohash);
  init_geohash_class(m);
}
