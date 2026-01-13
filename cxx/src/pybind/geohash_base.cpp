// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/pybind/geohash.hpp"

namespace pyinterp::pybind {

void init_geohash(nanobind::module_& m) {
  auto geohash_module = m.def_submodule("geohash", "GeoHash encoding/decoding");
  geohash::pybind::init_class(geohash_module);
  geohash::pybind::init_string(geohash_module);
}

}  // namespace pyinterp::pybind
