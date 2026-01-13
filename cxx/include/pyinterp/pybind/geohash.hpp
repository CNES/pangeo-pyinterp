// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <nanobind/nanobind.h>

namespace pyinterp::geohash::pybind {

/// @brief Initialize geohash string bindings
/// @param[in,out] m Python module
auto init_string(nanobind::module_& m) -> void;

/// @brief Initialize GeoHash class bindings
/// @param[in,out] m Python module
auto init_class(nanobind::module_& m) -> void;

}  // namespace pyinterp::geohash::pybind

namespace pyinterp::pybind {

/// @brief Initialize geohash bindings
/// @param[in,out] m Python module
void init_geohash(nanobind::module_& m);

}  // namespace pyinterp::pybind
