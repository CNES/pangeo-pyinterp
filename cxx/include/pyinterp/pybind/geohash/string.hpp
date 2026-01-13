// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <nanobind/nanobind.h>

namespace pyinterp::geohash::pybind {

/// @brief Initialize geohash string module bindings
/// @param m Nanobind module to bind to
auto init_string(nanobind::module_& m) -> void;

}  // namespace pyinterp::geohash::pybind
