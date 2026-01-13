// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <nanobind/nanobind.h>

namespace pyinterp::pybind {

/// @brief Initialize fill module
/// @param[in,out] m The nanobind module
auto init_fill(nanobind::module_& m) -> void;

}  // namespace pyinterp::pybind
