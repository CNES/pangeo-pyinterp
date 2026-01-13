// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <nanobind/nanobind.h>

namespace pyinterp::geometry::pybind {

/// @brief Initialize the satellite submodule.
/// @param[in,out] m The parent nanobind module.
auto init_satellite(nanobind::module_& m) -> void;

}  // namespace pyinterp::geometry::pybind
