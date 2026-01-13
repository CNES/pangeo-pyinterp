// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <nanobind/nanobind.h>

namespace pyinterp::pybind {

/// @brief Initialize geographic coordinate bindings
/// @param[in,out] m Python module
void init_geometry(nanobind::module_& m);

}  // namespace pyinterp::pybind
