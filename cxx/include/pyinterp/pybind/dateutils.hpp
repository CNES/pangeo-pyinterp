// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <nanobind/nanobind.h>

namespace pyinterp::pybind {

/// @brief Bind dateutils functions for datetime manipulation
/// @param[in,out] m Python module
auto init_dateutils(nanobind::module_& m) -> void;

}  // namespace pyinterp::pybind
