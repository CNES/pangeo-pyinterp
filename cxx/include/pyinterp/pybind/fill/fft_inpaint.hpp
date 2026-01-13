// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <nanobind/nanobind.h>

namespace pyinterp::fill::pybind {

/// Bind FFT Inpaint functions to Python module.
/// @param[in,out] m Python module
void bind_fft_inpaint(nanobind::module_& m);

}  // namespace pyinterp::fill::pybind
