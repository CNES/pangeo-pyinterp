// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/pybind/fill.hpp"

#include "pyinterp/pybind/fill/fft_inpaint.hpp"
#include "pyinterp/pybind/fill/gauss_seidel.hpp"
#include "pyinterp/pybind/fill/loess.hpp"
#include "pyinterp/pybind/fill/matrix.hpp"
#include "pyinterp/pybind/fill/multigrid.hpp"

namespace nb = nanobind;

namespace pyinterp::pybind {

auto init_fill(nb::module_& m) -> void {
  auto fill = m.def_submodule("fill", "Methods to fill missing data.");

  fill::pybind::bind_fft_inpaint(fill);
  fill::pybind::bind_gauss_seidel(fill);
  fill::pybind::bind_loess(fill);
  fill::pybind::bind_matrix(fill);
  fill::pybind::bind_multigrid(fill);
}

}  // namespace pyinterp::pybind
