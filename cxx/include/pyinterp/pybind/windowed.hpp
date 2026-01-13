// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <nanobind/nanobind.h>

namespace pyinterp {
namespace windowed::pybind {

/// @brief Initialize univariate bindings.
/// @param[in,out] m Nanobind module
auto init_univariate(nanobind::module_& m) -> void;

/// @brief Initialize bivariate bindings.
/// @param[in,out] m Nanobind module
auto init_bivariate(nanobind::module_& m) -> void;

/// @brief Initialize trivariate bindings.
/// @param[in,out] m Nanobind module
auto init_trivariate(nanobind::module_& m) -> void;

/// @brief Initialize quadrivariate bindings.
/// @param[in,out] m Nanobind module
auto init_quadrivariate(nanobind::module_& m) -> void;

}  // namespace windowed::pybind

namespace pybind {

/// @brief Initialize windowed bindings.
/// @param[in,out] m Nanobind module
inline auto init_windowed(nanobind::module_& m) -> void {
  windowed::pybind::init_univariate(m);
  windowed::pybind::init_bivariate(m);
  windowed::pybind::init_trivariate(m);
  windowed::pybind::init_quadrivariate(m);
}

}  // namespace pybind
}  // namespace pyinterp
