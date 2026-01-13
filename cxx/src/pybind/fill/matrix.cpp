// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/fill/matrix.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include <concepts>

#include "pyinterp/pybind/fill/matrix.hpp"

namespace nb = nanobind;

namespace pyinterp::fill::pybind {

constexpr const char* const kMatrixDoc = R"doc(
Fills in the gaps between defined values in a matrix with interpolated
values.

Args:
    x: The data to be processed.
    fill_value: Value to use for missing data.
)doc";

constexpr const char* const kVectorDoc = R"doc(
Fills in the gaps between defined values in a vector with interpolated
values.

Args:
    array: Array of dates.
    fill_value: Value to use for missing data.
)doc";

template <std::floating_point T>
auto bind_matrix(nb::module_& m) -> void {
  m.def("matrix", &pyinterp::fill::matrix<T>, nb::arg("x"),
        nb::arg("fill_value"), kMatrixDoc,
        nanobind::call_guard<nanobind::gil_scoped_release>());

  m.def("vector", &pyinterp::fill::vector<T>, nb::arg("array"),
        nb::arg("fill_value"), kVectorDoc,
        nanobind::call_guard<nanobind::gil_scoped_release>());
}

void bind_matrix(nb::module_& m) {
  bind_matrix<float>(m);
  bind_matrix<double>(m);
}

}  // namespace pyinterp::fill::pybind
