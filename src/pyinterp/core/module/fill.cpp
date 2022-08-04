// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/fill.hpp"

#include <pybind11/pybind11.h>

#include <cctype>

namespace py = pybind11;

template <class T>
using Grid3D = pyinterp::Grid3D<T, double>;

template <class T>
using Grid4D = pyinterp::Grid4D<T, double>;

template <class T>
using TemporalGrid3D = pyinterp::Grid3D<T, int64_t>;

template <class T>
using TemporalGrid4D = pyinterp::Grid4D<T, int64_t>;

template <typename Type>
void implement_fill_functions(py::module &m, const std::string &suffix) {
  auto function_suffix = suffix;
  function_suffix[0] = static_cast<char>(std::tolower(function_suffix[0]));

  m.def(("loess_" + function_suffix).c_str(), &pyinterp::fill::loess<Type>,
        py::arg("grid"), py::arg("nx") = 3, py::arg("ny") = 3,
        py::arg("value_type") = pyinterp::fill::kUndefined,
        py::arg("num_threads") = 0, (R"__doc__(
Fills undefined values using a locally weighted regression function or
LOESS. The weight function used for LOESS is the tri-cube weight function,
:math:`w(x)=(1-|d|^3)^3`.

Args:
    grid: Grid function on a uniform 2-dimensional grid to be filled.
    nx: Number of points of the half-window to be taken into account along the
        X-axis. Defaults to ``3``.
    ny: Number of points of the half-window to be taken into account along the
        Y-axis. Defaults to ``3``.
    value_type: Type of values processed by the filter
    num_threads: The number of threads to use for the computation. If 0 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Defaults to ``0``.

Returns:
    The grid will have all the NaN filled with extrapolated values.
)__doc__"));

  m.def(("gauss_seidel_" + function_suffix).c_str(),
        &pyinterp::fill::gauss_seidel<Type>, py::arg("grid"),
        py::arg("first_guess") = pyinterp::fill::kZonalAverage,
        py::arg("is_circle") = true, py::arg("max_iterations") = 2000,
        py::arg("epsilon") = 1e-4, py::arg("relaxation") = 1.0,
        py::arg("num_threads") = 0,
        R"__doc__(
Replaces all undefined values (NaN) in a grid using the Gauss-Seidel
method by relaxation.

Args:
    grid: Grid function on a uniform 2-dimensional grid to be filled.
    is_circle: True if the X axis of the grid defines a circle. Defaults to
        ``True``.
    max_iterations: Maximum number of iterations to be used by relaxation.
        Defaults to ``2000``.
    epsilon: Tolerance for ending relaxation before the maximum number of
        iterations limit. Defaults to ``1e-4``.
    relaxation: Relaxation constant. Defaults to ``1``.
    num_threads: The number of threads to use for the computation. If 0 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Defaults to ``0``.

Returns:
    The number of iterations performed and the maximum residual value.
)__doc__",
        py::call_guard<py::gil_scoped_release>());
}

template <typename Type, typename AxisType, typename GridType>
void implement_loess(py::module &m, const std::string &prefix,
                     const std::string &suffix) {
  auto function_suffix = suffix;
  function_suffix[0] = static_cast<char>(std::tolower(function_suffix[0]));

  m.def(
      ("loess_" + function_suffix).c_str(),
      [](const GridType &grid, const uint32_t nx, const uint32_t ny,
         const pyinterp::fill::ValueType value_type, const size_t num_threads) {
        return pyinterp::fill::loess<Type, AxisType>(grid, nx, ny, value_type,
                                                     num_threads);
      },
      py::arg("grid"), py::arg("nx") = 3, py::arg("ny") = 3,
      py::arg("value_type") = pyinterp::fill::kUndefined,
      py::arg("num_threads") = 0,
      R"__doc__(
Fills undefined values using a locally weighted regression function or
LOESS. The weight function used for LOESS is the tri-cube weight function,
:math:`w(x)=(1-|d|^3)^3`.

Args:
    grid: Grid containing the values to be filtered.
    nx: Number of points of the half-window to be taken into
        account along the X-axis. Defaults to ``3``.
    ny: Number of points of the half-window to be taken into
        account along the Y-axis. Defaults to ``3``.
    value_type: Type of values processed by the filter
    num_threads: The number of threads to use for the computation. If 0 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Defaults to ``0``.

Returns:
    The grid will have all the NaN filled with extrapolated values.
)__doc__");
}

void init_fill(py::module &m) {
  implement_fill_functions<double>(m, "Float64");
  implement_fill_functions<float>(m, "Float32");

  implement_loess<double, double, Grid3D<double>>(m, "", "Float64");
  implement_loess<float, double, Grid3D<float>>(m, "", "Float32");
  implement_loess<double, int64_t, TemporalGrid3D<double>>(m, "Temporal",
                                                           "Float64");
  implement_loess<float, int64_t, TemporalGrid3D<float>>(m, "Temporal",
                                                         "Float32");

  implement_loess<double, double, Grid4D<double>>(m, "", "Float64");
  implement_loess<float, double, Grid4D<float>>(m, "", "Float32");
  implement_loess<double, int64_t, TemporalGrid4D<double>>(m, "Temporal",
                                                           "Float64");
  implement_loess<float, int64_t, TemporalGrid4D<float>>(m, "Temporal",
                                                         "Float32");
}
