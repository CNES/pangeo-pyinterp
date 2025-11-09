// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/fill.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
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
Fill undefined values using locally weighted regression (LOESS).

The weight function used for LOESS is the tri-cube weight function,
:math:`w(x)=(1-|d|^3)^3`.

Args:
    grid: Grid function on a uniform 2-dimensional grid to be filled.
    nx: Number of points of the half-window to be taken into account along the
        X-axis. Defaults to ``3``.
    ny: Number of points of the half-window to be taken into account along the
        Y-axis. Defaults to ``3``.
    value_type: Type of values processed by the filter.
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
Replace all undefined values in a grid using Gauss-Seidel method.

Uses the Gauss-Seidel method by relaxation to fill all NaN values.

Args:
    grid: Grid function on a uniform 2-dimensional grid to be filled.
    first_guess: Method to use for the first guess.
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

  m.def(("multigrid_" + function_suffix).c_str(),
        &pyinterp::fill::multigrid<Type>, py::arg("grid"),
        py::arg("first_guess") = pyinterp::fill::kZonalAverage,
        py::arg("is_circle") = true, py::arg("max_iterations") = 500,
        py::arg("epsilon") = 1e-4, py::arg("pre_smooth") = 2,
        py::arg("post_smooth") = 2, py::arg("num_threads") = 0,
        R"__doc__(
Replace all undefined values (NaN) in a grid using multigrid method.

Args:
    grid: The grid to be processed
    first_guess: Method to use for the first guess.
    is_circle: If true, uses periodic boundaries along the X-axis. If false,
        uses reflective boundaries.
    max_iterations: Maximum number of iterations.
    epsilon: Tolerance for ending relaxation.
    pre_smooth: Number of pre-smoothing iterations per level (default: 2).
    post_smooth: Number of post-smoothing iterations per level (default: 2).
    num_threads: The number of threads to use for the computation.

Returns:
    A tuple containing the number of iterations performed and the maximum
    residual value.
)__doc__",
        py::call_guard<py::gil_scoped_release>());

  m.def(("fft_inpaint_" + function_suffix).c_str(),
        &pyinterp::fill::fft_inpaint<Type>, py::arg("grid"),
        py::arg("first_guess") = pyinterp::fill::kZonalAverage,
        py::arg("is_circle") = true, py::arg("max_iterations") = 500,
        py::arg("epsilon") = 1e-4, py::arg("sigma") = 10.0,
        py::arg("num_threads") = 0,
        R"__doc__(
Replace all undefined values (NaN) in a grid using spectral in-painting.

Args:
    grid: The grid to be processed
    first_guess: Method to use for the first guess.
    is_circle: If true, uses a Fast Fourier Transform (FFT) assuming periodic
        boundaries. If false, uses a Discrete Cosine Transform (DCT) assuming
        reflective boundaries.
    max_iterations: Maximum number of iterations.
    epsilon: Tolerance for ending relaxation.
    sigma: Standard deviation of the Gaussian low-pass filter in pixel units.
        Controls the smoothness of the fill.
    num_threads: The number of threads to use for the computation.

Returns:
    A tuple containing the number of iterations performed and the maximum
    residual value.
)__doc__",
        py::call_guard<py::gil_scoped_release>());

  m.def(("matrix_" + function_suffix).c_str(), &pyinterp::fill::matrix<Type>,
        py::arg("x"),
        py::arg("fill_value") = std::numeric_limits<Type>::quiet_NaN(),
        R"__doc__(
Fill gaps in a matrix with interpolated values.

Args:
    x: Data to be interpolated.
    fill_value: Value used to detect gaps in the matrix. Defaults to
        ``NaN``.
)__doc__",
        py::call_guard<py::gil_scoped_release>());

  m.def(("vector_" + function_suffix).c_str(), &pyinterp::fill::vector<Type>,
        py::arg("x"),
        py::arg("fill_value") = std::numeric_limits<Type>::quiet_NaN(),
        R"__doc__(
Fill gaps in a vector with interpolated values.

Args:
    x: Data to be interpolated.
    fill_value: Value used to detect gaps in the vector. Defaults to
        ``NaN``.
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
Fill undefined values using locally weighted regression (LOESS).

The weight function used for LOESS is the tri-cube weight function,
:math:`w(x)=(1-|d|^3)^3`.

Args:
    grid: Grid containing the values to be filtered.
    nx: Number of points of the half-window to be taken into
        account along the X-axis. Defaults to ``3``.
    ny: Number of points of the half-window to be taken into
        account along the Y-axis. Defaults to ``3``.
    value_type: Type of values processed by the filter.
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

  m.def("vector_int64", &pyinterp::fill::vector<int64_t>, py::arg("x"),
        py::arg("fill_value"),
        R"__doc__(
Fill gaps in a vector with interpolated values.

Args:
    x: Vector to be filled.
    fill_value: Value used to detect gaps in the vector.
)__doc__",
        py::call_guard<py::gil_scoped_release>());
}
