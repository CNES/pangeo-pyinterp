// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/fill.hpp"
#include <cctype>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename Type>
void implement_fill_functions(py::module& m, const std::string& suffix) {
  auto function_suffix = suffix;
  function_suffix[0] = std::tolower(function_suffix[0]);

  m.def(("loess_" + function_suffix).c_str(), &pyinterp::fill::loess<Type>,
        py::arg("grid"), py::arg("nx") = 3, py::arg("ny") = 3,
        py::arg("num_threads") = 0,
        (R"__doc__(
Fills undefined values using a locally weighted regression function or
LOESS. The weight function used for LOESS is the tri-cube weight function,
:math:`w(x)=(1-|d|^3)^3`

Args:
    grid (pyinterp.core.Grid2D)__doc__" +
         suffix +
         R"__doc__() : Grid function on a uniform 2-dimensional grid to be
        filled.
    nx (int, optional): Number of points of the half-window to be taken into
        account along the X-axis. Defaults to ``3``.
    ny (int, optional): Number of points of the half-window to be taken into
        account along the Y-axis. Defaults to ``3``.
    num_threads (int, optional): The number of threads to use for the
        computation. If 0 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging.
        Defaults to ``0``.

Return:
    numpy.ndarray: the grid will have all the NaN filled with extrapolated
    values.
)__doc__")
            .c_str());

  m.def(("poisson_" + function_suffix).c_str(), &pyinterp::fill::poisson<Type>,
        py::arg("grid"), py::arg("first_guess") = pyinterp::fill::kZonalAverage,
        py::arg("is_circle") = true, py::arg("max_iterations") = 200,
        py::arg("epsilon") = 1e-4, py::arg("relaxation") = 0.6,
        py::arg("num_thread") = 0,
        R"__doc__(
Replaces all undefined values (NaN) in a grid with values derived from solving
Poisson's equation via relaxation.

Args:
    grid (numpy.ndarray): Grid function on a uniform 2-dimensional grid to be
        filled.
    is_circle (bool, optional): True if the X axis of the grid defines a
        circle. Defaults to ``True``.
    max_iterations (int, optional): Maximum number of iterations to be used by
        relaxation. Defaults to ``200``.
    epsilon (float, optional): Tolerance for ending relaxation before the
        maximum number of iterations limit. Defaults to ``1e-4``.
    relaxation (float, opional): Relaxation constant.
        Usually, ``0.45 <= relaxation <= 0.6``. Defaults to ``0.6``.
    num_threads (int, optional): The number of threads to use for the
        computation. If 0 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging.
        Defaults to ``0``.

Return:
    tuple: the number of iterations performed and the maximum residual value.

.. note::

    The code was written from the routine code `fish_grid_fill
    <https://www.ncl.ucar.edu/Document/Functions/Built-in/poisson_grid_fill.shtml>`_
    available in NCAR Common Language.
)__doc__",
        py::call_guard<py::gil_scoped_release>());
}

void init_fill(py::module& m) {
  py::enum_<pyinterp::fill::FirstGuess>(
      m, "FirstGuess", "Type of first guess grid to solve Poisson's equation.")
      .value("Zero", pyinterp::fill::kZero, "Use 0.0 as an initial guess")
      .value("ZonalAverage", pyinterp::fill::kZonalAverage,
             "Use zonal average in x direction");

  implement_fill_functions<double>(m, "Float64");
  implement_fill_functions<float>(m, "Float32");
}