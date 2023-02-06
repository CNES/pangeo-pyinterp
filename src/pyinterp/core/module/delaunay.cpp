// Copyright (c) 2023 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/delaunay.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_delaunay(py::module &m) {
  py::class_<pyinterp::Delaunay>(
      m, "Delaunay",
      "Delaunay(self, coordinates: numpy.ndarray[numpy.float64],"
      " values: numpy.ndarray[numpy.float64],"
      " spheroid: Optional[pyinterp.geodetic.Spheroid] = None)"
      R"__doc__(

Natural neighbor interpolation.

This class is used to perform natural neighbor interpolation on a set of
points. The interpolation is performed using the CGAL library.

Args:
    coordinates: Coordinates of the points.
    values: Values of the points.
    spheroid: Spheroid used to convert the coordinates from geodetic to
        cartesian. Default is the WGS84 spheroid.

Example:
    >>> import numpy as np
    >>> from pyinterp import Delaunay
    >>> coordinates = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
    >>> values = np.array([0, 1, 2, 3], dtype=np.float64)
    >>> delaunay = Delaunay(coordinates, values)
    >>> delaunay.natural_neighbor_interpolation([[0.5, 0.5]])
    array([1.5])
)__doc__")
      .def(py::init<
               const pyinterp::NDArray &, const py::array_t<double> &,
               const std::optional<pyinterp::detail::geodetic::Spheroid> &>(),
           py::arg("coordinates"), py::arg("values"),
           py::arg("spheroid") = std::nullopt)
      .def("natural_neighbor_interpolation",
           &pyinterp::Delaunay::natural_neighbor_interpolation,
           py::arg("coordinates"), py::arg("num_threads") = 0, R"__doc__(
Natural neighbor interpolation.

Args:
    coordinates: Coordinates of the points to interpolate.
    num_threads: Number of threads to use for the interpolation.

Returns:
    Interpolated values.
)__doc__");
}
