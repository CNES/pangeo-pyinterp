// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/rtree.hpp"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template <typename Coordinate, typename Type>
static void implement_rtree(py::module& m, const char* const class_name) {
  py::class_<pyinterp::RTree<Coordinate, Type>>(m, class_name, R"__doc__(
RTree spatial index for geodetic scalar values
)__doc__")
      .def(py::init<std::optional<pyinterp::geodetic::System>>(),
           py::arg("system"),
           R"__doc__(
Default constructor

Args:
    system (pyinterp.core.geodetic.System, optional): WGS of the
        coordinate system used to transform equatorial spherical positions
        (longitudes, latitudes, altitude) into ECEF coordinates. If not set
        the geodetic system used is WGS-84.
)__doc__")
      .def(
          "bounds",
          [](const pyinterp::RTree<Coordinate, Type>& self) {
            auto bounds = self.equatorial_bounds();
            if (bounds) {
              return py::make_tuple(
                  py::make_tuple(boost::geometry::get<0>(bounds->min_corner()),
                                 boost::geometry::get<1>(bounds->min_corner()),
                                 boost::geometry::get<2>(bounds->min_corner())),
                  py::make_tuple(
                      boost::geometry::get<0>(bounds->max_corner()),
                      boost::geometry::get<1>(bounds->max_corner()),
                      boost::geometry::get<2>(bounds->max_corner())));
            }
            return py::make_tuple();
          },
          R"__doc__(
Returns the box able to contain all values stored in the container.

Return:
    tuple: A box defined by 3 coordinates able to contain all values stored
    in the container or None if there are no values in the container.
)__doc__")
      .def("__len__", &pyinterp::RTree<Coordinate, Type>::size)
      .def("__bool__",
           [](const pyinterp::RTree<Coordinate, Type>& self) {
             return !self.empty();
           })
      .def("clear", &pyinterp::RTree<Coordinate, Type>::clear,
           "Removes all values stored in the container.")
      .def("packing", &pyinterp::RTree<Coordinate, Type>::packing,
           py::arg("coordinates"), py::arg("values"),
           R"__doc__(
The tree is created using packing algorithm (The old data is erased
before construction.)

Args:
    coordinates (numpy.ndarray): A matrix ``(n, 2)`` to add points defined by
        their longitudes and latitudes or a matrix ``(n, 3)`` to add points
        defined by their longitudes, latitudes and altitudes.
    values (numpy.ndarray): An array of size ``(n)`` containing the values
        associated with the coordinates provided
)__doc__")
      .def("insert", &pyinterp::RTree<Coordinate, Type>::insert,
           py::arg("coordinates"), py::arg("values"),
           R"__doc__(
Insert new data into the search tree.

Args:
    coordinates (numpy.ndarray): A matrix ``(n, 2)`` to add points defined by
        their longitudes and latitudes or a matrix ``(n, 3)`` to add points
        defined by their longitudes, latitudes and altitudes.
    values (numpy.ndarray): An array of size ``(n)`` containing the values
        associated with the coordinates provided
)__doc__")
      .def(
          "query",
          [](const pyinterp::RTree<Coordinate, Type>& self,
             const py::array_t<double>& coordinates, const uint32_t k,
             const bool within, const size_t num_threads) -> py::tuple {
            return self.query(coordinates, k, within, num_threads);
          },
          py::arg("coordinates"), py::arg("k") = 4, py::arg("within") = false,
          py::arg("num_threads") = 0,
          R"__doc__(
Search for the nearest K nearest neighbors of a given point.

Args:
    coordinates (numpy.ndarray): A matrix ``(n, 2)`` to search points defined
        by their longitudes and latitudes or a matrix ``(n, 3)`` to search
        points defined by their longitudes, latitudes and altitudes.
    k (int, optional): The number of nearest neighbors to be used for
        calculating the interpolated value. Defaults to ``4``.
    within (bool, optional): If true, the method ensures that the neighbors
        found are located within the point of interest. Defaults to
        ``false``.
    num_threads (int, optional): The number of threads to use for the
        computation. If 0 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging.
        Defaults to ``0``.
Return:
    tuple: A tuple containing a matrix describing for each provided position,
    the distance, in meters, between the provided position and the found
    neighbors and a matrix containing the value of the different neighbors
    found for all provided positions.
)__doc__")
      .def("inverse_distance_weighting",
           &pyinterp::RTree<Coordinate, Type>::inverse_distance_weighting,
           py::arg("coordinates"),
           py::arg("radius") = std::numeric_limits<Coordinate>::max(),
           py::arg("k") = 4, py::arg("p") = 2, py::arg("within") = true,
           py::arg("num_threads") = 0,
           R"__doc__(
Interpolation of the value at the requested position by inverse distance
weighting method.

Args:
    coordinates (numpy.ndarray): A matrix ``(n, 2)`` to interpolate points
        defined by their longitudes and latitudes or a matrix ``(n, 3)`` to
        interpolate points defined by their longitudes, latitudes and
        altitudes.
    radius (float, optional): The maximum radius of the search (m).
        Defaults The maximum distance between two points.
    k (int, optional): The number of nearest neighbors to be used for
        calculating the interpolated value. Defaults to ``4``.
    p (float, optional): The power parameters. Defaults to ``2``.
    within (bool, optional): If true, the method ensures that the neighbors
        found are located around the point of interest. In other words, this
        parameter ensures that the calculated values will not be extrapolated.
        Defaults to ``true``.
    num_threads (int, optional): The number of threads to use for the
        computation. If 0 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging.
        Defaults to ``0``.
Return:
    tuple: The interpolated value and the number of neighbors used in the
    calculation.
)__doc__")
      .def(py::pickle(
          [](const pyinterp::RTree<Coordinate, Type>& self) {
            return self.getstate();
          },
          [](const py::tuple& state) {
            return pyinterp::RTree<Coordinate, Type>::setstate(state);
          }));
}

void init_rtree(py::module& m) {
  implement_rtree<double, double>(m, "RTreeFloat64");
  implement_rtree<float, float>(m, "RTreeFloat32");
}
