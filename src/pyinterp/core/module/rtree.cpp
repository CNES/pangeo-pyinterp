// Copyright (c) 2021 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/rtree.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

namespace py = pybind11;

template <size_t N>
static auto coordinates_help() -> std::string {
  auto ss = std::stringstream();
  if (N == 3) {
    ss << "coordinates (numpy.ndarray): a matrix ``(n, 3)`` where ``n`` is\n"
          "        the number of observations and 3 is the number of\n"
          "        coordinates in order: longitude and latitude in degrees\n"
          "        and altitude in meters. If the shape of the matrix is\n"
          "        ``(n, 2)`` then the method considers the altitude\n"
          "        constant and equal to zero.\n";
  } else {
    ss << "coordinates (numpy.ndarray): a matrix ``(n, " << N
       << ")`` where ``n`` is\n"
          "        the number of observations and 3 is the number of\n"
          "        coordinates in order: longitude and latitude in degrees,\n"
          "        altitude in meters and then the other coordinates\n"
          "        defined in Euclidean space. If the shape of the matrix\n"
          "        is ``(n, "
       << N - 1
       << ")`` then the method considers the\n"
          "        altitude constant and equal to zero.";
  }
  return ss.str();
}

template <size_t N>
static auto class_name(const char* const suffix) -> std::string {
  return "RTree" + std::to_string(N) + "D" + suffix;
}

template <typename CoordinateType, typename Type, size_t N>
static void implement_rtree(py::module& m, const char* const suffix) {
  py::class_<pyinterp::RTree<CoordinateType, Type, N>>(
      m, class_name<N>(suffix).c_str(),
      R"__doc__(
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
      .def("bounds",
           &pyinterp::RTree<CoordinateType, Type, N>::equatorial_bounds,
           R"__doc__(
Returns the box able to contain all values stored in the container.

Returns:
    tuple: A tuple that contains the coordinates of the minimum and
    maximum corners of the box able to contain all values stored in the
    container or an empty tuple if there are no values in the container.
)__doc__")
      .def("__len__", &pyinterp::RTree<CoordinateType, Type, N>::size,
           "Called to implement the built-in function ``len()``")
      .def(
          "__bool__",
          [](const pyinterp::RTree<CoordinateType, Type, N>& self) {
            return !self.empty();
          },
          "Called to implement truth value testing and the built-in operation "
          "``bool()``.")
      .def("clear", &pyinterp::RTree<CoordinateType, Type, N>::clear,
           "Removes all values stored in the container.")
      .def("packing", &pyinterp::RTree<CoordinateType, Type, N>::packing,
           py::arg("coordinates"), py::arg("values"),
           (R"__doc__(
The tree is created using packing algorithm (The old data is erased
before construction.)

Args:
    )__doc__" +
            coordinates_help<N>() + R"__doc__(
    values (numpy.ndarray): An array of size ``(n)`` containing the values
        associated with the coordinates provided.
)__doc__")
               .c_str(),
           py::call_guard<py::gil_scoped_release>())
      .def("insert", &pyinterp::RTree<CoordinateType, Type, N>::insert,
           py::arg("coordinates"), py::arg("values"),
           (R"__doc__(
Insert new data into the search tree.

Args:
    )__doc__" +
            coordinates_help<N>() + R"__doc__(
    values (numpy.ndarray): An array of size ``(n)`` containing the values
        associated with the coordinates provided.
)__doc__")
               .c_str(),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "query",
          [](const pyinterp::RTree<CoordinateType, Type, N>& self,
             const py::array_t<CoordinateType>& coordinates, const uint32_t k,
             const bool within, const size_t num_threads) -> py::tuple {
            return self.query(coordinates, k, within, num_threads);
          },
          py::arg("coordinates"), py::arg("k") = 4, py::arg("within") = false,
          py::arg("num_threads") = 0,
          (R"__doc__(
Search for the nearest K nearest neighbors of a given point.

Args:
    )__doc__" +
           coordinates_help<N>() + R"__doc__(
    k (int, optional): The number of nearest neighbors to be used for
        calculating the interpolated value. Defaults to ``4``.
    within (bool, optional): If true, the method ensures that the neighbors
        found are located within the point of interest. Defaults to
        ``false``.
    num_threads (int, optional): The number of threads to use for the
        computation. If 0 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging.
        Defaults to ``0``.
Returns:
    tuple: A tuple containing a matrix describing for each provided position,
    the distance, in meters, between the provided position and the found
    neighbors and a matrix containing the value of the different neighbors
    found for all provided positions.
)__doc__")
              .c_str())
      .def(
          "inverse_distance_weighting",
          &pyinterp::RTree<CoordinateType, Type, N>::inverse_distance_weighting,
          py::arg("coordinates"), py::arg("radius"), py::arg("k") = 9,
          py::arg("p") = 2, py::arg("within") = true,
          py::arg("num_threads") = 0,
          (R"__doc__(
Interpolation of the value at the requested position by inverse distance
weighting method.

Args:
    )__doc__" +
           coordinates_help<N>() + R"__doc__(
    radius (float, optional): The maximum radius of the search (m).
        Defaults The maximum distance between two points.
    k (int, optional): The number of nearest neighbors to be used for
        calculating the interpolated value. Defaults to ``9``.
    p (float, optional): The power parameters. Defaults to ``2``.
    within (bool, optional): If true, the method ensures that the neighbors
        found are located around the point of interest. In other words, this
        parameter ensures that the calculated values will not be extrapolated.
        Defaults to ``true``.
    num_threads (int, optional): The number of threads to use for the
        computation. If 0 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging.
        Defaults to ``0``.
Returns:
    tuple: The interpolated value and the number of neighbors used in the
    calculation.
)__doc__")
              .c_str())
      .def(
          "radial_basis_function",
          &pyinterp::RTree<CoordinateType, Type, N>::radial_basis_function,
          py::arg("coordinates"), py::arg("radius"), py::arg("k") = 9,
          py::arg("rbf") = pyinterp::RadialBasisFunction::Multiquadric,
          py::arg("epsilon") = std::optional<
              typename pyinterp::RTree<CoordinateType, Type, N>::promotion_t>(),
          py::arg("smooth") = 0, py::arg("within") = true,
          py::arg("num_threads") = 0,
          (R"__doc__(
Interpolation of the value at the requested position by radial basis function
interpolation.

Args:
    )__doc__" +
           coordinates_help<N>() + R"__doc__(
    radius (float, optional): The maximum radius of the search (m).
        Default to the largest value that can be represented on a float.
    k (int, optional): The number of nearest neighbors to be used for
        calculating the interpolated value. Defaults to ``9``.
    rbf (pyinterp.core.RadialBasisFunction, optional): The radial basis
        function, based on the radius, r, given by the distance between points.
        Default to :py:attr:`pyinterp.core.RadialBasisFunction.Multiquadric`.
    epsilon (float, optional): Adjustable constant for gaussian or
        multiquadrics functions. Default to the average distance between nodes.
    smooth (float, optional): Values greater than zero increase the smoothness
        of the approximation.
    within (bool, optional): If true, the method ensures that the neighbors
        found are located around the point of interest. Defaults to ``true``.
    num_threads (int, optional): The number of threads to use for the
        computation. If 0 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging.
        Defaults to ``0``.
Returns:
    tuple: The interpolated value and the number of neighbors used for the
    calculation.
)__doc__")
              .c_str())
      .def(py::pickle(
          [](const pyinterp::RTree<CoordinateType, Type, N>& self) {
            return self.getstate();
          },
          [](const py::tuple& state) {
            return pyinterp::RTree<CoordinateType, Type, N>::setstate(state);
          }));
}

void init_rtree(py::module& m) {
  py::enum_<pyinterp::RadialBasisFunction>(m, "RadialBasisFunction",
                                           "Radial basis functions")
      .value("Cubic", pyinterp::RadialBasisFunction::Cubic,
             ":math:`\\varphi(r) = r^3`")
      .value("Gaussian", pyinterp::RadialBasisFunction::Gaussian,
             R"(:math:`\varphi(r) = e^{-(\dfrac{r}{\varepsilon})^2}`)")
      .value("InverseMultiquadric",
             pyinterp::RadialBasisFunction::InverseMultiquadric,
             ":math:`\\varphi(r) = \\dfrac{1}"
             "{\\sqrt{1+(\\dfrac{r}{\\varepsilon})^2}}`")
      .value("Linear", pyinterp::RadialBasisFunction::Linear,
             ":math:`\\varphi(r) = r`")
      .value("Multiquadric", pyinterp::RadialBasisFunction::Multiquadric,
             R"(:math:`\varphi(r) = \sqrt{1+(\dfrac{r}{\varepsilon}^2})`)")
      .value("ThinPlate", pyinterp::RadialBasisFunction::ThinPlate,
             ":math:`\\varphi(r) = r^2 \\ln(r)`.");

  implement_rtree<double, double, 3>(m, "Float64");
  implement_rtree<float, float, 3>(m, "Float32");
}
