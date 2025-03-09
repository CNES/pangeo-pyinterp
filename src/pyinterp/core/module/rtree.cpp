// Copyright (c) 2025 CNES
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
  static_assert(N == 3, "The dimension must be 3.");
  ss << "coordinates: a matrix of shape ``(n, 3)``, where ``n`` is the\n"
        "        number of observations and 3 represents the coordinates in\n"
        "        the order: x, y, and z.\n"
        "        If the matrix shape is ``(n, 2)``, the z-coordinate is\n"
        "        assumed to be zero.\n"
        "        The coordinates (x, y, z) are in the Cartesian coordinate\n"
        "        system (ECEF) if the instance is configured to use this\n"
        "        system (ecef keyword set to True during construction).\n"
        "        Otherwise, the coordinates are in the geodetic system\n"
        "        (longitude, latitude, and altitude) in degrees, degrees,\n"
        "        and meters, respectively.";
  return ss.str();
}

template <size_t N>
static auto class_name(const char *const suffix) -> std::string {
  return "RTree" + std::to_string(N) + "D" + suffix;
}

template <typename Point, typename Type>
static void implement_rtree(py::module &m, const char *const suffix) {
  using RTree = pyinterp::RTree<Point, Type>;
  using dimension_t = typename RTree::dimension_t;

  py::class_<RTree>(
      m, class_name<dimension_t::value>(suffix).c_str(),
      (class_name<dimension_t::value>(suffix) +
       "(self, spheroid: Optional[pyinterp.core.geodetic.Spheroid] = None)" +
       R"__doc__(

RTree spatial index for geodetic scalar values

Args:
    spheroid: WGS of the coordinate system used to transform equatorial spherical
        positions (longitudes, latitudes, altitude) into ECEF coordinates. If
        not set the geodetic system used is WGS-84.
    ecef: If true, the coordinates managed by this instance are always in the
        Cartesian coordinate system (ECEF). If true, no conversion to the LLA
        system is performed. Defaults to ``False``.

Raises:
    ValueError: If a Cartesian coordinate system (ECEF) is desired, there is no
        need to use a geodetic system for LLA to ECEF conversion. Please set
        either the spheroid or ecef parameter, but not both.
)__doc__")
          .c_str())
      .def(py::init<std::optional<pyinterp::geodetic::Spheroid>, bool>(),
           py::arg("spheroid") = std::nullopt, py::arg("ecef") = false)
      .def("bounds", &RTree::equatorial_bounds,
           R"__doc__(
Returns the box able to contain all values stored in the container.

Returns:
    tuple: A tuple that contains the coordinates of the minimum and
    maximum corners of the box able to contain all values stored in the
    container or an empty tuple if there are no values in the container.
  )__doc__")
      .def(
          "__copy__", [](const RTree &self) { return RTree(self); },
          "Implements the shallow copy operation.",
          py::call_guard<py::gil_scoped_release>())
      .def("__len__", &RTree::size,
           "Called to implement the built-in function ``len()``")
      .def(
          "__bool__", [](const RTree &self) { return !self.empty(); },
          "Called to implement truth value testing and the built-in operation "
          "``bool()``.")
      .def("clear", &RTree::clear,
           "Removes all values stored in the container.")
      .def("packing", &RTree::packing, py::arg("coordinates"),
           py::arg("values"),
           (R"__doc__(
The tree is created using packing algorithm (The old data is erased
before construction.)

Args:
    )__doc__" +
            coordinates_help<dimension_t::value>() + R"__doc__(
    values: An array of size ``(n)`` containing the values associated with the
        coordinates provided.
)__doc__")
               .c_str())
      .def("insert", &RTree::insert, py::arg("coordinates"), py::arg("values"),
           (R"__doc__(
Insert new data into the search tree.

Args:
    )__doc__" +
            coordinates_help<dimension_t::value>() + R"__doc__(
    values: An array of size ``(n)`` containing the values associated with the
        coordinates provided.
)__doc__")
               .c_str())
      .def(
          "query",
          [](const RTree &self, const py::array_t<Type> &coordinates,
             const uint32_t k, const bool within,
             const size_t num_threads) -> py::tuple {
            return self.query(coordinates, k, within, num_threads);
          },
          py::arg("coordinates"), py::arg("k") = 4, py::arg("within") = false,
          py::arg("num_threads") = 0,
          (R"__doc__(
Search for the nearest K nearest neighbors of a given point.

Args:
    )__doc__" +
           coordinates_help<dimension_t::value>() + R"__doc__(
    k: The number of nearest neighbors to be used for calculating the
        interpolated value. Defaults to ``4``.
    within: If true, the method ensures that the neighbors found are located
        within the point of interest. Defaults to ``false``.
    num_threads: The number of threads to use for the computation. If 0 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Defaults to ``0``.
Returns:
    A tuple containing a matrix describing for each provided position, the
    distance between the provided position and the found neighbors (in meters if
    the RTree handles LLA coordinates, otherwise in Cartesian units) and a
    matrix containing the value of the different neighbors found for all
    provided positions.
)__doc__")
              .c_str())
      .def("inverse_distance_weighting", &RTree::inverse_distance_weighting,
           py::arg("coordinates"), py::arg("radius") = std::nullopt,
           py::arg("k") = 9, py::arg("p") = 2, py::arg("within") = true,
           py::arg("num_threads") = 0,
           (R"__doc__(
Interpolation of the value at the requested position by inverse distance
weighting method.

Args:
    )__doc__" +
            coordinates_help<dimension_t::value>() + R"__doc__(
    radius: The maximum radius of the search (m). Defaults The maximum
    distance
        between two points.
    k: The number of nearest neighbors to be used for calculating the
        interpolated value. Defaults to ``9``.
    p: The power parameters. Defaults to ``2``. within (bool, optional): If
        true, the method ensures that the neighbors found are located around the
        point of interest. In other words, this parameter ensures that the
        calculated values will not be extrapolated. Defaults to ``true``.
    num_threads: The number of threads to use for the computation. If 0 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Defaults to ``0``.
Returns:
    The interpolated value and the number of neighbors used in the
    calculation.
)__doc__")
               .c_str())
      .def("radial_basis_function", &RTree::radial_basis_function,
           py::arg("coordinates"), py::arg("radius") = std::nullopt,
           py::arg("k") = 9,
           py::arg("rbf") = pyinterp::RadialBasisFunction::Multiquadric,
           py::arg("epsilon") = std::optional<typename RTree::promotion_t>(),
           py::arg("smooth") = 0, py::arg("within") = true,
           py::arg("num_threads") = 0,
           (R"__doc__(
Interpolation of the value at the requested position by radial basis function
interpolation.

Args:
    )__doc__" +
            coordinates_help<dimension_t::value>() + R"__doc__(
    radius: The maximum radius of the search (m). Default to the largest value
        that can be represented on a float.
    k: The number of nearest neighbors to be used for calculating the
        interpolated value. Defaults to ``9``.
    rbf: The radial basis function, based on the radius, r, given by the
        distance between points. Default to
        :py:attr:`pyinterp.core.RadialBasisFunction.Multiquadric`.
    epsilon: Adjustable constant for gaussian or multiquadrics functions.
        Default to the average distance between nodes.
    smooth: Values greater than zero increase the smoothness of the
        approximation.
    within: If true, the method ensures that the neighbors found are located
        around the point of interest. Defaults to ``true``.
    num_threads: The number of threads to use for the computation. If 0 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Defaults to ``0``.
Returns:
    The interpolated value and the number of neighbors used for the calculation.
)__doc__")
               .c_str())
      .def("window_function", &RTree::window_function, py::arg("coordinates"),
           py::arg("radius") = std::nullopt, py::arg("k") = 9,
           py::arg("wf") = pyinterp::WindowFunction::kHamming,
           py::arg("arg") = std::nullopt, py::arg("within") = true,
           py::arg("num_threads") = 0,
           (R"__doc__(
Interpolation of the value at the requested position by window function.

Args:
    )__doc__" +
            coordinates_help<dimension_t::value>() + R"__doc__(
    radius: The maximum radius of the search (m). Default to the largest value
        that can be represented on a float.
    k: The number of nearest neighbors to be used for calculating the
        interpolated value. Defaults to ``9``.
    wf: The window function to be used. Defaults to
        :py:attr:`pyinterp.core.WindowFunction.Hamming`.
    arg: The optional argument of the window function. Defaults to ``None``.
    within: If true, the method ensures that the neighbors found are located
        around the point of interest. Defaults to ``true``.
    num_threads: The number of threads to use for the computation. If 0 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Defaults to ``0``.
Returns:
    The interpolated value and the number of neighbors used for the calculation.
)__doc__")
               .c_str())
      .def("universal_kriging", &RTree::universal_kriging,
           py::arg("coordinates"), py::arg("radius") = std::nullopt,
           py::arg("k") = 9,
           py::arg("covariance") = pyinterp::CovarianceFunction::kMatern_32,
           py::arg("sigma") = 1, py::arg("alpha") = 1'000'000,
           py::arg("within") = true, py::arg("num_threads") = 0,
           (R"__doc__(
Universal Kriging interpolation of the value at the requested position.

Args:
    )__doc__" +
            coordinates_help<dimension_t::value>() + R"__doc__(
    radius: The maximum radius of the search (m). Default to the largest value
        that can be represented on a float.
    k: The number of nearest neighbors to be used for calculating the
        interpolated value. Defaults to ``9``.
    covariance: The covariance function to be used. Defaults to
        :py:attr:`pyinterp.core.CovarianceFunction.Matern_52`.
    sigma: The magnitude parameter. Determines the overall scale of the
        covariance function. It represents the maximum possible covariance
        between two points. Defaults to ``1``.
    alpha: Decay rate parameter. Determines the rate at which the covariance
        decreases. It represents the spatial scale of the covariance function
        and can be used to control the smoothness of the spatial dependence
        structure.
    within: If true, the method ensures that the neighbors found are located
        around the point of interest. Defaults to ``true``.
    num_threads: The number of threads to use for the computation. If 0 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Defaults to ``0``.
Returns:
    The interpolated value and the number of neighbors used for the calculation.
)__doc__")
               .c_str())
      .def("value", &RTree::value, py::arg("coordinates"),
           py::arg("radius") = std::nullopt, py::arg("k") = 4,
           py::arg("within") = false, py::arg("num_threads") = 0,
           (R"__doc__(
Get the K nearest neighbors of a given point.

Args:
    )__doc__" +
            coordinates_help<dimension_t::value>() + R"__doc__(
    radius: The maximum radius of the search (m). By default, no limit is
        applied.
    k: The number of nearest neighbors to be used for calculating the
        interpolated value. Defaults to ``4``.
    within: If true, the method ensures that the neighbors found are located
        around the point of interest. Defaults to ``false``.
    num_threads: The number of threads to use for the computation. If 0 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Defaults to ``0``.
Returns:
    A tuple of matrices containing the coordinates and values of the
    different neighbors found for all provided positions.

.. note::
    The matrix containing the coordinates of the neighbors is a matrix of
    dimension ``(k, n)`` where ``n`` is equal to 2 if the provided coordinates
    matrix defines only x and y, and 3 if the defines x, y, and z.
)__doc__")
               .c_str())
      .def(py::pickle([](const RTree &self) { return self.getstate(); },
                      [](const py::tuple &state) {
                        return pyinterp::RTree<Point, Type>::setstate(state);
                      }));
}

void init_rtree(py::module &m) {
  implement_rtree<pyinterp::detail::geometry::Point3D<double>, double>(
      m, "Float64");
  implement_rtree<pyinterp::detail::geometry::Point3D<float>, float>(m,
                                                                     "Float32");
}
