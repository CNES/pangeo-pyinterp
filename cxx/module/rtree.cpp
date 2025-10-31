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
  ss << "coordinates: Array of shape ``(n, 3)`` or ``(n, 2)`` containing\n"
        "        observation coordinates. Here ``n`` is the number of\n"
        "        observations and each row represents a coordinate in the\n"
        "        order x, y, and optionally z. If the matrix shape is\n"
        "        ``(n, 2)``, the z-coordinate is assumed to be zero.\n"
        "        The coordinate system depends on the instance configuration:\n"
        "        If ``ecef=True``, coordinates are in the Cartesian\n"
        "        coordinate system (ECEF). Otherwise, coordinates are in the\n"
        "        geodetic system (longitude, latitude, altitude) in degrees,\n"
        "        degrees, and meters, respectively.";
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
       "(self, spheroid: pyinterp.core.geodetic.Spheroid | None = None)" +
       R"__doc__(
Create a spatial index for geodetic scalar values.

This class implements an R*-tree spatial index to efficiently query and
interpolate values on the sphere. It supports both geodetic (longitude,
latitude, altitude) and Cartesian (ECEF) coordinate systems.

Args:
    spheroid: World Geodetic System (WGS) of the coordinate system used to
        transform geodetic positions (longitude, latitude, altitude) into
        ECEF coordinates. If not set, WGS-84 is used. Defaults to None.
    ecef: If True, coordinates managed by this instance are in the Cartesian
        coordinate system (ECEF) and no conversion from geodetic coordinates
        is performed. Defaults to False.

Raises:
    ValueError: If both spheroid and ecef are specified. Either the spheroid
        or ecef parameter should be set, but not both.
)__doc__")
          .c_str())
      .def(py::init<std::optional<pyinterp::geodetic::Spheroid>, bool>(),
           py::arg("spheroid") = std::nullopt, py::arg("ecef") = false)
      .def("bounds", &RTree::equatorial_bounds,
           R"__doc__(
Get the bounding box containing all stored values.

Returns the box able to contain all values stored in the container.

Returns:
    Tuple of coordinates of the minimum and maximum corners of the bounding box,
    or an empty tuple if the container is empty.
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
Build the spatial index using a packing algorithm.

Constructs the R*-tree using the packing algorithm. This method erases any
existing data before construction.

Args:
    )__doc__" +
            coordinates_help<dimension_t::value>() + R"__doc__(
    values: Array of size (n) containing the values associated with the
        coordinates provided.
)__doc__")
               .c_str())
      .def("insert", &RTree::insert, py::arg("coordinates"), py::arg("values"),
           (R"__doc__(
Add new data into the spatial index.

Inserts coordinates and associated values into the R*-tree.

Args:
    )__doc__" +
            coordinates_help<dimension_t::value>() + R"__doc__(
    values: Array of size (n) containing the values associated with the
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
Find the K nearest neighbors of given points.

Searches for the nearest neighbors to the provided coordinates in the spatial
index.

Args:
    )__doc__" +
           coordinates_help<dimension_t::value>() + R"__doc__(
    k: Number of nearest neighbors to return. Defaults to 4.
    within: If True, ensures that neighbors are located around the point of
        interest (no extrapolation). Defaults to False.
    num_threads: Number of threads to use for computation. If 0, all CPUs
        are used. If 1, no parallel computing is used (useful for debugging).
        Defaults to 0.

Returns:
    Tuple containing (1) a matrix of distances between each provided position
    and the found neighbors (in meters for LLA coordinates, Cartesian units
    for ECEF), and (2) a matrix of values for the neighbors found.
)__doc__")
              .c_str())
      .def("inverse_distance_weighting", &RTree::inverse_distance_weighting,
           py::arg("coordinates"), py::arg("radius") = std::nullopt,
           py::arg("k") = 9, py::arg("p") = 2, py::arg("within") = true,
           py::arg("num_threads") = 0,
           (R"__doc__(
Interpolate values using inverse distance weighting.

Performs inverse distance weighted interpolation at the requested positions
using the K nearest neighbors found within the specified search radius.

Args:
    )__doc__" +
            coordinates_help<dimension_t::value>() + R"__doc__(
    radius: Maximum search radius in meters. Defaults to the largest
        representable value.
    k: Number of nearest neighbors to use for interpolation. Defaults to 9.
    p: Power parameter for inverse distance weighting. Defaults to 2.
    within: If True, ensures neighbors are located around the point of
        interest (no extrapolation). Defaults to True.
    num_threads: Number of threads to use for computation. If 0, all CPUs
        are used. If 1, no parallel computing is used (useful for debugging).
        Defaults to 0.

Returns:
    Tuple containing the interpolated value and the number of neighbors used
    in the calculation.
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
Interpolate values using radial basis functions.

Performs radial basis function (RBF) interpolation at the requested positions
using the K nearest neighbors found within the specified search radius.

Args:
    )__doc__" +
            coordinates_help<dimension_t::value>() + R"__doc__(
    radius: Maximum search radius in meters. Defaults to the largest
        representable value.
    k: Number of nearest neighbors to use for interpolation. Defaults to 9.
    rbf: The radial basis function to use (e.g., Multiquadric, Gaussian).
        Defaults to Multiquadric.
    epsilon: Adjustable constant for Gaussian or Multiquadrics functions.
        Defaults to the average distance between nodes.
    smooth: Smoothness parameter. Values greater than zero increase the
        smoothness of the approximation. Defaults to 0.
    within: If True, ensures neighbors are located around the point of
        interest (no extrapolation). Defaults to True.
    num_threads: Number of threads to use for computation. If 0, all CPUs
        are used. If 1, no parallel computing is used (useful for debugging).
        Defaults to 0.

Returns:
    Tuple containing the interpolated value and the number of neighbors used
    for the calculation.
)__doc__")
               .c_str())
      .def("window_function", &RTree::window_function, py::arg("coordinates"),
           py::arg("radius") = std::nullopt, py::arg("k") = 9,
           py::arg("wf") = pyinterp::WindowFunction::kHamming,
           py::arg("arg") = std::nullopt, py::arg("within") = true,
           py::arg("num_threads") = 0,
           (R"__doc__(
Interpolate values using a window function.

Performs window function interpolation at the requested positions using the K
nearest neighbors found within the specified search radius.

Args:
    )__doc__" +
            coordinates_help<dimension_t::value>() + R"__doc__(
    radius: Maximum search radius in meters. Defaults to the largest
        representable value.
    k: Number of nearest neighbors to use for interpolation. Defaults to 9.
    wf: Window function to be used (e.g., Hamming, Hanning, Blackman).
        Defaults to Hamming.
    arg: Optional argument of the window function. Defaults to None.
    within: If True, ensures neighbors are located around the point of
        interest (no extrapolation). Defaults to True.
    num_threads: Number of threads to use for computation. If 0, all CPUs
        are used. If 1, no parallel computing is used (useful for debugging).
        Defaults to 0.

Returns:
    Tuple containing the interpolated value and the number of neighbors used
    for the calculation.
)__doc__")
               .c_str())
      .def("kriging", &RTree::kriging, py::arg("coordinates"),
           py::arg("radius") = std::nullopt, py::arg("k") = 9,
           py::arg("covariance") = pyinterp::CovarianceFunction::kMatern_32,
           py::arg("drift_function") = std::nullopt, py::arg("sigma") = 1,
           py::arg("alpha") = 1'000'000, py::arg("nugget") = 0,
           py::arg("within") = true, py::arg("num_threads") = 0,
           (R"__doc__(
Interpolate values using kriging.

Performs kriging interpolation at the requested positions using the K nearest
neighbors found within the specified search radius. Supports both simple kriging
(with zero mean) and universal kriging (with a drift function).

Args:
    )__doc__" +
            coordinates_help<dimension_t::value>() + R"__doc__(
    radius: Maximum search radius in meters. Defaults to the largest
        representable value.
    k: Number of nearest neighbors to use for interpolation. Defaults to 9.
    covariance: Covariance function to use (e.g., Matern, Gaussian).
        Defaults to Matern_32.
    drift_function: Drift function for universal kriging. If not provided,
        simple kriging is used. Defaults to None.
    sigma: Magnitude parameter determining the overall scale of the covariance
        function. Defaults to 1.
    alpha: Decay rate parameter controlling the rate at which covariance
        decreases. Defaults to 1,000,000.
    nugget: Nugget effect term for numerical stability. Defaults to 0.
    within: If True, ensures neighbors are located around the point of
        interest (no extrapolation). Defaults to True.
    num_threads: Number of threads to use for computation. If 0, all CPUs
        are used. If 1, no parallel computing is used (useful for debugging).
        Defaults to 0.

Returns:
    Tuple containing the interpolated value and the number of neighbors used
    for the calculation.

Note:
    Universal kriging is used if a drift function is provided. Otherwise,
    simple kriging with a known (zero) mean is used.
)__doc__")
               .c_str())
      .def("value", &RTree::value, py::arg("coordinates"),
           py::arg("radius") = std::nullopt, py::arg("k") = 4,
           py::arg("within") = false, py::arg("num_threads") = 0,
           (R"__doc__(
Get the K nearest neighbors of given points.

Retrieves the coordinates and values of the K nearest neighbors for each
provided position.

Args:
    )__doc__" +
            coordinates_help<dimension_t::value>() + R"__doc__(
    radius: Maximum search radius in meters. By default, no limit is applied.
    k: Number of nearest neighbors to retrieve. Defaults to 4.
    within: If True, ensures neighbors are located around the point of
        interest (no extrapolation). Defaults to False.
    num_threads: Number of threads to use for computation. If 0, all CPUs
        are used. If 1, no parallel computing is used (useful for debugging).
        Defaults to 0.

Returns:
    Tuple of (coordinates, values) arrays for the K nearest neighbors found
    for each provided position. The coordinates array has shape (k, 2) or
    (k, 3) depending on whether z-coordinates were provided.

Note:
    The neighbor coordinates array has dimension (k, n) where n is 2 if the
    input coordinates only define x and y, and 3 if they define x, y, and z.
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
