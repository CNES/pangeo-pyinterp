// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <string>
#include <string_view>

#include "pyinterp/pybind/dtype_to_str.hpp"
#include "pyinterp/pybind/rtree.hpp"

namespace nb = nanobind;

namespace pyinterp::pybind {

constexpr const char* const kRTree3DDoc =
    R"(Spatial index for 3D point data with interpolation methods.

Create a spatial R-tree index for 3D point data with support for various
interpolation methods including k-nearest neighbor search, inverse distance
weighting, kriging, radial basis functions, and window functions.

Parameters:
    spheroid: Optional spheroid for geodetic coordinate conversions.
        If provided, input coordinates are assumed to be (lon, lat, alt) in
        degrees/degrees/meters, and will be converted to ECEF internally.
        If None, input coordinates are treated as Cartesian without any
        transformation. These can represent ECEF coordinates, planar
        coordinates (with Z=0), or any other Cartesian system.
        Users must ensure unit consistency across all coordinates and values.
        Defaults to None.

Examples:
    >>> import numpy as np
    >>> import pyinterp

    Create RTree for Cartesian coordinates (e.g., planar system)

    >>> coords = np.array([
    ...     [0.0, 0.0, 0.0],
    ...     [1.0, 1.0, 0.0]
    ... ], dtype='float64')
    >>> values = np.array([10.5, 20.3], dtype='float64')
    >>> tree = pyinterp.RTree3D()
    >>> tree.packing(coords, values)

    Query k-nearest neighbors

    >>> query_coords = np.array([[0.5, 0.5, 0.0]])
    >>> distances, values = tree.query(query_coords, k=2)

    Create RTree with geodetic coordinates (lon, lat, alt)

    >>> geodetic_coords = np.array([
    ...     [0.0, 45.0, 100.0],
    ...     [1.0, 46.0, 200.0]
    ... ], dtype='float64')
    >>> tree_geodetic = pyinterp.RTree3D(spheroid=pyinterp.Spheroid())
    >>> tree_geodetic.packing(geodetic_coords, values)
)";

constexpr const char* const kPackingDoc =
    R"(Bulk-load points using STR packing algorithm.

Efficiently inserts all points at once using the STR (Space-Tree Recursive)
packing algorithm. This method erases any existing data in the tree.

Args:
    coordinates: Matrix of shape (n, 3) or (n, 2) containing point coordinates.
        If spheroid was provided: (lon, lat, alt) in degrees/degrees/meters
        If spheroid was not provided: Cartesian coordinates in any consistent
        unit system (e.g., meters for ECEF, units on a plane, etc.)
        If shape is (n, 2), the third coordinate is assumed to be zero.
    values: Vector of size n containing values at each point.

Note:
    When no spheroid is provided, coordinates are treated as Cartesian without
    any transformation. Users must ensure all coordinates and search radii use
    consistent units.
)";

constexpr const char* const kInsertDoc =
    R"(Insert points into the tree.

Add new points to an existing tree. Can be called multiple times to add
points incrementally.

Args:
    coordinates: Matrix of shape (n, 3) or (n, 2) containing point coordinates.
        If spheroid was provided: (lon, lat, alt) in degrees/degrees/meters
        If spheroid was not provided: Cartesian coordinates in any consistent
        unit system (e.g., meters for ECEF, units on a plane, etc.)
        If shape is (n, 2), the third coordinate is assumed to be zero.
    values: Vector of size n containing values at each point.

Note:
    When no spheroid is provided, coordinates are treated as Cartesian without
    any transformation. Users must ensure all coordinates and search radii use
    consistent units.
)";

constexpr const char* const kQueryDoc =
    R"(Query k-nearest neighbors for multiple points.

Find the k-nearest neighbors for each query point.

Args:
    coordinates: Query coordinates, shape (n, 3) or (n, 2).
        Must use the same coordinate system as the points in the tree.
    k: Number of neighbors to find.
    boundary_check: Type of boundary verification: 'none' (default),
        'envelope' (AABB), or 'convex_hull'.
    num_threads: Number of threads for parallel computation. 0 = auto.

Returns:
    Tuple of (distances, values) matrices of shape [n_points x k].

    - distances: Distance from each query point to its k-nearest neighbors.
      When no spheroid is provided, distances are in the same units
      as the Cartesian coordinates.
    - values: Values at those k-nearest neighbors

Note:
    Distance calculations depend on the coordinate system. With a spheroid,
    distances are geodetic; without one, distances are Euclidean in the
    provided Cartesian coordinate system.
)";

constexpr const char* const kInverseDistanceWeightingDoc =
    R"(Inverse distance weighting interpolation.

Interpolate values at query locations using inverse distance weighting (IDW).

Args:
    coordinates: Query coordinates, shape (n, 3) or (n, 2).
    config: Configuration for IDW interpolation.

Returns:
    Tuple of (values, neighbor_counts) where:
    - values: Interpolated values at query points
    - neighbor_counts: Number of neighbors used in interpolation
)";

constexpr const char* const kKrigingDoc =
    R"(Kriging interpolation.

Interpolate values at query locations using kriging.

Args:
    coordinates: Query coordinates, shape (n, 3) or (n, 2).
    config: Configuration for kriging interpolation.

Returns:
    Tuple of (values, neighbor_counts) where:
    - values: Kriged values at query points
    - neighbor_counts: Number of neighbors used in interpolation
)";

constexpr const char* const kRadialBasisFunctionDoc =
    R"(Radial basis function interpolation.

Interpolate values at query locations using radial basis functions.

Args:
    coordinates: Query coordinates, shape (n, 3) or (n, 2).
    config: Configuration for RBF interpolation.

Returns:
    Tuple of (values, neighbor_counts) where:
    - values: RBF-interpolated values at query points
    - neighbor_counts: Number of neighbors used in interpolation
)";

constexpr const char* const kWindowFunctionDoc =
    R"(Window function based interpolation.

Interpolate values at query locations using window functions.

Args:
    coordinates: Query coordinates, shape (n, 3) or (n, 2).
    config: Configuration for window function interpolation.

Returns:
    Tuple of (values, neighbor_counts) where:
    - values: Interpolated values at query points
    - neighbor_counts: Number of neighbors used in interpolation
)";

/// Common RTree3D operations
template <typename T>
void implement_rtree_3d_methods(nb::class_<RTree3D<T>>& rtree3d) {
  rtree3d
      .def(
          "size", [](const RTree3D<T>& self) -> auto { return self.size(); },
          "Return the number of points in the tree.",
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "empty", [](const RTree3D<T>& self) -> auto { return self.empty(); },
          "Check if the tree is empty.",
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "clear", [](RTree3D<T>& self) -> auto { self.clear(); },
          "Remove all points from the tree.",
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "bounds",
          [](const RTree3D<T>& self)
              -> std::optional<
                  std::tuple<Eigen::Vector<T, 3>, Eigen::Vector<T, 3>>> {
            auto bounds = self.bounds();
            if (!bounds.has_value()) {
              return std::nullopt;
            }
            const auto& box = bounds.value();
            return std::make_tuple(
                Eigen::Vector<T, 3>{box.min_corner().template get<0>(),
                                    box.min_corner().template get<1>(),
                                    box.min_corner().template get<2>()},
                Eigen::Vector<T, 3>{box.max_corner().template get<0>(),
                                    box.max_corner().template get<1>(),
                                    box.max_corner().template get<2>()});
          },
          "Return the bounding box containing all stored values, or None if "
          "the tree is empty.",
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "packing",
          [](RTree3D<T>& self,
             const Eigen::Ref<const typename RTree3D<T>::CoordinateMatrix>&
                 coordinates,
             const Eigen::Ref<const typename RTree3D<T>::ValueVector>& values)
              -> auto { self.packing(coordinates, values); },
          nb::arg("coordinates"), nb::arg("values"), kPackingDoc,
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "insert",
          [](RTree3D<T>& self,
             const Eigen::Ref<const typename RTree3D<T>::CoordinateMatrix>&
                 coordinates,
             const Eigen::Ref<const typename RTree3D<T>::ValueVector>& values)
              -> auto { self.insert(coordinates, values); },
          nb::arg("coordinates"), nb::arg("values"), kInsertDoc,
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "query",
          [](const RTree3D<T>& self,
             const Eigen::Ref<const typename RTree3D<T>::CoordinateMatrix>&
                 coordinates,
             const std::optional<config::rtree::Query>& config) -> auto {
            return self.query(coordinates,
                              config.value_or(config::rtree::Query{}));
          },
          nb::arg("coordinates"), nb::arg("config") = std::nullopt, kQueryDoc,
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "inverse_distance_weighting",
          [](const RTree3D<T>& self,
             const Eigen::Ref<const typename RTree3D<T>::CoordinateMatrix>&
                 coordinates,
             const std::optional<config::rtree::InverseDistanceWeighting>&
                 config) -> auto {
            return self.inverse_distance_weighting(
                coordinates,
                config.value_or(config::rtree::InverseDistanceWeighting{}));
          },
          nb::arg("coordinates"), nb::arg("config") = std::nullopt,
          kInverseDistanceWeightingDoc,
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "kriging",
          [](const RTree3D<T>& self,
             const Eigen::Ref<const typename RTree3D<T>::CoordinateMatrix>&
                 coordinates,
             const std::optional<config::rtree::Kriging>& config) -> auto {
            return self.kriging(coordinates,
                                config.value_or(config::rtree::Kriging{}));
          },
          nb::arg("coordinates"), nb::arg("config") = std::nullopt, kKrigingDoc,
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "radial_basis_function",
          [](const RTree3D<T>& self,
             const Eigen::Ref<const typename RTree3D<T>::CoordinateMatrix>&
                 coordinates,
             const std::optional<config::rtree::RadialBasisFunction>& config)
              -> auto {
            return self.radial_basis_function(
                coordinates,
                config.value_or(config::rtree::RadialBasisFunction{}));
          },
          nb::arg("coordinates"), nb::arg("config") = std::nullopt,
          kRadialBasisFunctionDoc, nb::call_guard<nb::gil_scoped_release>())

      .def(
          "window_function",
          [](const RTree3D<T>& self,
             const Eigen::Ref<const typename RTree3D<T>::CoordinateMatrix>&
                 coordinates,
             const std::optional<config::rtree::InterpolationWindow>& config)
              -> auto {
            return self.window_function(
                coordinates,
                config.value_or(config::rtree::InterpolationWindow{}));
          },
          nb::arg("coordinates"), nb::arg("config") = std::nullopt,
          kWindowFunctionDoc, nb::call_guard<nb::gil_scoped_release>())

      .def("__getstate__", &RTree3D<T>::getstate, "Get the state for pickling.")
      .def(
          "__setstate__",
          [](RTree3D<T>& self, nb::tuple& state) -> void {
            new (&self) RTree3D<T>(std::move(RTree3D<T>::setstate(state)));
          },
          nb::arg("state"), "Set the state for unpickling.");
}

template <typename T>
void init_rtree_3d(nb::module_& m, std::string_view suffix) {
  auto rtree3d = nb::class_<RTree3D<T>>(
      m, std::format("RTree3D{}", suffix).c_str(), kRTree3DDoc);

  rtree3d.def(
      nb::init<const std::optional<geometry::geographic::Spheroid>&>(),
      nb::arg("spheroid") = std::nullopt,
      "Initialize the RTree3D with optional spheroid for geodetic conversions.",
      nb::call_guard<nb::gil_scoped_release>());

  rtree3d.def_prop_ro(
      "spheroid",
      [](const RTree3D<T>& self) -> auto { return self.spheroid(); },
      "Get the spheroid used for geodetic conversions, or None for ECEF.",
      nb::call_guard<nb::gil_scoped_release>());

  implement_rtree_3d_methods(rtree3d);
}

/// @brief RTree3D factory function that accepts dtype parameter
auto rtree_3d_factory(
    const std::optional<geometry::geographic::Spheroid>& spheroid,
    const nb::object& dtype) -> nb::object {
  auto dtype_str = dtype_to_str(dtype).value_or("float64");

  // Create appropriate RTree3D based on dtype string
  if (dtype_str == "float32") {
    return nb::cast(RTree3D<float>(spheroid), nb::rv_policy::move);
  }
  if (dtype_str == "float64") {
    return nb::cast(RTree3D<double>(spheroid), nb::rv_policy::move);
  }
  throw std::invalid_argument("dtype must be 'float32' or 'float64', got: " +
                              dtype_str);
}

constexpr const char* const kRTree3DFactoryDoc =
    R"(Spatial index for 3D point data with interpolation methods.

Create a spatial R-tree index for 3D point data with support for various
interpolation methods including k-nearest neighbor search, inverse distance
weighting, kriging, radial basis functions, and window functions.

Parameters:
    spheroid: Optional spheroid for geodetic coordinate conversions.
        If provided, input coordinates are assumed to be (lon, lat, alt) in
        degrees/degrees/meters, and will be converted to ECEF internally.
        If None, input coordinates are treated as Cartesian without any
        transformation. These can represent ECEF coordinates, planar
        coordinates (with Z=0), or any other Cartesian system.
        Users must ensure unit consistency across all coordinates and values.
        Defaults to None.
    dtype: Data type for internal storage, either 'float32' or 'float64'.
        Determines precision and memory usage. Defaults to 'float64'.

Examples:
    >>> import numpy as np
    >>> import pyinterp

    >>> # Create RTree for Cartesian coordinates with float64 (default)
    >>> coords = np.array([
    ...     [0.0, 0.0, 0.0],
    ...     [1.0, 1.0, 0.0]
    ... ], dtype='float64')
    >>> values = np.array([10.5, 20.3], dtype='float64')
    >>> tree = pyinterp.RTree3D()
    >>> tree.packing(coords, values)

    >>> # Create RTree with float32 for reduced memory usage
    >>> coords_f32 = coords.astype('float32')
    >>> values_f32 = values.astype('float32')
    >>> tree_f32 = pyinterp.RTree3D(dtype='float32')
    >>> tree_f32.packing(coords_f32, values_f32)

    >>> # Query k-nearest neighbors
    >>> query_coords = np.array([[0.5, 0.5, 0.0]])
    >>> distances, values = tree.query(query_coords, k=2)

    >>> # Create RTree with geodetic coordinates (lon, lat, alt)
    >>> geodetic_coords = np.array([
    ...     [0.0, 45.0, 100.0],
    ...     [1.0, 46.0, 200.0]
    ... ], dtype='float64')
    >>> tree_geodetic = pyinterp.RTree3D(spheroid=pyinterp.Spheroid())
    >>> tree_geodetic.packing(geodetic_coords, values)
)";

void init_rtree_3d(nanobind::module_& m) {
  // Register the concrete RTree3D classes
  init_rtree_3d<float>(m, "Float32");
  init_rtree_3d<double>(m, "Float64");

  // Register the factory function
  m.def("RTree3D", &rtree_3d_factory, kRTree3DFactoryDoc,
        nb::arg("spheroid") = std::nullopt, nb::arg("dtype") = nb::none());
}

}  // namespace pyinterp::pybind
