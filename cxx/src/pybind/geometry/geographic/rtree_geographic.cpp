// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include <optional>

#include "pyinterp/pybind/geometry/geographic/rtree.hpp"

namespace nb = nanobind;

namespace pyinterp::geometry::geographic::pybind {

namespace {

constexpr auto kRTreeDoc = R"doc(
Spatial index for geographic longitude/latitude points.

Build an R-tree on geographic coordinates expressed directly in degrees.
All distance calculations and interpolations are performed on the spheroid;
no conversion to ECEF is applied.

Examples:
	>>> import numpy as np
	>>> from pyinterp.core.geometry.geographic import RTree

	>>> coords = np.array([
	...     [2.0, 48.0],
	...     [3.0, 49.0]
	... ])
	>>> values = np.array([10.5, 20.3])
	>>> tree = RTree()
	>>> tree.packing(coords, values)

	>>> query = np.array([[2.5, 48.5]])
	>>> distances, neighbors = tree.query(query, k=2)
)doc";

constexpr auto kPackingDoc = R"doc(
Bulk-load points using the STR packing algorithm.

Args:
	coordinates: Matrix of shape (n, 2) with (lon, lat) in degrees.
	values: Vector of size n containing the value at each coordinate.
)doc";

constexpr auto kInsertDoc = R"doc(
Insert points into the tree.

Args:
	coordinates: Matrix of shape (n, 2) with (lon, lat) in degrees.
	values: Vector of size n containing the value at each coordinate.
)doc";

constexpr auto kQueryDoc = R"doc(
Query k-nearest neighbors for multiple points.

Args:
	coordinates: Matrix of shape (n, 2) with (lon, lat) in degrees.
	config: Query configuration (k neighbors, radius, boundary check, threads).

Returns:
	Tuple of (distances, values) with shape [n_points x k]. Distances are
	geodesic on the spheroid (meters); values mirror the stored data.
)doc";

constexpr auto kIdwDoc = R"doc(
Inverse distance weighting interpolation on geographic coordinates.

Args:
	coordinates: Matrix of shape (n, 2) with (lon, lat) in degrees.
	config: IDW configuration (k neighbors, radius, power, threads).

Returns:
	Tuple (values, neighbor_counts).
)doc";

constexpr auto kKrigingDoc = R"doc(
Kriging interpolation on geographic coordinates.

Args:
	coordinates: Matrix of shape (n, 2) with (lon, lat) in degrees.
	config: Kriging configuration (model parameters, k neighbors, radius,
		boundary check, threads).

Returns:
	Tuple (values, neighbor_counts).
)doc";

constexpr auto kRbfDoc = R"doc(
Radial basis function interpolation on geographic coordinates.

Args:
	coordinates: Matrix of shape (n, 2) with (lon, lat) in degrees.
	config: RBF configuration (kernel, epsilon, smooth, k, radius, threads).

Returns:
	Tuple (values, neighbor_counts).
)doc";

constexpr auto kWindowDoc = R"doc(
Window function interpolation on geographic coordinates.

Args:
	coordinates: Matrix of shape (n, 2) with (lon, lat) in degrees.
	config: Window configuration (window function, argument, k, radius,
		boundary check, threads).

Returns:
	Tuple (values, neighbor_counts).
)doc";

}  // namespace

auto init_rtree(nb::module_& m) -> void {
  auto rtree = nb::class_<RTree>(m, "RTree", kRTreeDoc);

  rtree
      .def(nb::init<>(), "Create an empty geographic RTree.",
           nb::call_guard<nb::gil_scoped_release>())

      .def(
          "size", [](const RTree& self) -> size_t { return self.size(); },
          "Return the number of points in the tree.",
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "empty", [](const RTree& self) -> bool { return self.empty(); },
          "Check if the tree is empty.",
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "clear", [](RTree& self) -> void { self.clear(); },
          "Remove all points from the tree.",
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "bounds",
          [](const RTree& self)
              -> std::optional<std::tuple<Eigen::Vector2d, Eigen::Vector2d>> {
            auto bounds = self.bounds();
            if (!bounds.has_value()) {
              return std::nullopt;
            }
            const auto& box = bounds.value();
            return std::make_tuple(
                Eigen::Vector2d{box.min_corner().template get<0>(),
                                box.min_corner().template get<1>()},
                Eigen::Vector2d{box.max_corner().template get<0>(),
                                box.max_corner().template get<1>()});
          },
          "Return the bounding box containing all stored points, or None if the"
          " tree is empty.",
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "packing",
          [](RTree& self,
             const Eigen::Ref<const RTree::CoordinateMatrix>& coordinates,
             const Eigen::Ref<const RTree::ValueVector>& values) -> void {
            self.packing(coordinates, values);
          },
          nb::arg("coordinates"), nb::arg("values"), kPackingDoc,
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "insert",
          [](RTree& self,
             const Eigen::Ref<const RTree::CoordinateMatrix>& coordinates,
             const Eigen::Ref<const RTree::ValueVector>& values) -> void {
            self.insert(coordinates, values);
          },
          nb::arg("coordinates"), nb::arg("values"), kInsertDoc,
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "query",
          [](const RTree& self,
             const Eigen::Ref<const RTree::CoordinateMatrix>& coordinates,
             const std::optional<config::rtree::Query>& config)
              -> std::tuple<Matrix<RTree::distance_t>,
                            Matrix<RTree::promotion_t>> {
            return self.query(coordinates,
                              config.value_or(config::rtree::Query{}));
          },
          nb::arg("coordinates"), nb::arg("config") = std::nullopt, kQueryDoc,
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "inverse_distance_weighting",
          [](const RTree& self,
             const Eigen::Ref<const RTree::CoordinateMatrix>& coordinates,
             const std::optional<config::rtree::InverseDistanceWeighting>&
                 config) -> std::tuple<RTree::ValueVector, Vector<uint32_t>> {
            return self.inverse_distance_weighting(
                coordinates,
                config.value_or(config::rtree::InverseDistanceWeighting{}));
          },
          nb::arg("coordinates"), nb::arg("config") = std::nullopt, kIdwDoc,
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "kriging",
          [](const RTree& self,
             const Eigen::Ref<const RTree::CoordinateMatrix>& coordinates,
             const std::optional<config::rtree::Kriging>& config)
              -> std::tuple<RTree::ValueVector, Vector<uint32_t>> {
            return self.kriging(coordinates,
                                config.value_or(config::rtree::Kriging{}));
          },
          nb::arg("coordinates"), nb::arg("config") = std::nullopt, kKrigingDoc,
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "radial_basis_function",
          [](const RTree& self,
             const Eigen::Ref<const RTree::CoordinateMatrix>& coordinates,
             const std::optional<config::rtree::RadialBasisFunction>& config)
              -> std::tuple<RTree::ValueVector, Vector<uint32_t>> {
            return self.radial_basis_function(
                coordinates,
                config.value_or(config::rtree::RadialBasisFunction{}));
          },
          nb::arg("coordinates"), nb::arg("config") = std::nullopt, kRbfDoc,
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "window_function",
          [](const RTree& self,
             const Eigen::Ref<const RTree::CoordinateMatrix>& coordinates,
             const std::optional<config::rtree::InterpolationWindow>& config)
              -> std::tuple<RTree::ValueVector, Vector<uint32_t>> {
            return self.window_function(
                coordinates,
                config.value_or(config::rtree::InterpolationWindow{}));
          },
          nb::arg("coordinates"), nb::arg("config") = std::nullopt, kWindowDoc,
          nb::call_guard<nb::gil_scoped_release>())

      .def("__getstate__", &RTree::getstate, "Get the state for pickling.")

      .def(
          "__setstate__",
          [](RTree& self, nb::tuple& state) -> void {
            new (&self) RTree(RTree::setstate(state));
          },
          nb::arg("state"), "Set the state for unpickling.");
}

}  // namespace pyinterp::geometry::geographic::pybind
