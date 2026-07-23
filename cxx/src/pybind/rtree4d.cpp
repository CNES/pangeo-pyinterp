// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/pybind/rtree4d.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <format>
#include <string>
#include <string_view>

#include "pyinterp/pybind/dtype_to_str.hpp"

namespace nb = nanobind;

namespace pyinterp::pybind {

constexpr const char* const kRTree4DDoc =
    R"(Spatial index for 4D point data with per-observation error variance.

This is the indexing primitive feeding the :class:`Optimal Interpolation
(OI / BLUE) <pyinterp.OptimalInterpolation>` estimator. The tree is purely
Cartesian — no spheroid / geodetic conversion. Each indexed item carries an
observed value and its measurement-error variance ``σ²_obs``, which becomes
the diagonal of the matrix ``R`` in ``(C_oo + R) w = c_og``.

Compared to :class:`RTree3D`, this tree does **not** provide
inverse-distance-weighting, kriging, RBF or window-function methods. Use it
for k-nearest-neighbour lookups in 4D space-time, or call its built-in
``optimal_interpolation`` method (the estimator behind
:class:`pyinterp.OptimalInterpolation`).

Parameters:
    dtype: Data type for internal storage, either ``'float32'`` or
        ``'float64'``. Defaults to ``'float64'``.
)";

constexpr const char* const kPackingDoc =
    R"(Bulk-load observations using STR packing.

Erases any existing data and rebuilds the index from the given
``(coordinates, values, sigma2)`` triple.

Args:
    coordinates: Matrix of shape ``(n, 4)`` — the four Cartesian
        coordinates of each observation. The user is responsible for unit
        consistency.
    values: Vector of length ``n`` with the observed values.
    sigma2: Vector of length ``n`` with the per-observation error
        variance. Must be strictly positive.
)";

constexpr const char* const kInsertDoc =
    R"(Insert observations into the existing tree.

Args:
    coordinates: Matrix of shape ``(n, 4)``.
    values: Vector of length ``n``.
    sigma2: Vector of length ``n``. Must be strictly positive.
)";

constexpr const char* const kQueryDoc =
    R"(Query k-nearest neighbours for many points.

Args:
    coordinates: Query coordinates of shape ``(n, 4)``.
    config: Optional :class:`pyinterp.core.config.rtree.Query` instance
        (``k``, ``radius``, ``num_threads``). Defaults to a fresh
        ``Query()`` with ``k = 8`` and no radius limit.

Returns:
    Tuple ``(distances, values, sigma2)`` of shape ``(n, k)``. Cells
    beyond the actual neighbour count are filled with ``NaN``.
)";

constexpr const char* const kOptimalInterpolationDoc =
    R"(Optimal Interpolation (BLUE) at many query points.

For each query point this method retrieves up to ``config.k()`` neighbours
from the 4D tree, builds the anisotropic covariance system with per-query
length scales and field standard deviation, solves it via Cholesky (LDLT
fallback) and returns the analysed value, the formal error standard
deviation, and the number of neighbours actually used.

Args:
    coordinates: Query points, shape ``(m, 4)``.
    lx: Decorrelation length along axis 0, shape ``(m,)``.
    ly: Decorrelation length along axis 1, shape ``(m,)``.
    lz: Decorrelation length along axis 2, shape ``(m,)``.
    lt: Decorrelation length along axis 3, shape ``(m,)``.
    sigma: Field standard deviation, shape ``(m,)``.
    config: Optional :class:`...config.rtree.OptimalInterpolation` instance
        (``covariance_model``, ``k``, ``radius``, ``num_threads``).

Returns:
    Tuple ``(values, errors, neighbors)`` of shape ``(m,)`` — analysed
    value, formal error standard deviation, and neighbour count. Cells
    with no neighbour return ``NaN`` / ``0``.
)";

template <typename T>
void implement_rtree_4d_methods(nb::class_<RTree4D<T>>& cls) {
  cls.def(
         "size", [](const RTree4D<T>& self) { return self.size(); },
         "Return the number of observations in the tree.",
         nb::call_guard<nb::gil_scoped_release>())
      .def(
          "empty", [](const RTree4D<T>& self) { return self.empty(); },
          "Check whether the tree is empty.",
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "clear", [](RTree4D<T>& self) { self.clear(); },
          "Remove all observations from the tree.",
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "bounds",
          [](const RTree4D<T>& self)
              -> std::optional<
                  std::tuple<Eigen::Vector<T, 4>, Eigen::Vector<T, 4>>> {
            auto bounds = self.bounds();
            if (!bounds.has_value()) {
              return std::nullopt;
            }
            const auto& box = bounds.value();
            return std::make_tuple(
                Eigen::Vector<T, 4>{box.min_corner().template get<0>(),
                                    box.min_corner().template get<1>(),
                                    box.min_corner().template get<2>(),
                                    box.min_corner().template get<3>()},
                Eigen::Vector<T, 4>{box.max_corner().template get<0>(),
                                    box.max_corner().template get<1>(),
                                    box.max_corner().template get<2>(),
                                    box.max_corner().template get<3>()});
          },
          "Return the 4D bounding box of all stored observations, or None.",
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "packing",
          [](RTree4D<T>& self,
             const Eigen::Ref<const typename RTree4D<T>::CoordinateMatrix>&
                 coordinates,
             const Eigen::Ref<const typename RTree4D<T>::ValueVector>& values,
             const Eigen::Ref<const typename RTree4D<T>::ValueVector>& sigma2) {
            self.packing(coordinates, values, sigma2);
          },
          nb::arg("coordinates"), nb::arg("values"), nb::arg("sigma2"),
          kPackingDoc, nb::call_guard<nb::gil_scoped_release>())
      .def(
          "insert",
          [](RTree4D<T>& self,
             const Eigen::Ref<const typename RTree4D<T>::CoordinateMatrix>&
                 coordinates,
             const Eigen::Ref<const typename RTree4D<T>::ValueVector>& values,
             const Eigen::Ref<const typename RTree4D<T>::ValueVector>& sigma2) {
            self.insert(coordinates, values, sigma2);
          },
          nb::arg("coordinates"), nb::arg("values"), nb::arg("sigma2"),
          kInsertDoc, nb::call_guard<nb::gil_scoped_release>())
      .def(
          "query",
          [](const RTree4D<T>& self,
             const Eigen::Ref<const typename RTree4D<T>::CoordinateMatrix>&
                 coordinates,
             const std::optional<config::rtree::Query>& config) {
            return self.query(coordinates,
                              config.value_or(config::rtree::Query{}));
          },
          nb::arg("coordinates"), nb::arg("config") = std::nullopt, kQueryDoc,
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "optimal_interpolation",
          [](const RTree4D<T>& self,
             const Eigen::Ref<const typename RTree4D<T>::CoordinateMatrix>&
                 coordinates,
             const Eigen::Ref<const typename RTree4D<T>::ValueVector>& lx,
             const Eigen::Ref<const typename RTree4D<T>::ValueVector>& ly,
             const Eigen::Ref<const typename RTree4D<T>::ValueVector>& lz,
             const Eigen::Ref<const typename RTree4D<T>::ValueVector>& lt,
             const Eigen::Ref<const typename RTree4D<T>::ValueVector>& sigma,
             const std::optional<config::rtree::OptimalInterpolation>& config) {
            return self.optimal_interpolation(
                coordinates, lx, ly, lz, lt, sigma,
                config.value_or(config::rtree::OptimalInterpolation{}));
          },
          nb::arg("coordinates"), nb::arg("lx"), nb::arg("ly"), nb::arg("lz"),
          nb::arg("lt"), nb::arg("sigma"), nb::arg("config") = std::nullopt,
          kOptimalInterpolationDoc, nb::call_guard<nb::gil_scoped_release>())
      .def("__getstate__", &RTree4D<T>::getstate, "Get the state for pickling.")
      .def(
          "__setstate__",
          [](RTree4D<T>& self, nb::tuple& state) {
            new (&self) RTree4D<T>(std::move(RTree4D<T>::setstate(state)));
          },
          nb::arg("state"), "Set the state for unpickling.");
}

template <typename T>
void init_rtree_4d_impl(nb::module_& m, std::string_view suffix) {
  auto cls = nb::class_<RTree4D<T>>(m, std::format("RTree4D{}", suffix).c_str(),
                                    kRTree4DDoc);
  cls.def(nb::init<>(), "Initialize a fresh 4D Cartesian R-tree.",
          nb::call_guard<nb::gil_scoped_release>());
  implement_rtree_4d_methods(cls);
}

/// Factory that picks the dtype-specialised RTree4D class.
auto rtree_4d_factory(const nb::object& dtype) -> nb::object {
  auto dtype_str = dtype_to_str(dtype).value_or("float64");
  if (dtype_str == "float32") {
    return nb::cast(RTree4D<float>(), nb::rv_policy::move);
  }
  if (dtype_str == "float64") {
    return nb::cast(RTree4D<double>(), nb::rv_policy::move);
  }
  throw std::invalid_argument("dtype must be 'float32' or 'float64', got: " +
                              dtype_str);
}

constexpr const char* const kRTree4DFactoryDoc =
    R"(Spatial index for 4D point data with per-observation error variance.

Create a Cartesian 4D R-tree designed to back the Optimal Interpolation
estimator. Each observation is stored together with its measurement-error
variance ``σ²_obs``.

Parameters:
    dtype: ``'float32'`` or ``'float64'`` (default).

Examples:
    >>> import numpy as np
    >>> import pyinterp
    >>> tree = pyinterp.RTree4D()
    >>> tree.packing(
    ...     np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    ...     np.array([10.0, 20.0]),
    ...     np.array([0.01, 0.02]),
    ... )
    >>> from pyinterp.core.config.rtree import Query
    >>> distances, values, sigma2 = tree.query(
    ...     np.array([[0.5, 0.0, 0.0, 0.0]]), Query().with_k(2),
    ... )
)";

void init_rtree_4d(nanobind::module_& m) {
  init_rtree_4d_impl<float>(m, "Float32");
  init_rtree_4d_impl<double>(m, "Float64");
  m.def("RTree4D", &rtree_4d_factory, kRTree4DFactoryDoc,
        nb::arg("dtype") = nb::none());
}

}  // namespace pyinterp::pybind
