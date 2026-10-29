// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/pybind/regrid.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include <concepts>
#include <cstdint>
#include <optional>
#include <tuple>

#include "pyinterp/eigen.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"
#include "pyinterp/regrid/conservative.hpp"

namespace nb = nanobind;

namespace pyinterp::pybind {
namespace {

using geometry::geographic::Spheroid;

constexpr const char* const kOverlapMatrixDoc = R"doc(
Compute the overlaps between the cells of two axes.

Args:
    source_bounds: Bounds of the source cells, shape (n, 2). The cells must not
        overlap.
    target_bounds: Bounds of the target cells, shape (m, 2).
    period: Period of the axis (360 for longitudes), 0 if the axis is not
        periodic. The overlaps of a periodic axis are computed modulo the
        period.

Returns:
    The sparse (m, n) overlap matrix in the CSR format ``(indptr, indices,
    weights)``: a weight is the length of the overlap divided by the length of
    the target cell.
)doc";

constexpr const char* const kLatitudeMeasureDoc = R"doc(
Transform latitude bounds into the sines of the authalic latitudes.

The area of the cell [λ1, λ2] × [φ1, φ2] is R²·(λ2 − λ1)·(μ(φ2) − μ(φ1)),
where μ is the sine of the authalic latitude and R the authalic radius. On a
sphere, μ is the sine of the latitude.

Args:
    bounds: Latitude bounds, in degrees.
    spheroid: Ellipsoid of the coordinates. Defaults to a sphere.

Returns:
    The sines of the authalic latitudes of the bounds.
)doc";

constexpr const char* const kConservativeDoc = R"doc(
First order conservative regridding between rectilinear grids.

Each target cell receives the mean of the source values weighted by the area
of their overlap with the target cell.

Args:
    data: Fields to regrid, a C-contiguous array of shape (fields × nx, ny):
        the fields follow each other, each one being a (nx, ny) matrix.
    source_x_bounds: Bounds of the source cells along the first axis, shape
        (nx, 2).
    source_y_bounds: Bounds of the source cells along the second axis, shape
        (ny, 2).
    target_x_bounds: Bounds of the target cells along the first axis, shape
        (mx, 2).
    target_y_bounds: Bounds of the target cells along the second axis, shape
        (my, 2).
    geodetic: If true, the first axis is a longitude, periodic with a period
        of 360 degrees, and the second axis a latitude, in degrees. Otherwise,
        the axes are Cartesian.
    spheroid: Ellipsoid of a geodetic grid. Defaults to a sphere.
    min_valid_fraction: Minimum fraction of the target cell covered by defined
        source values; below, the result is NaN.
    normalize: If true, undefined source values are ignored: the result is the
        mean of the defined values. Otherwise, the weighted sum is divided by
        the area of the whole target cell, as if they were zero.
    num_threads: Number of threads, 0 for all CPUs.

Returns:
    The regridded fields, an array of shape (fields × mx, my).
)doc";

template <typename T>
auto to_vector(const std::vector<T>& values) -> Vector<T> {
  return Eigen::Map<const Vector<T>>(values.data(),
                                     static_cast<int64_t>(values.size()));
}

auto overlap_matrix(const regrid::Bounds& source_bounds,
                    const regrid::Bounds& target_bounds, const double period)
    -> std::tuple<Vector<int64_t>, Vector<int64_t>, Vector<double>> {
  const auto matrix =
      regrid::overlap_matrix(source_bounds, target_bounds, period);
  return {to_vector(matrix.indptr), to_vector(matrix.indices),
          to_vector(matrix.weights)};
}

/// Overlaps along both axes of a grid
auto overlaps(const regrid::Bounds& source_x_bounds,
              const regrid::Bounds& source_y_bounds,
              const regrid::Bounds& target_x_bounds,
              const regrid::Bounds& target_y_bounds, const bool geodetic,
              const std::optional<Spheroid>& spheroid)
    -> std::tuple<regrid::OverlapMatrix, regrid::OverlapMatrix> {
  if (!geodetic) {
    return {regrid::overlap_matrix(source_x_bounds, target_x_bounds),
            regrid::overlap_matrix(source_y_bounds, target_y_bounds)};
  }
  const auto source_y = regrid::latitude_measure(source_y_bounds, spheroid);
  const auto target_y = regrid::latitude_measure(target_y_bounds, spheroid);
  return {regrid::overlap_matrix(source_x_bounds, target_x_bounds, 360.0),
          regrid::overlap_matrix(source_y, target_y)};
}

template <std::floating_point T>
auto bind_conservative(nb::module_& m) -> void {
  m.def(
      "conservative",
      [](const Eigen::Ref<const RowMajorMatrix<T>>& data,
         const regrid::Bounds& source_x_bounds,
         const regrid::Bounds& source_y_bounds,
         const regrid::Bounds& target_x_bounds,
         const regrid::Bounds& target_y_bounds, const bool geodetic,
         const std::optional<Spheroid>& spheroid,
         const double min_valid_fraction, const bool normalize,
         const int64_t num_threads) -> RowMajorMatrix<T> {
        const auto [wx, wy] =
            overlaps(source_x_bounds, source_y_bounds, target_x_bounds,
                     target_y_bounds, geodetic, spheroid);
        if (data.cols() != source_y_bounds.rows()) {
          throw std::invalid_argument(
              "the shape of the data does not match the source axes");
        }
        return regrid::conservative<T>(wx, wy, data, source_x_bounds.rows(),
                                       min_valid_fraction, normalize,
                                       num_threads);
      },
      nb::arg("data"), nb::arg("source_x_bounds"), nb::arg("source_y_bounds"),
      nb::arg("target_x_bounds"), nb::arg("target_y_bounds"), nb::kw_only(),
      nb::arg("geodetic") = true, nb::arg("spheroid") = nb::none(),
      nb::arg("min_valid_fraction") = 0.0, nb::arg("normalize") = true,
      nb::arg("num_threads") = 0, kConservativeDoc,
      nb::call_guard<nb::gil_scoped_release>());
}

}  // namespace

auto init_regrid(nb::module_& m) -> void {
  auto submodule = m.def_submodule(
      "regrid", "Conservative regridding between rectilinear grids.");

  submodule.def("overlap_matrix", &overlap_matrix, nb::arg("source_bounds"),
                nb::arg("target_bounds"), nb::arg("period") = 0.0,
                kOverlapMatrixDoc, nb::call_guard<nb::gil_scoped_release>());

  submodule.def(
      "latitude_measure",
      [](const regrid::Bounds& bounds,
         const std::optional<Spheroid>& spheroid) -> RowMajorMatrix<double> {
        return regrid::latitude_measure(bounds, spheroid);
      },
      nb::arg("bounds"), nb::arg("spheroid") = nb::none(), kLatitudeMeasureDoc,
      nb::call_guard<nb::gil_scoped_release>());

  bind_conservative<float>(submodule);
  bind_conservative<double>(submodule);
}

}  // namespace pyinterp::pybind
