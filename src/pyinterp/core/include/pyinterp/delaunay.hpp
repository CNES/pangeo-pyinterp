// Copyright (c) 2023 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_3.h>
#include <CGAL/interpolation_functions.h>
#include <CGAL/natural_neighbor_coordinates_3.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "pyinterp/detail/geodetic/coordinates.hpp"

namespace pyinterp {

using NDArray = pybind11::array_t<double, pybind11::array::c_style |
                                              pybind11::array::forcecast>;

/// @brief Natural neighbor interpolation
class Delaunay {
 public:
  using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
  using CoordinateType = Kernel::FT;
  using KPoint = Kernel::Point_3;
  using KVector = Kernel::Vector_3;
  using DelaunayTriangulation = CGAL::Delaunay_triangulation_3<Kernel>;
  using CoordinateVector = std::vector<std::pair<KPoint, CoordinateType>>;
  using PointValueMap = std::map<KPoint, CoordinateType, Kernel::Less_xy_3>;

  /// @brief Constructor
  /// @param coordinates Coordinates of the points.
  /// @param values Values of the points.
  /// @param spheroid Spheroid used to convert the coordinates from
  /// geodetic to cartesian.
  Delaunay(const NDArray& coordinates, const pybind11::array_t<double>& values,
           const std::optional<detail::geodetic::Spheroid>& spheroid);

  /// @brief Natural neighbor interpolation
  /// @param coordinates Coordinates of the points to interpolate.
  /// @param num_threads Number of threads to use for the interpolation.
  /// @return Interpolated values.
  auto natural_neighbor_interpolation(const NDArray& coordinates,
                                      size_t num_threads) const
      -> pybind11::array_t<double>;

 private:
  /// @brief Delaunay triangulation
  DelaunayTriangulation triangulation_;
  /// @brief Map of the Cartesian points and their values
  PointValueMap point_values_;
  /// @brief Spheroid used to convert the coordinates from geodetic to
  /// cartesian.
  detail::geodetic::Coordinates coordinates_;
};

}  // namespace pyinterp
