// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/bivariate.hpp"
#include "pyinterp/trivariate.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace geometry = pyinterp::detail::geometry;

void init_grid(py::module& m) {
  pyinterp::implement_bivariate_interpolator<geometry::EquatorialPoint2D,
                                             double>(m, "2D");
  pyinterp::implement_bivariate_interpolator<geometry::EquatorialPoint3D,
                                             double>(m, "3D");

  pyinterp::implement_bivariate<geometry::EquatorialPoint2D, double, double>(
      m, "BivariateFloat64");
  pyinterp::implement_bivariate<geometry::EquatorialPoint2D, double, float>(
      m, "BivariateFloat32");
  pyinterp::implement_bivariate<geometry::EquatorialPoint2D, double, int64_t>(
      m, "BivariateInt64");
  pyinterp::implement_bivariate<geometry::EquatorialPoint2D, double, uint64_t>(
      m, "BivariateUInt64");
  pyinterp::implement_bivariate<geometry::EquatorialPoint2D, double, int32_t>(
      m, "BivariateInt32");
  pyinterp::implement_bivariate<geometry::EquatorialPoint2D, double, uint32_t>(
      m, "BivariateUInt32");
  pyinterp::implement_bivariate<geometry::EquatorialPoint2D, double, int16_t>(
      m, "BivariateInt16");
  pyinterp::implement_bivariate<geometry::EquatorialPoint2D, double, uint16_t>(
      m, "BivariateUInt16");
  pyinterp::implement_bivariate<geometry::EquatorialPoint2D, double, int8_t>(
      m, "BivariateInt8");
  pyinterp::implement_bivariate<geometry::EquatorialPoint2D, double, uint8_t>(
      m, "BivariateUInt8");

  pyinterp::implement_trivariate<geometry::EquatorialPoint3D, double, double>(
      m, "TrivariateFloat64");
  pyinterp::implement_trivariate<geometry::EquatorialPoint3D, double, float>(
      m, "TrivariateFloat32");
  pyinterp::implement_trivariate<geometry::EquatorialPoint3D, double, int64_t>(
      m, "TrivariateInt64");
  pyinterp::implement_trivariate<geometry::EquatorialPoint3D, double, uint64_t>(
      m, "TrivariateUInt64");
  pyinterp::implement_trivariate<geometry::EquatorialPoint3D, double, int32_t>(
      m, "TrivariateInt32");
  pyinterp::implement_trivariate<geometry::EquatorialPoint3D, double, uint32_t>(
      m, "TrivariateUInt32");
  pyinterp::implement_trivariate<geometry::EquatorialPoint3D, double, int16_t>(
      m, "TrivariateInt16");
  pyinterp::implement_trivariate<geometry::EquatorialPoint3D, double, uint16_t>(
      m, "TrivariateUInt16");
  pyinterp::implement_trivariate<geometry::EquatorialPoint3D, double, int8_t>(
      m, "TrivariateInt8");
  pyinterp::implement_trivariate<geometry::EquatorialPoint3D, double, uint8_t>(
      m, "TrivariateUInt8");
}
