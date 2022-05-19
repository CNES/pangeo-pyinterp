// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <pybind11/pybind11.h>

#include "pyinterp/detail/axis.hpp"
#include "pyinterp/detail/math/radial_basis_functions.hpp"
#include "pyinterp/detail/math/window_functions.hpp"
#include "pyinterp/fill.hpp"

namespace axis = pyinterp::axis;
namespace fill = pyinterp::fill;
namespace math = pyinterp::detail::math;
namespace py = pybind11;

void init_enum(py::module& core, py::module& fill) {
  py::enum_<axis::Boundary>(core, "AxisBoundary", "Type of boundary handling.")
      .value("Expand", axis::kExpand, "*Expand the boundary as a constant*.")
      .value("Wrap", axis::kWrap, "*Circular boundary conditions*.")
      .value("Sym", axis::kSym, "*Symmetrical boundary conditions*.")
      .value("Undef", axis::kUndef, "*Boundary violation is not defined*.");

  py::enum_<math::RadialBasisFunction>(core, "RadialBasisFunction",
                                       "Radial basis functions")
      .value("Cubic", math::RadialBasisFunction::Cubic)
      .value("Gaussian", math::RadialBasisFunction::Gaussian)
      .value("InverseMultiquadric",
             math::RadialBasisFunction::InverseMultiquadric)
      .value("Linear", math::RadialBasisFunction::Linear)
      .value("Multiquadric", math::RadialBasisFunction::Multiquadric)
      .value("ThinPlate", math::RadialBasisFunction::ThinPlate);

  py::enum_<math::window::Function>(core, "WindowFunction", "Window functions")
      .value("Blackman", math::window::Function::kBlackman)
      .value("BlackmanHarris", math::window::Function::kBlackmanHarris)
      .value("Boxcar", math::window::Function::kBoxcar)
      .value("FlatTop", math::window::Function::kFlatTop)
      .value("Gaussian", math::window::Function::kGaussian)
      .value("Hamming", math::window::Function::kHamming)
      .value("Lanczos", math::window::Function::kLanczos)
      .value("Nuttall", math::window::Function::kNuttall)
      .value("Parzen", math::window::Function::kParzen)
      .value("ParzenSWOT", math::window::Function::kParzenSWOT);

  py::enum_<fill::FirstGuess>(
      fill, "FirstGuess",
      "Type of first guess grid to solve Poisson's equation.")
      .value("Zero", fill::kZero, "Use 0.0 as an initial guess")
      .value("ZonalAverage", fill::kZonalAverage,
             "Use zonal average in x direction");

  py::enum_<pyinterp::fill::ValueType>(fill, "ValueType",
                                       R"__doc__(
Type of values processed by the loess filter
)__doc__")
      .value("Undefined", fill::kUndefined,
             "*Undefined values (fill undefined values)*.")
      .value("Defined", fill::kDefined, "*Defined values (smooth values)*.")
      .value("All", fill::kAll, "*Smooth and fill values*.");
}
