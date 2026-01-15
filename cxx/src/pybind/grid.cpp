// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/pybind/grid.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <variant>

namespace pyinterp::pybind {

namespace {

/// @brief Axis variant type for factory functions
using AxisVariant = std::variant<Axis<double>, TemporalAxis>;

/// @brief Create a 1D grid from axes and array
auto create_grid_1d(const AxisVariant& x_var, const nanobind::object& array)
    -> GridHolder {
  if (std::holds_alternative<TemporalAxis>(x_var)) {
    throw std::invalid_argument("1D grids do not support temporal axes");
  }
  const auto& x = std::get<Axis<double>>(x_var);
  return grid_factory<MathSpatialAxis<>>(std::make_tuple(x), array);
}

/// @brief Create a 2D grid from axes and array
auto create_grid_2d(const AxisVariant& x_var, const AxisVariant& y_var,
                    const nanobind::object& array) -> GridHolder {
  if (std::holds_alternative<TemporalAxis>(x_var) ||
      std::holds_alternative<TemporalAxis>(y_var)) {
    throw std::invalid_argument("2D grids do not support temporal axes");
  }
  const auto& x = std::get<Axis<double>>(x_var);
  const auto& y = std::get<Axis<double>>(y_var);
  return grid_factory<MathSpatialAxis<>, MathSpatialAxis<>>(
      std::make_tuple(x, y), array);
}

/// @brief Create a 3D grid from axes and array
auto create_grid_3d(const AxisVariant& x_var, const AxisVariant& y_var,
                    const AxisVariant& z_var, const nanobind::object& array)
    -> GridHolder {
  // Check for invalid temporal axis placement
  if (std::holds_alternative<TemporalAxis>(x_var) ||
      std::holds_alternative<TemporalAxis>(y_var)) {
    throw std::invalid_argument(
        "Temporal axis must be on the z-axis for 3D grids");
  }
  const auto& x = std::get<Axis<double>>(x_var);
  const auto& y = std::get<Axis<double>>(y_var);

  if (std::holds_alternative<TemporalAxis>(z_var)) {
    // Temporal grid
    const auto& z = std::get<TemporalAxis>(z_var);
    return grid_factory<MathSpatialAxis<>, MathSpatialAxis<>, MathTemporalAxis>(
        std::make_tuple(x, y, z), array);
  }
  // Spatial grid
  const auto& z = std::get<Axis<double>>(z_var);
  return grid_factory<MathSpatialAxis<>, MathSpatialAxis<>, MathSpatialAxis<>>(
      std::make_tuple(x, y, z), array);
}

/// @brief Create a 4D grid from axes and array
auto create_grid_4d(const AxisVariant& x_var, const AxisVariant& y_var,
                    const AxisVariant& z_var, const AxisVariant& u_var,
                    const nanobind::object& array) -> GridHolder {
  // Check for invalid temporal axis placement
  if (std::holds_alternative<TemporalAxis>(x_var) ||
      std::holds_alternative<TemporalAxis>(y_var) ||
      std::holds_alternative<TemporalAxis>(u_var)) {
    throw std::invalid_argument(
        "Temporal axis must be on the z-axis for 4D grids");
  }
  const auto& x = std::get<Axis<double>>(x_var);
  const auto& y = std::get<Axis<double>>(y_var);
  const auto& u = std::get<Axis<double>>(u_var);

  if (std::holds_alternative<TemporalAxis>(z_var)) {
    // Temporal grid
    const auto& z = std::get<TemporalAxis>(z_var);
    return grid_factory<MathSpatialAxis<>, MathSpatialAxis<>, MathTemporalAxis,
                        MathSpatialAxis<>>(std::make_tuple(x, y, z, u), array);
  }
  // Spatial grid
  const auto& z = std::get<Axis<double>>(z_var);
  return grid_factory<MathSpatialAxis<>, MathSpatialAxis<>, MathSpatialAxis<>,
                      MathSpatialAxis<>>(std::make_tuple(x, y, z, u), array);
}

constexpr const char* kGridDocstring = R"doc(
N-dimensional Cartesian grid with runtime dtype detection.

This class represents a grid of values defined on N spatial or temporal axes.
The grid automatically detects the data type from the input array and creates
the appropriate internal representation.
)doc";

}  // namespace

auto init_grids(nanobind::module_& m) -> void {
  namespace nb = nanobind;

  // Helper to convert axis object to variant
  auto axis_to_variant = [](const nb::object& axis) -> AxisVariant {
    if (nb::isinstance<Axis<double>>(axis)) {
      return nb::cast<Axis<double>>(axis);
    }
    if (nb::isinstance<TemporalAxis>(axis)) {
      return nb::cast<TemporalAxis>(axis);
    }
    throw std::invalid_argument("Axis must be Axis or TemporalAxis");
  };

  // Bind the GridHolder class
  nb::class_<GridHolder>(m, "GridHolder", kGridDocstring)
      .def_prop_ro("ndim", &GridHolder::ndim, "Number of dimensions")
      .def_prop_ro(
          "dtype",
          [](const GridHolder& self) -> std::string {
            return std::string(self.dtype_str());
          },
          "Data type of the grid values")
      .def_prop_ro("has_temporal_axis", &GridHolder::has_temporal_axis,
                   "Whether this grid has a temporal axis")
      .def_prop_ro("temporal_axis_index", &GridHolder::temporal_axis_index,
                   "Index of the temporal axis, or -1 if none")
      .def_prop_ro(
          "shape",
          [](const GridHolder& self) -> nb::object {
            return nb::cast(self.shape());
          },
          "Shape of the grid")
      .def_prop_ro(
          "x",
          [](const GridHolder& self) -> nb::object {
            if (self.ndim() < 1) {
              return nb::none();
            }
            return self.pybind_axis_object(0);
          },
          "X-axis (first axis)")
      .def_prop_ro(
          "y",
          [](const GridHolder& self) -> nb::object {
            if (self.ndim() < 2) {
              return nb::none();
            }
            return self.pybind_axis_object(1);
          },
          "Y-axis (second axis)")
      .def_prop_ro(
          "z",
          [](const GridHolder& self) -> nb::object {
            if (self.ndim() < 3) {
              return nb::none();
            }
            return self.pybind_axis_object(2);
          },
          "Z-axis (third axis)")
      .def_prop_ro(
          "u",
          [](const GridHolder& self) -> nb::object {
            if (self.ndim() < 4) {
              return nb::none();
            }
            return self.pybind_axis_object(3);
          },
          "U-axis (fourth axis)")
      .def_prop_ro("array", &GridHolder::array_object,
                   "The underlying data array")
      .def("__repr__", &GridHolder::repr,
           "Return the string representation of this Grid.")
      .def("__getstate__", &GridHolder::getstate,
           "Get the state for pickling (axes..., array).");

  // 1D grid factory
  m.def(
      "Grid",
      [axis_to_variant](const nb::object& x, const nb::object& array)
          -> GridHolder { return create_grid_1d(axis_to_variant(x), array); },
      nb::arg("x"), nb::arg("array"),
      "Create a 1D grid with automatic dtype detection.");

  // 2D grid factory
  m.def(
      "Grid",
      [axis_to_variant](const nb::object& x, const nb::object& y,
                        const nb::object& array) -> GridHolder {
        return create_grid_2d(axis_to_variant(x), axis_to_variant(y), array);
      },
      nb::arg("x"), nb::arg("y"), nb::arg("array"),
      "Create a 2D grid with automatic dtype detection.");

  // 3D grid factory
  m.def(
      "Grid",
      [axis_to_variant](const nb::object& x, const nb::object& y,
                        const nb::object& z,
                        const nb::object& array) -> GridHolder {
        return create_grid_3d(axis_to_variant(x), axis_to_variant(y),
                              axis_to_variant(z), array);
      },
      nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("array"),
      "Create a 3D grid with automatic dtype detection. "
      "Use TemporalAxis for z to create a temporal grid.");

  // 4D grid factory
  m.def(
      "Grid",
      [axis_to_variant](const nb::object& x, const nb::object& y,
                        const nb::object& z, const nb::object& u,
                        const nb::object& array) -> GridHolder {
        return create_grid_4d(axis_to_variant(x), axis_to_variant(y),
                              axis_to_variant(z), axis_to_variant(u), array);
      },
      nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("u"), nb::arg("array"),
      "Create a 4D grid with automatic dtype detection. "
      "Use TemporalAxis for z to create a temporal grid.");
}

}  // namespace pyinterp::pybind
