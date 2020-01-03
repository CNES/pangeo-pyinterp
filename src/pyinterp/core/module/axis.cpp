// Copyright (c) 2020 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <pybind11/eigen.h>
#include "pyinterp/axis.hpp"
#include "pyinterp/detail/broadcast.hpp"
namespace py = pybind11;

template <typename T>
void implement_axis(py::module& m, const std::string& prefix) {
  auto class_name = prefix + "Axis";
  auto axis = py::class_<pyinterp::Axis<T>, std::shared_ptr<pyinterp::Axis<T>>>(
      m, class_name.c_str(), R"__doc__(
A coordinate axis is a Variable that specifies one of the coordinates
of a variable's values.
)__doc__");

  axis.def(py::init<py::array_t<T, py::array::c_style>&, T, bool>(),
           py::arg("values"), py::arg("epsilon") = static_cast<T>(1e-6),
           py::arg("is_circle") = false,
           R"__doc__(
Create a coordinate axis from values.

Args:
    values (numpy.ndarray): Axis values.
    epsilon (float, optional): Maximum allowed difference between two real
        numbers in order to consider them equal. Defaults to ``1e-6``.
    is_circle (bool, optional): True, if the axis can represent a
        circle. Defaults to ``false``.
)__doc__")
      .def("__len__",
           [](const pyinterp::Axis<T>& self) -> size_t { return self.size(); })
      .def("__getitem__",
           [](const pyinterp::Axis<T>& self, size_t index) -> T {
             return self.coordinate_value(index);
           })
      .def("__getitem__", &pyinterp::Axis<T>::coordinate_values)
      .def("min_value", &pyinterp::Axis<T>::min_value, R"__doc__(
Get the minimum coordinate value.

Return:
    float: The minimum coordinate value.
)__doc__")
      .def("max_value", &pyinterp::Axis<T>::max_value, R"__doc__(
Get the maximum coordinate value.

Return:
    float: The maximum coordinate value.
)__doc__")
      .def("is_regular",
           [](const pyinterp::Axis<T>& self) -> bool {
             return self.is_regular();
           },
           R"__doc__(
Check if this axis values are spaced regularly

Return:
  bool: True if this axis values are spaced regularly
)__doc__")
      .def("flip",
           [](std::shared_ptr<pyinterp::Axis<T>>& self,
              const bool inplace) -> std::shared_ptr<pyinterp::Axis<T>> {
             if (inplace) {
               self->flip();
               return self;
             }
             auto result = std::make_shared<pyinterp::Axis<T>>(
                 pyinterp::Axis<T>::setstate(self->getstate()));
             result->flip();
             return result;
           },
           py::arg("inplace") = false,
           (R"__doc__(
Reverse the order of elements in this axis

Args:
    inplace (bool, optional): If true, this instance will be modified,
        otherwise the modification will be made on a copy. Default to
        ``False``.

Return:
    pyinterp.core.)__doc__" +
            class_name +
            R"__doc__(: The flipped axis
)__doc__")
               .c_str())
      .def("find_index",
           [](const pyinterp::Axis<T>& self, const py::array_t<T>& coordinates,
              const bool bounded) -> py::array_t<int64_t> {
             return self.find_index(coordinates, bounded);
           },
           py::arg("coordinates"), py::arg("bounded") = false, R"__doc__(
Given coordinate positions, find what grid elements contains them, or is
closest to them.

Args:
    coordinates (numpy.ndarray): Positions in this coordinate system
    bounded (bool, optional): True if you want to obtain the closest value to
        a coordinate outside the axis definition range.
Return:
    numpy.ndarray: index of the grid points containing them or -1 if the
    ``bounded`` parameter is set to false and if one of the searched indexes
    is out of the definition range of the axis, otherwise the index of the
    closest value of the coordinate is returned.
)__doc__")
      .def("front", &pyinterp::Axis<T>::front, R"__doc__(
Get the first value of this axis

Return:
    float: The first value
)__doc__")
      .def("back", &pyinterp::Axis<T>::back, R"__doc__(
Get the last value of this axis

Return:
    float: The last value
)__doc__")
      .def("is_ascending", &pyinterp::Axis<T>::is_ascending, R"__doc__(
Test if the data is sorted in ascending order.

Return:
    bool: True if the data is sorted in ascending order.
)__doc__")
      .def("increment", &pyinterp::Axis<T>::increment, R"__doc__(
Get increment value if is_regular()

Raises:
    RuntimeError: if this instance does not represent a regular axis
Return:
    float: Increment value
)__doc__")
      .def_property_readonly("is_circle",
                             [](const pyinterp::Axis<T>& self) -> bool {
                               return self.is_circle();
                             },
                             R"__doc__(
Test if this axis represents a circle.

Return:
    bool: True if this axis represents a circle
)__doc__")
      .def("__eq__",
           [](const pyinterp::Axis<T>& self,
              const pyinterp::Axis<T>& rhs) -> bool { return self == rhs; },
           py::arg("other"),
           "Overrides the default behavior of the ``==`` operator.")
      .def("__ne__",
           [](const pyinterp::Axis<T>& self,
              const pyinterp::Axis<T>& rhs) -> bool { return self != rhs; },
           py::arg("other"),
           "Overrides the default behavior of the ``!=`` operator.")
      .def("__repr__",
           [](const pyinterp::Axis<T>& self) -> std::string {
             return static_cast<std::string>(self);
           },
           "Called by the ``repr()`` built-in function to compute the string "
           "representation of an Axis.")
      .def(py::pickle(
          [](const pyinterp::Axis<T>& self) { return self.getstate(); },
          [](const py::tuple& state) {
            return pyinterp::Axis<T>::setstate(state);
          }));
}

void init_axis(py::module& m) {
  py::enum_<pyinterp::axis::Boundary>(m, "AxisBoundary", R"__doc__(
Type of boundary handling.
)__doc__")
      .value("Expand", pyinterp::axis::kExpand,
             "*Expand the boundary as a constant*.")
      .value("Wrap", pyinterp::axis::kWrap, "*Circular boundary conditions*.")
      .value("Sym", pyinterp::axis::kSym, "*Symmetrical boundary conditions*.")
      .value("Undef", pyinterp::axis::kUndef,
             "*Boundary violation is not defined*.");

  implement_axis<double>(m, "");
  implement_axis<int64_t>(m, "Temporal");
}
