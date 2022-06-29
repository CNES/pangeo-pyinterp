// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/axis.hpp"

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/temporal_axis.hpp"

namespace py = pybind11;

using AxisInt64 = pyinterp::Axis<int64_t>;

template <class Axis, class Coordinates>
auto implement_axis(py::class_<Axis, std::shared_ptr<Axis>> &axis,
                    const std::string &name) {
  axis.def(
          "__repr__",
          [](const Axis &self) -> std::string {
            return static_cast<std::string>(self);
          },
          "Called by the ``repr()`` built-in function to compute the string "
          "representation of an Axis.")
      .def(
          "__copy__", [](const Axis &self) { return Axis(self); },
          "Implements the shallow copy operation.",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "__getitem__",
          [](const Axis &self, size_t index) {
            return self.coordinate_value(index);
          },
          py::arg("index"))
      .def("__getitem__", &Axis::coordinate_values, py::arg("indices"))
      .def(
          "__len__", [](const Axis &self) { return self.size(); },
          "Called to implement the built-in function ``len()``")
      .def(
          "is_regular",
          [](const Axis &self) -> bool { return self.is_regular(); },
          R"__doc__(
Check if this axis values are spaced regularly.

Returns:
  True if this axis values are spaced regularly.
)__doc__")
      .def(
          "flip",
          [](std::shared_ptr<Axis> &self,
             const bool inplace) -> std::shared_ptr<Axis> {
            if (inplace) {
              self->flip();
              return self;
            }
            auto result =
                std::make_shared<Axis>(Axis::setstate(self->getstate()));
            result->flip();
            return result;
          },
          py::arg("inplace") = false,
          (R"__doc__(
Reverse the order of elements in this axis.

Args:
    inplace: If true, this instance will be modified, otherwise the
        modification will be made on a copy. Default to ``False``.

Returns:
    )__doc__" +
           name + R"__doc__(: The flipped axis.
)__doc__")
              .c_str())
      .def(
          "find_index",
          [](const Axis &self, Coordinates &coordinates,
             const bool bounded) -> py::array_t<int64_t> {
            return self.find_index(coordinates, bounded);
          },
          py::arg("coordinates"), py::arg("bounded") = false, R"__doc__(
Given coordinate positions, find what grid elements contains them, or is
closest to them.

Args:
    coordinates: Positions in this coordinate system.
    bounded: True if you want to obtain the closest value to a coordinate
        outside the axis definition range.
Returns:
    Index of the grid points containing them or -1 if the ``bounded`` parameter
    is set to false and if one of the searched indexes is out of the definition
    range of the axis, otherwise the index of the closest value of the
    coordinate is returned.
)__doc__")
      .def(
          "find_indexes",
          [](const Axis &self,
             Coordinates &coordinates) -> py::array_t<int64_t> {
            return self.find_indexes(coordinates);
          },
          py::arg("coordinates"), R"__doc__(
For all coordinate positions, search for the axis elements around them. This
means that for n coordinate ``ix`` of the provided array, the method searches
the indexes ``i0`` and ``i1`` as follow:

.. code::

  self[i0] <= coordinates[ix] <= self[i1]

The provided coordinates located outside the axis definition range are set to
``-1``.

Args:
    coordinates: Positions in this coordinate system.
Returns:
    A matrix of shape ``(n, 2)``. The first column of the matrix contains the
    indexes ``i0`` and the second column the indexes ``i1`` found.
)__doc__")
      .def("is_ascending", &Axis::is_ascending, R"__doc__(
Test if the data is sorted in ascending order.

Returns:
    True if the data is sorted in ascending order.
)__doc__")
      .def(
          "__eq__",
          [](const Axis &self, const Axis &rhs) -> bool { return self == rhs; },
          py::arg("other"),
          "Overrides the default behavior of the ``==`` operator.")
      .def(
          "__ne__",
          [](const Axis &self, const Axis &rhs) -> bool { return self != rhs; },
          py::arg("other"),
          "Overrides the default behavior of the ``!=`` operator.")
      .def(py::pickle(
          [](const Axis &self) { return self.getstate(); },
          [](const py::tuple &state) { return Axis::setstate(state); }));
}

static void init_core_axis(py::module &m) {
  using Axis = pyinterp::Axis<double>;

  auto axis = py::class_<Axis, std::shared_ptr<Axis>>(m, "Axis", R"__doc__(
Axis(self, values: numpy.ndarray[numpy.float64], epsilon: float = 1e-06, is_circle: bool = False)

A coordinate axis is a Variable that specifies one of the coordinates
of a variable's values.

Args:
    values: Axis values.
    epsilon: Maximum allowed difference between two real
        numbers in order to consider them equal. Defaults to ``1e-6``.
    is_circle: True, if the axis can represent a circle. Defaults to ``false``.
)__doc__");

  axis.def(py::init<>([](py::array_t<double, py::array::c_style> &values,
                         const double epsilon, const bool is_circle) {
             return std::make_shared<Axis>(values, epsilon, is_circle);
           }),
           py::arg("values"), py::arg("epsilon") = 1e-6,
           py::arg("is_circle") = false)
      .def_property_readonly(
          "is_circle",
          [](const Axis &self) -> bool { return self.is_circle(); },
          R"__doc__(
Test if this axis represents a circle.

Returns:
    True if this axis represents a circle.
)__doc__")
      .def("front", &Axis::front, R"__doc__(
Get the first value of this axis.

Returns:
    The first value.
)__doc__")
      .def("back", &Axis::back, R"__doc__(
Get the last value of this axis.

Returns:
    The last value.
)__doc__")
      .def("increment", &Axis::increment, R"__doc__(
Get increment value if is_regular().

Raises:
    RuntimeError: if this instance does not represent a regular axis.
Returns:
    float: Increment value.
)__doc__")
      .def("min_value", &Axis::min_value, R"__doc__(
Get the minimum coordinate value.

Returns:
    The minimum coordinate value.
)__doc__")
      .def("max_value", &Axis::max_value, R"__doc__(
Get the maximum coordinate value.

Returns:
    The maximum coordinate value.
)__doc__");
  implement_axis<Axis, const py::array_t<double>>(axis, "pyinterp.core.Axis");
}

auto init_axis_int64(py::module &m)
    -> py::class_<AxisInt64, std::shared_ptr<AxisInt64>> {
  auto axis = py::class_<AxisInt64, std::shared_ptr<AxisInt64>>(m, "AxisInt64",
                                                                R"__doc__(
AxisInt64(self, values: numpy.ndarray[numpy.int64])

A coordinate axis is a Variable that specifies one of the coordinates
of a variable's values.

Args:
    values: Axis values.
)__doc__");

  axis.def(py::init<>(
               [](const py::array_t<int64_t, py::array::c_style> &values) {
                 return std::make_shared<AxisInt64>(values, 0, false);
               }),
           py::arg("values"))
      .def("front", &AxisInt64::front, R"__doc__(
Get the first value of this axis.

Returns:
    The first value.
)__doc__")
      .def("back", &AxisInt64::back, R"__doc__(
Get the last value of this axis.

Returns:
    The last value.
)__doc__")
      .def("increment", &AxisInt64::increment, R"__doc__(
Get increment value if is_regular().

Raises:
    RuntimeError: if this instance does not represent a regular axis.
Returns:
    Increment value.
)__doc__")
      .def("min_value", &AxisInt64::min_value, R"__doc__(
Get the minimum coordinate value.

Returns:
    The minimum coordinate value.
)__doc__")
      .def("max_value", &AxisInt64::max_value, R"__doc__(
Get the maximum coordinate value.

Returns:
    The maximum coordinate value.
)__doc__");

  implement_axis<AxisInt64, const py::array_t<int64_t>>(
      axis, "pyinterp.core.AxisInt64");
  return axis;
}

void init_temporal_axis(
    py::module &m,
    const py::class_<AxisInt64, std::shared_ptr<AxisInt64>> &base_class) {
  auto axis = py::class_<pyinterp::TemporalAxis,
                         std::shared_ptr<pyinterp::TemporalAxis>>(
      m, "TemporalAxis", base_class,
      R"__doc__(
TemporalAxis(self, values: numpy.ndarray)

Time axis

Args:
    values: Items representing the datetimes or timedeltas of the axis.

Raises:
    TypeError: if the array data type is not a datetime64 or timedelta64
        subtype.

Examples:

    >>> import datetime
    >>> import numpy as np
    >>> import pyinterp
    >>> start = datetime.datetime(2000, 1, 1)
    >>> values = np.array([
    ...     start + datetime.timedelta(hours=index)
    ...     for index in range(86400)
    ... ],
    ...                   dtype="datetime64[us]")
    >>> axis = pyinterp.TemporalAxis(values)
    >>> axis
    <pyinterp.core.TemporalAxis>
      min_value: 2000-01-01T00:00:00.000000
      max_value: 2009-11-08T23:00:00.000000
      step     : 3600000000 microseconds
    >>> values = np.array([
    ...     datetime.timedelta(hours=index)
    ...     for index in range(86400)
    ... ],
    ...                   dtype="timedelta64[us]")
    >>> axis = pyinterp.TemporalAxis(values)
    >>> axis
    <pyinterp.core.TemporalAxis>
      min_value: 0 microseconds
      max_value: 311036400000000 microseconds
      step     : 3600000000 microseconds

)__doc__");

  axis.def(py::init<>([](const py::array &values) {
             return std::make_shared<pyinterp::TemporalAxis>(values);
           }),
           py::arg("values"))
      .def("dtype", &pyinterp::TemporalAxis::dtype, R"__doc__(
Data-type of the axis's elements.

Returns:
    Numpy dtype object.
)__doc__")
      .def("front", &pyinterp::TemporalAxis::front, R"__doc__(
Get the first value of this axis.

Returns:
    The first value.
)__doc__")
      .def("back", &pyinterp::TemporalAxis::back, R"__doc__(
Get the last value of this axis.

Returns:
    The last value.
)__doc__")
      .def("increment", &pyinterp::TemporalAxis::increment, R"__doc__(
Get increment value if is_regular().

Raises:
    RuntimeError: if this instance does not represent a regular axis.
Returns:
    Increment value.
)__doc__")
      .def("min_value", &pyinterp::TemporalAxis::min_value, R"__doc__(
Get the minimum coordinate value.

Returns:
    The minimum coordinate value.
)__doc__")
      .def("max_value", &pyinterp::TemporalAxis::max_value, R"__doc__(
Get the maximum coordinate value.

Returns:
    The maximum coordinate value.
)__doc__")
      .def(
          "safe_cast",
          [](const pyinterp::TemporalAxis &self, const pybind11::array &array)
              -> py::array { return self.safe_cast("values", array); },
          py::arg("values"),
          R"__doc__(
Convert the values of the vector in the same unit as the time axis
handled by this instance.

Args:
    values: Values to convert.

Returns:
    Values converted.

Raises:
    UserWarning: If the implicit conversion of the supplied values to the
        resolution of the axis truncates the values (e.g. converting
        microseconds to seconds).
)__doc__");

  implement_axis<pyinterp::TemporalAxis, py::array>(
      axis, "pyinterp.core.TemporalAxis");
}

void init_axis(py::module &m) {
  init_core_axis(m);
  init_temporal_axis(m, init_axis_int64(m));
}
