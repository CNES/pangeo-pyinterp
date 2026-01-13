// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <Eigen/Core>
#include <boost/geometry.hpp>
#include <format>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "pyinterp/geometry/cartesian/linestring.hpp"
#include "pyinterp/geometry/cartesian/point.hpp"
#include "pyinterp/pybind/ndarray_serialization.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

using pyinterp::pybind::NanobindArray1DUInt8;
using pyinterp::pybind::reader_from_ndarray;
using pyinterp::pybind::writer_to_ndarray;

namespace pyinterp::geometry::cartesian::pybind {

constexpr auto kLineStringClassDoc = R"doc(
A linestring in Cartesian coordinates.

A LineString is an ordered sequence of points describing an open path on a
flat plane. Unlike a Ring, the linestring is not automatically closed.

Examples:
    >>> import numpy as np
    >>> from pyinterp.cartesian import LineString
    >>> x = np.array([0.0, 5.0, 10.0])
    >>> y = np.array([0.0, 5.0, 0.0])
    >>> line = LineString(x, y)
    >>> len(line)
    3
)doc";

constexpr auto kLineStringInitDoc = R"doc(
Construct a linestring from x and y arrays.

Args:
    x: Array of x coordinate values.
    y: Array of y coordinate values (must have same size as x).

Raises:
    ValueError: If x and y arrays have different sizes.
)doc";

auto init_linestring(nb::module_& m) -> void {
  nb::class_<LineString>(m, "LineString", kLineStringClassDoc)
      .def(nb::init<>(), "Construct an empty linestring.")

      .def(
          "__init__",
          [](LineString* self, const Eigen::Ref<const Eigen::VectorXd>& xs,
             const Eigen::Ref<const Eigen::VectorXd>& ys) -> void {
            nb::gil_scoped_release release;
            new (self) LineString(xs, ys);
          },
          "xs"_a, "ys"_a, kLineStringInitDoc)

      // Container operations
      .def("__len__", &LineString::size,
           "Return the number of points in the linestring.")

      .def(
          "__getitem__",
          [](const LineString& self, Eigen::Index idx) -> Point {
            if (idx < 0 || std::cmp_greater_equal(idx, self.size())) {
              throw std::out_of_range("LineString index out of range");
            }
            return self[static_cast<size_t>(idx)];
          },
          "idx"_a, "Get the point at the given index.")

      .def(
          "__setitem__",
          [](LineString& self, Eigen::Index idx, const Point& point) -> void {
            if (idx < 0 || std::cmp_greater_equal(idx, self.size())) {
              throw std::out_of_range("LineString index out of range");
            }
            self[static_cast<size_t>(idx)] = point;
          },
          "idx"_a, "point"_a, "Set the point at the given index.")

      .def(
          "append",
          [](LineString& self, const Point& point) -> void {
            self.push_back(point);
          },
          "point"_a, "Append a point to the linestring.")

      .def("clear", &LineString::clear,
           "Remove all points from the linestring.")

      .def(
          "__bool__",
          [](const LineString& self) -> bool { return !self.empty(); },
          "Return True if the linestring is not empty.")

      // Iteration support
      .def("__iter__",
           [](const LineString& self) -> nb::object {
             nb::list result;
             for (const auto& point : self) {
               result.append(point);
             }
             return result.attr("__iter__")();
           })

      // Comparison operators
      .def("__eq__",
           [](const LineString& self, const LineString& other) -> bool {
             return boost::geometry::equals(self, other);
           })

      .def("__ne__",
           [](const LineString& self, const LineString& other) -> bool {
             return !boost::geometry::equals(self, other);
           })

      // String representation
      .def("__repr__",
           [](const LineString& self) -> std::string {
             return std::format("LineString({} points)", self.size());
           })

      .def("__str__",
           [](const LineString& self) -> std::string {
             std::ostringstream oss;
             oss << "LineString[";
             for (size_t i = 0; i < self.size(); ++i) {
               if (i > 0) oss << ", ";
               oss << "(" << self[i].x() << ", " << self[i].y() << ")";
               if (i >= 3 && self.size() > 5) {
                 oss << ", ...";
                 break;
               }
             }
             oss << "]";
             return oss.str();
           })

      // Pickle support
      .def("__getstate__",
           [](const LineString& self) -> nb::tuple {
             serialization::Writer state;
             {
               nb::gil_scoped_release release;
               state = self.pack();
             }
             return nb::make_tuple(writer_to_ndarray(std::move(state)));
           })

      .def("__setstate__",
           [](LineString* self, const nb::tuple& state) -> void {
             if (state.size() != 1) {
               throw std::invalid_argument("Invalid state");
             }
             auto array = nanobind::cast<NanobindArray1DUInt8>(state[0]);
             auto reader = reader_from_ndarray(array);
             {
               nb::gil_scoped_release release;
               new (self) LineString(LineString::unpack(reader));
             }
           });
}

}  // namespace pyinterp::geometry::cartesian::pybind
