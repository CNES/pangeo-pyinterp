// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <boost/geometry.hpp>
#include <format>

#include "pyinterp/geometry/cartesian/point.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::geometry::cartesian::pybind {

constexpr auto kPointClassDoc = R"doc(
A point in Cartesian coordinates.

The point is represented using Cartesian coordinates (x, y) on a flat plane.

Examples:
    >>> from pyinterp.geometry.cartesian import Point
    >>> point = Point(100.0, 200.0)
    >>> point.x
    100.0
    >>> point.y
    200.0
)doc";

constexpr auto kPointInitDoc = R"doc(
Construct a point from x and y coordinates.

Args:
    x: X coordinate (default: 0.0).
    y: Y coordinate (default: 0.0).
)doc";

constexpr auto kXDoc = R"doc(
X coordinate.
)doc";

constexpr auto kYDoc = R"doc(
Y coordinate.
)doc";

auto init_point(nb::module_& m) -> void {
  nb::class_<Point>(m, "Point", kPointClassDoc)
      .def(nb::init<double, double>(), "x"_a = 0.0, "y"_a = 0.0, kPointInitDoc)

      // Properties
      .def_prop_rw(
          "x", [](const Point& self) -> double { return self.x(); },
          [](Point& self, double value) -> void { self.x() = value; }, kXDoc)
      .def_prop_rw(
          "y", [](const Point& self) -> double { return self.y(); },
          [](Point& self, double value) -> void { self.y() = value; }, kYDoc)

      // Comparison operators
      .def(
          "__eq__",
          [](const Point& self, const Point& other) -> bool {
            return boost::geometry::equals(self, other);
          },
          "other"_a, "Check if two points are equal.")

      .def(
          "__ne__",
          [](const Point& self, const Point& other) -> bool {
            return !boost::geometry::equals(self, other);
          },
          "other"_a, "Check if two points are not equal.")

      // String representation
      .def(
          "__repr__",
          [](const Point& self) -> std::string {
            return std::format("Point(x={}, y={})", self.x(), self.y());
          },
          "Return the official string representation of the point.")

      .def(
          "__str__",
          [](const Point& self) -> std::string {
            return std::format("({}, {})", self.x(), self.y());
          },
          "Return the informal string representation of the point.")

      // Hash support for use in sets/dicts
      .def(
          "__hash__",
          [](const Point& self) -> size_t {
            std::hash<Point> hasher;
            return hasher(self);
          },
          "Return the hash value of the point.")

      // Pickle support
      .def(
          "__getstate__",
          [](const Point& self) -> std::tuple<double, double> {
            return std::make_tuple(self.x(), self.y());
          },
          "Return the state of the point for pickling.")

      .def(
          "__setstate__",
          [](Point* self, const std::tuple<double, double>& state) -> void {
            new (self) Point(std::get<0>(state), std::get<1>(state));
          },
          "state"_a, "Restore the state of the point from pickling.");
}

}  // namespace pyinterp::geometry::cartesian::pybind
