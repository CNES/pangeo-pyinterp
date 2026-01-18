// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <boost/geometry.hpp>
#include <format>

#include "pyinterp/geometry/geographic/point.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::geometry::geographic::pybind {

constexpr auto kPointClassDoc = R"doc(
A point in geodetic coordinates (longitude, latitude).

The point is represented using geographic coordinates (i.e., on the
ellipsoid surface). Longitude and latitude are expressed in degrees.

Examples:
    >>> from pyinterp.geometry.geographic import Point
    >>> point = Point(2.3, 48.9)
    >>> point.lon
    2.3
    >>> point.lat
    48.9
)doc";

constexpr auto kPointInitDoc = R"doc(
Construct a point from longitude and latitude.

Args:
    lon: Longitude in degrees (default: 0.0).
    lat: Latitude in degrees (default: 0.0).
)doc";

constexpr auto kLonDoc = R"doc(
Longitude in degrees.
)doc";

constexpr auto kLatDoc = R"doc(
Latitude in degrees.
)doc";

auto init_point(nb::module_& m) -> void {
  nb::class_<Point>(m, "Point", kPointClassDoc)
      .def(nb::init<double, double>(), "lon"_a = 0.0, "lat"_a = 0.0,
           kPointInitDoc)

      // Properties
      .def_prop_rw(
          "lon", [](const Point& self) -> double { return self.lon(); },
          [](Point& self, double value) -> void { self.lon() = value; },
          kLonDoc)
      .def_prop_rw(
          "lat", [](const Point& self) -> double { return self.lat(); },
          [](Point& self, double value) -> void { self.lat() = value; },
          kLatDoc)

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
            return std::format("Point(lon={}, lat={})", self.lon(), self.lat());
          },
          "Return the official string representation of the Point.")

      .def(
          "__str__",
          [](const Point& self) -> std::string {
            return std::format("({}, {})", self.lon(), self.lat());
          },
          "Return the string representation of the Point.")

      // Hash support for use in sets/dicts
      .def(
          "__hash__",
          [](const Point& self) -> size_t {
            std::hash<Point> hasher;
            return hasher(self);
          },
          "Return the hash value of the Point.")

      // Pickle support
      .def(
          "__getstate__",
          [](const Point& self) -> std::tuple<double, double> {
            return std::make_tuple(self.lon(), self.lat());
          },
          "Return the state of the Point for pickling.")

      .def(
          "__setstate__",
          [](Point* self, const std::tuple<double, double>& state) -> void {
            new (self) Point(std::get<0>(state), std::get<1>(state));
          },
          "state"_a, "Restore the state of the Point from pickling.");
}

}  // namespace pyinterp::geometry::geographic::pybind
