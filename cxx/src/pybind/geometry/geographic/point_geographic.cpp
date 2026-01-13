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
      .def(nb::init<>(), "Construct a point at (0, 0).")
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
      .def("__eq__",
           [](const Point& self, const Point& other) -> bool {
             return boost::geometry::equals(self, other);
           })

      .def("__ne__",
           [](const Point& self, const Point& other) -> bool {
             return !boost::geometry::equals(self, other);
           })

      // String representation
      .def("__repr__",
           [](const Point& self) -> std::string {
             return std::format("Point(lon={}, lat={})", self.lon(),
                                self.lat());
           })

      .def("__str__",
           [](const Point& self) -> std::string {
             return std::format("({}, {})", self.lon(), self.lat());
           })

      // Hash support for use in sets/dicts
      .def("__hash__",
           [](const Point& self) -> size_t {
             std::hash<Point> hasher;
             return hasher(self);
           })

      // Pickle support
      .def("__getstate__",
           [](const Point& self) -> std::tuple<double, double> {
             return std::make_tuple(self.lon(), self.lat());
           })

      .def("__setstate__",
           [](Point* self, const std::tuple<double, double>& state) -> void {
             new (self) Point(std::get<0>(state), std::get<1>(state));
           });
}

}  // namespace pyinterp::geometry::geographic::pybind
