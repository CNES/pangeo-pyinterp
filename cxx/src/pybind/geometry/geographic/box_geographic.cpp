// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <Eigen/Core>
#include <boost/geometry.hpp>
#include <format>

#include "pyinterp/geometry/geographic/box.hpp"
#include "pyinterp/geometry/geographic/point.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::geometry::geographic::pybind {

constexpr auto kBoxClassDoc = R"doc(
A box (rectangle) in geographic coordinates.

The box is represented by two corner points: the minimum corner (southwest)
and the maximum corner (northeast). Coordinates are in degrees.

Examples:
    >>> import numpy as np
    >>> from pyinterp.geodetic import Box
    >>> # Create box from (-10, 40) to (10, 50)
    >>> lon = (-10.0, 40.0)
    >>> lat = (10.0, 50.0)
    >>> box = Box(lon, lat)
    >>> box.min_corner()
    Point(lon=-10.0, lat=40.0)
    >>> box.max_corner()
    Point(lon=10.0, lat=50.0)
)doc";

constexpr auto kBoxInitDoc = R"doc(
Construct a box from longitude and latitude tuples.

Args:
    min_corner: Tuple of (lon_min, lat_min) for the minimum corner.
    max_corner: Tuple of (lon_max, lat_max) for the maximum corner.
)doc";

constexpr auto kMinCornerDoc = R"doc(
Minimum corner (southwest) of the box.
)doc";

constexpr auto kMaxCornerDoc = R"doc(
Maximum corner (northeast) of the box.
)doc";

constexpr auto kCentroidDoc = R"doc(
Get the centroid (center point) of the box.

Returns:
    Point: The center point of the box.
)doc";

auto init_box(nb::module_& m) -> void {
  nb::class_<Box>(m, "Box", kBoxClassDoc)
      .def(nb::init<>(), "Construct an empty box.")

      .def(
          "__init__",
          [](Box* self, const std::tuple<double, double>& min_corner,
             const std::tuple<double, double>& max_corner) -> void {
            new (self)
                Box(Point(std::get<0>(min_corner), std::get<1>(min_corner)),
                    Point(std::get<0>(max_corner), std::get<1>(max_corner)));
          },
          "min_corner"_a, "max_corner"_a, kBoxInitDoc)

      // Corner accessors (read-only to avoid returning references)
      .def_prop_rw(
          "min_corner",
          [](const Box& self) -> pyinterp::geometry::geographic::Point {
            return self.min_corner();
          },
          [](Box& self, const Point& pt) -> void { self.min_corner() = pt; },
          kMinCornerDoc)

      .def_prop_rw(
          "max_corner",
          [](const Box& self) -> pyinterp::geometry::geographic::Point {
            return self.max_corner();
          },
          [](Box& self, const Point& pt) -> void { self.max_corner() = pt; },
          kMaxCornerDoc)

      .def("centroid", &Box::centroid, kCentroidDoc)

      // Comparison operators
      .def("__eq__",
           [](const Box& self, const Box& other) -> bool {
             return boost::geometry::equals(self, other);
           })

      .def("__ne__",
           [](const Box& self, const Box& other) -> bool {
             return !boost::geometry::equals(self, other);
           })

      // String representation
      .def("__repr__",
           [](const Box& self) -> std::string {
             const auto& min = self.min_corner();
             const auto& max = self.max_corner();
             return std::format("Box(min=({}, {}), max=({}, {}))", min.lon(),
                                min.lat(), max.lon(), max.lat());
           })

      .def("__str__",
           [](const Box& self) -> std::string {
             const auto& min = self.min_corner();
             const auto& max = self.max_corner();
             return std::format("[({}, {}) to ({}, {})]", min.lon(), min.lat(),
                                max.lon(), max.lat());
           })

      // Hash support
      .def("__hash__",
           [](const Box& self) -> size_t {
             const auto& min = self.min_corner();
             const auto& max = self.max_corner();
             auto h1 = std::hash<double>{}(min.lon());
             auto h2 = std::hash<double>{}(min.lat());
             auto h3 = std::hash<double>{}(max.lon());
             auto h4 = std::hash<double>{}(max.lat());
             return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
           })

      // Pickle support
      .def("__getstate__",
           [](const Box& self) -> std::tuple<double, double, double, double> {
             const auto& min = self.min_corner();
             const auto& max = self.max_corner();
             return std::make_tuple(min.lon(), min.lat(), max.lon(), max.lat());
           })

      .def("__setstate__",
           [](Box* self,
              const std::tuple<double, double, double, double>& state) -> void {
             new (self) Box(Point(std::get<0>(state), std::get<1>(state)),
                            Point(std::get<2>(state), std::get<3>(state)));
           });
}

}  // namespace pyinterp::geometry::geographic::pybind
