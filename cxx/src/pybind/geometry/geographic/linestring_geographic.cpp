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

#include "pyinterp/geometry/geographic/linestring.hpp"
#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/pybind/ndarray_serialization.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

using pyinterp::pybind::NanobindArray1DUInt8;
using pyinterp::pybind::reader_from_ndarray;
using pyinterp::pybind::writer_to_ndarray;

namespace pyinterp::geometry::geographic::pybind {

constexpr auto kLineStringClassDoc = R"doc(
A linestring in geographic coordinates.

A LineString is an ordered sequence of points describing an open path on the
ellipsoid surface. Unlike a Ring, the linestring is not automatically closed.

Examples:
    >>> import numpy as np
    >>> from pyinterp.geometry.geographic import LineString
    >>> lon = np.array([0.0, 5.0, 10.0])
    >>> lat = np.array([0.0, 5.0, 0.0])
    >>> line = LineString(lon, lat)
    >>> len(line)
    3
)doc";

constexpr auto kLineStringInitDoc = R"doc(
Construct a linestring from longitude and latitude arrays.

Args:
    lon: Array of longitude values in degrees.
    lat: Array of latitude values in degrees (must have same size as lon).

Raises:
    ValueError: If lon and lat arrays have different sizes.
)doc";

auto init_linestring(nb::module_& m) -> void {
  nb::class_<LineString>(m, "LineString", kLineStringClassDoc)
      .def(nb::init<>(), "Construct an empty linestring.")

      .def(
          "__init__",
          [](LineString* self, const Eigen::Ref<const Eigen::VectorXd>& lon,
             const Eigen::Ref<const Eigen::VectorXd>& lat) -> void {
            if (lon.size() != lat.size()) {
              throw std::invalid_argument(
                  "lon and lat arrays must have the same size");
            }
            nb::gil_scoped_release release;
            new (self) LineString(lon, lat);
          },
          "lon"_a, "lat"_a, kLineStringInitDoc)

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

      // Iteration support - return list of points
      .def(
          "__iter__",
          [](const LineString& self) -> nb::object {
            nb::list result;
            for (auto i : self) {
              result.append(i);
            }
            return result.attr("__iter__")();
          },
          "Return an iterator over the points in the linestring.")

      // Comparison operators
      .def(
          "__eq__",
          [](const LineString& self, const LineString& other) -> bool {
            return boost::geometry::equals(self, other);
          },
          "other"_a, "Check if two linestrings are equal.")

      .def(
          "__ne__",
          [](const LineString& self, const LineString& other) -> bool {
            return !boost::geometry::equals(self, other);
          },
          "other"_a, "Check if two linestrings are not equal.")

      // String representation
      .def(
          "__repr__",
          [](const LineString& self) -> std::string {
            return std::format("LineString({} points)", self.size());
          },
          "Return a string representation of the linestring.")

      .def(
          "__str__",
          [](const LineString& self) -> std::string {
            std::ostringstream oss;
            oss << "LineString[";
            for (size_t i = 0; i < self.size(); ++i) {
              if (i > 0) {
                oss << ", ";
              }
              oss << "(" << self[i].lon() << ", " << self[i].lat() << ")";
              if (i >= 3 && self.size() > 5) {
                oss << ", ...";
                break;
              }
            }
            oss << "]";
            return oss.str();
          },
          "Return an informal string representation of the linestring.")

      // Pickle support
      .def(
          "__getstate__",
          [](const LineString& self) -> nb::tuple {
            serialization::Writer state;
            {
              nb::gil_scoped_release release;
              state = self.pack();
            }
            return nb::make_tuple(writer_to_ndarray(std::move(state)));
          },
          "Return the serialized state for pickling.")

      .def(
          "__setstate__",
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
          },
          "state"_a, "Restore the linestring from the serialized state.");
}

}  // namespace pyinterp::geometry::geographic::pybind
