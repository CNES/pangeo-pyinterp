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
#include <utility>

#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/geometry/geographic/ring.hpp"
#include "pyinterp/pybind/ndarray_serialization.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

using pyinterp::pybind::NanobindArray1DUInt8;
using pyinterp::pybind::reader_from_ndarray;
using pyinterp::pybind::writer_to_ndarray;

namespace pyinterp::geometry::geographic::pybind {

constexpr auto kRingClassDoc = R"doc(
A ring (closed linestring) in geographic coordinates.

A ring is a closed sequence of points used as polygon boundaries. The ring
automatically closes itself (the last point connects back to the first).

Examples:
    >>> import numpy as np
    >>> from pyinterp.geometry.geographic import Ring
    >>> # Create a square ring
    >>> lon = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    >>> lat = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    >>> ring = Ring(lon, lat)
    >>> len(ring)
    5
)doc";

constexpr auto kRingInitDoc = R"doc(
Construct a ring from longitude and latitude arrays.

Args:
    lon: Array of longitude values in degrees.
    lat: Array of latitude values in degrees (must have same size as lon).

Raises:
    ValueError: If lon and lat arrays have different sizes.
)doc";

auto init_ring(nb::module_& m) -> void {
  nb::class_<Ring>(m, "Ring", kRingClassDoc)
      .def(nb::init<>(), "Construct an empty ring.")

      .def(
          "__init__",
          [](Ring* self, const Eigen::Ref<const Eigen::VectorXd>& lon,
             const Eigen::Ref<const Eigen::VectorXd>& lat) -> void {
            if (lon.size() != lat.size()) {
              throw std::invalid_argument(
                  "lon and lat arrays must have the same size");
            }
            new (self) Ring(lon, lat);
          },
          "lon"_a, "lat"_a, kRingInitDoc)

      // Container operations
      .def("__len__", &Ring::size, "Return the number of points in the ring.")

      .def(
          "__getitem__",
          [](const Ring& self, Eigen::Index idx) -> Point {
            if (idx < 0 || std::cmp_greater_equal(idx, self.size())) {
              throw std::out_of_range("Ring index out of range");
            }
            return self[static_cast<size_t>(idx)];
          },
          "idx"_a, "Get the point at the given index.")

      .def(
          "__setitem__",
          [](Ring& self, Eigen::Index idx, const Point& point) -> void {
            if (idx < 0 || std::cmp_greater_equal(idx, self.size())) {
              throw std::out_of_range("Ring index out of range");
            }
            self[static_cast<size_t>(idx)] = point;
          },
          "idx"_a, "point"_a, "Set the point at the given index.")

      .def(
          "append",
          [](Ring& self, const Point& point) -> void { self.push_back(point); },
          "point"_a, "Append a point to the ring.")

      .def("clear", &Ring::clear, "Remove all points from the ring.")

      .def(
          "__bool__", [](const Ring& self) -> bool { return !self.empty(); },
          "Return True if the ring is not empty.")

      // Iteration support - return list of points
      .def(
          "__iter__",
          [](const Ring& self) -> nb::object {
            nb::list result;
            for (auto i : self) {
              result.append(i);
            }
            return result.attr("__iter__")();
          },
          "Return an iterator over the points in the ring.")

      // Comparison operators
      .def(
          "__eq__",
          [](const Ring& self, const Ring& other) -> bool {
            return boost::geometry::equals(self, other);
          },
          "other"_a, "Check if two rings are equal.")

      .def(
          "__ne__",
          [](const Ring& self, const Ring& other) -> bool {
            return !boost::geometry::equals(self, other);
          },
          "other"_a, "Check if two rings are not equal.")

      // String representation
      .def(
          "__repr__",
          [](const Ring& self) -> std::string {
            return std::format("Ring({} points)", self.size());
          },
          "Return the official string representation of the ring.")

      .def(
          "__str__",
          [](const Ring& self) -> std::string {
            std::ostringstream oss;
            oss << "Ring[";
            for (size_t i = 0; i < self.size(); ++i) {
              if (i > 0) oss << ", ";
              oss << "(" << self[i].lon() << ", " << self[i].lat() << ")";
              if (i >= 3 && self.size() > 5) {
                oss << ", ...";
                break;
              }
            }
            oss << "]";
            return oss.str();
          },
          "Return the string representation of the ring.")

      // Pickle support
      .def(
          "__getstate__",
          [](const Ring& self) -> nb::tuple {
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
          [](Ring* self, const nb::tuple& state) -> void {
            if (state.size() != 1) {
              throw std::invalid_argument("Invalid state");
            }
            auto array = nanobind::cast<NanobindArray1DUInt8>(state[0]);
            auto reader = reader_from_ndarray(array);
            {
              nb::gil_scoped_release release;
              new (self) Ring(Ring::unpack(reader));
            }
          },
          "state"_a, "Restore the ring from the serialized state.");
}

}  // namespace pyinterp::geometry::geographic::pybind
