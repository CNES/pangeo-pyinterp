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

#include "pyinterp/geometry/cartesian/point.hpp"
#include "pyinterp/geometry/cartesian/ring.hpp"
#include "pyinterp/pybind/ndarray_serialization.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

using pyinterp::pybind::NanobindArray1DUInt8;
using pyinterp::pybind::reader_from_ndarray;
using pyinterp::pybind::writer_to_ndarray;

namespace pyinterp::geometry::cartesian::pybind {

constexpr auto kRingClassDoc = R"doc(
A ring (closed linestring) in Cartesian coordinates.

A ring is a closed sequence of points used as polygon boundaries. The ring
automatically closes itself (the last point connects back to the first).

Examples:
    >>> import numpy as np
    >>> from pyinterp.geometry.cartesian import Ring

    Create a square ring

    >>> x = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    >>> y = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    >>> ring = Ring(x, y)
    >>> len(ring)
    5
)doc";

constexpr auto kRingInitDoc = R"doc(
Construct a ring from x and y arrays.

Args:
    x: Array of x coordinate values.
    y: Array of y coordinate values (must have same size as x).

Raises:
    ValueError: If x and y arrays have different sizes.
)doc";

auto init_ring(nb::module_& m) -> void {
  nb::class_<Ring>(m, "Ring", kRingClassDoc)
      .def(nb::init<>(), "Construct an empty ring.")

      .def(
          "__init__",
          [](Ring* self, const Eigen::Ref<const Eigen::VectorXd>& xs,
             const Eigen::Ref<const Eigen::VectorXd>& ys) -> void {
            nb::gil_scoped_release release;
            new (self) Ring(xs, ys);
          },
          "xs"_a, "ys"_a, kRingInitDoc)

      // Container operations
      .def("__len__", &Ring::size, "Return the number of points in the ring.")

      .def(
          "__getitem__",
          [](const Ring& self, int64_t idx) -> Point {
            if (idx < 0) {
              idx += self.size();
            }
            if (idx < 0 || std::cmp_greater_equal(idx, self.size())) {
              throw std::out_of_range("Ring index out of range");
            }
            return self[static_cast<size_t>(idx)];
          },
          "idx"_a, "Get the point at the given index.")

      .def(
          "__setitem__",
          [](Ring& self, int64_t idx, const Point& point) -> void {
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
            for (const auto& point : self) {
              result.append(point);
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
              oss << "(" << self[i].x() << ", " << self[i].y() << ")";
              if (i >= 3 && self.size() > 5) {
                oss << ", ...";
                break;
              }
            }
            oss << "]";
            return oss.str();
          },
          "Return the informal string representation of the ring.")

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
          "Return the state of the ring for pickling.")

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
          "state"_a, "Restore the state of the ring from pickling.");
}

}  // namespace pyinterp::geometry::cartesian::pybind
