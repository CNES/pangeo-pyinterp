// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <boost/geometry.hpp>
#include <format>
#include <sstream>
#include <stdexcept>

#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/geometry/geographic/segment.hpp"
#include "pyinterp/pybind/ndarray_serialization.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

using pyinterp::pybind::NanobindArray1DUInt8;
using pyinterp::pybind::reader_from_ndarray;
using pyinterp::pybind::writer_to_ndarray;

namespace pyinterp::geometry::geographic::pybind {

constexpr auto kSegmentClassDoc = R"doc(
A segment in geographic coordinates.

A Segment is defined by two endpoints (points) on the ellipsoid surface.

Examples:
    >>> from pyinterp.geometry.geographic import Segment
    >>> s = Segment((0.0, 0.0), (10.0, 5.0))
    >>> len(s)
    2
)doc";

constexpr auto kSegmentInitDoc = R"doc(
Construct a segment from two endpoints or four coordinates.

Args:
    a: a tuple of two floats representing the first endpoint as lon, lat.
    b: a tuple of two floats representing the second endpoint as lon, lat.
)doc";

auto init_segment(nb::module_& m) -> void {
  nb::class_<Segment>(m, "Segment", kSegmentClassDoc)
      // Construct from two Points
      .def(
          "__init__",
          [](Segment* self, const std::optional<std::tuple<double, double>>& a,
             const std::optional<std::tuple<double, double>>& b) -> void {
            auto default_point = std::make_tuple(0.0, 0.0);
            const auto& a_tuple = a.value_or(default_point);
            const auto& b_tuple = b.value_or(default_point);
            new (self)
                Segment(Point(std::get<0>(a_tuple), std::get<1>(a_tuple)),
                        Point(std::get<0>(b_tuple), std::get<1>(b_tuple)));
          },
          "a"_a = std::nullopt, "b"_a = std::nullopt, kSegmentInitDoc)

      // Length is fixed to 2 endpoints
      .def(
          "__len__",
          []([[maybe_unused]] const Segment& self) -> int { return 2; },
          "Return the number of endpoints (always 2).")

      // Indexing support: 0 -> a, 1 -> b
      .def(
          "__getitem__",
          [](const Segment& self, int idx) -> Point {
            if (idx == 0) return self.a();
            if (idx == 1) return self.b();
            throw std::out_of_range("Segment index out of range");
          },
          "idx"_a, "Get endpoint at index (0 or 1).")

      .def(
          "__setitem__",
          [](Segment& self, int idx, const Point& p) -> void {
            if (idx == 0) {
              self.a() = p;
            } else if (idx == 1) {
              self.b() = p;
            } else {
              throw std::out_of_range("Segment index out of range");
            }
          },
          "idx"_a, "point"_a, "Set endpoint at index (0 or 1).")

      // Accessors
      .def_prop_rw(
          "a", [](Segment& self) -> Point { return self.a(); },
          [](Segment& self, const Point& p) -> void { self.a() = p; },
          "First endpoint.")

      .def_prop_rw(
          "b", [](Segment& self) -> Point { return self.b(); },
          [](Segment& self, const Point& p) -> void { self.b() = p; },
          "Second endpoint.")

      // Truthiness: segment is non-empty if endpoints are not both default
      .def(
          "__bool__",
          [](const Segment& self) -> bool {
            // Consider default-constructed points as empty; otherwise true
            const auto a = self.a();
            const auto b = self.b();
            return !(a.lon() == 0.0 && a.lat() == 0.0 && b.lon() == 0.0 &&
                     b.lat() == 0.0);
          },
          "Return True if the segment has non-default endpoints.")

      // Equality via boost geometry
      .def(
          "__eq__",
          [](const Segment& s1, const Segment& s2) -> bool {
            return boost::geometry::equals(s1, s2);
          },
          "other"_a, "Check if two segments are equal.")
      .def(
          "__ne__",
          [](const Segment& s1, const Segment& s2) -> bool {
            return !boost::geometry::equals(s1, s2);
          },
          "other"_a, "Check if two segments are not equal.")

      // String representation
      .def(
          "__repr__",
          [](const Segment& self) -> std::string {
            const auto& a = self.a();
            const auto& b = self.b();
            return std::format("Segment(a=({}, {}), b=({}, {}))", a.lon(),
                               a.lat(), b.lon(), b.lat());
          },
          "Return the official string representation of the segment.")
      .def(
          "__str__",
          [](const Segment& self) -> std::string {
            std::ostringstream oss;
            const auto& a = self.a();
            const auto& b = self.b();
            oss << "Segment[('" << a.lon() << ", " << a.lat() << "') -> ('"
                << b.lon() << ", " << b.lat() << "')]";
            return oss.str();
          },
          "Return the informal string representation of the segment.")

      // Pickle support
      .def(
          "__getstate__",
          [](const Segment& self) -> nb::tuple {
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
          [](Segment* self, const nb::tuple& state) -> void {
            if (state.size() != 1) {
              throw std::invalid_argument("Invalid state");
            }
            auto array = nanobind::cast<NanobindArray1DUInt8>(state[0]);
            auto reader = reader_from_ndarray(array);
            {
              nb::gil_scoped_release release;
              new (self) Segment(Segment::unpack(reader));
            }
          },
          "state"_a, "Restore the segment from the serialized state.");
}

}  // namespace pyinterp::geometry::geographic::pybind
