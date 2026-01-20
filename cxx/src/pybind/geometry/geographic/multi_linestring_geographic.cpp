// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
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
#include "pyinterp/geometry/geographic/multi_linestring.hpp"
#include "pyinterp/pybind/geometry/container_view.hpp"
#include "pyinterp/pybind/ndarray_serialization.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

using pyinterp::geometry::pybind::bind_container_view;
using pyinterp::geometry::pybind::ContainerView;
using pyinterp::pybind::NanobindArray1DUInt8;
using pyinterp::pybind::reader_from_ndarray;
using pyinterp::pybind::writer_to_ndarray;

namespace pyinterp::geometry::geographic::pybind {

constexpr auto kMultiLineStringClassDoc = R"doc(
A collection of linestrings in geographic coordinates.

Behaves like a Python container of `LineString` objects.
)doc";

constexpr auto kMultiLineStringInitDoc = R"doc(
Construct a multilinestring from an optional sequence of linestrings.

Args:
    lines: Optional sequence of `LineString` objects.
)doc";

// Traits for LinesView
struct LinesTraits {
  static auto size_getter(MultiLineString* ml) -> size_t { return ml->size(); }

  static auto item_getter(MultiLineString* ml, size_t idx) -> LineString& {
    return (*ml)[idx];
  }

  static void item_setter(MultiLineString* ml, size_t idx,
                          const LineString& ls) {
    (*ml)[idx] = ls;
  }

  static void appender(MultiLineString* ml, const LineString& ls) {
    ml->push_back(ls);
  }

  static void clearer(MultiLineString* ml) { ml->clear(); }
};

// Proxy view over the lines container.
class LinesView
    : public ContainerView<MultiLineString, LineString, LinesTraits> {
 public:
  using ContainerView<MultiLineString, LineString, LinesTraits>::ContainerView;

  explicit LinesView(MultiLineString* owner)
      : ContainerView<MultiLineString, LineString, LinesTraits>(
            owner, "MultiLineString index out of range") {}
};

auto init_multilinestring(nb::module_& m) -> void {
  nb::class_<MultiLineString>(m, "MultiLineString", kMultiLineStringClassDoc)
      .def(
          "__init__",
          [](MultiLineString* self,
             std::optional<std::vector<LineString>> lines) -> void {
            if (lines) {
              new (self) MultiLineString(std::move(*lines));
            } else {
              new (self) MultiLineString();
            }
          },
          "lines"_a = std::nullopt, kMultiLineStringInitDoc)

      // Container-like operations
      .def("__len__", &MultiLineString::size, "Number of linestrings.")

      .def(
          "__getitem__",
          [](MultiLineString& self, int64_t idx) -> LineString& {
            if (idx < 0) {
              idx += self.size();
            }
            if (idx < 0 || std::cmp_greater_equal(idx, self.size())) {
              throw std::out_of_range("MultiLineString index out of range");
            }
            return self[static_cast<size_t>(idx)];
          },
          nb::rv_policy::reference_internal, "Get linestring at index.")

      .def(
          "__setitem__",
          [](MultiLineString& self, int64_t idx, const LineString& ls) -> void {
            if (idx < 0 || std::cmp_greater_equal(idx, self.size())) {
              throw std::out_of_range("MultiLineString index out of range");
            }
            self[static_cast<size_t>(idx)] = ls;
          },
          "idx"_a, "ls"_a, "Set linestring at index.")

      .def("append", &MultiLineString::push_back, "ls"_a,
           "Append a linestring to the collection.")

      .def("clear", &MultiLineString::clear,
           "Remove all linestrings from the collection.")

      .def(
          "__bool__",
          [](const MultiLineString& self) -> bool { return !self.empty(); },
          "Return True if not empty.")

      .def(
          "__iter__",
          [](MultiLineString& self) -> nb::object {
            nb::list items;
            for (size_t i = 0; i < self.size(); ++i) {
              items.append(self[static_cast<size_t>(i)]);
            }
            return items.attr("__iter__")();
          },
          "Iterate over linestrings.")

      // View property over the underlying container
      .def_prop_rw(
          "lines",
          [](MultiLineString& self) -> nb::object {
            return nb::cast(LinesView(&self), nb::rv_policy::reference);
          },
          [](MultiLineString& self, const nb::list& items) -> void {
            std::vector<LineString> ls;
            ls.reserve(items.size());
            for (const auto& item : items) {
              ls.push_back(nb::cast<LineString>(item));
            }
            self.clear();
            for (const auto& x : ls) self.push_back(x);
          },
          nb::keep_alive<0, 1>(), nb::rv_policy::reference_internal,
          "View over lines bound to multilinestring lifetime.")

      // Equality via boost geometry
      .def(
          "__eq__",
          [](const MultiLineString& a, const MultiLineString& b) -> bool {
            return boost::geometry::equals(a, b);
          },
          "other"_a, "Check if two multilinestrings are equal.")
      .def(
          "__ne__",
          [](const MultiLineString& a, const MultiLineString& b) -> bool {
            return !boost::geometry::equals(a, b);
          },
          "other"_a, "Check if two multilinestrings are not equal.")
      // Repr/str
      .def(
          "__repr__",
          [](const MultiLineString& self) -> std::string {
            return std::format("MultiLineString({} lines)", self.size());
          },
          "Return the official string representation of the multilinestring.")
      .def(
          "__str__",
          [](const MultiLineString& self) -> std::string {
            std::ostringstream oss;
            oss << "MultiLineString[n=" << self.size() << "]";
            return oss.str();
          },
          "Return the string representation of the multilinestring.")

      // Pickle support
      .def(
          "__getstate__",
          [](const MultiLineString& self) -> nb::tuple {
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
          [](MultiLineString* self, const nb::tuple& state) -> void {
            if (state.size() != 1) {
              throw std::invalid_argument("Invalid state");
            }
            auto array = nanobind::cast<NanobindArray1DUInt8>(state[0]);
            auto reader = reader_from_ndarray(array);
            {
              nb::gil_scoped_release release;
              new (self) MultiLineString(MultiLineString::unpack(reader));
            }
          },
          "state"_a, "Restore the multilinestring from the serialized state.");

  // Bind view class
  bind_container_view<LinesView, LineString>(m, "_LinesView", "linestring");
}

}  // namespace pyinterp::geometry::geographic::pybind
