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

#include "pyinterp/geometry/cartesian/multi_polygon.hpp"
#include "pyinterp/geometry/cartesian/polygon.hpp"
#include "pyinterp/pybind/geometry/container_view.hpp"
#include "pyinterp/pybind/ndarray_serialization.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

using pyinterp::geometry::pybind::bind_container_view;
using pyinterp::geometry::pybind::ContainerView;
using pyinterp::pybind::NanobindArray1DUInt8;
using pyinterp::pybind::reader_from_ndarray;
using pyinterp::pybind::writer_to_ndarray;

namespace pyinterp::geometry::cartesian::pybind {

constexpr auto kMultiPolygonClassDoc = R"doc(
A collection of polygons in Cartesian coordinates.

Behaves like a Python container of `Polygon` objects.
)doc";

constexpr auto kMultiPolygonInitDoc = R"doc(
Construct a multipolygon from an optional sequence of polygons.

Args:
    polygons: Optional sequence of `Polygon` objects.
)doc";

// Traits for PolygonsView
struct PolygonsTraits {
  static auto size_getter(MultiPolygon* mp) -> size_t { return mp->size(); }

  static auto item_getter(MultiPolygon* mp, size_t idx) -> Polygon& {
    return (*mp)[idx];
  }

  static void item_setter(MultiPolygon* mp, size_t idx, const Polygon& poly) {
    (*mp)[idx] = poly;
  }

  static void appender(MultiPolygon* mp, const Polygon& poly) {
    mp->push_back(poly);
  }

  static void clearer(MultiPolygon* mp) { mp->clear(); }
};

// Proxy view over the polygons container.
class PolygonsView
    : public ContainerView<MultiPolygon, Polygon, PolygonsTraits> {
 public:
  using ContainerView<MultiPolygon, Polygon, PolygonsTraits>::ContainerView;

  explicit PolygonsView(MultiPolygon* owner)
      : ContainerView<MultiPolygon, Polygon, PolygonsTraits>(
            owner, "MultiPolygon index out of range") {}
};

auto init_multipolygon(nb::module_& m) -> void {
  nb::class_<MultiPolygon>(m, "MultiPolygon", kMultiPolygonClassDoc)
      .def(
          "__init__",
          [](MultiPolygon* self,
             std::optional<std::vector<Polygon>> polygons) -> void {
            if (polygons) {
              new (self) MultiPolygon(std::move(*polygons));
            } else {
              new (self) MultiPolygon();
            }
          },
          "polygons"_a = std::nullopt, kMultiPolygonInitDoc)

      // Container-like operations
      .def("__len__", &MultiPolygon::size, "Number of polygons.")

      .def(
          "__getitem__",
          [](MultiPolygon& self, int64_t idx) -> Polygon& {
            if (idx < 0) {
              idx += self.size();
            }
            if (idx < 0 || std::cmp_greater_equal(idx, self.size())) {
              throw std::out_of_range("MultiPolygon index out of range");
            }
            return self[static_cast<size_t>(idx)];
          },
          nb::rv_policy::reference_internal, "Get polygon at index.")

      .def(
          "__setitem__",
          [](MultiPolygon& self, int64_t idx, const Polygon& poly) -> void {
            if (idx < 0 || std::cmp_greater_equal(idx, self.size())) {
              throw std::out_of_range("MultiPolygon index out of range");
            }
            self[static_cast<size_t>(idx)] = poly;
          },
          "idx"_a, "poly"_a, "Set polygon at index.")

      .def(
          "append",
          [](MultiPolygon& self, Polygon poly) -> void {
            self.push_back(std::move(poly));
          },
          "poly"_a, "Append a polygon to the collection.")

      .def("clear", &MultiPolygon::clear,
           "Remove all polygons from the collection.")

      .def(
          "__bool__",
          [](const MultiPolygon& self) -> bool { return !self.empty(); },
          "Return True if not empty.")

      .def(
          "__iter__",
          [](MultiPolygon& self) -> nb::object {
            nb::list items;
            for (size_t i = 0; i < self.size(); ++i) {
              items.append(self[static_cast<size_t>(i)]);
            }
            return items.attr("__iter__")();
          },
          "Iterate over polygons.")

      // View property over the underlying container
      .def_prop_rw(
          "polygons",
          [](MultiPolygon& self) -> nb::object {
            return nb::cast(PolygonsView(&self), nb::rv_policy::reference);
          },
          [](MultiPolygon& self, const nb::list& items) -> void {
            std::vector<Polygon> polys;
            polys.reserve(items.size());
            for (const auto& item : items) {
              polys.push_back(nb::cast<Polygon>(item));
            }
            self.clear();
            for (const auto& x : polys) self.push_back(x);
          },
          nb::keep_alive<0, 1>(), nb::rv_policy::reference_internal,
          "View over polygons bound to multipolygon lifetime.")

      // Equality via boost geometry
      .def(
          "__eq__",
          [](const MultiPolygon& self, const MultiPolygon& other) -> bool {
            return boost::geometry::equals(self, other);
          },
          "other"_a, "Check if two multipolygons are equal.")

      .def(
          "__ne__",
          [](const MultiPolygon& self, const MultiPolygon& other) -> bool {
            return !boost::geometry::equals(self, other);
          },
          "other"_a, "Check if two multipolygons are not equal.")

      // String representation
      .def(
          "__repr__",
          [](const MultiPolygon& self) -> std::string {
            return std::format("MultiPolygon({} polygons)", self.size());
          },
          "Return the official string representation of the MultiPolygon.")

      .def(
          "__str__",
          [](const MultiPolygon& self) -> std::string {
            std::ostringstream oss;
            oss << "MultiPolygon[" << self.size() << " polygons]";
            return oss.str();
          },
          "Return the informal string representation of the MultiPolygon.")

      // Pickle support
      .def(
          "__getstate__",
          [](const MultiPolygon& self) -> nb::tuple {
            serialization::Writer state;
            {
              nb::gil_scoped_release release;
              state = self.pack();
            }
            return nb::make_tuple(writer_to_ndarray(std::move(state)));
          },
          "Return the state of the MultiPolygon for pickling.")

      .def(
          "__setstate__",
          [](MultiPolygon* self, const nb::tuple& state) -> void {
            if (state.size() != 1) {
              throw std::invalid_argument("Invalid state");
            }
            auto array = nanobind::cast<NanobindArray1DUInt8>(state[0]);
            auto reader = reader_from_ndarray(array);
            {
              nb::gil_scoped_release release;
              new (self) MultiPolygon(MultiPolygon::unpack(reader));
            }
          },
          "state"_a, "Restore the state of the MultiPolygon from pickling.");

  // Bind the PolygonsView helper
  bind_container_view<PolygonsView, Polygon>(
      m, "PolygonsView", "View over polygons in MultiPolygon.");
}

}  // namespace pyinterp::geometry::cartesian::pybind
