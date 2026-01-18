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
#include <string>

#include "pyinterp/geometry/geographic/polygon.hpp"
#include "pyinterp/geometry/geographic/ring.hpp"
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

constexpr auto kPolygonClassDoc = R"doc(
A polygon in geographic coordinates.

A Polygon is defined by an exterior ring and zero or more interior rings
(holes). The exterior ring is indexed at position 0; interior rings start at
index 1.

Examples:
  >>> import numpy as np
  >>> from pyinterp.geometry.geographic import Polygon, Ring
  >>> outer = Ring(np.array([0.0, 10.0, 10.0, 0.0, 0.0]),
  ...              np.array([0.0, 0.0, 10.0, 10.0, 0.0]))
  >>> poly = Polygon(outer)
  >>> len(poly)
  1
)doc";

constexpr auto kPolygonInitDoc = R"doc(
Construct a polygon from an exterior ring and optional interior rings.

Args:
    exterior: Exterior ring defining the polygon boundary.
    interiors: Optional sequence of interior rings (holes).
)doc";

// Traits for InnerRingsView
struct InnerRingsTraits {
  static auto size_getter(Polygon* poly) -> size_t {
    return poly->inners().size();
  }

  static auto item_getter(Polygon* poly, size_t idx) -> Ring& {
    return poly->inners()[idx];
  }

  static void item_setter(Polygon* poly, size_t idx, const Ring& ring) {
    poly->inners()[idx] = ring;
  }

  static void appender(Polygon* poly, const Ring& ring) {
    poly->inners().push_back(ring);
  }

  static void clearer(Polygon* poly) { poly->inners().clear(); }
};

// Proxy view over the interior rings container.
class InnerRingsView : public ContainerView<Polygon, Ring, InnerRingsTraits> {
 public:
  using ContainerView<Polygon, Ring, InnerRingsTraits>::ContainerView;

  explicit InnerRingsView(Polygon* owner)
      : ContainerView<Polygon, Ring, InnerRingsTraits>(
            owner, "Interior ring index out of range") {}
};

auto init_polygon(nb::module_& m) -> void {
  nb::class_<Polygon>(m, "Polygon", kPolygonClassDoc)
      .def(
          "__init__",
          [](Polygon* self, std::optional<Ring> exterior,
             std::optional<std::vector<Ring>>& interiors) -> void {
            auto exteriog_ring = Ring{};
            auto interior_rings = std::vector<Ring>{};
            if (exterior) {
              exteriog_ring = std::move(*exterior);
            }
            if (interiors) {
              interior_rings = std::move(*interiors);
            }
            new (self)
                Polygon(std::move(exteriog_ring), std::move(interior_rings));
          },
          "exterior"_a, "interiors"_a = std::vector<Ring>{}, kPolygonInitDoc)

      // Accessors
      .def_prop_rw(
          "outer", [](Polygon& self) -> Ring& { return self.outer(); },
          [](Polygon& self, const Ring& ring) -> void { self.outer() = ring; },
          nb::rv_policy::reference_internal, "Exterior ring of the polygon.")

      .def_prop_rw(
          "inners",
          // Getter returns a proxy view; keep the view alive via keep_alive
          [](Polygon& self) -> nb::object {
            return nb::cast(InnerRingsView(&self), nb::rv_policy::reference);
          },
          // Setter accepts a sequence of rings to replace interiors
          [](Polygon& self, const nb::list& rings) -> void {
            std::vector<Ring> new_inners;
            new_inners.reserve(rings.size());
            for (const auto& ring : rings) {
              new_inners.push_back(nb::cast<Ring>(ring));
            }
            self.inners() = std::move(new_inners);
          },
          nb::keep_alive<0, 1>(), nb::rv_policy::reference_internal,
          "View over interior rings bound to polygon lifetime.")

      .def(
          "append",
          [](Polygon& self, Ring&& ring) -> void {
            self.inners().push_back(std::move(ring));
          },
          "ring"_a, "Append an interior ring (hole).")

      .def(
          "clear",
          [](Polygon& self) -> void {
            self.outer() = Ring{};
            self.inners().clear();
          },
          "Remove the exterior and all interior rings.")

      // Comparison operators
      .def(
          "__eq__",
          [](const Polygon& self, const Polygon& other) -> bool {
            return boost::geometry::equals(self, other);
          },
          "other"_a, "Check if two polygons are equal.")

      .def(
          "__ne__",
          [](const Polygon& self, const Polygon& other) -> bool {
            return !boost::geometry::equals(self, other);
          },
          "other"_a, "Check if two polygons are not equal.")

      // String representation
      .def(
          "__repr__",
          [](const Polygon& self) -> std::string {
            const auto count =
                self.outer().empty() ? 0 : 1 + self.inners().size();
            return std::format("Polygon({} rings)", count);
          },
          "Return the official string representation of the polygon.")

      .def(
          "__str__",
          [](const Polygon& self) -> std::string {
            std::ostringstream oss;
            oss << "Polygon[outer=" << self.outer().size()
                << " points, inners=" << self.inners().size() << "]";
            return oss.str();
          },
          "Return the string representation of the polygon.")

      // Pickle support
      .def(
          "__getstate__",
          [](const Polygon& self) -> nb::tuple {
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
          [](Polygon* self, const nb::tuple& state) -> void {
            if (state.size() != 1) {
              throw std::invalid_argument("Invalid state");
            }
            auto array = nanobind::cast<NanobindArray1DUInt8>(state[0]);
            auto reader = reader_from_ndarray(array);
            {
              nb::gil_scoped_release release;
              new (self) Polygon(Polygon::unpack(reader));
            }
          },
          "state"_a, "Restore the polygon from the serialized state.");

  // Bind the view class
  bind_container_view<InnerRingsView, Ring>(m, "_InnerRingsView",
                                            "interior ring");
}

}  // namespace pyinterp::geometry::geographic::pybind
