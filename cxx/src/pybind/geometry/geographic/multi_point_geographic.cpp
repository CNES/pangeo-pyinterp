// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/eigen/dense.h>
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

#include "pyinterp/geometry/geographic/multi_point.hpp"
#include "pyinterp/geometry/geographic/point.hpp"
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

constexpr auto kMultiPointClassDoc = R"doc(
A collection of points in geographic coordinates.

Behaves like a Python container of `Point` objects.
)doc";

constexpr auto kMultiPointInitDoc = R"doc(
Construct a multipoint from an optional sequence of points.

Args:
    points: Optional sequence of `Point` objects.
)doc";

constexpr auto kMultiPointInitFromLonsLatsDoc = R"doc(
Construct a multipoint from separate longitude and latitude arrays.

Args:
    lons: Longitude array.
    lats: Latitude array.
)doc";

// Traits for PointsView
struct PointsTraits {
  static auto size_getter(MultiPoint* mp) -> size_t { return mp->size(); }

  static auto item_getter(MultiPoint* mp, size_t idx) -> Point& {
    return (*mp)[idx];
  }

  static void item_setter(MultiPoint* mp, size_t idx, const Point& pt) {
    (*mp)[idx] = pt;
  }

  static void appender(MultiPoint* mp, const Point& pt) { mp->push_back(pt); }

  static void clearer(MultiPoint* mp) { mp->clear(); }
};

// Proxy view over the points container.
class PointsView : public ContainerView<MultiPoint, Point, PointsTraits> {
 public:
  using ContainerView<MultiPoint, Point, PointsTraits>::ContainerView;

  explicit PointsView(MultiPoint* owner)
      : ContainerView<MultiPoint, Point, PointsTraits>(
            owner, "MultiPoint index out of range") {}
};

auto init_multipoint(nb::module_& m) -> void {
  nb::class_<MultiPoint>(m, "MultiPoint", kMultiPointClassDoc)
      .def(
          "__init__",
          [](MultiPoint* self,
             std::optional<std::vector<Point>> points) -> void {
            if (points) {
              new (self) MultiPoint(std::move(*points));
            } else {
              new (self) MultiPoint();
            }
          },
          "points"_a = std::nullopt, kMultiPointInitDoc)
      .def(
          "__init__",
          [](MultiPoint* self, const Eigen::Ref<const Vector<double>>& lons,
             const Eigen::Ref<const Vector<double>>& lats) -> void {
            if (lons.size() != lats.size()) {
              throw std::invalid_argument(
                  "lons and lats arrays must have the same size");
            }
            nb::gil_scoped_release release;
            new (self) MultiPoint(lons, lats);
          },
          "lons"_a, "lats"_a, kMultiPointInitFromLonsLatsDoc)

      // Container-like operations
      .def("__len__", &MultiPoint::size, "Number of points.")

      .def(
          "__getitem__",
          [](MultiPoint& self, Eigen::Index idx) -> Point& {
            if (idx < 0 || std::cmp_greater_equal(idx, self.size())) {
              throw std::out_of_range("MultiPoint index out of range");
            }
            return self[static_cast<size_t>(idx)];
          },
          nb::rv_policy::reference_internal, "Get point at index.")

      .def(
          "__setitem__",
          [](MultiPoint& self, Eigen::Index idx, const Point& pt) -> void {
            if (idx < 0 || std::cmp_greater_equal(idx, self.size())) {
              throw std::out_of_range("MultiPoint index out of range");
            }
            self[static_cast<size_t>(idx)] = pt;
          },
          "idx"_a, "pt"_a, "Set point at index.")

      .def("append", &MultiPoint::push_back, "pt"_a,
           "Append a point to the collection.")

      .def("clear", &MultiPoint::clear,
           "Remove all points from the collection.")

      .def(
          "__bool__",
          [](const MultiPoint& self) -> bool { return !self.empty(); },
          "Return True if not empty.")

      .def(
          "__iter__",
          [](MultiPoint& self) -> nb::object {
            nb::list items;
            for (size_t i = 0; i < self.size(); ++i) {
              items.append(self[static_cast<size_t>(i)]);
            }
            return items.attr("__iter__")();
          },
          "Iterate over points.")

      // View property over the underlying container
      .def_prop_rw(
          "points",
          [](MultiPoint& self) -> nb::object {
            return nb::cast(PointsView(&self), nb::rv_policy::reference);
          },
          [](MultiPoint& self, const nb::list& items) -> void {
            std::vector<Point> pts;
            pts.reserve(items.size());
            for (const auto& item : items) {
              pts.push_back(nb::cast<Point>(item));
            }
            self.clear();
            for (const auto& p : pts) self.push_back(p);
          },
          nb::keep_alive<0, 1>(), nb::rv_policy::reference_internal,
          "View over points bound to multipoint lifetime.")

      // Equality via boost geometry
      .def(
          "__eq__",
          [](const MultiPoint& a, const MultiPoint& b) -> bool {
            return boost::geometry::equals(a, b);
          },
          "other"_a, "Check if two multipoints are equal.")
      .def(
          "__ne__",
          [](const MultiPoint& a, const MultiPoint& b) -> bool {
            return !boost::geometry::equals(a, b);
          },
          "other"_a, "Check if two multipoints are not equal.")

      // Repr/str
      .def(
          "__repr__",
          [](const MultiPoint& self) -> std::string {
            return std::format("MultiPoint({} points)", self.size());
          },
          "Return the official string representation of the multipoint.")
      .def(
          "__str__",
          [](const MultiPoint& self) -> std::string {
            std::ostringstream oss;
            oss << "MultiPoint[n=" << self.size() << "]";
            return oss.str();
          },
          "Return the string representation of the multipoint.")

      // Pickle support
      .def(
          "__getstate__",
          [](const MultiPoint& self) -> nb::tuple {
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
          [](MultiPoint* self, const nb::tuple& state) -> void {
            if (state.size() != 1) {
              throw std::invalid_argument("Invalid state");
            }
            auto array = nanobind::cast<NanobindArray1DUInt8>(state[0]);
            auto reader = reader_from_ndarray(array);
            {
              nb::gil_scoped_release release;
              new (self) MultiPoint(MultiPoint::unpack(reader));
            }
          },
          "state"_a, "Restore the multipoint from the serialized state.");

  // Bind view class
  bind_container_view<PointsView, Point>(m, "_PointsView", "point");
}

}  // namespace pyinterp::geometry::geographic::pybind
