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

#include "pyinterp/geometry/cartesian/multi_point.hpp"
#include "pyinterp/geometry/cartesian/point.hpp"
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

constexpr auto kMultiPointClassDoc = R"doc(
A collection of points in Cartesian coordinates.

Behaves like a Python container of `Point` objects.
)doc";

constexpr auto kMultiPointInitDoc = R"doc(
Construct a multipoint from an optional sequence of points.

Args:
    points: Optional sequence of `Point` objects.
)doc";

constexpr auto kMultiPointInitFromLonsLatsDoc = R"doc(
Construct a multipoint from separate X and Y coordinate arrays.

Args:
    xs: X coordinate array.
    ys: Y coordinate array.
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
      .def(nb::init<>(), "Construct an empty multipoint.")
      .def(
          "__init__",
          [](MultiPoint* self, std::vector<Point> points) -> void {
            new (self) MultiPoint(std::move(points));
          },
          "points"_a = std::vector<Point>{}, kMultiPointInitDoc)
      .def(
          "__init__",
          [](MultiPoint* self, const Eigen::Ref<const Vector<double>>& xs,
             const Eigen::Ref<const Vector<double>>& ys) -> void {
            nb::gil_scoped_release release;
            new (self) MultiPoint(xs, ys);
          },
          "xs"_a, "ys"_a, kMultiPointInitFromLonsLatsDoc)

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
            for (const auto& point : self) {
              items.append(point);
            }
            return items.attr("__iter__")();
          },
          "Iterate over points.")

      // Comparison operators
      .def("__eq__",
           [](const MultiPoint& self, const MultiPoint& other) -> bool {
             return boost::geometry::equals(self, other);
           })

      .def("__ne__",
           [](const MultiPoint& self, const MultiPoint& other) -> bool {
             return !boost::geometry::equals(self, other);
           })

      // String representation
      .def("__repr__",
           [](const MultiPoint& self) -> std::string {
             return std::format("MultiPoint({} points)", self.size());
           })

      .def("__str__",
           [](const MultiPoint& self) -> std::string {
             std::ostringstream oss;
             oss << "MultiPoint[";
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
           })

      // Pickle support
      .def("__getstate__",
           [](const MultiPoint& self) -> nb::tuple {
             serialization::Writer state;
             {
               nb::gil_scoped_release release;
               state = self.pack();
             }
             return nb::make_tuple(writer_to_ndarray(std::move(state)));
           })

      .def("__setstate__",
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
           });

  // Bind the PointsView helper
  bind_container_view<PointsView, Point>(m, "PointsView",
                                         "View over points in MultiPoint.");
}

}  // namespace pyinterp::geometry::cartesian::pybind
