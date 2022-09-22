// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyinterp/geodetic/algorithm.hpp"
#include "pyinterp/geodetic/box.hpp"
#include "pyinterp/geodetic/coordinates.hpp"
#include "pyinterp/geodetic/crossover.hpp"
#include "pyinterp/geodetic/line_string.hpp"
#include "pyinterp/geodetic/multipolygon.hpp"
#include "pyinterp/geodetic/point.hpp"
#include "pyinterp/geodetic/polygon.hpp"
#include "pyinterp/geodetic/rtree.hpp"
#include "pyinterp/geodetic/spheroid.hpp"
#include "pyinterp/geodetic/swath.hpp"

namespace geodetic = pyinterp::geodetic;
namespace math = pyinterp::detail::math;
namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(geodetic::MultiPolygon)

static inline auto parse_distance_strategy(const std::string &strategy)
    -> geodetic::DistanceStrategy {
  if (strategy == "andoyer") {
    return geodetic::kAndoyer;
  }
  if (strategy == "thomas") {
    return geodetic::kThomas;
  }
  if (strategy == "vincenty") {
    return geodetic::kVincenty;
  }
  throw std::invalid_argument("Invalid strategy: " + strategy);
}

static void init_geodetic_point(py::module &m) {
  py::class_<geodetic::Point>(m, "Point", R"__doc__(
Point(self, lon: float = 0, lat: float = 0)

Handle a point in a geographic coordinates system in degrees.

Args:
    lon: Longitude in degrees.
    lat: Latitude in degrees.
)__doc__")
      .def(py::init<>())
      .def(py::init<double, double>(), py::arg("lon"), py::arg("lat"))
      .def_property("lon",
                    static_cast<double (geodetic::Point::*)() const>(
                        &geodetic::Point::lon),
                    static_cast<void (geodetic::Point::*)(const double)>(
                        &geodetic::Point::lon),
                    "Longitude coordinate in degrees.")
      .def_property("lat",
                    static_cast<double (geodetic::Point::*)() const>(
                        &geodetic::Point::lat),
                    static_cast<void (geodetic::Point::*)(const double)>(
                        &geodetic::Point::lat),
                    "Latitude coordinate in degrees.")
      .def(
          "distance",
          [](const geodetic::Point &self, const geodetic::Point &other,
             const std::string &strategy,
             const std::optional<geodetic::Spheroid> &wgs) -> double {
            return self.distance(other, parse_distance_strategy(strategy), wgs);
          },
          py::arg("other"), py::arg("strategy") = "thomas",
          py::arg("wgs") = std::nullopt,
          R"__doc__(
Calculate the distance between the two points.

Args:
    other: The other point to consider.
    strategy: The calculation method used to calculate the distance. This
        parameter can take the values ``andoyer``, ``thomas`` or ``vincenty``.
    wgs: The spheroid used to calculate the distance. If not provided, the
        WGS-84 spheroid is used.

Returns:
    The distance between the two points in meters.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def("to_geojson", &geodetic::Point::to_geojson,
           R"__doc__(
Return the point as a GeoJSON type.

Returns:
    The point as a GeoJSON type.
)__doc__")
      .def(
          "wkt",
          [](const geodetic::Point &self) -> std::string {
            auto ss = std::stringstream();
            ss << boost::geometry::wkt(self);
            return ss.str();
          },
          R"__doc__(
Gets the OGC Well-Known Text (WKT) representation of this instance.

Returns:
    The WKT representation.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def_static(
          "read_wkt",
          [](const std::string &wkt) -> geodetic::Point {
            auto point = geodetic::Point();
            boost::geometry::read_wkt(wkt, point);
            return point;
          },
          py::arg("wkt"), R"__doc__(
Parses OGC Well-Known Text (WKT) into a Point.

Args:
    wkt: the WKT representation of the Point.
Returns:
    The point defined by the WKT representation.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def("__repr__", &geodetic::Point::to_string,
           "Called by the ``repr()`` built-in function to compute the string "
           "representation of a point.")
      .def(
          "__copy__",
          [](const geodetic::Point &self) { return geodetic::Point(self); },
          "Implements the shallow copy operation.",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "__eq__",
          [](const geodetic::Point &self, const geodetic::Point &rhs) -> bool {
            return boost::geometry::equals(self, rhs);
          },
          py::arg("other"),
          "Overrides the default behavior of the ``==`` operator.")
      .def(
          "__ne__",
          [](const geodetic::Point &self, const geodetic::Point &rhs) -> bool {
            return !boost::geometry::equals(self, rhs);
          },
          py::arg("other"),
          "Overrides the default behavior of the ``!=`` operator.")
      .def(py::pickle(
          [](const geodetic::Point &self) { return self.getstate(); },
          [](const py::tuple &state) {
            return geodetic::Point::setstate(state);
          }));
}

static void init_geodetic_box(py::class_<geodetic::Box> &class_) {
  class_.def(py::init<>())
      .def(py::init<geodetic::Point, geodetic::Point>(), py::arg("min_corner"),
           py::arg("max_corner"))
      .def_static("from_geojson", &geodetic::Box::from_geojson,
                  py::arg("array"), R"__doc__(
Creates a box from a GeoJSON coordinates array.

Args:
    array: the GeoJSON coordinate array.
Returns:
    The box defined by the GeoJSON coordinate array.
)__doc__")
      .def_property(
          "min_corner",
          [](const geodetic::Box &self) { return self.min_corner(); },
          [](geodetic::Box &self, const geodetic::Point &point) -> void {
            self.min_corner() = point;
          },
          "The minimal corner (lower left) of the box.")
      .def_property(
          "max_corner",
          [](const geodetic::Box &self) { return self.max_corner(); },
          [](geodetic::Box &self, const geodetic::Point &point) -> void {
            self.max_corner() = point;
          },
          "The maximal corner (upper right) of the box.")
      .def_static("whole_earth", &geodetic::Box::whole_earth,
                  "Returns the box covering the whole earth.")
      .def(
          "as_polygon",
          [](const geodetic::Box &self) -> geodetic::Polygon {
            return static_cast<geodetic::Polygon>(self);
          },
          "Returns the box as a polygon.")
      .def("centroid", &geodetic::Box::centroid,
           R"__doc__(
Computes the centroid of the box.

Returns:
    The centroid of the box.
)__doc__")
      .def(
          "covered_by",
          [](const geodetic::Box &self, const geodetic::Point &point) -> bool {
            return self.covered_by(point);
          },
          py::arg("point"), R"__doc__(
Test if the given point is inside or on border of this box.

Args:
    point: point to test.
Returns:
    True if the given point is inside or on border of this box.
)__doc__")
      .def(
          "covered_by",
          [](const geodetic::Box &self,
             const Eigen::Ref<const Eigen::VectorXd> &lon,
             const Eigen::Ref<const Eigen::VectorXd> &lat,
             const size_t num_threads) -> py::array_t<bool> {
            return self.covered_by(lon, lat, num_threads);
          },
          py::arg("lon"), py::arg("lat"), py::arg("num_threads") = 1,
          R"__doc__(
Test if the coordinates of the points provided are located inside or at the
edge of this box.

Args:
    lon: Longitudes coordinates in degrees to check
    lat: Latitude coordinates in degrees to check
    num_threads: The number of threads to use for the computation. If 0 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Default to 1.
Returns:
    A vector containing a flag equal to 1 if the coordinate is located in the
    box or at the edge otherwise 0.
)__doc__")
      .def("area", &geodetic::Box::area, py::arg("wgs") = std::nullopt,
           R"__doc__(
Calculates the area.

Args:
    wgs: The spheroid used to calculate the area. If not provided, the
        WGS-84 spheroid is used.

Returns:
    The calculated area.
)__doc__",
           py::call_guard<py::gil_scoped_release>())
      .def(
          "distance",
          [](const geodetic::Box &self, const geodetic::Box &other) -> double {
            return self.distance(other);
          },
          py::arg("other"),
          R"__doc__(
Calculate the distance between the two boxes.

Args:
    other: The other box to consider.

Returns:
    The distance between the two boxes in meters.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "distance",
          [](const geodetic::Box &self, const geodetic::Point &point)
              -> double { return self.distance(point); },
          py::arg("point"),
          R"__doc__(
Calculate the distance between this instance and a point.

Args:
    point: The point to consider.

Returns:
    The distance between this box and the provided point.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def("to_geojson", &geodetic::Box::to_geojson,
           R"__doc__(
Return the box as a GeoJSON type.

Returns:
    The box as a GeoJSON type.
)__doc__")
      .def(
          "wkt",
          [](const geodetic::Box &self) -> std::string {
            auto ss = std::stringstream();
            ss << boost::geometry::wkt(self);
            return ss.str();
          },
          R"__doc__(
Gets the OGC Well-Known Text (WKT) representation of this instance.

Returns:
    The WKT representation.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def_static(
          "read_wkt",
          [](const std::string &wkt) -> geodetic::Box {
            auto box = geodetic::Box();
            boost::geometry::read_wkt(wkt, box);
            return box;
          },
          py::arg("wkt"), R"__doc__(
Parses OGC Well-Known Text (WKT) into a box.

Args:
    wkt: the WKT representation of the box.
Returns:
    The box defined by the WKT representation.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "__copy__",
          [](const geodetic::Box &self) { return geodetic::Box(self); },
          "Implements the shallow copy operation.",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "__eq__",
          [](const geodetic::Box &self, const geodetic::Box &rhs) -> bool {
            return boost::geometry::equals(self, rhs);
          },
          py::arg("other"),
          "Overrides the default behavior of the ``==`` operator.")
      .def(
          "__ne__",
          [](const geodetic::Box &self, const geodetic::Box &rhs) -> bool {
            return !boost::geometry::equals(self, rhs);
          },
          py::arg("other"),
          "Overrides the default behavior of the ``!=`` operator.")
      .def("__repr__", &geodetic::Box::to_string,
           "Called by the ``repr()`` built-in function to compute the string "
           "representation of a box.")
      .def(py::pickle([](const geodetic::Box &self) { return self.getstate(); },
                      [](const py::tuple &state) {
                        return geodetic::Box::setstate(state);
                      }));
}

static void init_geodetic_polygon(py::class_<geodetic::Polygon> &class_) {
  class_
      .def(py::init([](const py::list &outer,
                       std::optional<const py::list> &inners) {
             return geodetic::Polygon(outer, inners.value_or(py::list()));
           }),
           py::arg("outer"), py::arg("inners") = std::nullopt)
      .def_property_readonly("outer", &geodetic::Polygon::outer,
                             "The outer ring.")
      .def_property_readonly("inners", &geodetic::Polygon::inners,
                             "The inner rings.")
      .def_static("from_geojson", &geodetic::Polygon::from_geojson,
                  py::arg("array"), R"__doc__(
Creates a polygon from a GeoJSON coordinates array.

Args:
    array: The GeoJSON coordinates array.
Returns:
    The polygon defined by the GeoJSON coordinate array.
)__doc__")
      .def(
          "__copy__",
          [](const geodetic::Polygon &self) { return geodetic::Polygon(self); },
          "Implements the shallow copy operation.",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "__eq__",
          [](const geodetic::Polygon &self, const geodetic::Polygon &rhs)
              -> bool { return boost::geometry::equals(self, rhs); },
          py::arg("other"),
          "Overrides the default behavior of the ``==`` operator.")
      .def(
          "__ne__",
          [](const geodetic::Polygon &self, const geodetic::Polygon &rhs)
              -> bool { return !boost::geometry::equals(self, rhs); },
          py::arg("other"),
          "Overrides the default behavior of the ``!=`` operator.")
      .def("__repr__", &geodetic::Polygon::to_string,
           "Called by the ``repr()`` built-in function to compute the string "
           "representation of a point.")
      .def("envelope", &geodetic::Polygon::envelope,
           R"__doc__(
Calculates the envelope of this polygon.

Returns:
    The envelope of this instance.
)__doc__",
           py::call_guard<py::gil_scoped_release>())
      .def("num_interior_rings", &geodetic::Polygon::num_interior_rings,
           "Returns the number of the interior rings.")
      .def(
          "union",
          [](const geodetic::Polygon &self, const geodetic::Polygon &other)
              -> geodetic::MultiPolygon { return self.union_(other); },
          py::arg("other"),
          R"__doc__(
Computes the union of this polygon with another.

Args:
    other: The polygon to compute the union with.
Returns:
    The union of this polygon with the provided polygon.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "intersection",
          [](const geodetic::Polygon &self, const geodetic::Polygon &other)
              -> geodetic::MultiPolygon { return self.intersection(other); },
          py::arg("other"),
          R"__doc__(
Computes the intersection of this polygon with another.

Args:
    other: The polygon to compute the intersection with.
Returns:
    The intersection of this polygon with the provided polygon.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "intersects",
          [](const geodetic::Polygon &self, const geodetic::Polygon &other)
              -> bool { return self.intersects(other); },
          py::arg("other"),
          R"__doc__(
Checks if this polygon intersects another.

Args:
    other: The polygon to check for intersection with.
Returns:
    True if this polygon intersects the provided polygon, False otherwise.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "touches",
          [](const geodetic::Polygon &self, const geodetic::Polygon &other)
              -> bool { return self.touches(other); },
          py::arg("other"),
          R"__doc__(
Checks if this polygon touches another.

Args:
    other: The polygon to check for touch with.
Returns:
    True if this polygon touches the provided polygon, False otherwise.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "covered_by",
          [](const geodetic::Polygon &self, const geodetic::Point &point)
              -> bool { return self.covered_by(point); },
          py::arg("point"), R"__doc__(
Test if the given point is inside or on border of this polygon.

Args:
    point: point to test.
Returns:
    True if the given point is inside or on border of this polygon.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "covered_by",
          [](const geodetic::Polygon &self,
             const Eigen::Ref<const Eigen::VectorXd> &lon,
             const Eigen::Ref<const Eigen::VectorXd> &lat,
             const size_t num_threads) -> py::array_t<bool> {
            return self.covered_by(lon, lat, num_threads);
          },
          py::arg("lon"), py::arg("lat"), py::arg("num_threads") = 1,
          R"__doc__(
Test if the coordinates of the points provided are located inside or at the
edge of this polygon.

Args:
    lon: Longitudes coordinates in degrees to check.
    lat: Latitude coordinates in degrees to check.
    num_threads: The number of threads to use for the computation. If 0 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Default to 1.
Returns:
    A vector containing a flag equal to 1 if the coordinate is located in the
    polygon or at the edge otherwise 0.
)__doc__")
      .def("area", &geodetic::Polygon::area, py::arg("wgs") = std::nullopt,
           R"__doc__(
Calculates the area.

Args:
    wgs: The spheroid used to calculate the distance. If not provided, the
        WGS-84 spheroid is used.

Returns:
    The calculated area.
)__doc__",
           py::call_guard<py::gil_scoped_release>())
      .def(
          "distance",
          [](const geodetic::Polygon &self, const geodetic::Polygon &other)
              -> double { return self.distance(other); },
          py::arg("other"),
          R"__doc__(
Calculate the distance between the two polygons.

Args:
    other: The other polygon to consider.

Returns:
    The distance between the two polygons in meters.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "distance",
          [](const geodetic::Polygon &self, const geodetic::Point &point)
              -> double { return self.distance(point); },
          py::arg("point"),
          R"__doc__(
Calculate the distance between this instance and a point.

Args:
    point: The point to consider.

Returns:
    The distance between this polygon and the provided point.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def("to_geojson", &geodetic::Polygon::to_geojson,
           R"__doc__(
Return the polygon as a GeoJSON type.

Returns:
    The polygon as a GeoJSON type.
)__doc__")
      .def(
          "wkt",
          [](const geodetic::Polygon &self) -> std::string {
            auto ss = std::stringstream();
            ss << boost::geometry::wkt(self);
            return ss.str();
          },
          R"__doc__(
Gets the OGC Well-Known Text (WKT) representation of this instance.

Returns:
    The WKT representation.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def_static(
          "read_wkt",
          [](const std::string &wkt) -> geodetic::Polygon {
            auto polygon = geodetic::Polygon();
            boost::geometry::read_wkt(wkt, polygon);
            return polygon;
          },
          py::arg("wkt"), R"__doc__(
Parses OGC Well-Known Text (WKT) into a polygon.

Args:
    wkt: the WKT representation of the polygon.
Returns:
    The polygon defined by the WKT representation.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(py::pickle(
          [](const geodetic::Polygon &self) { return self.getstate(); },
          [](const py::tuple &state) {
            return geodetic::Polygon::setstate(state);
          }));
}

static void init_geodetic_multipolygon(
    py::class_<geodetic::MultiPolygon> &class_) {
  class_.def(py::init<>(), "Defaults to an empty MultiPolygon.")
      .def(py::init<const py::list &>(), py::arg("polygons"), R"__doc__(
Initializes a MultiPolygon from a list of polygons.

Args:
    polygons: The polygons to use.
)__doc__")
      .def_static("from_geojson", &geodetic::MultiPolygon::from_geojson,
                  py::arg("array"), R"__doc__(
Initializes a MultiPolygon from a GeoJSON coordinate array.

Args:
    array: The GeoJSON coordinate array.
Returns:
    The MultiPolygon initialized from the GeoJSON coordinate array.
)__doc__")
      .def("num_interior_rings", &geodetic::MultiPolygon::num_interior_rings,
           "Returns the number of the interior rings of all polygons.")
      .def(
          "union",
          [](const geodetic::MultiPolygon &self, const geodetic::Polygon &other)
              -> geodetic::MultiPolygon { return self.union_(other); },
          py::arg("other"),
          R"__doc__(
Computes the union of this multi-polygon with a polygon.

Args:
    other: The polygon to compute the union with.
Returns:
    The union of this multi-polygon with the provided polygon.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "union",
          [](const geodetic::MultiPolygon &self,
             const geodetic::MultiPolygon &other) -> geodetic::MultiPolygon {
            return self.union_(other);
          },
          py::arg("other"),
          R"__doc__(
Computes the union of this multi-polygon with another multi-polygon.

Args:
    other: The multi-polygon to compute the union with.
Returns:
    The union of this multi-polygon with the provided multi-polygon.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "intersection",
          [](const geodetic::MultiPolygon &self, const geodetic::Polygon &other)
              -> geodetic::MultiPolygon { return self.intersection(other); },
          py::arg("other"),
          R"__doc__(
Computes the intersection of this multi-polygon with a polygon.

Args:
    other: The polygon to compute the intersection with.
Returns:
    The intersection of this multi-polygon with the provided polygon.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "intersection",
          [](const geodetic::MultiPolygon &self,
             const geodetic::MultiPolygon &other) -> geodetic::MultiPolygon {
            return self.intersection(other);
          },
          py::arg("other"),
          R"__doc__(
Computes the intersection of this multi-polygon with another multi-polygon.

Args:
    other: The multi-polygon to compute the intersection with.
Returns:
    The intersection of this multi-polygon with the provided multi-polygon.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "intersects",
          [](const geodetic::MultiPolygon &self, const geodetic::Polygon &other)
              -> bool { return self.intersects(other); },
          py::arg("other"),
          R"__doc__(
Checks if this multi-polygon intersects with a polygon.

Args:
    other: The polygon to check for intersection with.
Returns:
    True if this multi-polygon intersects with the provided polygon, False
    otherwise.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "intersects",
          [](const geodetic::MultiPolygon &self,
             const geodetic::MultiPolygon &other) -> bool {
            return self.intersects(other);
          },
          py::arg("other"),
          R"__doc__(
Checks if this multi-polygon intersects with another multi-polygon.

Args:
    other: The multi-polygon to check for intersection with.
Returns:
    True if this multi-polygon intersects with the provided multi-polygon,
    False otherwise.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "touches",
          [](const geodetic::MultiPolygon &self, const geodetic::Polygon &other)
              -> bool { return self.touches(other); },
          py::arg("other"),
          R"__doc__(
Checks if this multi-polygon touches a polygon.

Args:
    other: The polygon to check for touches with.
Returns:
    True if this multi-polygon touches the provided polygon, False otherwise.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "touches",
          [](const geodetic::MultiPolygon &self,
             const geodetic::MultiPolygon &other) -> bool {
            return self.touches(other);
          },
          py::arg("other"),
          R"__doc__(
Checks if this multi-polygon touches another multi-polygon.

Args:
    other: The multi-polygon to check for touches with.
Returns:
    True if this multi-polygon touches the provided multi-polygon, False
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "append",
          [](geodetic::MultiPolygon &self, geodetic::Polygon polygon) -> void {
            self.append(std::move(polygon));
          },
          py::arg("polygon"), R"__doc__(
Appends a polygon to this instance.

Args:
    polygon: The polygon to append.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "__copy__",
          [](const geodetic::MultiPolygon &self) {
            return geodetic::MultiPolygon(self);
          },
          "Implements the shallow copy operation.",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "__add__",
          [](const geodetic::MultiPolygon &lhs,
             const geodetic::MultiPolygon &rhs) -> geodetic::MultiPolygon {
            auto result = geodetic::MultiPolygon(lhs);
            result += rhs;
            return result;
          },
          py::arg("other"),
          "Overrides the + operator to concatenate two MultiPolygons.",
          py::call_guard<py::gil_scoped_release>())
      .def("__iadd__", &geodetic::MultiPolygon::operator+=, py::arg("other"),
           "Overrides the default behavior of the ``+=`` operator.",
           py::call_guard<py::gil_scoped_release>())
      .def("__len__", &geodetic::MultiPolygon::size,
           "Returns the number of polygons in this instance.")
      .def("__getitem__", &geodetic::MultiPolygon::operator(), py::arg("index"),
           "Returns the polygon at the given index.")
      .def(
          "__contains__",
          [](const geodetic::MultiPolygon &self,
             const geodetic::Polygon &polygon) {
            return self.contains(polygon);
          },
          py::arg("polygon"),
          "True if the multi-polygon has the specified polygon, else False",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "__iter__",
          [](const geodetic::MultiPolygon &self) {
            return py::make_iterator(self.begin(), self.end());
          },
          py::keep_alive<0, 1>())
      .def(
          "__eq__",
          [](const geodetic::MultiPolygon &self,
             const geodetic::MultiPolygon &rhs) -> bool {
            return boost::geometry::equals(self, rhs);
          },
          py::arg("other"),
          "Overrides the default behavior of the ``==`` operator.",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "__ne__",
          [](const geodetic::MultiPolygon &self,
             const geodetic::MultiPolygon &rhs) -> bool {
            return !boost::geometry::equals(self, rhs);
          },
          py::arg("other"),
          "Overrides the default behavior of the ``!=`` operator.",
          py::call_guard<py::gil_scoped_release>())
      .def("__repr__", &geodetic::MultiPolygon::to_string,
           "Called by the ``repr()`` built-in function to compute the string "
           "representation of a point.")
      .def("envelope", &geodetic::MultiPolygon::envelope,
           R"__doc__(
Calculates the envelope of this multi-polygon.

Returns:
    The envelope of this instance.
)__doc__",
           py::call_guard<py::gil_scoped_release>())
      .def(
          "covered_by",
          [](const geodetic::MultiPolygon &self, const geodetic::Point &point)
              -> bool { return self.covered_by(point); },
          py::arg("point"), R"__doc__(
Test if the given point is inside or on border of this multi-polygon.

Args:
    point: point to test.
Returns:
    True if the given point is inside or on border of this multi-polygon.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "covered_by",
          [](const geodetic::MultiPolygon &self,
             const Eigen::Ref<const Eigen::VectorXd> &lon,
             const Eigen::Ref<const Eigen::VectorXd> &lat,
             const size_t num_threads) -> py::array_t<bool> {
            return self.covered_by(lon, lat, num_threads);
          },
          py::arg("lon"), py::arg("lat"), py::arg("num_threads") = 1,
          R"__doc__(
Test if the coordinates of the points provided are located inside or at the
edge of this multi-polygon.

Args:
    lon: Longitudes coordinates in degrees to check.
    lat: Latitude coordinates in degrees to check.
    num_threads: The number of threads to use for the computation. If 0 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Default to 1.
Returns:
    A vector containing a flag equal to 1 if the coordinate is located in the
    multi-polygon or at the edge otherwise 0.
)__doc__")
      .def("area", &geodetic::MultiPolygon::area, py::arg("wgs") = std::nullopt,
           R"__doc__(
Calculates the area.

Args:
    wgs: The spheroid used to calculate the area. If not provided, the
        WGS-84 spheroid is used.

Returns:
    The calculated area.
)__doc__",
           py::call_guard<py::gil_scoped_release>())
      .def(
          "distance",
          [](const geodetic::MultiPolygon &self,
             const geodetic::MultiPolygon &other) -> double {
            return self.distance(other);
          },
          py::arg("other"),
          R"__doc__(
Calculate the distance between the two multi-polygons.

Args:
    other: The other multi-polygon to consider.

Returns:
    The distance between the two multi-polygons in meters.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "distance",
          [](const geodetic::MultiPolygon &self, const geodetic::Polygon &other)
              -> double { return self.distance(other); },
          py::arg("other"),
          R"__doc__(
Calculate the distance between this instance and a polygon.

Args:
    other: The other multi-polygon to consider.

Returns:
    The distance between this instance and the polygon in meters.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "distance",
          [](const geodetic::MultiPolygon &self, const geodetic::Point &point)
              -> double { return self.distance(point); },
          py::arg("point"),
          R"__doc__(
Calculate the distance between this instance and a point.

Args:
    point: The point to consider.

Returns:
    The distance between this multi-polygon and the provided point.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def("to_geojson", &geodetic::MultiPolygon::to_geojson,
           R"__doc__(
Return the multi-polygon as a GeoJSON type.

Returns:
    The multi-polygon as a GeoJSON type.
)__doc__")
      .def(
          "wkt",
          [](const geodetic::MultiPolygon &self) -> std::string {
            auto ss = std::stringstream();
            ss << boost::geometry::wkt(self);
            return ss.str();
          },
          R"__doc__(
Gets the OGC Well-Known Text (WKT) representation of this instance.

Returns:
    The WKT representation.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def_static(
          "read_wkt",
          [](const std::string &wkt) -> geodetic::MultiPolygon {
            auto multipolygon = geodetic::MultiPolygon();
            boost::geometry::read_wkt(wkt, multipolygon);
            return multipolygon;
          },
          py::arg("wkt"), R"__doc__(
Parses OGC Well-Known Text (WKT) into a multi-polygon.

Args:
    wkt: the WKT representation of the multi-polygon.
Returns:
    The multi-polygon defined by the WKT representation.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(py::pickle(
          [](const geodetic::MultiPolygon &self) { return self.getstate(); },
          [](const py::tuple &state) {
            return geodetic::MultiPolygon::setstate(state);
          }));
}

static void init_geodetic_linestring(py::module &m) {
  py::class_<geodetic::LineString>(
      m, "LineString",
      R"__doc__(LineString(self, lon: numpy.ndarray, lat: numpy.ndarray)

A linestring (named so by OGC) is a collection of points.

Args:
    lon: Longitudes coordinates in degrees.
    lat: Latitude coordinates in degrees.
)__doc__")
      .def(py::init<>())
      .def(py::init<const py::list &>(), py::arg("points"))
      .def(py::init<const Eigen::Ref<const pyinterp::Vector<double>> &,
                    const Eigen::Ref<const pyinterp::Vector<double>> &>(),
           py::arg("lon"), py::arg("lat"),
           py::call_guard<py::gil_scoped_release>())
      .def_static("from_geojson", &geodetic::LineString::from_geojson,
                  py::arg("array"), R"__doc__(
Creates a line string from a GeoJSON coordinates array.

Args:
    array: the GeoJSON coordinate array.
Returns:
    The line string defined by the GeoJSON coordinate array.
)__doc__")
      .def("to_geojson", &geodetic::LineString::to_geojson,
           R"__doc__(
Return the line string as a GeoJSON type.

Returns:
    The line string as a GeoJSON type.
)__doc__")
      .def(
          "wkt",
          [](const geodetic::LineString &self) -> std::string {
            auto ss = std::stringstream();
            ss << boost::geometry::wkt(self);
            return ss.str();
          },
          R"__doc__(
Gets the OGC Well-Known Text (WKT) representation of this instance.

Returns:
    The WKT representation.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def_static(
          "read_wkt",
          [](const std::string &wkt) -> geodetic::LineString {
            auto result = geodetic::LineString();
            boost::geometry::read_wkt(wkt, result);
            return result;
          },
          py::arg("wkt"), R"__doc__(
Parses OGC Well-Known Text (WKT) into a LineString.

Args:
    wkt: the WKT representation of the LineString.
Returns:
    The line string defined by the WKT representation.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "append",
          [](geodetic::LineString &self, geodetic::Point point) -> void {
            self.append(point);
          },
          py::arg("point"), R"__doc__(
Appends a point to this instance.

Args:
    point: The point to append.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "__copy__",
          [](const geodetic::LineString &self) {
            return geodetic::LineString(self);
          },
          "Implements the shallow copy operation.",
          py::call_guard<py::gil_scoped_release>())
      .def("__len__", &geodetic::LineString::size,
           "Called to implement the built-in function ``len()``")
      .def(
          "__getitem__",
          [](const geodetic::LineString &self,
             size_t index) -> geodetic::Point { return self(index); },
          py::arg("index"), "Returns the point at the given index.")
      .def(
          "__eq__",
          [](const geodetic::LineString &self, const geodetic::LineString &rhs)
              -> bool { return boost::geometry::equals(self, rhs); },
          py::arg("other"),
          "Overrides the default behavior of the ``==`` operator.")
      .def(
          "__ne__",
          [](const geodetic::LineString &self, const geodetic::LineString &rhs)
              -> bool { return !boost::geometry::equals(self, rhs); },
          py::arg("other"),
          "Overrides the default behavior of the ``!=`` operator.")
      .def("__repr__", &geodetic::LineString::to_string,
           "Called by the ``repr()`` built-in function to compute the string "
           "representation of a point.")
      .def("__iter__",
           [](const geodetic::LineString &self) {
             return py::make_iterator(self.begin(), self.end(),
                                      py::keep_alive<0, 1>());
           })
      .def(
          "curvilinear_distance",
          [](const geodetic::LineString &self, const std::string &strategy,
             const std::optional<geodetic::Spheroid> &spheroid) {
            return self.curvilinear_distance(parse_distance_strategy(strategy),
                                             spheroid);
          },
          py::arg("strategy") = "thomas", py::arg("wgs") = std::nullopt,
          R"__doc__(
Computes the curvilinear distance between the points of this instance.

Args:
    strategy: the distance strategy to use. This parameter can take the values
        ``andoyer``, ``thomas`` or ``vincenty``
    wgs: the spheroid to use. If not provided, the WGS84 spheroid is used.

Returns:
    The curvilinear distance between the points of this instance.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def("intersects", &geodetic::LineString::intersects, py::arg("rhs"),
           py::arg("wgs") = std::nullopt,
           R"__doc__(
Test if this linestring intersects with another linestring.

Args:
    rhs: The linestring to test.
    wgs: If specified, searches for the intersection using geographic
    coordinates with the specified spheroid, otherwise searches for the
    intersection using spherical coordinates.
Returns:
    True if the linestring intersects this instance.
)__doc__",
           py::call_guard<py::gil_scoped_release>())
      .def("intersection", &geodetic::LineString::intersection, py::arg("rhs"),
           py::arg("wgs") = std::nullopt,
           R"__doc__(
Computes the intersection between this linestring and another linestring.

Args:
    rhs: The linestring to test.
    wgs: The World Geodetic System to use. Defaults to WGS84.

Returns:
    The intersection between this linestring and the other linestring.
)__doc__",
           py::call_guard<py::gil_scoped_release>())
      .def(py::pickle(
          [](const geodetic::LineString &self) { return self.getstate(); },
          [](const py::tuple &state) {
            return geodetic::LineString::setstate(state);
          }));
}

void init_geodetic_crossover(py::module &m) {
  py::class_<geodetic::Crossover>(
      m, "Crossover",
      "Crossover(self,"
      " half_orbit_1: pyinterp.core.geodetic.LineString,"
      " half_orbit_2: pyinterp.core.geodetic.LineString)"
      R"__doc__(

Calculate the crossover between two half-orbits.

Args:
    half_orbit_1: The first half-orbit.
    half_orbit_2: The second half-orbit.
)__doc__")
      .def(py::init<geodetic::LineString, geodetic::LineString>(),
           py::arg("half_orbit_1"), py::arg("half_orbit_2"),
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("half_orbit_1",
                             &geodetic::Crossover::get_half_orbit_1,
                             "Returns the first half-orbit.")
      .def_property_readonly("half_orbit_2",
                             &geodetic::Crossover::get_half_orbit_2,
                             "Returns the second half-orbit.")
      .def("search", &geodetic::Crossover::search,
           py::arg("wgs") = std::nullopt,
           R"__doc__(
Search for the crossover between the two half-orbits.

Args:
    wgs: If specified, searches for the intersection using geographic
        coordinates with the specified spheroid, otherwise searches for the
        intersection using spherical coordinates.

Returns:
    The crossover or None if there is no crossover.
)__doc__",
           py::call_guard<py::gil_scoped_release>())
      .def("exists", &geodetic::Crossover::exists,
           py::arg("wgs") = std::nullopt,
           R"__doc__(
Test if there is a crossover between the two half-orbits.

Args:
    wgs: If specified, searches for the intersection using geographic
        coordinates with the specified spheroid, otherwise searches for the
        intersection using spherical coordinates.

Returns:
    True if there is a crossover.
)__doc__",
           py::call_guard<py::gil_scoped_release>())
      .def(
          "nearest",
          [](const geodetic::Crossover &self, const geodetic::Point &point,
             const std::optional<double> &predicate,
             const std::string &strategy,
             const std::optional<geodetic::Spheroid> &wgs)
              -> std::optional<std::tuple<size_t, size_t>> {
            return self.nearest(point, predicate.value_or(40'075'000.0),
                                parse_distance_strategy(strategy), wgs);
          },
          py::arg("point"), py::arg("predicate") = std::nullopt,
          py::arg("strategy") = "thomas", py::arg("wgs") = std::nullopt,
          R"__doc__(
Find the nearest indices on the two half-orbits from a given point.

Args:
    point: The point to consider.
    predicate: The distance predicate, in meters.
    strategy: The distance calculation strategy.
    wgs: The spheroid used to calculate the distance. If not provided, the
        WGS-84 spheroid is used.

Returns:
    The indices of the nearest points or None if no intersection is found.
)__doc__",
          py::call_guard<py::gil_scoped_release>())
      .def(py::pickle(
          [](const geodetic::Crossover &self) { return self.getstate(); },
          [](const py::tuple &state) {
            return geodetic::Crossover::setstate(state);
          }));
}

static auto init_geodetic_rtree(py::module &m) {
  py::class_<geodetic::RTree>(
      m, "RTree",
      "RTree(self, spheroid: Optional[pyinterp.core.geodetic.Spheroid] = None)"
      R"__doc__(
R*Tree spatial index.

Args:
    spheroid: WGS of the coordinate system used to calculate the distance.
)__doc__")
      .def(py::init<std::optional<geodetic::Spheroid>>(),
           py::arg("spheroid") = std::nullopt)
      .def(
          "__copy__",
          [](const geodetic::RTree &self) { return geodetic::RTree(self); },
          "Implements the shallow copy operation.",
          py::call_guard<py::gil_scoped_release>())
      .def("__len__", &geodetic::RTree::size,
           "Called to implement the built-in function ``len()``")
      .def(
          "__bool__", [](const geodetic::RTree &self) { return !self.empty(); },
          "Called to implement truth value testing and the built-in "
          "operation "
          "``bool()``.")
      .def("clear", &geodetic::RTree::clear,
           "Removes all values stored in the container.")
      .def("packing", &geodetic::RTree::packing, py::arg("lon"), py::arg("lat"),
           py::arg("values"),
           R"__doc__(
The tree is created using packing algorithm (The old data is erased
before construction.)

Args:
    lon: The longitude, in degrees, of the points to insert.
    lat: The latitude, in degrees, of the points to insert.
    values: The values to insert.
)__doc__",
           py::call_guard<py::gil_scoped_release>())
      .def("insert", &geodetic::RTree::insert, py::arg("lon"), py::arg("lat"),
           py::arg("values"),
           R"__doc__(
Insert new data into the search tree.

Args:
    lon: The longitude, in degrees, of the points to insert.
    lat: The latitude, in degrees, of the points to insert.
    values: The values to insert.
)__doc__",
           py::call_guard<py::gil_scoped_release>())
      .def(
          "query",
          [](const geodetic::RTree &self,
             const Eigen::Ref<pyinterp::Vector<double>> &lon,
             const Eigen::Ref<pyinterp::Vector<double>> &lat, const uint32_t k,
             const bool within, const size_t num_threads) -> py::tuple {
            return self.query(lon, lat, k, within, num_threads);
          },
          py::arg("lon"), py::arg("lat"), py::arg("k") = 4,
          py::arg("within") = false, py::arg("num_threads") = 0,
          R"__doc__(
Search for the nearest K nearest neighbors of a given point.

Args:
    lon: The longitude of the points in degrees.
    lat: The latitude of the points in degrees.
    k: The number of nearest neighbors to be used for calculating the
        interpolated value. Defaults to ``4``.
    within: If true, the method ensures that the neighbors found are located
        within the point of interest. Defaults to ``false``.
    num_threads: The number of threads to use for the computation. If 0 all
        CPUs are used. If 1 is given, no parallel computing code is used at
        all, which is useful for debugging. Defaults to ``0``.
Returns:
    A tuple containing a matrix describing for each provided position, the
    distance, in meters, between the provided position and the found neighbors
    and a matrix containing the value of the different neighbors found for all
    provided positions.
)__doc__")
      .def("inverse_distance_weighting",
           &geodetic::RTree::inverse_distance_weighting, py::arg("lon"),
           py::arg("lat"), py::arg("radius") = std::nullopt, py::arg("k") = 9,
           py::arg("p") = 2, py::arg("within") = true,
           py::arg("num_threads") = 0,
           R"__doc__(
Interpolation of the value at the requested position by inverse distance
weighting method.

Args:
    lon: The longitude of the points, in degrees, to be interpolated.
    lat: The latitude of the points, in degrees, to be interpolated.
    radius: The maximum radius of the search (m). Defaults The maximum distance
        between two points.
    k: The number of nearest neighbors to be used for calculating the
        interpolated value. Defaults to ``9``.
    p: The power parameters. Defaults to ``2``. within (bool, optional): If
        true, the method ensures that the neighbors found are located around
        the point of interest. In other words, this parameter ensures that the
        calculated values will not be extrapolated. Defaults to ``true``.
    within: If true, the method ensures that the neighbors found are located
        within the point of interest. Defaults to ``false``.
    num_threads: The number of threads to use for the computation. If 0 all
        CPUs are used. If 1 is given, no parallel computing code is used at
        all, which is useful for debugging. Defaults to ``0``.
Returns:
    The interpolated value and the number of neighbors used in the
    calculation.
)__doc__")
      .def("radial_basis_function", &geodetic::RTree::radial_basis_function,
           py::arg("lon"), py::arg("lat"), py::arg("radius"), py::arg("k") = 9,
           py::arg("rbf") = math::RadialBasisFunction::Multiquadric,
           py::arg("epsilon") = std::optional<double>(), py::arg("smooth") = 0,
           py::arg("within") = true, py::arg("num_threads") = 0,
           R"__doc__(
Interpolation of the value at the requested position by radial basis
function interpolation.

Args:
    lon: The longitude of the points, in degrees, to be interpolated.
    lat: The latitude of the points, in degrees, to be interpolated.
    radius: The maximum radius of the search (m). Default to the
    largest value
        that can be represented on a float.
    k: The number of nearest neighbors to be used for calculating the
        interpolated value. Defaults to ``9``.
    rbf: The radial basis function, based on the radius, r, given by
        the distance between points. Default to
        :py:attr:`pyinterp.core.RadialBasisFunction.Multiquadric`.
    epsilon: Adjustable constant for gaussian or multiquadrics
        functions. Default to the average distance between nodes.
    smooth: Values greater than zero increase the smoothness of the
        approximation.
    within: If true, the method ensures that the neighbors found are
        located around the point of interest. Defaults to ``true``.
    num_threads: The number of threads to use for the computation. If 0
        all CPUs are used. If 1 is given, no parallel computing code is used at
        all, which is useful for debugging. Defaults to ``0``.
Returns:
    The interpolated value and the number of neighbors used for the
    calculation.
)__doc__")
      .def("window_function", &geodetic::RTree::window_function, py::arg("lon"),
           py::arg("lat"), py::arg("radius"), py::arg("k") = 9,
           py::arg("wf") = math::window::Function::kHamming,
           py::arg("arg") = std::nullopt, py::arg("within") = true,
           py::arg("num_threads") = 0,
           R"__doc__(
Interpolation of the value at the requested position by window function.

Args:
    lon: The longitude of the points, in degrees, to be interpolated.
    lat: The latitude of the points, in degrees, to be interpolated.
    radius: The maximum radius of the search (m). Default to the largest value
        that can be represented on a float.
    k: The number of nearest neighbors to be used for calculating the
        interpolated value. Defaults to ``9``.
    wf: The window function to be used. Defaults to
        :py:attr:`pyinterp.core.WindowFunction.Hamming`.
    arg: The optional argument of the window function. Defaults to ``None``.
    within: If true, the method ensures that the neighbors found are located
        around the point of interest. Defaults to ``true``.
    num_threads: The number of threads to use for the computation. If 0 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Defaults to ``0``.
Returns:
    The interpolated value and the number of neighbors used for the calculation.
  )__doc__")

      .def(py::pickle(
          [](const geodetic::RTree &self) { return self.getstate(); },
          [](const py::tuple &state) {
            return geodetic::RTree::setstate(state);
          }));
}

void init_geodetic(py::module &m) {
  auto _spheroid = py::class_<pyinterp::detail::geodetic::Spheroid>(
      m, "_Spheroid", "C++ implementation of the WGS system.");

  py::class_<geodetic::Spheroid, pyinterp::detail::geodetic::Spheroid>(
      m, "Spheroid",
      R"(
Spheroid(self, semi_major_axis: float, flattening: float)

World Geodetic System (WGS).

Args:
    semi_major_axis: Semi-major axis of ellipsoid, in meters.
    flattening: Flattening of ellipsoid.
.. note::
    The default constructor initializes a WGS-84 ellipsoid.
)")
      .def(py::init<>())
      .def(py::init<double, double>(), py::arg("semi_major_axis"),
           py::arg("flattening"))
      .def_property_readonly(
          "semi_major_axis", &geodetic::Spheroid::semi_major_axis,
          "Semi-major axis of ellipsoid, in meters (:math:`a`).")
      .def_property_readonly(
          "flattening", &geodetic::Spheroid::flattening,
          "Flattening of ellipsoid (:math:`f=\\frac{a-b}{a}`).")
      .def("semi_minor_axis", &geodetic::Spheroid::semi_minor_axis,
           R"__doc__(
Gets the semiminor axis.

Returns:
    :math:`b=a(1-f)`
)__doc__")
      .def("first_eccentricity_squared",
           &geodetic::Spheroid::first_eccentricity_squared, R"__doc__(
Gets the first eccentricity squared.

Returns:
    :math:`e^2=\frac{a^2-b^2}{a^2}`
)__doc__")
      .def("second_eccentricity_squared",
           &geodetic::Spheroid::second_eccentricity_squared, R"__doc__(
Gets the second eccentricity squared.

Returns:
    float: :math:`e^2=\frac{a^2-b^2}{b^2}`
)__doc__")
      .def("equatorial_circumference",
           &geodetic::Spheroid::equatorial_circumference,
           py::arg("semi_major_axis") = true, R"__doc__(
Gets the equatorial circumference.

Args:
    semi_major_axis: True to get the equatorial circumference for the
        semi-majors axis, False for the semi-minor axis. Defaults to ``true``.
Returns:
    :math:`2\pi \times a` if semi_major_axis is true otherwise
    :math:`2\pi \times b`.
)__doc__")
      .def("polar_radius_of_curvature",
           &geodetic::Spheroid::polar_radius_of_curvature,
           R"__doc__(
Gets the polar radius of curvature.

Returns:
    :math:`\frac{a^2}{b}`
)__doc__")
      .def("equatorial_radius_of_curvature",
           &geodetic::Spheroid::equatorial_radius_of_curvature,
           R"__doc__(
Gets the equatorial radius of curvature for a meridian.

Returns:
    :math:`\frac{b^2}{a}`
)__doc__")
      .def("axis_ratio", &geodetic::Spheroid::axis_ratio, R"__doc__(
Gets the axis ratio.

Returns:
    :math:`\frac{b}{a}`
)__doc__")
      .def("linear_eccentricity", &geodetic::Spheroid::linear_eccentricity,
           R"__doc__(
Gets the linear eccentricity.

Returns:
    :math:`E=\sqrt{{a^2}-{b^2}}`
)__doc__")
      .def("mean_radius", &geodetic::Spheroid::mean_radius, R"__doc__(
Gets the mean radius.

Returns:
    :math:`R_1=\frac{2a+b}{3}`
)__doc__")
      .def("geocentric_radius", &geodetic::Spheroid::geocentric_radius,
           py::arg("lat"),
           R"__doc__(
Gets the geocentric radius at the given latitude $\phi$.

Args:
    lat: The latitude, in degrees.

Returns:
    .. math::

        R(\phi)=\sqrt{\frac{{(a^{2}\cos(\phi))}^{2} + \\
        (b^{2}\sin(\phi))^{2}}{(a\cos(\phi))^{2} + (b\cos(\phi))^{2}}}
)__doc__")
      .def("authalic_radius", &geodetic::Spheroid::authalic_radius,
           R"__doc__(
Gets the authalic radius.

Returns:
    :math:`R_2=\sqrt{\frac{a^2+\frac{ab^2}{E}ln(\frac{a + E}{b})}{2}}`
)__doc__")
      .def("volumetric_radius", &geodetic::Spheroid::volumetric_radius,
           R"__doc__(
Gets the volumetric radius.

Returns:
    :math:`R_3=\sqrt[3]{a^{2}b}`
)__doc__")
      .def("__eq__", &geodetic::Spheroid::operator==, py::arg("other"),
           "Overrides the default behavior of the ``==`` operator.")
      .def("__ne__", &geodetic::Spheroid::operator!=, py::arg("other"),
           "Overrides the default behavior of the ``!=`` operator.")
      .def(py::pickle(
          [](const geodetic::Spheroid &self) { return self.getstate(); },
          [](const py::tuple &state) {
            return geodetic::Spheroid::setstate(state);
          }));

  py::class_<geodetic::Coordinates>(
      m, "Coordinates",
      "Coordinates(self, "
      "spheroid: Optional[pyinterp.core.geodetic.Spheroid] = None)"
      R"(

World Geodetic Coordinates System.

Args:
    spheroid: Optional spheroid to use. Defaults to WGS-84.
)")
      .def(py::init<std::optional<geodetic::Spheroid>>(),
           py::arg("spheroid") = std::nullopt)
      .def_property_readonly("spheroid", &geodetic::Coordinates::spheroid,
                             "WGS used to transform the coordinates.")
      .def("ecef_to_lla", &geodetic::Coordinates::ecef_to_lla<double>,
           py::arg("x"), py::arg("y"), py::arg("z"), py::arg("num_threads") = 0,
           R"__doc__(
Converts Cartesian coordinates to Geographic latitude, longitude, and
altitude. Cartesian coordinates should be in meters. The returned
latitude and longitude are in degrees, and the altitude will be in
meters.

Args:
    x: X-coordinates in meters.
    y: Y-coordinates in meters.
    z: Z-coordinates in meters.
    num_threads: The number of threads to use for the computation. If 0 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Defaults to ``0``.
Returns:
    Longitudes, latitudes and altitudes in the coordinate system defined by this
    instance.

.. seealso::

    Olson, D.K. "Converting earth-Centered, Earth-Fixed Coordinates to
    Geodetic Coordinates," IEEE Transactions on Aerospace and Electronic
    Systems, Vol. 32, No. 1, January 1996, pp. 473-476.
  )__doc__")
      .def("lla_to_ecef", &geodetic::Coordinates::lla_to_ecef<double>,
           py::arg("lon"), py::arg("lat"), py::arg("alt"),
           py::arg("num_threads") = 0,
           R"__doc__(
Converts Geographic coordinates latitude, longitude, and altitude to
Cartesian coordinates. The latitude and longitude should be in degrees
and the altitude in meters. The returned ECEF coordinates will be in
meters.

Args:
    lon: Longitudes in degrees.
    lat: Latitudes in degrees.
    alt: Altitudes in meters.
    num_threads: The number of threads to use for the computation. If 0 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Defaults to ``0``.
Returns:
    tuple: X, Y and Z ECEF coordinates in meters.
)__doc__")
      .def("transform", &geodetic::Coordinates::transform<double>,
           py::arg("target"), py::arg("lon"), py::arg("lat"), py::arg("alt"),
           py::arg("num_threads") = 0,
           R"__doc__(
Transforms the positions, provided in degrees and meters, from one WGS
system to another.

Args:
    target: WGS target.
    lon: Longitudes in degrees.
    lat: Latitudes in degrees.
    alt: Altitudes in meters.
    num_threads: The number of threads to use for the computation. If 0 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Defaults to ``0``.
Returns:
    Longitudes, latitudes and altitudes in the new coordinate system.
)__doc__")
      .def(py::pickle(
          [](const geodetic::Coordinates &self) { return self.getstate(); },
          [](const py::tuple &state) {
            return geodetic::Coordinates::setstate(state);
          }));

  auto box = py::class_<geodetic::Box>(m, "Box", R"__doc__(
Box(self, min_corner: Point, max_corner: Point)

Defines a box made of two describing points.

Args:
    min_corner: the minimum corner point (lower left) of the box.
    max_corner: the maximum corner point (upper right) of the box.
)__doc__");

  auto polygon = py::class_<geodetic::Polygon>(
      m, "Polygon",
      R"(Polygon(self, outer: list, inners: Optional[list] = None)

The polygon contains an outer ring and zero or more inner rings.

Args:
    outer: outer ring.
    inners: list of inner rings.
Raises:
    ValueError: if outer is not a list of pyinterp.geodetic.Point.
    ValueError: if inners is not a list of list of pyinterp.geodetic.Point.
)");

  auto multipolygon = py::class_<geodetic::MultiPolygon>(m, "MultiPolygon",
                                                         R"__doc__(
A MultiPolygon is a collection of polygons.

Args:
    polygons: The polygons to use.
)__doc__");

  init_geodetic_point(m);
  init_geodetic_box(box);
  init_geodetic_polygon(polygon);
  init_geodetic_multipolygon(multipolygon);
  init_geodetic_linestring(m);
  init_geodetic_crossover(m);
  init_geodetic_rtree(m);

  m.def(
      "normalize_longitudes",
      [](Eigen::Ref<Eigen::VectorXd> &lon, const double min_lon) -> void {
        auto *lon_ptr = lon.data();
        std::for_each(lon_ptr, lon_ptr + lon.size(), [min_lon](double &x) {
          x = pyinterp::detail::math::normalize_angle(x, min_lon, 360.0);
        });
      },
      py::arg("lon"), py::arg("min_lon") = -180.0,
      R"__doc__(
Normalizes longitudes to the range ``[min_lon, min_lon + 360)`` in place.

Args:
    lon: Longitudes in degrees.
    min_lon: Minimum longitude. Defaults to ``-180.0``.
)__doc__",
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "normalize_longitudes",
      [](const Eigen::Ref<const Eigen::VectorXd> &lon,
         const double min_lon) -> Eigen::VectorXd {
        return lon.unaryExpr([min_lon](double x) {
          return pyinterp::detail::math::normalize_angle(x, min_lon, 360.0);
        });
      },
      py::arg("lon"), py::arg("min_lon") = -180.0,
      R"__doc__(
Normalizes longitudes to the range ``[min_lon, min_lon + 360)``.

Args:
    lon: Longitudes in degrees.
    min_lon: Minimum longitude. Defaults to ``-180.0``.

Returns:
    Longitudes normalized to the range ``[min_lon, min_lon + 360)``.
)__doc__",
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "coordinate_distances",
      [](const Eigen::Ref<const Eigen::VectorXd> &lon1,
         const Eigen::Ref<const Eigen::VectorXd> &lat1,
         const Eigen::Ref<const Eigen::VectorXd> &lon2,
         const Eigen::Ref<const Eigen::VectorXd> &lat2,
         const std::string &strategy,
         const std::optional<geodetic::Spheroid> &wgs,
         const size_t num_threads) -> py::array_t<double> {
        return geodetic::coordinate_distances<geodetic::Point>(
            lon1, lat1, lon2, lat2, parse_distance_strategy(strategy), wgs,
            num_threads);
      },
      py::arg("lon1"), py::arg("lat1"), py::arg("lon2"), py::arg("lat2"),
      py::arg("strategy") = "thomas", py::arg("wgs") = std::nullopt,
      py::arg("num_threads") = 0, R"__doc__(
Returns the distance between the given coordinates.

Args:
    lon1: Longitudes in degrees.
    lat1: Latitudes in degrees.
    lon2: Longitudes in degrees.
    lat2: Latitudes in degrees.
    strategy: The calculation method used to calculate the distance. This
        parameter can take the values "andoyer", "thomas" or "vincenty".
    wgs: The spheroid used to calculate the distance. Defaults to ``None``,
        which means the WGS-84 spheroid is used.
    num_threads: The number of threads to use for the computation. If 0 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Defaults to ``0``.
Returns:
    An array containing the distances ``[..., distance_i, ...]``, corresponding
    to the distances between the coordinates
    ``[..., (Point(lon1_i, lat1_i), Point(lon2_i, lat2_i)), ...]``.
)__doc__");

  m.def("calculate_swath", &geodetic::calculate_swath<double>,
        R"__doc__(
Calculate the swath coordinates from the nadir coordinates.

Args:
    lon_nadir: Longitudes in degrees of the nadir points.
    lat_nadir: Latitudes in degrees of the nadir points.
    delta_ac: Acrosstrack distance in meters.
    hal_gap: The gap between the nadir and the first point of the swath in
        meters.
    half_swath: The half swath width in meters.
    spheroid: The spheroid to use for the calculation. Defaults to ``None``,
        which means the WGS-84 spheroid is used.

Returns:
    A tuple containing the longitudes and latitudes of the swath points.
)__doc__",
        py::arg("lon_nadir"), py::arg("lat_nadir"), py::arg("delta_ac"),
        py::arg("half_gap"), py::arg("half_swath"),
        py::arg("spheroid") = std::nullopt,
        py::call_guard<py::gil_scoped_release>());

  m.def(
      "calculate_crossover",
      [](const Eigen::Ref<const Eigen::VectorXd> &lon1,
         const Eigen::Ref<const Eigen::VectorXd> &lat1,
         const Eigen::Ref<const Eigen::VectorXd> &lon2,
         const Eigen::Ref<const Eigen::VectorXd> &lat2,
         const std::optional<double> &predicate, const std::string &strategy,
         const std::optional<geodetic::Spheroid> &wgs,
         const bool cartesian_plane) {
        return geodetic::crossover(
            lon1, lat1, lon2, lat2, predicate.value_or(40'075'000.0),
            parse_distance_strategy(strategy), wgs, cartesian_plane);
      },
      R"__doc__(
Calculate the crossover coordinates from the nadir coordinates.

Args:
    lon1: Longitudes in degrees of the nadir points of the first line.
    lat1: Latitudes in degrees of the nadir points of the first line.
    lon2: Longitudes in degrees of the nadir points of the second line.
    lat2: Latitudes in degrees of the nadir points of the second line.
    predicate: The maximum distance allowed between the nadir points closest
        to the crossing point.
    strategy: The calculation method used to calculate the distance. This
        parameter can take the values "andoyer", "thomas" or "vincenty".
    wgs: The spheroid used to calculate the distance. Defaults to ``None``,
        which means the WGS-84 spheroid is used.
    cartesian_plane: If ``True``, the crossing point is calculated in the
        cartesian plane. Defaults to ``False``.

Return:
    A tuple containing the crossover coordinates and the indices of the nearest
    nadir points on the first and second line.
)__doc__",
      py::arg("lon1"), py::arg("lat1"), py::arg("lon2"), py::arg("lat2"),
      py::arg("predicate") = std::nullopt, py::arg("strategy") = "thomas",
      py::arg("wgs") = std::nullopt, py::arg("cartesian_plane") = true,
      py::call_guard<py::gil_scoped_release>());
}
