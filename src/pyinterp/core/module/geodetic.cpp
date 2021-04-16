// Copyright (c) 2021 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyinterp/geodetic/box.hpp"
#include "pyinterp/geodetic/coordinates.hpp"
#include "pyinterp/geodetic/point.hpp"
#include "pyinterp/geodetic/polygon.hpp"
#include "pyinterp/geodetic/system.hpp"

namespace geodetic = pyinterp::geodetic;
namespace py = pybind11;

static void init_geodetic_point(py::module& m) {
  py::class_<geodetic::Point>(m, "Point", R"__doc__(
    Handle a point in a geographic coordinates system in degrees.
)__doc__")
      .def(py::init<>())
      .def(py::init<double, double>(), py::arg("lon"), py::arg("lat"),
           R"__doc__(
Build a new point with the coordinates provided.

Args:
    lon (float): Longitude in degrees
    lat (float): Latitude in degrees
)__doc__")
      .def_property("lon",
                    static_cast<double (geodetic::Point::*)() const>(
                        &geodetic::Point::lon),
                    static_cast<void (geodetic::Point::*)(const double)>(
                        &geodetic::Point::lon),
                    "Longitude coordinate in degrees")
      .def_property("lat",
                    static_cast<double (geodetic::Point::*)() const>(
                        &geodetic::Point::lat),
                    static_cast<void (geodetic::Point::*)(const double)>(
                        &geodetic::Point::lat),
                    "Latitude coordinate in degrees")
      .def(
          "wkt",
          [](const geodetic::Point& self) -> std::string {
            auto ss = std::stringstream();
            ss << boost::geometry::wkt(self);
            return ss.str();
          },
          R"__doc__(
Gets the OGC Well-Known Text (WKT) representation of this instance

Return:
    str: the WTK representation
)__doc__")
      .def_static(
          "read_wkt",
          [](const std::string& wkt) -> geodetic::Point {
            auto point = geodetic::Point();
            boost::geometry::read_wkt(wkt, point);
            return point;
          },
          py::arg("wkt"), R"__doc__(
Parses OGC Well-Known Text (WKT) into a Point

Args:
    wkt (str): the WKT representation of the Point
Return:
    pyinterp.geodetic.Point: The point defined by the WKT representation.
)__doc__")
      .def("__repr__", &geodetic::Point::to_string,
           "Called by the ``repr()`` built-in function to compute the string "
           "representation of a point.")
      .def(
          "__eq__",
          [](const geodetic::Point& self, const geodetic::Point& rhs) -> bool {
            return boost::geometry::equals(self, rhs);
          },
          py::arg("other"),
          "Overrides the default behavior of the ``==`` operator.")
      .def(
          "__ne__",
          [](const geodetic::Point& self, const geodetic::Point& rhs) -> bool {
            return !boost::geometry::equals(self, rhs);
          },
          py::arg("other"),
          "Overrides the default behavior of the ``!=`` operator.")
      .def(py::pickle(
          [](const geodetic::Point& self) { return self.getstate(); },
          [](const py::tuple& state) {
            return geodetic::Point::setstate(state);
          }));
}

static void init_geodetic_box(py::module& m) {
  py::class_<geodetic::Box>(m, "Box", R"__doc__(
    Defines a box made of two describing points.
)__doc__")
      .def(py::init<>())
      .def(py::init<geodetic::Point, geodetic::Point>(), py::arg("min_corner"),
           py::arg("max_corner"),
           R"__doc__(
Constructor taking the minimum corner point and the maximum corner point.

Args:
    min_corner (pyinterp.core.geodetic.Point2D): the minimum corner point
        (lower left) of the box
    max_corner (pyinterp.core.geodetic.Point2D): the maximum corner point
        (upper right) of the box
)__doc__")
      .def_property(
          "min_corner",
          [](const geodetic::Box& self) { return self.min_corner(); },
          [](geodetic::Box& self, const geodetic::Point& point) -> void {
            self.min_corner() = point;
          },
          "The minimal corner (lower left) of the box")
      .def_property(
          "max_corner",
          [](const geodetic::Box& self) { return self.max_corner(); },
          [](geodetic::Box& self, const geodetic::Point& point) -> void {
            self.max_corner() = point;
          },
          "The maximal corner (upper right) of the box")
      .def_static("whole_earth", &geodetic::Box::whole_earth,
                  "Returns the box covering the whole earth.")
      .def(
          "covered_by",
          [](const geodetic::Box& self, const geodetic::Point& point) -> bool {
            return self.covered_by(point);
          },
          py::arg("point"), R"__doc__(
Test if the given point is inside or on border of this box

Args:
    point (pyinterp.geodectic.Point2D): point to test
Return:
    bool: True if the given point is inside or on border of this
    box
)__doc__")
      .def(
          "covered_by",
          [](const geodetic::Box& self,
             const Eigen::Ref<const Eigen::VectorXd>& lon,
             const Eigen::Ref<const Eigen::VectorXd>& lat,
             const size_t num_threads) -> py::array_t<int8_t> {
            return self.covered_by(lon, lat, num_threads);
          },
          py::arg("lon"), py::arg("lat"), py::arg("num_theads") = 1, R"__doc__(
Test if the coordinates of the points provided are located inside or at the
edge of this box.

Args:
    lon (numpy.ndarray): Longitudes coordinates in degrees to check
    lat (numpy.ndarray): Latitude coordinates in degrees to check
    num_threads (int, optional): The number of threads to use for the
        computation. If 0 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging.
        Default to 1.
Return:
    numpy.ndarray: a vector containing a flag equal to 1 if the coordinate
    is located in the box or at the edge otherwise 0.
)__doc__")
      .def("area", &geodetic::Box::area, py::arg("wgs") = py::none(),
           R"__doc__(
Calculates the area.

Args:
    (pyinterp.core.geodetic.System, optional): WGS system used for the
        calculation, default to WGS84

Return:
    float: The calculated area
)__doc__")
      .def(
          "wkt",
          [](const geodetic::Box& self) -> std::string {
            auto ss = std::stringstream();
            ss << boost::geometry::wkt(self);
            return ss.str();
          },
          R"__doc__(
Gets the OGC Well-Known Text (WKT) representation of this instance

Return:
    str: the WTK representation
)__doc__")
      .def_static(
          "read_wkt",
          [](const std::string& wkt) -> geodetic::Box {
            auto box = geodetic::Box();
            boost::geometry::read_wkt(wkt, box);
            return box;
          },
          py::arg("wkt"), R"__doc__(
Parses OGC Well-Known Text (WKT) into a box

Args:
    wkt (str): the WKT representation of the box
Return:
    pyinterp.geodetic.Box: The box defined by the WKT
    representation.
)__doc__")
      .def(
          "__eq__",
          [](const geodetic::Box& self, const geodetic::Box& rhs) -> bool {
            return boost::geometry::equals(self, rhs);
          },
          py::arg("other"),
          "Overrides the default behavior of the ``==`` operator.")
      .def(
          "__ne__",
          [](const geodetic::Box& self, const geodetic::Box& rhs) -> bool {
            return !boost::geometry::equals(self, rhs);
          },
          py::arg("other"),
          "Overrides the default behavior of the ``!=`` operator.")
      .def("__repr__", &geodetic::Box::to_string,
           "Called by the ``repr()`` built-in function to compute the string "
           "representation of a box.")
      .def(py::pickle([](const geodetic::Box& self) { return self.getstate(); },
                      [](const py::tuple& state) {
                        return geodetic::Box::setstate(state);
                      }));
}

static void init_geodetic_polygon(py::module& m) {
  py::class_<geodetic::Polygon>(
      m, "Polygon",
      "The polygon contains an outer ring and zero or more inner rings")
      .def(py::init([](const py::list& outer,
                       std::optional<const py::list>& inners) {
             return geodetic::Polygon(outer, inners.value_or(py::list()));
           }),
           py::arg("outer"), py::arg("inners") = py::none(), R"(
Constructor filling the polygon

Args:
  outer (list): outer ring
  inners (list, optional): list of inner rings
Raises:
  ValueError: if outer is not a list of pyinterp.geodetic.Point
  ValueError: if inners is not a list of list of pyinterp.geodetic.Point
)")
      .def(
          "__eq__",
          [](const geodetic::Polygon& self, const geodetic::Polygon& rhs)
              -> bool { return boost::geometry::equals(self, rhs); },
          py::arg("other"),
          "Overrides the default behavior of the ``==`` operator.")
      .def(
          "__ne__",
          [](const geodetic::Polygon& self, const geodetic::Polygon& rhs)
              -> bool { return !boost::geometry::equals(self, rhs); },
          py::arg("other"),
          "Overrides the default behavior of the ``!=`` operator.")
      .def("__repr__", &geodetic::Polygon::to_string,
           "Called by the ``repr()`` built-in function to compute the string "
           "representation of a point.")
      .def(
          "envelope",
          [](const geodetic::Polygon& self) -> geodetic::Box {
            auto box = geodetic::Box();
            boost::geometry::envelope(self, box);
            return box;
          },
          R"__doc__(
Calculates the envelope of this polygon.

Return:
  pyinterp.geodetic.Box: The envelope of this instance
)__doc__")
      .def(
          "covered_by",
          [](const geodetic::Polygon& self, const geodetic::Point& point) -> bool {
            return self.covered_by(point);
          },
          py::arg("point"), R"__doc__(
Test if the given point is inside or on border of this polygon

Args:
    point (pyinterp.geodectic.Point2D): point to test
Return:
    bool: True if the given point is inside or on border of this box
)__doc__")
      .def(
          "covered_by",
          [](const geodetic::Polygon& self,
             const Eigen::Ref<const Eigen::VectorXd>& lon,
             const Eigen::Ref<const Eigen::VectorXd>& lat,
             const size_t num_threads) -> py::array_t<int8_t> {
            return self.covered_by(lon, lat, num_threads);
          },
          py::arg("lon"), py::arg("lat"), py::arg("num_theads") = 1, R"__doc__(
Test if the coordinates of the points provided are located inside or at the
edge of this polygon.

Args:
    lon (numpy.ndarray): Longitudes coordinates in degrees to check
    lat (numpy.ndarray): Latitude coordinates in degrees to check
    num_threads (int, optional): The number of threads to use for the
        computation. If 0 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging.
        Default to 1.
Return:
    numpy.ndarray: a vector containing a flag equal to 1 if the coordinate
    is located in the box or at the edge otherwise 0.
)__doc__")
      .def("area", &geodetic::Polygon::area, py::arg("wgs") = py::none(),
           R"__doc__(
Calculates the area.

Args:
    (pyinterp.core.geodetic.System, optional): WGS system used for the
        calculation, default to WGS84

Return:
    float: The calculated area
)__doc__")
      .def(
          "wkt",
          [](const geodetic::Polygon& self) -> std::string {
            auto ss = std::stringstream();
            ss << boost::geometry::wkt(self);
            return ss.str();
          },
          R"__doc__(
Gets the OGC Well-Known Text (WKT) representation of this instance

Return:
    str: the WTK representation
)__doc__")
      .def_static(
          "read_wkt",
          [](const std::string& wkt) -> geodetic::Polygon {
            auto polygon = geodetic::Polygon();
            boost::geometry::read_wkt(wkt, polygon);
            return polygon;
          },
          py::arg("wkt"), R"__doc__(
Parses OGC Well-Known Text (WKT) into a polygon

Args:
    wkt (str): the WKT representation of the polygon
Return:
    pyinterp.geodetic.Box: The polygon defined by the WKT
    representation.
)__doc__")
      .def(py::pickle(
          [](const geodetic::Polygon& self) { return self.getstate(); },
          [](const py::tuple& state) {
            return geodetic::Polygon::setstate(state);
          }));
}

void init_geodetic(py::module& m) {
  auto _system = py::class_<pyinterp::detail::geodetic::System>(
      m, "_System", "C++ implementation of the WGS system.");

  py::class_<geodetic::System, pyinterp::detail::geodetic::System>(
      m, "System", "World Geodetic System (WGS).")
      .def(py::init<>())
      .def(py::init<double, double>(), py::arg("semi_major_axis"),
           py::arg("flattening"), R"__doc__(
Args:
    semi_major_axis (float): Semi-major axis of ellipsoid, in meters
    flattening (float): Flattening of ellipsoid
.. note::
    The default constructor initializes a WGS-84 ellipsoid.
)__doc__")
      .def_property_readonly(
          "semi_major_axis", &geodetic::System::semi_major_axis,
          "Semi-major axis of ellipsoid, in meters (:math:`a`)")
      .def_property_readonly(
          "flattening", &geodetic::System::flattening,
          "Flattening of ellipsoid (:math:`f=\\frac{a-b}{a}`)")
      .def("semi_minor_axis", &geodetic::System::semi_minor_axis, R"__doc__(
Gets the semiminor axis

Return:
    float: :math:`b=a(1-f)`
)__doc__")
      .def("first_eccentricity_squared",
           &geodetic::System::first_eccentricity_squared, R"__doc__(
Gets the first eccentricity squared

Return:
    float: :math:`e^2=\frac{a^2-b^2}{a^2}`
)__doc__")
      .def("second_eccentricity_squared",
           &geodetic::System::second_eccentricity_squared, R"__doc__(
Gets the second eccentricity squared

Return:
    float: :math:`e^2=\frac{a^2-b^2}{b^2}`
)__doc__")
      .def("equatorial_circumference",
           &geodetic::System::equatorial_circumference,
           py::arg("semi_major_axis") = true, R"__doc__(
Gets the equatorial circumference

Args:
    semi_major_axis (bool, optional): True to get the equatorial circumference
        for the semi-majors axis, False for the semi-minor axis. Defaults to
        ``true``.
Return:
    float: :math:`2\pi \times a` if semi_major_axis is true otherwise
    :math:`2\pi \times b`
)__doc__")
      .def("polar_radius_of_curvature",
           &geodetic::System::polar_radius_of_curvature,
           R"__doc__(
Gets the polar radius of curvature

Return:
    float: :math:`\frac{a^2}{b}`
)__doc__")
      .def("equatorial_radius_of_curvature",
           &geodetic::System::equatorial_radius_of_curvature,
           R"__doc__(
Gets the equatorial radius of curvature for a meridian

Return:
    float: :math:`\frac{b^2}{a}`
)__doc__")
      .def("axis_ratio", &geodetic::System::axis_ratio, R"__doc__(
Gets the axis ratio

Return:
    float: :math:`\frac{b}{a}`
)__doc__")
      .def("linear_eccentricity", &geodetic::System::linear_eccentricity,
           R"__doc__(
Gets the linear eccentricity

Return:
    float :math:`E=\sqrt{{a^2}-{b^2}}`
)__doc__")
      .def("mean_radius", &geodetic::System::mean_radius, R"__doc__(
Gets the mean radius

Return:
    float: :math:`R_1=\frac{2a+b}{3}`
)__doc__")
      .def("authalic_radius", &geodetic::System::authalic_radius, R"__doc__(
Gets the authalic radius

Return:
    float: :math:`R_2=\sqrt{\frac{a^2+\frac{ab^2}{E}ln(\frac{a + E}{b})}{2}}`
)__doc__")
      .def("volumetric_radius", &geodetic::System::volumetric_radius,
           R"__doc__(
Gets the volumetric radius

Return:
    float: :math:`R_3=\sqrt[3]{a^{2}b}`
)__doc__")
      .def("__eq__", &geodetic::System::operator==, py::arg("other"),
           "Overrides the default behavior of the ``==`` operator.")
      .def("__ne__", &geodetic::System::operator!=, py::arg("other"),
           "Overrides the default behavior of the ``!=`` operator.")
      .def(py::pickle(
          [](const geodetic::System& self) { return self.getstate(); },
          [](const py::tuple& state) {
            return geodetic::System::setstate(state);
          }));

  py::class_<geodetic::Coordinates>(m, "Coordinates",
                                    "World Geodetic Coordinates System.")
      .def(py::init<std::optional<geodetic::System>>(), py::arg("system"),
           R"__doc__(
Default constructor

Args:
    system (pyinterp.core.geodetic.System, optional): WGS System. If this
        option is not defined, the instance manages a WGS84 ellipsoid.
)__doc__")
      .def("ecef_to_lla", &geodetic::Coordinates::ecef_to_lla<double>,
           py::arg("x"), py::arg("y"), py::arg("z"), py::arg("num_threads") = 0,
           R"__doc__(
Converts Cartesian coordinates to Geographic latitude, longitude, and
altitude. Cartesian coordinates should be in meters. The returned
latitude and longitude are in degrees, and the altitude will be in
meters.

Args:
    x (numpy.ndarray): X-coordinates in meters
    y (numpy.ndarray): Y-coordinates in meters
    z (numpy.ndarray): Z-coordinates in meters
    num_threads (int, optional): The number of threads to use for the
        computation. If 0 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging.
        Defaults to ``0``.
Return:
    tuple: Longitudes, latitudes and altitudes in the coordinate system
    defined by this instance.

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
    lon (numpy.ndarray): Longitudes in degrees
    lat (numpy.ndarray): Latitudes in degrees
    alt (numpy.ndarray): Altitudes in meters
    num_threads (int, optional): The number of threads to use for the
        computation. If 0 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging.
        Defaults to ``0``.
Return:
    tuple: X, Y and Z ECEF coordinates in meters.
)__doc__")
      .def("transform", &geodetic::Coordinates::transform<double>,
           py::arg("target"), py::arg("lon"), py::arg("lat"), py::arg("alt"),
           py::arg("num_threads") = 0,
           R"__doc__(
Transforms the positions, provided in degrees and meters, from one WGS
system to another.

Args:
    target (pyinterp.core.geodetic.System): WGS target
    lon (numpy.ndarray): Longitudes in degrees
    lat (numpy.ndarray): Latitudes in degrees
    alt (numpy.ndarray): Altitudes in meters
    num_threads (int, optional): The number of threads to use for the
        computation. If 0 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging.
        Defaults to ``0``.
Return:
    tuple: Longitudes, latitudes and altitudes in the new coordinate system.
)__doc__")
      .def(py::pickle(
          [](const geodetic::Coordinates& self) { return self.getstate(); },
          [](const py::tuple& state) {
            return geodetic::Coordinates::setstate(state);
          }));

  init_geodetic_point(m);
  init_geodetic_box(m);
  init_geodetic_polygon(m);
}
