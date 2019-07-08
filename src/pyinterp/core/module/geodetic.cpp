// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geodetic/coordinates.hpp"
#include "pyinterp/geodetic/system.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace geodetic = pyinterp::geodetic;
namespace py = pybind11;

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

Return
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
      .def("__eq__", &geodetic::System::operator==)
      .def("__ne__", &geodetic::System::operator!=)
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
Returns:
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
}