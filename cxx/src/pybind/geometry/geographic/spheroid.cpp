// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geometry/geographic/spheroid.hpp"

#include <nanobind/nanobind.h>

#include "pyinterp/pybind/geometry/geographic.hpp"

namespace nb = nanobind;

namespace pyinterp::geometry::geographic::pybind {

constexpr const char *const kSpheroidDoc =
    R"(Define a reference ellipsoid for geodetic calculations.

Args:
    semi_major_axis: Semi-major axis of ellipsoid, in meters.
    flattening: Flattening of ellipsoid.

Note:
    The default constructor initializes a WGS-84 ellipsoid.
)";

constexpr const char *const kSemiMinorAxisDoc =
    R"(Get the semi-minor axis.

Returns:
    :math:`b=a(1-f)`
)";

constexpr const char *const kFirstEccentricitySquaredDoc =
    R"(Get the first eccentricity squared.

Returns:
    :math:`e^2=\frac{a^2-b^2}{a^2}`
)";

constexpr const char *const kSecondEccentricitySquaredDoc =
    R"(Get the second eccentricity squared.

Returns:
    :math:`e^2=\frac{a^2-b^2}{b^2}`
)";

constexpr const char *const kEquatorialCircumferenceDoc =
    R"(Get the equatorial circumference.

Args:
    semi_major_axis: True to get the equatorial circumference for the
        semi-majors axis, False for the semi-minor axis. Defaults to ``true``.

Returns:
    :math:`2\pi \times a` if semi_major_axis is true otherwise
    :math:`2\pi \times b`.
)";

constexpr const char *const kPolarRadiusOfCurvatureDoc =
    R"(Get the polar radius of curvature.

Returns:
    :math:`\frac{a^2}{b}`
)";

constexpr const char *const kEquatorialRadiusOfCurvatureDoc =
    R"(Get the equatorial radius of curvature for a meridian.

Returns:
    :math:`\frac{b^2}{a}`
)";

constexpr const char *const kAxisRatioDoc =
    R"(Get the axis ratio.

Returns:
    :math:`\frac{b}{a}`
)";

constexpr const char *const kLinearEccentricityDoc =
    R"(Get the linear eccentricity.

Returns:
    :math:`E=\sqrt{{a^2}-{b^2}}`
)";

constexpr const char *const kMeanRadiusDoc =
    R"(Get the mean radius.

Returns:
    :math:`R_1=\frac{2a+b}{3}`
)";

constexpr const char *const kGeocentricRadiusDoc =
    R"(Get the geocentric radius at the given latitude :math:`\phi`.

Args:
    lat: The latitude, in degrees.

Returns:
    .. math::

        R(\phi)=\sqrt{\frac{{(a^{2}\cos(\phi))}^{2} + \\
        (b^{2}\sin(\phi))^{2}}{(a\cos(\phi))^{2} + (b\cos(\phi))^{2}}}
)";

constexpr const char *const kAuthalicRadiusDoc =
    R"(Get the authalic radius.

Returns:
    :math:`R_2=\sqrt{\frac{a^2+\frac{ab^2}{E}ln(\frac{a + E}{b})}{2}}`
)";

constexpr const char *const kVolumetricRadiusDoc =
    R"(Get the volumetric radius.

Returns:
    :math:`R_3=\sqrt[3]{a^{2}b}`
)";

auto init_spheroid(nanobind::module_ &m) -> void {
  auto spheroid =
      nb::class_<geometry::geographic::Spheroid>(m, "Spheroid", kSpheroidDoc);

  spheroid
      .def(nb::init<>(), "Initialize the spheroid with WGS-84 ellipsoid.",
           nb::call_guard<nb::gil_scoped_release>())
      .def(nb::init<double, double>(), nb::arg("semi_major_axis"),
           nb::arg("flattening"),
           "Initialize the spheroid with custom parameters.",
           nb::call_guard<nb::gil_scoped_release>())

      .def_prop_ro(
          "semi_major_axis",
          [](const geometry::geographic::Spheroid &self) -> double {
            return self.semi_major_axis();
          },
          "Semi-major axis of ellipsoid, in meters (:math:`a`).",
          nb::call_guard<nb::gil_scoped_release>())

      .def_prop_ro(
          "flattening",
          [](const geometry::geographic::Spheroid &self) -> double {
            return self.flattening();
          },
          "Flattening of ellipsoid (:math:`f=\\frac{a-b}{a}`).",
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "semi_minor_axis",
          [](const geometry::geographic::Spheroid &self) -> double {
            return self.semi_minor_axis();
          },
          kSemiMinorAxisDoc, nb::call_guard<nb::gil_scoped_release>())

      .def(
          "first_eccentricity_squared",
          [](const geometry::geographic::Spheroid &self) -> double {
            return self.first_eccentricity_squared();
          },
          kFirstEccentricitySquaredDoc,
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "second_eccentricity_squared",
          [](const geometry::geographic::Spheroid &self) -> double {
            return self.second_eccentricity_squared();
          },
          kSecondEccentricitySquaredDoc,
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "equatorial_circumference",
          [](const geometry::geographic::Spheroid &self,
             const bool semi_major_axis) -> double {
            return self.equatorial_circumference(semi_major_axis);
          },
          nb::arg("semi_major_axis") = true, kEquatorialCircumferenceDoc,
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "polar_radius_of_curvature",
          [](const geometry::geographic::Spheroid &self) -> double {
            return self.polar_radius_of_curvature();
          },
          kPolarRadiusOfCurvatureDoc, nb::call_guard<nb::gil_scoped_release>())

      .def(
          "equatorial_radius_of_curvature",
          [](const geometry::geographic::Spheroid &self) -> double {
            return self.equatorial_radius_of_curvature();
          },
          kEquatorialRadiusOfCurvatureDoc,
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "axis_ratio",
          [](const geometry::geographic::Spheroid &self) -> double {
            return self.axis_ratio();
          },
          kAxisRatioDoc, nb::call_guard<nb::gil_scoped_release>())

      .def(
          "linear_eccentricity",
          [](const geometry::geographic::Spheroid &self) -> double {
            return self.linear_eccentricity();
          },
          kLinearEccentricityDoc, nb::call_guard<nb::gil_scoped_release>())

      .def(
          "mean_radius",
          [](const geometry::geographic::Spheroid &self) -> double {
            return self.mean_radius();
          },
          kMeanRadiusDoc, nb::call_guard<nb::gil_scoped_release>())

      .def(
          "geocentric_radius",
          [](const geometry::geographic::Spheroid &self, const double lat)
              -> double { return self.geocentric_radius(lat); },
          nb::arg("lat"), kGeocentricRadiusDoc,
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "authalic_radius",
          [](const geometry::geographic::Spheroid &self) -> double {
            return self.authalic_radius();
          },
          kAuthalicRadiusDoc, nb::call_guard<nb::gil_scoped_release>())

      .def(
          "volumetric_radius",
          [](const geometry::geographic::Spheroid &self) -> double {
            return self.volumetric_radius();
          },
          kVolumetricRadiusDoc, nb::call_guard<nb::gil_scoped_release>())

      .def(
          "__eq__",
          [](const geometry::geographic::Spheroid &self,
             const geometry::geographic::Spheroid &other) -> bool {
            return self == other;
          },
          nb::arg("other"),
          "Override the default behavior of the ``==`` operator.",
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "__ne__",
          [](const geometry::geographic::Spheroid &self,
             const geometry::geographic::Spheroid &other) -> bool {
            return self != other;
          },
          nb::arg("other"),
          "Override the default behavior of the ``!=`` operator.",
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "__getstate__",
          [](const geometry::geographic::Spheroid &self) -> nb::tuple {
            return nb::make_tuple(self.semi_major_axis(), self.flattening());
          },
          "Get the state for pickling.")
      .def(
          "__setstate__",
          [](geometry::geographic::Spheroid &self, nb::tuple &state) -> void {
            if (state.size() != 2) {
              throw std::invalid_argument("Invalid state");
            }
            auto semi_major_axis = nb::cast<double>(state[0]);
            auto flattening = nb::cast<double>(state[1]);
            new (&self)
                geometry::geographic::Spheroid(semi_major_axis, flattening);
          },
          nb::arg("state"), "Set the state for unpickling.");
}

}  // namespace pyinterp::geometry::geographic::pybind
