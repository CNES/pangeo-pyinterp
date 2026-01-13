// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include "pyinterp/pybind/geometry/geographic/coordinate.hpp"

namespace nb = nanobind;

namespace pyinterp::geometry::geographic::pybind {

constexpr const char *const kCoordinatesDoc =
    R"(Create a coordinate transformation system.

Args:
    spheroid: Optional spheroid to use. Defaults to WGS-84.
)";

constexpr const char *const kEcefToLlaDoc =
    R"(Convert Cartesian ECEF coordinates to geographic coordinates.

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

    Vermeille, H. (2002). Direct transformation from geocentric to geodetic
    coordinates. Journal of Geodesy, 76(8), 451-454.
    DOI: https://doi.org/10.1007/s00190-002-0273-6
)";

constexpr const char *const kLlaToEcefDoc =
    R"(Convert geographic coordinates to Cartesian ECEF coordinates.

Args:
    lon: Longitudes in degrees.
    lat: Latitudes in degrees.
    alt: Altitudes in meters.
    num_threads: The number of threads to use for the computation. If 0 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Defaults to ``0``.
Returns:
    tuple: X, Y and Z ECEF coordinates in meters.
)";

constexpr const char *const kTransformDoc =
    R"(Transform positions from one coordinate system to another.

Args:
    target: Target coordinate system.
    lon: Longitudes in degrees.
    lat: Latitudes in degrees.
    alt: Altitudes in meters.
    num_threads: The number of threads to use for the computation. If 0 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Defaults to ``0``.
Returns:
    Longitudes, latitudes and altitudes in the new coordinate system.
)";

auto init_coordinates(nanobind::module_ &m) -> void {
  auto coordinates = nb::class_<Coordinates>(m, "Coordinates", kCoordinatesDoc);

  coordinates
      .def(nb::init<std::optional<geometry::geographic::Spheroid>>(),
           nb::arg("spheroid") = std::nullopt,
           "Initialize the coordinates system with optional spheroid.",
           nb::call_guard<nb::gil_scoped_release>())

      .def_prop_ro(
          "spheroid",
          [](const Coordinates &self) -> Spheroid { return self.spheroid(); },
          "WGS used to transform the coordinates.",
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "ecef_to_lla",
          [](const Coordinates &self, const Eigen::Ref<const Vector<double>> &x,
             const Eigen::Ref<const Vector<double>> &y,
             const Eigen::Ref<const Vector<double>> &z, const int num_threads)
              -> std::tuple<Vector<double>, Vector<double>, Vector<double>> {
            return self.ecef_to_lla<double>(x, y, z, num_threads);
          },
          nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("num_threads") = 0,
          kEcefToLlaDoc, nb::call_guard<nb::gil_scoped_release>())

      .def(
          "lla_to_ecef",
          [](const Coordinates &self,
             const Eigen::Ref<const Vector<double>> &lon,
             const Eigen::Ref<const Vector<double>> &lat,
             const Eigen::Ref<const Vector<double>> &alt, const int num_threads)
              -> std::tuple<Vector<double>, Vector<double>, Vector<double>> {
            return self.lla_to_ecef<double>(lon, lat, alt, num_threads);
          },
          nb::arg("lon"), nb::arg("lat"), nb::arg("alt"),
          nb::arg("num_threads") = 0, kLlaToEcefDoc,
          nb::call_guard<nb::gil_scoped_release>())

      .def(
          "transform",
          [](const Coordinates &self, const Coordinates &target,
             const Eigen::Ref<const Vector<double>> &lon,
             const Eigen::Ref<const Vector<double>> &lat,
             const Eigen::Ref<const Vector<double>> &alt, const int num_threads)
              -> std::tuple<Vector<double>, Vector<double>, Vector<double>> {
            return self.transform<double>(target, lon, lat, alt, num_threads);
          },
          nb::arg("target"), nb::arg("lon"), nb::arg("lat"), nb::arg("alt"),
          nb::arg("num_threads") = 0, kTransformDoc,
          nb::call_guard<nb::gil_scoped_release>())

      .def("__getstate__", &Coordinates::getstate,
           "Get the state for pickling.",
           nb::call_guard<nb::gil_scoped_release>())
      .def(
          "__setstate__",
          [](Coordinates &self, nb::tuple &state) -> void {
            new (&self) Coordinates(Coordinates::setstate(state));
          },
          nb::arg("state"), "Set the state for unpickling.",
          nb::call_guard<nb::gil_scoped_release>());
}

}  // namespace pyinterp::geometry::geographic::pybind
