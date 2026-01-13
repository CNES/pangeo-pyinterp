// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "pyinterp/geohash.hpp"

namespace nb = nanobind;

using nb::literals::operator""_a;

namespace pyinterp::geohash::pybind {

constexpr auto kGeohashClassDoc = R"doc(
Geohash: Base32 string encoding a rectangular geographic area.

Geohashing is a geocoding method used to encode geographic coordinates
(latitude and longitude) into a short string of digits and letters
delineating an area on a map, which is called a cell, with varying
resolutions.

The more characters in the string, the more precise the location. The table
below gives the correspondence between the number of characters, the size of
the boxes of the grid at the equator and the total number of boxes:

    =========  ===============  ==========
    precision  lng/lat (km)     samples
    =========  ===============  ==========
    1          4950/4950        32
    2          618.75/1237.50   1024
    3          154.69/154.69    32768
    4          19.34/38.67      1048576
    5          4.83/4.83        33554432
    6          0.60/1.21        1073741824
    =========  ===============  ==========

Geohashes use Base-32 alphabet encoding (characters can be 0 to 9 and A to
Z, excl A, I, L and O).

The geohash is a compact way of representing a location, and is useful for
storing a location in a database, or for indexing a location in a database.
)doc";

constexpr auto kIinitDoc = R"doc(
Construct a GeoHash from longitude, latitude with number of characters.

Args:
    lon: Longitude of the point.
    lat: Latitude of the point.
    precision: Number of characters in the geohash (must be <= 12).

Raises:
    ValueError: If precision > 12.
)doc";

constexpr auto kFromStringDoc = R"doc(
Construct a GeoHash from its string representation.

Args:
    code: String representation of the geohash.
    round: If True, the coordinates of the point will be rounded to the
        accuracy defined by the GeoHash. Defaults to False.

Returns:
    GeoHash: The constructed geohash instance.

Raises:
    ValueError: If the geohash is not valid or precision > 12.
)doc";

constexpr auto kBoundingBoxDoc = R"doc(
Returns the bounding box of the geohash.

Returns:
    Box: Geodetic box representing the geohash bounds.
)doc";

constexpr auto kCenterDoc = R"doc(
Returns the center point of this geohash.

Returns:
    Point: Geodetic point at the center of the geohash.
)doc";

constexpr auto kStringValue = R"doc(
String representation of the geohash.

Returns:
    str: The geohash code.
)doc";

constexpr auto kPPrecisionDoc = R"doc(
Number of characters in the geohash.

Returns:
    int: The precision.
)doc";

constexpr auto kNumberOfBitsDoc = R"doc(
Number of bits used to represent the geohash.

Returns:
    int: Number of bits (precision * 5).
)doc";

constexpr auto kIntegerValueDoc = R"doc(
Returns the value of the integer64 stored in the geohash.

Args:
    round: If True, returns rounded corner; otherwise centroid.
        Defaults to False.

Returns:
    int: 64-bit integer representation.
)doc";

constexpr auto kNeighborsDoc = R"doc(
Returns the eight neighbors of this geohash.

Returns:
    list[GeoHash]: Vector of GeoHash in the order N, NE, E, SE, S, SW, W, NW.
)doc";

constexpr auto kAreaDoc = R"doc(
Returns the area covered by this geohash.

Args:
    wgs: Optional spheroid for area calculation. Defaults to None.

Returns:
    float: The area of the geohash in square meters.
)doc";

constexpr auto kGridPropertiesDoc = R"doc(
Gets the property of the grid covering the given box.

Args:
    box: Geodetic box to cover.
    precision: Number of characters in the geohash.

Returns:
    tuple[GeoHash, int, int]: Tuple containing the GeoHash of the minimum
        corner point, the number of cells in longitudes, and the number of
        cells in latitudes.
)doc";

constexpr auto kErrorWithPrecisionDoc = R"doc(
Returns the precision in longitude/latitude degrees for the given precision.

Args:
    precision: Number of characters in the geohash.

Returns:
    tuple[float, float]: Tuple of (longitude_error, latitude_error) in degrees.

Raises:
    ValueError: If precision > 12.
)doc";

auto init_class(nb::module_& m) -> void {
  nb::class_<GeoHash>(m, "GeoHash", kGeohashClassDoc)
      .def(nb::init<double, double, uint32_t>(), "lon"_a, "lat"_a,
           "precision"_a, kIinitDoc)

      .def_static("from_string", &GeoHash::from_string, "code"_a,
                  "round"_a = false, kFromStringDoc)

      .def("bounding_box", &GeoHash::bounding_box, kBoundingBoxDoc)
      .def("center", &GeoHash::center, kCenterDoc)

      .def_prop_ro("code", &GeoHash::string_value, kStringValue)

      .def_prop_ro("precision", &GeoHash::precision, kPPrecisionDoc)

      .def_prop_ro("number_of_bits", &GeoHash::number_of_bits, kNumberOfBitsDoc)

      .def("integer_value", &GeoHash::integer_value, "round"_a = false,
           kIntegerValueDoc)

      .def("neighbors", &GeoHash::neighbors, kNeighborsDoc)

      .def("area", &GeoHash::area, "spheroid"_a = nb::none(), kAreaDoc)

      .def_static("grid_properties", &GeoHash::grid_properties, "box"_a,
                  "precision"_a, kGridPropertiesDoc)

      .def_static("error_with_precision", &GeoHash::error_with_precision,
                  "precision"_a, kErrorWithPrecisionDoc)

      .def("__repr__",
           [](const GeoHash& self) -> std::string {
             return "<GeoHash code='" + self.string_value() + "'>";
           })

      .def("__str__",
           [](const GeoHash& self) -> std::string { return std::string(self); })

      .def("__hash__",
           [](const GeoHash& self) -> size_t {
             return std::hash<std::string>{}(self.string_value());
           })

      .def("__eq__",
           [](const GeoHash& self, const GeoHash& other) -> bool {
             return self.string_value() == other.string_value();
           })

      .def("__getstate__", &GeoHash::getstate)

      .def("__setstate__",
           [](GeoHash* self,
              const std::tuple<double, double, uint32_t>& state) -> void {
             new (self) GeoHash(std::get<0>(state), std::get<1>(state),
                                std::get<2>(state));
           });
}

}  // namespace pyinterp::geohash::pybind
