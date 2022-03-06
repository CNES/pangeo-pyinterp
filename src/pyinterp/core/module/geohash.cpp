// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geohash.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace geohash = pyinterp::geohash;

void init_geohash_class(py::module &m) {
  py::class_<geohash::GeoHash>(m, "GeoHash", R"__doc__(
GeoHash(self, latitude: float, longitude: float, precision: int = 12)

Handle GeoHash encoded in base 32.

Args:
    longitude: Longitude of the point.
    latitude: Latitude of the point.
    precision: Number of characters in the geohash. Default is 12.

Throws:
    ValueError: If the precision is not in the range [1, 12].
)__doc__")
      .def(py::init<double, double, uint32_t>(), py::arg("longitude"),
           py::arg("latitude"), py::arg("precision") = 12)
      .def_static("from_string", geohash::GeoHash::from_string, py::arg("code"),
                  py::arg("round") = false, R"__doc__(
GeoHash from its string representation.

Args:
    code: String representation of the geohash.
    round: If true, the coordinates of the point will be rounded to the accuracy
        defined by the GeoHash.

Throws:
    ValueError: If the code is not a valid geohash.
)__doc__")
      .def("bounding_box", &geohash::GeoHash::bounding_box,
           R"__doc__(
Returns the bounding box of this GeoHash.

Returns:
    Bounding box.
)__doc__")
      .def("center", &geohash::GeoHash::center,
           R"__doc__(
Returns the center point of this GeoHash.

Returns:
    Bounding box.
)__doc__")
      .def("precision", &geohash::GeoHash::precision,
           R"__doc__(
Returns the precision of this GeoHash.

Returns:
    Precision.
)__doc__")
      .def("number_of_bits", &geohash::GeoHash::number_of_bits,
           R"__doc__(
Returns the number of bits of this GeoHash.

Returns:
    Number of bits.
)__doc__")
      .def("__str__", &geohash::GeoHash::string_value,
           R"__doc__(
Returns the string representation of this GeoHash.

Returns:
    The geohash code.
)__doc__")
      .def("neighbors", &geohash::GeoHash::neighbors,
           R"__doc__(
Returns the eight neighbors of this GeoHash.

Returns:
    A list of GeoHash in the order N, NE, E, SE, S, SW, W, NW.
)__doc__")
      .def("area", &geohash::GeoHash::area, py::arg("wgs") = std::nullopt,
           R"__doc__(
Returns the area covered by this.

Args:
    wgs: WGS used to calculate the area. Defaults to WGS84.

Returns:
    Calculated area in square meters.
)__doc__")
      .def("reduce", &geohash::GeoHash::reduce,
           R"__doc__(
Returns the arguments to rebuild this instance.

Returns:
    Longitude and latitude precisions.
)__doc__")
      .def_static("grid_properties", &geohash::GeoHash::grid_properties,
                  py::arg("box"), py::arg("precision") = 1,
                  R"__doc__(
Gets the property of the grid covering the given box.

Args:
  box: Bounding box.
  precision: Required accuracy. Default is 1.

Returns:
    A tuple of three elements containing: the GeoHash of the minimum corner
    point, the number of squares in longitudes and latitudes.
)__doc__")
      .def_static("error_with_precision",
                  &geohash::GeoHash::error_with_precision,
                  py::arg("precision") = 1, R"__doc__(
Returns the accuracy in longitude/latitude and degrees for the given precision.

Args:
    precision: Number of bits used to encode the geohash code. Default is 1.
Returns:
    Accuracy in longitude/latitude and degrees.
Raises:
    ValueError: If the given precision is not within [1, 12].
)__doc__");
}
