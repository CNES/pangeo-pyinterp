// Copyright (c) 2021 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geohash.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace geohash = pyinterp::geohash;

void init_geohash_class(py::module& m) {
  py::class_<geohash::GeoHash>(m, "GeoHash", R"__doc__(
Handle GeoHash encoded in base 32.

Geohashing is a geocoding method used to encode geographic coordinates
(latitude and longitude) into a short string of digits and letters
delineating an area on a map, which is called a cell, with varying
resolutions. The more characters in the string, the more precise the
location. The table below gives the correspondence between the number of
characters, the size of the boxes of the grid at the equator and the total
number of boxes.

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
storing a location in a database, or for indexing a location in a
database.
)__doc__")
      .def(py::init<double, double, uint32_t>(), py::arg("longitude"),
           py::arg("latitude"), py::arg("precision") = 12, R"__doc__(
GeoHash from longitude, latitude with number of characters.

Args:
    longitude (float): Longitude of the point.
    latitude (float): Latitude of the point.
    precision (int): Number of characters in the geohash. Default is 12.

Throws:
    ValueError: If the precision is not in the range [1, 12].
)__doc__")
      .def_static("from_string", geohash::GeoHash::from_string, py::arg("code"),
                  py::arg("round") = false, R"__doc__(
GeoHash from its string representation.

Args:
    code (str): String representation of the geohash.
    round (bool): If true, the coordinates of the point will be rounded to
        the accuracy defined by the GeoHash.

Throws:
    ValueError: If the code is not a valid geohash.
)__doc__")
      .def("bounding_box", &geohash::GeoHash::bounding_box,
           R"__doc__(
Returns the bounding box of this GeoHash.

Returns:
  pyinterp.geodetic.Box: Bounding box.
)__doc__")
      .def("center", &geohash::GeoHash::center,
           R"__doc__(
Returns the center point of this GeoHash.

Returns:
  pyinterp.geodetic.Point: Bounding box.
)__doc__")
      .def("precision", &geohash::GeoHash::precision,
           R"__doc__(
Returns the precision of this GeoHash.

Returns:
    int: Precision.
)__doc__")
      .def("number_of_bits", &geohash::GeoHash::number_of_bits,
           R"__doc__(
Returns the number of bits of this GeoHash.

Returns:
    int: Number of bits.
)__doc__")
      .def("__str__", &geohash::GeoHash::string_value,
           R"__doc__(
Returns the string representation of this GeoHash.

Returns:
  str: The geohash code.
)__doc__")
      .def("neighbors", &geohash::GeoHash::neighbors,
           R"__doc__(
Returns the eight neighbors of this GeoHash.

Returns:
  list: A list of GeoHash in the order N, NE, E, SE, S, SW, W, NW.
)__doc__")
      .def("area", &geohash::GeoHash::area, py::arg("wgs") = py::none(),
           R"__doc__(
Returns the area covered by this.

Args:
    wgs (optional, pyinterp.geodetic.System): WGS used to calculate the area.
        Defaults to WGS84.
    
Returns:
    float: calculated area in square meters.
)__doc__")
      .def("reduce", &geohash::GeoHash::reduce, 
R"__doc__(
Returns the arguments to rebuild this instance.

Returns:
    tuple: (longitude, latitude, precision)
)__doc__")
      .def_static("grid_properties", &geohash::GeoHash::grid_properties,
                  py::arg("box"), py::arg("precision") = 1,
                  R"__doc__(
Gets the property of the grid covering the given box.

Args:
  box (pyinterp.geodetic.Box): Bounding box.
  precision (int, optional): Required accuracy. Default is 1.
    
Returns:
    tuple: A tuple of three elements containing: the GeoHash of the minimum
    corner point, the number of squares in longitudes and latitudes.
)__doc__")
      .def_static("error_with_precision",
                  &geohash::GeoHash::error_with_precision,
                  py::arg("precision") = 1, R"__doc__(
Returns the accuracy in longitude/latitude and degrees for the given precision.

Args:
    precision (int, optional): Number of bits used to encode the geohash code.
        Default is 1.
Returns:
    tuple: Accuracy in longitude/latitude and degrees.
Raises:
    ValueError: If the given precision is not within [1, 12].
)__doc__");
}
