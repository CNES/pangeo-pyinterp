// Copyright (c) 2021 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geohash/int64.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>

#include "pyinterp/geodetic/box.hpp"
#include "pyinterp/geodetic/point.hpp"

namespace py = pybind11;
namespace geohash = pyinterp::geohash;
namespace geodetic = pyinterp::geodetic;

// Checking the value defining the precision of a geohash.
static inline auto check_range(uint32_t precision) -> void {
  if (precision < 1 || precision > 64) {
    throw std::invalid_argument("precision must be within [1, 64]");
  }
}

void init_geohash_int64(py::module& m) {
  m.def(
       "error",
       [](const uint32_t& precision) -> py::tuple {
         check_range(precision);
         auto lon_lat_err = geohash::int64::error_with_precision(precision);
         return py::make_tuple(std::get<1>(lon_lat_err),
                               std::get<0>(lon_lat_err));
       },
       py::arg("precision"),
       R"__doc__(
Returns the accuracy in longitude/latitude and degrees for the number of bits
used to encode the GeoHash.

Args:
    precision (int): Number of bits used to encode the geohash code.
Returns:
    tuple: Accuracy in longitude/latitude and degrees.
Raises:
    ValueError: If the given precision is not within [1, 64].
)__doc__")
      .def(
          "encode",
          [](const geodetic::Point& point, uint32_t precision) -> uint64_t {
            check_range(precision);
            return geohash::int64::encode(point, precision);
          },
          py::arg("point"), py::arg("precision") = 64,
          R"__doc__(
Encode a point into geohash code with the given precision.

Args:
  point (pyinterp.geodetic.Point) Point to encode.
  precision (int, optional) Number of bits used to encode the geohash code.
Returns:
  int: geohash code.
Raises:
  ValueError: If the given precision is not within [1, 64].
)__doc__")
      .def(
          "encode",
          [](const Eigen::Ref<const Eigen::VectorXd>& lon,
             const Eigen::Ref<const Eigen::VectorXd>& lat,
             const uint32_t precision) -> pyinterp::Vector<uint64_t> {
            check_range(precision);
            return geohash::int64::encode(lon, lat, precision);
          },
          py::arg("lon"), py::arg("lat"), py::arg("precision") = 64,
          R"__doc__(
Encode coordinates into geohash with the given precision.

Args:
  lon (numpy.ndarray) Longitudes in degrees.
  lat (numpy.ndarray) Latitudes in degrees.
  precision (int, optional) Number of bits used to encode the geohash code.
Returns:
  numpy.ndarray: geohash codes.
Raises:
  ValueError: If the given precision is not within [1, 64].
  ValueError: If the lon and lat vectors have different sizes.
)__doc__")
      .def(
          "decode",
          [](const uint64_t hash, const uint32_t precision,
             const bool round) -> geodetic::Point {
            check_range(precision);
            return geohash::int64::decode(hash, precision, round);
          },
          py::arg("hash"), py::arg("precision") = 64, py::arg("round") = false,
          R"__doc__(
Decode a hash into a geographic point with the given precision.

Args:
  hash (int): Geohash.
  precision (int, optional): Required accuracy.
  round (optional, bool): If true, the coordinates of the point will be
    rounded to the accuracy defined by the GeoHash.
Returns:
  pyinterp.geodetic.Point: decoded geographic point.
Raises:
    ValueError: If the given precision is not within [1, 64].
)__doc__")
      .def(
          "decode",
          [](const Eigen::Ref<const pyinterp::Vector<uint64_t>>& hash,
             const uint32_t precision,
             const bool round) -> std::tuple<Eigen::VectorXd, Eigen::VectorXd> {
            check_range(precision);
            return geohash::int64::decode(hash, precision, round);
          },
          py::arg("hash"), py::arg("precision") = 64, py::arg("round") = false,
          R"__doc__(
Decode hash into a geographic points with the given precision.

Args:
  hash (numpy.ndarray): Geohash.
  precision (int, optional): Required accuracy.
  round (optional, bool): If true, the coordinates of the point will be
    rounded to the accuracy defined by the GeoHash."
Returns:
  tuple: longitudes/latitudes of the decoded points.
Raises:
    ValueError: If the given precision is not within [1, 64].
)__doc__")
      .def(
          "bounding_box",
          [](const uint64_t hash, const uint32_t precision) -> geodetic::Box {
            check_range(precision);
            return geohash::int64::bounding_box(hash, precision);
          },
          py::arg("hash"), py::arg("precision") = 64,
          R"__doc__(
Returns the bounding box encoded by the geohash with the specified precision.

Args:
  hash (int): Geohash.
  precision (int, optional): Required accuracy.
Returns:
  pyinterp.geodetic.Box: Bounding box.
Raises:
    ValueError: If the given precision is not within [1, 64].
)__doc__")
      .def(
          "bounding_boxes",
          [](const std::optional<geodetic::Box>& box,
             const uint32_t precision) -> pyinterp::Vector<uint64_t> {
            check_range(precision);
            return geohash::int64::bounding_boxes(box, precision);
          },
          py::arg("box") = py::none(), py::arg("precision") = 5,
          R"__doc__(
Returns all geohash codes contained in the defined bounding box.

Args:
  box (pyinterp.geohash.Box, optional): Bounding box.
  precision (int, optional): Required accuracy.
Returns:
  numpy.ndarray: Geohash codes.
Raises:
    ValueError: If the given precision is not within [1, 64].
)__doc__")
      .def(
          "neighbors",
          [](const uint64_t hash,
             const uint32_t precision) -> Eigen::Matrix<uint64_t, 8, 1> {
            check_range(precision);
            return geohash::int64::neighbors(hash, precision);
          },
          py::arg("hash"), py::arg("precision") = 64,
          R"__doc__(
Returns all neighbors hash clockwise from north around northwest
at the given precision.

Args:
  hash (int): Geohash code.
  precision (int, optional): Required accuracy.
Returns:
  numpy.ndarray: Geohash codes.
Raises:
  ValueError: If the given precision is not within [1, 64].
)__doc__")
      .def(
          "grid_properties",
          [](const geodetic::Box& box,
             const uint32_t precision) -> std::tuple<uint64_t, size_t, size_t> {
            check_range(precision);
            return geohash::int64::grid_properties(box, precision);
          },
          py::arg("box") = py::none(), py::arg("precision") = 64,
          R"__doc__(
Returns the property of the grid covering the given bounding box.

Args:
  box (pyinterp.geodetic.Box, optional): Bounding box.
  precision (int, optional): Required accuracy.
Returns:
  tuple:: geohash of the minimum corner point, number of boxes in longitudes
    and latitudes.
Raises:
  ValueError: If the given precision is not within [1, 64].
)__doc__")
      .def("where", &geohash::int64::where, py::arg("hash"),
           R"__doc__(
Returns the start and end indexes of the different GeoHash boxes.

Args:
  hash (numpy.ndarray): Geohash codes.
Returns:
  dict: dictionary between successive identical geohash codes and start and
    end indexes in the table provided as input.
)__doc__");
}