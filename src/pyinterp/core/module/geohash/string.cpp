// Copyright (c) 2021 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geohash/string.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>

#include "pyinterp/geohash/int64.hpp"

namespace py = pybind11;
namespace geohash = pyinterp::geohash;
namespace geodetic = pyinterp::geodetic;

// Parsing of the string defining a GeoHash.
static inline const auto parse_str = [](const py::str& hash) -> auto {
  auto result = std::string(hash);
  if (result.length() < 1 || result.length() > 12) {
    throw std::invalid_argument("Geohash length must be within [1, 12]");
  }
  return result;
};

// Checking the value defining the precision of a geohash.
static inline auto check_range(uint32_t precision) -> void {
  if (precision < 1 || precision > 12) {
    throw std::invalid_argument("precision must be within [1, 12]");
  }
}

void init_geohash_string(py::module& m) {
  m.def(
       "error",
       [](const uint32_t& precision) -> py::tuple {
         check_range(precision);
         auto lon_lat_err = geohash::int64::error_with_precision(precision * 5);
         return py::make_tuple(std::get<1>(lon_lat_err),
                               std::get<0>(lon_lat_err));
       },
       py::arg("precision"),
       R"__doc__(
Returns the accuracy in longitude/latitude and degrees for the given precision.

Args:
    precision (int): Number of bits used to encode the geohash code.
Returns:
    tuple: Accuracy in longitude/latitude and degrees.
Raises:
    ValueError: If the given precision is not within [1, 12].
)__doc__")
      .def(
          "encode",
          [](const geodetic::Point& point,
             const uint32_t precision) -> py::handle {
            auto result = std::array<char, 12>();
            check_range(precision);
            geohash::string::encode(point, result.data(), precision);
            return PyBytes_FromStringAndSize(result.data(), precision);
          },
          py::arg("point"), py::arg("precision") = 12,
          R"__doc__(
Encode a point into geohash code with the given precision.

Args:
  point (pyinterp.geodetic.Point) Point to encode.
  precision (int, optional) Number of bits used to encode the geohash code.
Returns:
  int: geohash code.
Raises:
  ValueError: If the given precision is not within [1, 12].
)__doc__")
      .def(
          "encode",
          [](const Eigen::Ref<const Eigen::VectorXd>& lon,
             const Eigen::Ref<const Eigen::VectorXd>& lat,
             const uint32_t precision) -> pybind11::array {
            check_range(precision);
            return geohash::string::encode(lon, lat, precision);
          },
          py::arg("lon"), py::arg("lat"), py::arg("precision") = 12,
          R"__doc__(
Encode coordinates into geohash with the given precision.

Args:
  lon (numpy.ndarray) Longitudes in degrees.
  lat (numpy.ndarray) Latitudes in degrees.
  precision (int, optional) Number of bits used to encode the geohash code.
Returns:
  numpy.ndarray: geohash codes.
Raises:
  ValueError: If the given precision is not within [1, 12].
  ValueError: If the lon and lat vectors have different sizes.
)__doc__")
      .def(
          "decode",
          [](const py::str& hash, const bool round) -> geodetic::Point {
            auto buffer = parse_str(hash);
            return geohash::string::decode(buffer.data(), buffer.length(),
                                           round);
          },
          py::arg("hash"), py::arg("round") = false,
          R"__doc__(
Decode a hash into a geographic point.

Args:
  hash (str): Geohash.
  round (optional, bool): If true, the coordinates of the point will be
    rounded to the accuracy defined by the GeoHash.
Returns:
  pyinterp.geodetic.Point: decoded geographic point.
)__doc__")
      .def(
          "decode",
          [](const pybind11::array& hash,
             const bool round) -> std::tuple<Eigen::VectorXd, Eigen::VectorXd> {
            return geohash::string::decode(hash, round);
          },
          py::arg("hash"), py::arg("round") = false,
          R"__doc__(
Decode hashes into a geographic points.

Args:
  hash (numpy.ndarray): Geohash codes.
  round (optional, bool): If true, the coordinates of the point will be
    rounded to the accuracy defined by the GeoHash."
Returns:
  tuple: longitudes/latitudes of the decoded points.
)__doc__")
      .def(
          "area",
          [](const py::str& hash,
             const std::optional<geodetic::System>& wgs) -> double {
            auto buffer = parse_str(hash);
            return geohash::string::area(buffer.data(), buffer.length(), wgs);
          },
          py::arg("hash"), py::arg("wgs") = py::none(),
          R"__doc__(
Calculate the area covered by the GeoHash.

Args:
  hash (str): Geohash.
  wgs (optional, pyinterp.geodetic.System): WGS used to calculate the area.
    Default to WGS84.
Returns:
  double: calculated area.
)__doc__")
      .def(
          "area",
          [](const pybind11::array& hash,
             const std::optional<geodetic::System>& wgs) -> Eigen::VectorXd {
            return geohash::string::area(hash, wgs);
          },
          py::arg("hash"), py::arg("wgs") = py::none(),
          R"__doc__(
Calculated the area caovered by the GeoHash codes.

Args:
  hash (numpy.ndarray): Geohash codes.
  wgs (optional, pyinterp.geodetic.System): WGS used to calculate the area.
    Default to WGS84.

Returns:
  double: calculated areas.
)__doc__")
      .def(
          "bounding_box",
          [](const py::str& hash) -> geodetic::Box {
            auto buffer = parse_str(hash);
            return geohash::string::bounding_box(buffer.data(),
                                                 buffer.length());
          },
          py::arg("hash"),
          R"__doc__(
Returns the bounding box encoded by the geohash.

Args:
  hash (str): Geohash.
Returns:
  pyinterp.geodetic.Box: Bounding box.
)__doc__")
      .def(
          "bounding_boxes",
          [](const std::optional<geodetic::Box>& box,
             const uint32_t precision) -> py::array {
            check_range(precision);
            return geohash::string::bounding_boxes(box, precision);
          },
          py::arg("box") = py::none(), py::arg("precision") = 1,
          R"__doc__(
Returns all geohash codes contained in the defined bounding box.

Args:
  box (pyinterp.geohash.Box, optional): Bounding box.
  precision (int, optional): Required accuracy.
Returns:
  numpy.ndarray: Geohash codes.
Raises:
    ValueError: If the given precision is not within [1, 12].
    MemoryError: If the memory is not sufficient to store the result.
)__doc__")
      .def(
          "bounding_boxes",
          [](const geodetic::Polygon& polygon,
             const uint32_t precision) -> py::array {
            check_range(precision);
            return geohash::string::bounding_boxes(polygon, precision);
          },
          py::arg("box") = py::none(), py::arg("precision") = 1,
          R"__doc__(
Returns all geohash codes contained in the defined polygon.

Args:
  polygon (pyinterp.geodetic.Polygon): Polygon.
  precision (int, optional): Required accuracy.
Returns:
  numpy.ndarray: Geohash codes.
Raises:
    ValueError: If the given precision is not within [1, 12].
    MemoryError: If the memory is not sufficient to store the result.
)__doc__")
      .def(
          "neighbors",
          [](const py::str& hash) {
            auto buffer = parse_str(hash);
            return geohash::string::neighbors(buffer.data(), buffer.length());
          },
          py::arg("hash"),
          R"__doc__(
Returns all neighbors hash clockwise from north around northwest.

Args:
  hash (str): Geohash code.
Returns:
  numpy.ndarray: Geohash codes.
)__doc__")
      .def(
          "grid_properties",
          [](const geodetic::Box& box,
             const uint32_t precision) -> std::tuple<uint64_t, size_t, size_t> {
            check_range(precision);
            return geohash::int64::grid_properties(box, precision * 5);
          },
          py::arg("box") = py::none(), py::arg("precision") = 12,
          R"__doc__(
Returns the property of the grid covering the given bounding box.

Args:
  box (pyinterp.geodetic.Box, optional): Bounding box.
  precision (int, optional): Required accuracy.
Returns:
  tuple: geohash of the minimum corner point, number of boxes in longitudes
    and latitudes.
Raises:
  ValueError: If the given precision is not within [1, 12].
)__doc__")
      .def(
          "where",
          // We want to return an associative dictionary between bytes and
          // tuples and not str and tuples.
          [](const pybind11::array& hash) -> py::dict {
            auto result = py::dict();
            for (auto&& item : geohash::string::where(hash)) {
              auto key = py::bytes(item.first);
              result[key] = py::cast(item.second);
            }
            return result;
          },
          py::arg("hash"),
          R"__doc__(
Returns the start and end indexes of the different GeoHash boxes.

Args:
  hash (numpy.ndarray): Geohash codes.
Returns:
  dict: dictionary between successive identical geohash codes and start and
    end indexes in the table provided as input.
)__doc__")
      .def(
          "zoom_in",
          [](const py::array& hash, const uint32_t precision) -> py::array {
            check_range(precision);
            return geohash::string::zoom_in(hash, precision);
          },
          py::arg("hash"), py::arg("precision") = 1, R"__doc__(
Returns the geohash code corresponding to the given one with a higher
accuracy.

Args:
  hash (numpy.ndarray): Geohash codes.
  precision (int, optional): Required accuracy.
Returns:
  numpy.ndarray: Geohash codes with higher accuracy.
Raises:
  ValueError: If the given precision is not within [1, 12].
)__doc__")
      .def(
          "zoom_out",
          [](const py::array& hash, const uint32_t precision) -> py::array {
            check_range(precision);
            return geohash::string::zoom_out(hash, precision);
          },
          py::arg("hash"), py::arg("precision") = 1, R"__doc__(
Returns the geohash code corresponding to the given one with a lower
accuracy.

Args:
  hash (numpy.ndarray): Geohash codes.
  precision (int, optional): Required accuracy.
Returns:
  numpy.ndarray: Geohash codes with lower accuracy.
Raises:
  ValueError: If the given precision is not within [1, 12].
)__doc__");
}
