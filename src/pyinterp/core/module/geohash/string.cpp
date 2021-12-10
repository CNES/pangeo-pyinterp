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

// Checking the value defining the precision of a geohash.
static inline auto check_range(uint32_t precision) -> void {
  if (precision > 12) {
    throw std::invalid_argument("precision must be within [1, 12]");
  }
}

void init_geohash_string(py::module& m) {
  m.def(
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
    Defaults to 12.
Returns:
  numpy.ndarray: geohash codes.
Raises:
  ValueError: If the given precision is not within [1, 12].
  ValueError: If the lon and lat vectors have different sizes.
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
  hash (numpy.ndarray): GeoHash codes.
  round (optional, bool): If true, the coordinates of the point will be
    rounded to the accuracy defined by the GeoHash. Defaults to False.
Returns:
  tuple: longitudes/latitudes of the decoded points.
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
  hash (numpy.ndarray): GeoHash codes.
  wgs (optional, pyinterp.geodetic.System): WGS used to calculate the area.
    Defaults to WGS84.

Returns:
  double: calculated areas.
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
  box (pyinterp.geohash.Box, optional): Bounding box. Default to the
    global bounding box.
  precision (int, optional): Required accuracy. Defaults to 1.
Returns:
  numpy.ndarray: GeoHash codes.
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
  numpy.ndarray: GeoHash codes.
Raises:
    ValueError: If the given precision is not within [1, 12].
    MemoryError: If the memory is not sufficient to store the result.
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
  hash (numpy.ndarray): GeoHash codes.
Returns:
  dict: dictionary between successive identical geohash codes and start and
    end indexes in the table provided as input.
)__doc__")
      .def(
          "transform",
          [](const py::array& hash, const uint32_t precision) -> py::array {
            check_range(precision);
            return geohash::string::transform(hash, precision);
          },
          py::arg("hash"), py::arg("precision") = 1, R"__doc__(
Transforms the given codes from one precision to another. If the given
precision is higher than the precision of the given codes, the result contains
a zoom in, otherwise it contains a zoom out.

Args:
  hash (numpy.ndarray): GeoHash codes.
  precision (int, optional): Required accuracy. Defaults to 1.
Returns:
  numpy.ndarray: GeoHash codes transformed.
Raises:
  ValueError: If the given precision is not within [1, 12].
)__doc__");
}
