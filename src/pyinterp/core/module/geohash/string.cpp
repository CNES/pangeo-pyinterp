// Copyright (c) 2022 CNES
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
constexpr static auto check_range(uint32_t precision) -> void {
  if (precision > 12) {
    throw std::invalid_argument("precision must be within [1, 12]");
  }
}

void init_geohash_string(py::module &m) {
  m.def(
       "encode",
       [](const Eigen::Ref<const Eigen::VectorXd> &lon,
          const Eigen::Ref<const Eigen::VectorXd> &lat,
          const uint32_t precision) -> pybind11::array {
         check_range(precision);
         return geohash::string::encode(lon, lat, precision);
       },
       py::arg("lon"), py::arg("lat"), py::arg("precision") = 12,
       R"__doc__(
Encode coordinates into geohash with the given precision.

Args:
    lon: Longitudes in degrees.
    lat: Latitudes in degrees.
    precision: Number of bits used to encode the geohash code. Defaults to 12.
Returns:
    Geohash codes.
Raises:
    ValueError: If the given precision is not within [1, 12].
    ValueError: If the lon and lat vectors have different sizes.
)__doc__")
      .def(
          "decode",
          [](const pybind11::array &hash,
             const bool round) -> std::tuple<Eigen::VectorXd, Eigen::VectorXd> {
            return geohash::string::decode(hash, round);
          },
          py::arg("hash"), py::arg("round") = false,
          R"__doc__(
Decode hashes into a geographic points.

Args:
    hash: GeoHash codes.
    round: If true, the coordinates of the point will be rounded to the accuracy
        defined by the GeoHash. Defaults to False.
Returns:
    Longitudes/latitudes of the decoded points.
)__doc__")
      .def(
          "area",
          [](const pybind11::array &hash,
             const std::optional<geodetic::Spheroid> &wgs) -> Eigen::VectorXd {
            return geohash::string::area(hash, wgs);
          },
          py::arg("hash"), py::arg("wgs") = py::none(),
          R"__doc__(
Calculated the area caovered by the GeoHash codes.

Args:
    hash: GeoHash codes.
    wgs: WGS used to calculate the area. Defaults to WGS84.

Returns:
   Calculated areas.
)__doc__")
      .def(
          "bounding_boxes",
          [](const std::optional<geodetic::Box> &box,
             const uint32_t precision) -> py::array {
            check_range(precision);
            return geohash::string::bounding_boxes(box, precision);
          },
          py::arg("box") = py::none(), py::arg("precision") = 1,
          R"__doc__(
Returns all geohash codes contained in the defined bounding box.

Args:
    box: Bounding box. Default to the global bounding box.
    precision: Required accuracy. Defaults to 1.
Returns:
    GeoHash codes.
Raises:
    ValueError: If the given precision is not within [1, 12].
    MemoryError: If the memory is not sufficient to store the result.
)__doc__")
      .def(
          "bounding_boxes",
          [](const geodetic::Polygon &polygon, const uint32_t precision,
             const size_t num_threads) -> py::array {
            check_range(precision);
            return geohash::string::bounding_boxes(polygon, precision,
                                                   num_threads);
          },
          py::arg("polygon"), py::arg("precision") = 1,
          py::arg("num_threads") = 0,
          R"__doc__(
Returns all geohash codes contained in the defined polygon.

Args:
    polygon: Polygon.
    precision: Required accuracy. Defaults to ``1``.
    num_threads: The number of threads to use for the
        computation. If 0 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging.
        Defaults to ``0``.
Returns:
    GeoHash codes.
Raises:
    ValueError: If the given precision is not within [1, 12].
    MemoryError: If the memory is not sufficient to store the result.
)__doc__")
      .def(
          "bounding_boxes",
          [](const geodetic::MultiPolygon &polygons, const uint32_t precision,
             const size_t num_threads) -> py::array {
            check_range(precision);
            return geohash::string::bounding_boxes(polygons, precision,
                                                   num_threads);
          },
          py::arg("polygons"), py::arg("precision") = 1,
          py::arg("num_threads") = 0,
          R"__doc__(
Returns all geohash codes contained in one or more defined polygons.

Args:
    polygons: MultiPolygon.
    precision: Required accuracy. Defaults to ``1``.
    num_threads: The number of threads to use for the
        computation. If 0 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging.
        Defaults to ``0``.
Returns:
    GeoHash codes.
Raises:
    ValueError: If the given precision is not within [1, 12].
    MemoryError: If the memory is not sufficient to store the result.
)__doc__")
      .def(
          "where",
          // We want to return an associative dictionary between bytes and
          // tuples and not str and tuples.
          [](const pybind11::array &hash) -> py::dict {
            auto result = py::dict();
            for (auto &&item : geohash::string::where(hash)) {
              auto key = py::bytes(item.first);
              result[key] = py::cast(item.second);
            }
            return result;
          },
          py::arg("hash"),
          R"__doc__(
Returns the start and end indexes of the different GeoHash boxes.

Args:
    hash: GeoHash codes.
Returns:
    Dictionary between successive identical geohash codes and start and
    end indexes in the table provided as input.
)__doc__")
      .def(
          "transform",
          [](const py::array &hash, const uint32_t precision) -> py::array {
            check_range(precision);
            return geohash::string::transform(hash, precision);
          },
          py::arg("hash"), py::arg("precision") = 1, R"__doc__(
Transforms the given codes from one precision to another. If the given
precision is higher than the precision of the given codes, the result contains
a zoom in, otherwise it contains a zoom out.

Args:
    hash: GeoHash codes.
    precision: Required accuracy. Defaults to ``1``.
Returns:
    GeoHash codes transformed.
Raises:
    ValueError: If the given precision is not within [1, 12].
)__doc__");
}
