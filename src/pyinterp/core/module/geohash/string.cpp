// Copyright (c) 2025 CNES
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
Encode geographic coordinates into geohash strings.

This function encodes the given longitude and latitude coordinates into
geohash strings with the specified precision.

Args:
    lon: Longitudes in degrees.
    lat: Latitudes in degrees.
    precision: Number of characters used to encode the geohash code.
        Defaults to 12.

Returns:
    Geohash codes as strings.

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
Decode geohash strings into geographic coordinates.

This function decodes geohash strings into longitude and latitude coordinates.
Optionally rounds the coordinates to the accuracy defined by the geohash.

Args:
    hash: GeoHash codes to decode.
    round: If true, the coordinates of the point will be rounded to the
        accuracy defined by the GeoHash. Defaults to False.

Returns:
    Tuple of (longitudes, latitudes) of the decoded points.
)__doc__")
      .def(
          "area",
          [](const pybind11::array &hash,
             const std::optional<geodetic::Spheroid> &wgs) -> Eigen::VectorXd {
            return geohash::string::area(hash, wgs);
          },
          py::arg("hash"), py::arg("wgs") = py::none(),
          R"__doc__(
Calculate the area covered by geohash codes.

This function computes the area (in square meters) covered by the provided
geohash codes using the specified geodetic reference system.

Args:
    hash: GeoHash codes.
    wgs: WGS (World Geodetic System) used to calculate the area.
        Defaults to WGS84.

Returns:
    Array of calculated areas in square meters.
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
Get all geohash codes within a bounding box.

This function returns all geohash codes contained in the defined bounding box
at the specified precision level.

Args:
    box: Bounding box defining the region. Defaults to the global bounding box.
    precision: Required accuracy level. Defaults to 1.

Returns:
    Array of GeoHash codes.

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
Get all geohash codes within a polygon.

This function returns all geohash codes contained in the defined polygon at
the specified precision level. Supports parallel computation using multiple
threads.

Args:
    polygon: Polygon defining the region.
    precision: Required accuracy level. Defaults to 1.
    num_threads: Number of threads to use for computation. If 0, all CPUs
        are used. If 1, no parallel computing is used (useful for debugging).
        Defaults to 0.

Returns:
    Array of GeoHash codes.

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
Get all geohash codes within one or more polygons.

This function returns all geohash codes contained in one or more defined
polygons at the specified precision level. Supports parallel computation using
multiple threads.

Args:
    polygons: MultiPolygon defining one or more regions.
    precision: Required accuracy level. Defaults to 1.
    num_threads: Number of threads to use for computation. If 0, all CPUs
        are used. If 1, no parallel computing is used (useful for debugging).
        Defaults to 0.

Returns:
    Array of GeoHash codes.

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
Get the start and end indexes for successive geohash codes.

Returns a dictionary mapping successive identical geohash codes to their
start and end positions in the input array.

Args:
    hash: Array of GeoHash codes.

Returns:
    Dictionary where keys are geohash codes (as bytes) and values are tuples
    of (start_index, end_index) in the input array.
)__doc__")
      .def(
          "transform",
          [](const py::array &hash, const uint32_t precision) -> py::array {
            check_range(precision);
            return geohash::string::transform(hash, precision);
          },
          py::arg("hash"), py::arg("precision") = 1, R"__doc__(
Transform geohash codes between different precision levels.

Changes the precision of the given geohash codes. If the target precision is
higher than the current precision, the result contains a zoom in; otherwise
it contains a zoom out.

Args:
    hash: Array of GeoHash codes.
    precision: Target accuracy level. Defaults to 1.

Returns:
    Array of GeoHash codes at the new precision level.

Raises:
    ValueError: If the given precision is not within [1, 12].
)__doc__");
}
