// Copyright (c) 2022 CNES
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

// Checking the value defining the precision of a geohash.
constexpr static auto check_range(uint32_t precision) -> void {
  if (precision < 1 || precision > 64) {
    throw std::invalid_argument("precision must be within [1, 64]");
  }
}

void init_geohash_int64(py::module &m) {
  m.def(
       "encode",
       [](const Eigen::Ref<const Eigen::VectorXd> &lon,
          const Eigen::Ref<const Eigen::VectorXd> &lat,
          const uint32_t precision) -> pyinterp::Vector<uint64_t> {
         check_range(precision);
         return geohash::int64::encode(lon, lat, precision);
       },
       py::arg("lon"), py::arg("lat"), py::arg("precision") = 64,
       R"__doc__(
Encode coordinates into geohash with the given precision.

Args:
    lon: Longitudes in degrees.
    lat: Latitudes in degrees.
    precision: Number of bits used to encode the geohash code.
Returns:
    Geohash codes.
Raises:
    ValueError: If the given precision is not within [1, 64].
    ValueError: If the lon and lat vectors have different sizes.
)__doc__")
      .def(
          "decode",
          [](const Eigen::Ref<const pyinterp::Vector<uint64_t>> &hash,
             const uint32_t precision,
             const bool round) -> std::tuple<Eigen::VectorXd, Eigen::VectorXd> {
            check_range(precision);
            return geohash::int64::decode(hash, precision, round);
          },
          py::arg("hash"), py::arg("precision") = 64, py::arg("round") = false,
          R"__doc__(
Decode hash into a geographic points with the given precision.

Args:
    hash: GeoHash.
    precision: Required accuracy.
    round: If true, the coordinates of the point will be rounded to the accuracy
        defined by the GeoHash."
Returns:
    Longitudes/latitudes of the decoded points.
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
    hash: Geohash code.
    precision: Required accuracy.
Returns:
    Geohash codes.
Raises:
    ValueError: If the given precision is not within [1, 64].
)__doc__");
}
