// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/geohash/string.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>

#include <Eigen/Core>
#include <utility>

#include "pyinterp/geometry/geographic/box.hpp"
#include "pyinterp/geometry/geographic/multi_polygon.hpp"
#include "pyinterp/geometry/geographic/polygon.hpp"

namespace nb = nanobind;

namespace pyinterp::geohash::pybind {

using pyinterp::geohash::area;
using pyinterp::geohash::bounding_boxes;
using pyinterp::geohash::decode;
using pyinterp::geohash::encode;
using pyinterp::geohash::EncodedHashes;
using pyinterp::geohash::EncodedHashesView;
using pyinterp::geohash::HashRegionBounds;
using pyinterp::geohash::transform;
using pyinterp::geohash::where;

// Checking the value defining the precision of a geohash.
constexpr static auto check_range(uint32_t precision) -> void {
  if (precision == 0 || precision > 12) {
    throw std::invalid_argument("precision must be within [1, 12]");
  }
}

// Convert EncodedHashes to numpy string array (zero-copy via move)
static auto to_numpy(EncodedHashes&& hashes) -> nb::object {
  auto count = hashes.count;
  auto precision = hashes.precision;

  // Move buffer into heap-allocated vector
  auto ptr = std::make_unique<std::vector<char>>(std::move(hashes.buffer));
  auto* data = ptr->data();

  // Create capsule with ownership
  nb::capsule owner(ptr.get(), [](void* p) noexcept -> void {
    delete static_cast<std::vector<char>*>(p);
  });
  ptr.release();

  // Create the ndarray holder: a 1D array of characters
  auto array = nb::ndarray<nb::numpy, char, nb::ndim<1>>(
      data, {count * precision}, owner);
  // Finally return a view as string array
  return array.cast().attr("view")(nb::str("S{}").format(precision));
}

template <size_t NDIM>
struct ArrayInfo {
  std::array<size_t, NDIM> shape;
  std::array<int64_t, NDIM> strides;
};

template <size_t NDIM>
inline auto get_array_info(const nb::object& hash) -> ArrayInfo<NDIM> {
  auto ndim = nb::cast<size_t>(hash.attr("ndim"));
  if (ndim != NDIM) {
    throw std::invalid_argument(
        std::format("hash must be a {}-dimensional array", NDIM));
  }
  auto kind = nb::cast<std::string>(hash.attr("dtype").attr("kind"));
  if (kind != "S") {
    throw std::invalid_argument("hash must be a string array");
  }
  ArrayInfo<NDIM> info;
  for (size_t ix = 0; ix < NDIM; ++ix) {
    info.shape[ix] = nb::cast<size_t>(hash.attr("shape")[ix]);
    info.strides[ix] = nb::cast<int64_t>(hash.attr("strides")[ix]);
  }
  return info;
}

// Create a view into numpy string array as EncodedHashesView
// (zero-copy)
template <size_t NDIM>
static auto from_numpy(const nb::object& hash) -> EncodedHashesView {
  static_assert(NDIM == 1 || NDIM == 2, "NDIM must be 1 or 2");
  auto info = get_array_info<NDIM>(hash);
  if constexpr (NDIM == 1) {
    // 1D string array - precision is the stride
    auto precision = static_cast<uint32_t>(info.strides[0]);
    if (precision == 0 || precision > 12) {
      throw std::invalid_argument("string length must be within [1, 12]");
    }

    // View as uint8 and cast to typed ndarray
    auto viewed =
        nb::cast<nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>, nb::c_contig>>(
            hash.attr("view")("uint8"));

    return EncodedHashesView{
        .data = reinterpret_cast<const char*>(viewed.data()),
        .precision = precision,
        .count = info.shape[0],
    };
  } else if constexpr (NDIM == 2) {
    if (std::cmp_not_equal(info.strides[0], info.shape[1] * info.strides[1])) {
      throw std::invalid_argument("hash must be an array of strings");
    }
    auto precision = static_cast<uint32_t>(info.shape[1]);
    if (precision == 0 || precision > 12) {
      throw std::invalid_argument("string length must be within [1, 12]");
    }

    auto viewed =
        nb::cast<nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>, nb::c_contig>>(
            hash.attr("view")("uint8"));

    return EncodedHashesView{
        .data = reinterpret_cast<const char*>(viewed.data()),
        .precision = precision,
        .count = static_cast<size_t>(info.shape[0]),
    };
  }
}

constexpr const char* const kEncodeDoc = R"__doc__(
Encode geographic coordinates into geohash strings.

This function encodes the given longitude and latitude coordinates into
geohash strings with the specified precision.

Args:
    lon: Longitudes in degrees.
    lat: Latitudes in degrees.
    precision: Number of characters used to encode the geohash code.
        Defaults to 12.

Returns:
    Geohash codes as numpy string array (dtype ``S{precision}``).

Raises:
    ValueError: If the given precision is not within [1, 12].
    ValueError: If the lon and lat vectors have different sizes.
)__doc__";

constexpr const char* const kDecodeDoc = R"__doc__(
Decode geohash strings into geographic coordinates.

This function decodes geohash strings into longitude and latitude coordinates.
Optionally rounds the coordinates to the accuracy defined by the geohash.

Args:
    hash: GeoHash codes to decode (numpy string array).
    round: If true, the coordinates of the point will be rounded to the
        accuracy defined by the GeoHash. Defaults to False.

Returns:
    Tuple of (longitudes, latitudes) of the decoded points.
)__doc__";

constexpr const char* const kAreaDoc = R"__doc__(
Calculate the area covered by geohash codes.

This function computes the area (in square meters) covered by the provided
geohash codes using the specified geodetic reference system.

Args:
    hash: GeoHash codes (numpy string array).
    wgs: WGS (World Geodetic System) used to calculate the area.
        Defaults to WGS84.

Returns:
    Array of calculated areas in square meters.
)__doc__";

constexpr const char* const kBoundingBoxesBoxDoc = R"__doc__(
Get all geohash codes within a bounding box.

This function returns all geohash codes contained in the defined bounding box
at the specified precision level.

Args:
    box: Bounding box defining the region. Defaults to the global bounding box.
    precision: Required accuracy level. Defaults to 1.

Returns:
    Array of GeoHash codes (numpy string array).

Raises:
    ValueError: If the given precision is not within [1, 12].
    MemoryError: If the memory is not sufficient to store the result.
)__doc__";

constexpr const char* const kBoundingBoxesPolygonDoc = R"__doc__(
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
    Array of GeoHash codes (numpy string array).

Raises:
    ValueError: If the given precision is not within [1, 12].
    MemoryError: If the memory is not sufficient to store the result.
)__doc__";

constexpr const char* const kBoundingBoxesMultiPolygonDoc = R"__doc__(
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
    Array of GeoHash codes (numpy string array).

Raises:
    ValueError: If the given precision is not within [1, 12].
    MemoryError: If the memory is not sufficient to store the result.
)__doc__";

constexpr const char* const kWhereDoc = R"__doc__(
Get the start and end indexes for successive geohash codes.

Returns a dictionary mapping successive identical geohash codes to their
start and end positions in the input numpy string array.

Args:
    hash: Array of GeoHash codes (numpy string array).

Returns:
    Dictionary where keys are geohash codes (as bytes) and values are tuples
    of ((min_row, max_row), (min_col, max_col)) in the input array.
)__doc__";

constexpr const char* const kTransformDoc = R"__doc__(
Transform geohash codes between different precision levels.

Changes the precision of the given geohash codes. If the target precision is
higher than the current precision, the result contains a zoom in; otherwise
it contains a zoom out.

Args:
    hash: Array of GeoHash codes (numpy string array).
    precision: Target accuracy level. Defaults to 1.

Returns:
    Array of GeoHash codes at the new precision level (numpy string array).

Raises:
    ValueError: If the given precision is not within [1, 12].
)__doc__";

auto init_string(nb::module_& m) -> void {
  m.def(
      "encode",
      [](const Eigen::Ref<const Eigen::VectorXd>& lon,
         const Eigen::Ref<const Eigen::VectorXd>& lat,
         uint32_t precision) -> nb::object {
        check_range(precision);
        EncodedHashes result;
        {
          nb::gil_scoped_release release;
          result = encode(lon, lat, precision);
        }
        return to_numpy(std::move(result));
      },
      nb::arg("lon"), nb::arg("lat"), nb::arg("precision") = 12, kEncodeDoc);

  m.def(
      "decode",
      [](const nb::object& hash,
         bool round) -> std::tuple<Eigen::VectorXd, Eigen::VectorXd> {
        auto hashes = from_numpy<1>(hash);
        {
          nb::gil_scoped_release release;
          return decode(hashes, round);
        }
      },
      nb::arg("hash"), nb::arg("round") = false, kDecodeDoc);

  m.def(
      "area",
      [](const nb::object& hash,
         const std::optional<geometry::geographic::Spheroid>& spheroid)
          -> Eigen::VectorXd {
        auto hashes = from_numpy<1>(hash);
        {
          nb::gil_scoped_release release;
          return area(hashes, spheroid);
        }
      },
      nb::arg("hash"), nb::arg("spheroid") = nb::none(), kAreaDoc);

  m.def(
      "bounding_boxes",
      [](const std::optional<geometry::geographic::Box>& box,
         uint32_t precision) -> nb::object {
        check_range(precision);
        EncodedHashes result;
        {
          nb::gil_scoped_release release;
          result = bounding_boxes(box, precision);
        }
        return to_numpy(std::move(result));
      },
      nb::arg("box") = nb::none(), nb::arg("precision") = 1,
      kBoundingBoxesBoxDoc);

  m.def(
      "bounding_boxes",
      [](const geometry::geographic::Polygon& polygon, uint32_t precision,
         size_t num_threads) -> nb::object {
        check_range(precision);
        EncodedHashes result;
        {
          nb::gil_scoped_release release;
          result = bounding_boxes(polygon, precision, num_threads);
        }
        return to_numpy(std::move(result));
      },
      nb::arg("polygon"), nb::arg("precision") = 1, nb::arg("num_threads") = 0,
      kBoundingBoxesPolygonDoc);

  m.def(
      "bounding_boxes",
      [](const geometry::geographic::MultiPolygon& polygons, uint32_t precision,
         size_t num_threads) -> nb::object {
        check_range(precision);
        EncodedHashes result;
        {
          nb::gil_scoped_release release;
          result = bounding_boxes(polygons, precision, num_threads);
        }
        return to_numpy(std::move(result));
      },
      nb::arg("polygons"), nb::arg("precision") = 1, nb::arg("num_threads") = 0,
      kBoundingBoxesMultiPolygonDoc);

  m.def(
      "where",
      [](const nb::object& hash) -> nb::dict {
        auto hashes = from_numpy<2>(hash);
        HashRegionBounds result_map;
        {
          nb::gil_scoped_release release;
          result_map = where(hashes, hashes.count, hashes.precision);
        }

        // Convert to dict with bytes keys
        auto result = nb::dict();
        for (auto&& [key, value] : result_map) {
          result[nb::bytes(key.c_str(), key.size())] = nb::cast(value);
        }
        return result;
      },
      nb::arg("hash"), kWhereDoc);

  m.def(
      "transform",
      [](const nb::object& hash, uint32_t precision) -> nb::object {
        check_range(precision);
        auto hashes = from_numpy<1>(hash);
        EncodedHashes result;
        {
          nb::gil_scoped_release release;
          result = transform(hashes, precision);
        }
        return to_numpy(std::move(result));
      },
      nb::arg("hash"), nb::arg("precision") = 1, kTransformDoc);
}

}  // namespace pyinterp::geohash::pybind
