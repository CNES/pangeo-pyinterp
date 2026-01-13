// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "pyinterp/geometry/geographic/box.hpp"
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

#include "pyinterp/eigen.hpp"
#include "pyinterp/format_byte.hpp"
#include "pyinterp/geohash/int64.hpp"
#include "pyinterp/parallel_for.hpp"

// Ref: https://mmcloughlin.com/posts/geohash-assembly
namespace pyinterp::geohash::int64 {
namespace codec {

static constexpr auto exp232 = 4294967296.0;      // 2^32;
static constexpr auto inv_exp232 = 1.0 / exp232;  // 1 / 2^32;

// Returns true if the CPU supports Bit Manipulation Instruction Set 2 (BMI2)
inline auto has_bmi2() noexcept -> bool {
#if defined(__x86_64__) || defined(_M_X64)
  std::array<unsigned int, 4> registers{};
#ifdef _WIN32
  __cpuidex(reinterpret_cast<int *>(registers.data()), 7, 0);
#else
  // Use the __cpuid_count intrinsic for GCC/Clang
  __cpuid_count(7, 0, registers[0], registers[1], registers[2], registers[3]);
#endif
  // BMI2 is feature bit 8 in EBX for leaf 7.
  return (registers[1] & (1U << 8U)) != 0;
#else
  // Not an x86-64 architecture, so no BMI2.
  return false;
#endif
}

// Spread out the 32 bits of x into 64 bits, where the bits of x occupy even
// bit positions.
constexpr auto spread(const uint32_t x) noexcept -> uint64_t {
  auto result = static_cast<uint64_t>(x);
  result = (result | (result << 16U)) & 0X0000FFFF0000FFFFUL;
  result = (result | (result << 8U)) & 0X00FF00FF00FF00FFUL;
  result = (result | (result << 4U)) & 0X0F0F0F0F0F0F0F0FUL;
  result = (result | (result << 2U)) & 0X3333333333333333UL;
  result = (result | (result << 1U)) & 0X5555555555555555UL;
  return result;
}

// Squash the even bitlevels of X into a 32-bit word. Odd bitlevels of X are
// ignored, and may take any value.
constexpr auto squash(uint64_t x) noexcept -> uint32_t {
  x &= 0x5555555555555555UL;
  x = (x | (x >> 1U)) & 0X3333333333333333UL;
  x = (x | (x >> 2U)) & 0X0F0F0F0F0F0F0F0FUL;
  x = (x | (x >> 4U)) & 0X00FF00FF00FF00FFUL;
  x = (x | (x >> 8U)) & 0X0000FFFF0000FFFFUL;
  x = (x | (x >> 16U)) & 0X00000000FFFFFFFFUL;
  return static_cast<uint32_t>(x);
}

// Interleave the bits of x and y. In the result, x and y occupy even and odd
// bitlevels, respectively.
constexpr auto interleave(const uint32_t x, const uint32_t y) noexcept
    -> uint64_t {
  return spread(x) | (spread(y) << 1U);
}

// Deinterleave the bits of x into 32-bit words containing the even and odd
// bitlevels of x, respectively.
constexpr auto deinterleave(const uint64_t x) noexcept
    -> std::tuple<uint32_t, uint32_t> {
  return {squash(x), squash(x >> 1U)};
}

// Encode the position of x within the range -r to +r as a 32-bit integer.
constexpr auto encode_range(const double x, const double r) -> uint32_t {
  if (x >= r) {
    return std::numeric_limits<uint32_t>::max();
  }
  auto p = (x + r) / (2 * r);
  return static_cast<uint32_t>(p * exp232);
}

// Decode the 32-bit range encoding X back to a value in the range -r to +r.
constexpr auto decode_range(const uint32_t x, const double r) noexcept
    -> double {
  if (x == std::numeric_limits<uint32_t>::max()) {
    return r;
  }
  auto p = static_cast<double>(x) * inv_exp232;
  return 2 * r * p - r;
}

// Encode the position a 64-bit integer
constexpr auto encode(const double lat, const double lon) -> uint64_t {
  return interleave(encode_range(lat, 90), encode_range(lon, 180));
}

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("bmi2")))
#endif
inline auto encode_bmi2(const double lat, const double lon) noexcept
    -> uint64_t {
  auto shrq = [](const double val) {
    return std::bit_cast<uint64_t>(val) >> 20U;
  };

  // The explicit _pdep_u64 calls require BMI2 support.
  auto y = _pdep_u64(
      lat == 90.0 ? 0X3FFFFFFFFFF : shrq(1.5 + (lat * 0.005555555555555556)),
      0x5555555555555555UL);
  auto x = _pdep_u64(
      lon == 180.0 ? 0X3FFFFFFFFFF : shrq(1.5 + (lon * 0.002777777777777778)),
      0x5555555555555555UL);

  return (x << 1U) | y;
}
#else
inline auto encode_bmi2(const double lat, const double lon) -> uint64_t {
  throw std::runtime_error(
      "BMI2 instructions are not supported on this platform.");
}
#endif

// Deinterleave the bits of x into 32-bit words containing the even and odd
// bitlevels of x, respectively.
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("bmi2")))
#endif
inline auto deinterleave_bmi2(const uint64_t x) noexcept
    -> std::tuple<uint32_t, uint32_t> {
  // The explicit _pext_u64 calls require BMI2 support.
  auto lat = _pext_u64(x, 0x5555555555555555UL);
  auto lon = _pext_u64(x, 0XAAAAAAAAAAAAAAAAUL);

  return std::make_tuple(static_cast<uint32_t>(lat),
                         static_cast<uint32_t>(lon));
}
#else
inline auto deinterleave_bmi2(const uint64_t x)
    -> std::tuple<uint32_t, uint32_t> {
  throw std::runtime_error(
      "BMI2 instructions are not supported on this platform.");
}
#endif

}  // namespace codec

// Pointer to the GeoHash position encoding function.
using EncoderPtr = uint64_t (*)(double, double);

// Pointer to the bits extracting function.
using DecoderPtr = std::tuple<uint32_t, uint32_t> (*)(uint64_t);

// Can the CPU use the BMI2 instruction set?
static const bool have_bmi2 = codec::has_bmi2();

// Sets the encoding/decoding functions according to the CPU capacity
static EncoderPtr const encoder =
    have_bmi2 ? codec::encode_bmi2 : codec::encode;
static DecoderPtr const deinterleaver =
    have_bmi2 ? codec::deinterleave_bmi2 : codec::deinterleave;

// ============================================================================

static auto allocate_array(const size_t size) -> Vector<uint64_t> {
  try {
    return Eigen::Matrix<uint64_t, -1, 1>(size);
  } catch (const std::bad_alloc &) {
    throw std::runtime_error(std::format(
        "Unable to allocate {} for an array with shape ({},) and data "
        "type uint64",
        pyinterp::format_bytes(size * 8ULL), size));
  }
}

// ============================================================================

auto encode(const geometry::geographic::Point &point, const uint32_t precision)
    -> uint64_t {
  auto lon = point.lon();
  if (lon < -180.0 || lon > 180.0) {
    // GeoHash longitude must be in the interval [-180, 180]
    lon = math::normalize_period(lon, -180.0, 360.0);
  }
  auto result = encoder(point.lat(), lon);
  if (precision != 64) {
    result >>= (64 - precision);
  }
  return result;
}

// ============================================================================

auto bounding_box(const uint64_t hash, const uint32_t precision) noexcept
    -> geometry::geographic::Box {
  auto full_hash = hash << (64U - precision);
  auto [x_err, y_err] = error_with_precision(precision);
  auto [y, x] = deinterleaver(full_hash);
  auto lat = codec::decode_range(y, 90);
  auto lon = codec::decode_range(x, 180);

  return {
      {lon, lat},
      {lon + x_err, lat + y_err},
  };
}

// ============================================================================

auto neighbors(const uint64_t hash, const uint32_t precision)
    -> NeighborHashes {
  auto box = bounding_box(hash, precision);
  auto center = box.centroid();
  auto lon = center.lon();
  auto lat = center.lat();
  auto [lon_delta, lat_delta] = box.delta(false);

  auto lon_inc = lon + lon_delta;
  auto lon_dec = lon - lon_delta;
  auto lat_inc = lat + lat_delta;
  auto lat_dec = lat - lat_delta;

  return NeighborHashes({// N
                         encode({lon, lat_inc}, precision),
                         // NE,
                         encode({lon_inc, lat_inc}, precision),
                         // E,
                         encode({lon_inc, lat}, precision),
                         // SE,
                         encode({lon_inc, lat_dec}, precision),
                         // S,
                         encode({lon, lat_dec}, precision),
                         // SW,
                         encode({lon_dec, lat_dec}, precision),
                         // W,
                         encode({lon_dec, lat}, precision),
                         // NW
                         encode({lon_dec, lat_inc}, precision)});
}

// ============================================================================

auto grid_properties(const geometry::geographic::Box &box,
                     const uint32_t precision)
    -> std::tuple<uint64_t, size_t, size_t> {
  auto hash_sw = encode(box.min_corner(), precision);
  auto box_sw = bounding_box(hash_sw, precision);
  auto box_ne = bounding_box(encode(box.max_corner(), precision), precision);

  // Special case: the box is a single point
  if (box_sw == box_ne) {
    return std::make_tuple(hash_sw, 1, 1);
  }

  auto lon_offset = box.max_corner().lon() == 180 ? 1 : 0;
  auto lat_offset = box.max_corner().lat() == 90 ? 1 : 0;

  auto lng_lat_err = error_with_precision(precision);
  auto lon_step = static_cast<size_t>(
      std::round((box_ne.min_corner().lon() - box_sw.min_corner().lon()) /
                 (std::get<0>(lng_lat_err))));
  auto lat_step = static_cast<size_t>(
      std::round((box_ne.min_corner().lat() - box_sw.min_corner().lat()) /
                 (std::get<1>(lng_lat_err))));

  return {hash_sw, lon_step + lon_offset, lat_step + lat_offset};
}

// ============================================================================

// Calculate the intersection mask between the geometry and the GeoHash grid.
template <typename Geometry>
auto mask_cell(const geometry::geographic::Box &envelope,
               const Geometry &geometry, double lng_err, double lat_err,
               const geometry::geographic::Point &point_sw, size_t lon_step,
               size_t lat_step, uint32_t bits, size_t num_threads)
    -> Matrix<bool> {
  // Allocate the grid result
  auto result = Matrix<bool>(lon_step, lat_step);

  parallel_for(
      static_cast<int64_t>(lat_step),
      [&](int64_t start, int64_t end) -> void {
        for (auto lat = start; lat < end; ++lat) {
          auto point = geometry::geographic::Point(
              0, point_sw.lat() + static_cast<double>(lat) * lat_err);

          for (size_t lon = 0; lon < lon_step; ++lon) {
            point.lon() = point_sw.lon() + static_cast<double>(lon) * lng_err;
            result(lon, lat) = boost::geometry::intersects(
                bounding_box(encode(point, bits), bits), geometry);
          }
        }
      },
      static_cast<int64_t>(num_threads));

  return result;
}

// ============================================================================//

// Return all GeoHash codes selected by the mask.
static auto select_cell(double lng_err, double lat_err,
                        const geometry::geographic::Point &point_sw,
                        size_t lon_step, size_t lat_step, uint32_t bits,
                        uint32_t precision, const Matrix<bool> &mask)
    -> Vector<uint64_t> {
  // Count the number of cells that are enclosed by the polygon
  auto size = mask.cast<uint64_t>().sum();

  // Allocates the result
  auto result = allocate_array(size);

  // For each cell of the grid, if it is selected, we add the code to the
  // result
  size_t result_ix = 0;
  for (size_t lat = 0; lat < lat_step; ++lat) {
    auto point = geometry::geographic::Point(
        0, point_sw.lat() + static_cast<double>(lat) * lat_err);

    for (size_t lon = 0; lon < lon_step; ++lon) {
      if (mask(lon, lat)) {
        point.lon() = point_sw.lon() + static_cast<double>(lon) * lng_err;
        result(result_ix++) = int64::encode(point, bits);
      }
    }
  }
  return result;
}

// ============================================================================//

// Helper to safely compute envelope, correcting boost's anti-meridian
// normalization
template <typename Geometry>
auto safe_envelope(const Geometry &geometry) -> geometry::geographic::Box {
  if constexpr (std::is_same_v<Geometry, geometry::geographic::Box>) {
    return geometry;  // Use box directly
  } else {
    geometry::geographic::Box envelope;
    boost::geometry::envelope(geometry, envelope);

    // Check if boost incorrectly normalized the envelope
    // This happens when boost wraps longitudes > 180
    // (e.g., -180 becomes 180, -135 becomes 225)
    auto &lon_min = envelope.min_corner().lon();
    auto &lon_max = envelope.max_corner().lon();

    if (lon_max > 180.0) {
      // Boost incorrectly normalized.
      // Correct it by wrapping back to [-180, 180].
      // If max > 180, both min and max should be wrapped
      // e.g., (180, ...) to (225, ...) should become (-180, ...) to (-135, ...)
      lon_min -= 360.0;
      lon_max -= 360.0;
    }

    return envelope;
  }
}

// ============================================================================//

// Common implementation for bounding_boxes with geometry
template <typename Geometry>
auto bounding_boxes_impl(const Geometry &geometry, uint32_t precision,
                         size_t num_threads) -> Vector<uint64_t> {
  // Compute envelope safely, handling anti-meridian cases
  auto envelope = safe_envelope(geometry);

  // Grid resolution in degrees
  const auto [lng_err, lat_err] = error_with_precision(precision);

  // Property of the grid
  auto [hash_sw, lon_step, lat_step] = grid_properties(envelope, precision);
  const auto point_sw = decode(hash_sw, precision, false);

  Matrix<bool> mask;
  if constexpr (std::is_same_v<Geometry, geometry::geographic::Box>) {
    // If the geometry is a box, all cells are selected
    mask = Matrix<bool>(lon_step, lat_step);
    mask.setConstant(true);
  } else {
    // Otherwise, calculates the intersection mask between the geometry and the
    // GeoHash grid
    mask = mask_cell(envelope, geometry, lng_err, lat_err, point_sw, lon_step,
                     lat_step, precision, num_threads);
  }

  // Finally, selects the geohashes that are enclosed in the geometry
  return select_cell(lng_err, lat_err, point_sw, lon_step, lat_step, precision,
                     precision, mask);
}

// ============================================================================

auto bounding_boxes(const geometry::geographic::Box &box,
                    const uint32_t precision, const size_t num_threads)
    -> Vector<uint64_t> {
  return bounding_boxes_impl(box, precision, num_threads);
}

// ============================================================================

auto bounding_boxes(const geometry::geographic::Polygon &polygon,
                    const uint32_t precision, const size_t num_threads)
    -> Vector<uint64_t> {
  return bounding_boxes_impl(polygon, precision, num_threads);
}

// ============================================================================

auto bounding_boxes(const geometry::geographic::MultiPolygon &polygons,
                    const uint32_t precision, const size_t num_threads)
    -> Vector<uint64_t> {
  return bounding_boxes_impl(polygons, precision, num_threads);
}

}  // namespace pyinterp::geohash::int64
