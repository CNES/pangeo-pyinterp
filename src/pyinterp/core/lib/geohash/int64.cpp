// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geohash/int64.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

#include <array>
#include <bit>
#include <iomanip>
#include <iostream>
#include <sstream>

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
constexpr auto spread(const uint32_t x) -> uint64_t {
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
constexpr auto squash(uint64_t x) -> uint32_t {
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
constexpr auto interleave(const uint32_t x, const uint32_t y) -> uint64_t {
  return spread(x) | (spread(y) << 1U);
}

// Deinterleave the bits of x into 32-bit words containing the even and odd
// bitlevels of x, respectively.
inline auto deinterleave(const uint64_t x) -> std::tuple<uint32_t, uint32_t> {
  return std::make_tuple(squash(x), squash(x >> 1U));
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
constexpr auto decode_range(const uint32_t x, const double r) -> double {
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
inline auto encode_bmi2(const double lat, const double lon) -> uint64_t {
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
inline auto deinterleave_bmi2(const uint64_t x)
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
using encoder_t = uint64_t (*)(double, double);

// Pointer to the bits extracting function.
using deinterleaver_t = std::tuple<uint32_t, uint32_t> (*)(uint64_t);

// Can the CPU use the BMI2 instruction set?
static const bool have_bmi2 = codec::has_bmi2();

// Sets the encoding/decoding functions according to the CPU capacity
static encoder_t const encoder = have_bmi2 ? codec::encode_bmi2 : codec::encode;
static deinterleaver_t const deinterleaver =
    have_bmi2 ? codec::deinterleave_bmi2 : codec::deinterleave;

// ---------------------------------------------------------------------------
auto format_bytes(size_t bytes) -> std::string {
  struct Suffix {
    const char *suffix;
    size_t divisor;

    Suffix(const char *suffix, const size_t divisor)
        : suffix(suffix), divisor(divisor) {}
  };

  static std::array<Suffix, 7> suffixes = {
      Suffix("EiB", 1152921504606846976ULL),
      Suffix("PiB", 1125899906842624ULL),
      Suffix("TiB", 1099511627776ULL),
      Suffix("GiB", 1073741824ULL),
      Suffix("MiB", 1048576ULL),
      Suffix("KiB", 1024ULL),
      Suffix("B", 1ULL),
  };

  auto result = std::string{};

  for (const auto &item : suffixes) {
    if (bytes > item.divisor) {
      auto ss = std::stringstream();
      ss << std::setprecision(2) << std::fixed
         << (static_cast<double>(bytes) / static_cast<double>(item.divisor))
         << " " << item.suffix;
      result = ss.str();
      break;
    }
  }
  return result.empty() ? std::to_string(bytes) + " B" : result;
}

// ---------------------------------------------------------------------------
auto allocate_array(const size_t size) -> Vector<uint64_t> {
  try {
    return Eigen::Matrix<uint64_t, -1, 1>(size);
  } catch (const std::bad_alloc &) {
    auto ss = std::stringstream();
    ss << "Unable to allocate " << format_bytes(size * 8ULL)
       << " for an array with shape (" << size << ",) and data type uint64";
    PyErr_SetString(PyExc_MemoryError, ss.str().c_str());
    throw pybind11::error_already_set();
  }
}

// ---------------------------------------------------------------------------
auto encode(const geodetic::Point &point, const uint32_t precision)
    -> uint64_t {
  auto result = encoder(point.lat(), point.lon());
  if (precision != 64) {
    result >>= (64 - precision);
  }
  return result;
}

// ---------------------------------------------------------------------------
auto bounding_box(const uint64_t hash, const uint32_t precision)
    -> geodetic::Box {
  auto full_hash = hash << (64U - precision);
  auto lat_lng_int = deinterleaver(full_hash);
  auto lat = codec::decode_range(std::get<0>(lat_lng_int), 90);
  auto lon = codec::decode_range(std::get<1>(lat_lng_int), 180);
  auto lon_lat_err = error_with_precision(precision);

  return {
      {lon, lat},
      {lon + std::get<0>(lon_lat_err), lat + std::get<1>(lon_lat_err)},
  };
}

// ---------------------------------------------------------------------------
auto neighbors(const uint64_t hash, const uint32_t precision)
    -> Eigen::Matrix<uint64_t, 8, 1> {
  auto box = bounding_box(hash, precision);
  auto center = box.centroid();
  auto [lon_delta, lat_delta] = box.delta(false);

  return (Eigen::Matrix<uint64_t, 8, 1>() <<
              // N
              encode({center.lon(), center.lat() + lat_delta}, precision),
          // NE,
          encode({center.lon() + lon_delta, center.lat() + lat_delta},
                 precision),
          // E,
          encode({center.lon() + lon_delta, center.lat()}, precision),
          // SE,
          encode({center.lon() + lon_delta, center.lat() - lat_delta},
                 precision),
          // S,
          encode({center.lon(), center.lat() - lat_delta}, precision),
          // SW,
          encode({center.lon() - lon_delta, center.lat() - lat_delta},
                 precision),
          // W,
          encode({center.lon() - lon_delta, center.lat()}, precision),
          // NW
          encode({center.lon() - lon_delta, center.lat() + lat_delta},
                 precision))
      .finished();
}

// ---------------------------------------------------------------------------
auto grid_properties(const geodetic::Box &box, const uint32_t precision)
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

  return std::make_tuple(hash_sw, lon_step + lon_offset, lat_step + lat_offset);
}

// ---------------------------------------------------------------------------
auto bounding_boxes(const geodetic::Box &box, const uint32_t precision)
    -> Vector<uint64_t> {
  // Grid resolution in degrees
  const auto [lng_err, lat_err] = error_with_precision(precision);

  // Allocation of the vector storing the different codes of the matrix created
  auto [hash_sw, lon_step, lat_step] = grid_properties(box, precision);
  auto result = allocate_array(lon_step * lat_step);
  auto ix = static_cast<int64_t>(0);

  auto point_sw = decode(hash_sw, precision, false);

  for (size_t lat = 0; lat < lat_step; ++lat) {
    auto point =
        geodetic::Point(0, point_sw.lat() + static_cast<double>(lat) * lat_err);

    for (size_t lon = 0; lon < lon_step; ++lon) {
      point.lon(point_sw.lon() + static_cast<double>(lon) * lng_err);
      result(ix++) = encode(point, precision);
    }
  }
  return result;
}

}  // namespace pyinterp::geohash::int64
