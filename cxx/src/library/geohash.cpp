// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/geohash.hpp"

#include <stdexcept>

#include "pyinterp/geohash/int64.hpp"
#include "pyinterp/geohash/string.hpp"
#include "pyinterp/math.hpp"

namespace pyinterp {

GeoHash::GeoHash(const double longitude, const double latitude,
                 const uint32_t precision)
    : code_(precision, '\0') {
  if (precision > 12) {
    throw std::invalid_argument("GeoHash precision must be <= 12");
  }
  geohash::encode({math::normalize_period(longitude, -180.0, 360.0), latitude},
                  std::span<char>(code_.data(), precision));
}

auto GeoHash::from_string(const std::string& code, const bool round)
    -> GeoHash {
  auto precision = static_cast<uint32_t>(code.size());
  if (precision > 12) {
    throw std::invalid_argument("GeoHash precision must be <= 12");
  }
  if (!geohash::Base32().validate(
          std::span<const char>(code.data(), precision))) {
    throw std::invalid_argument("GeoHash is not valid");
  }
  auto result = GeoHash(precision);
  geohash::encode(
      geohash::decode(std::span<const char>(code.data(), precision), round),
      std::span<char>(result.code_.data(), precision));
  return result;
}

auto GeoHash::bounding_box() const -> geometry::geographic::Box {
  return geohash::bounding_box(
      std::span<const char>(code_.data(), precision()));
}

auto GeoHash::center() const -> geometry::geographic::Point {
  return bounding_box().centroid();
}

auto GeoHash::integer_value(const bool round) const -> uint64_t {
  return geohash::int64::encode(
      geohash::decode(std::span<const char>(code_.data(), precision()), round),
      number_of_bits());
}

auto GeoHash::neighbors() const -> std::vector<GeoHash> {
  auto neighbor_codes =
      geohash::neighbors(std::span<const char>(code_.data(), precision()));
  auto result = std::vector<GeoHash>();
  result.reserve(8);

  for (size_t ix = 0; ix < neighbor_codes.count; ++ix) {
    auto neighbor_span = neighbor_codes.get(ix);
    auto code = std::string(neighbor_span.begin(), neighbor_span.end());
    result.emplace_back(GeoHash::from_string(code, false));
  }
  return result;
}

auto GeoHash::area(
    const std::optional<geometry::geographic::Spheroid>& spheroid) const
    -> double {
  return geohash::area(std::span<const char>(code_.data(), precision()),
                       spheroid);
}

auto GeoHash::grid_properties(const geometry::geographic::Box& box,
                              const uint32_t precision)
    -> std::tuple<GeoHash, size_t, size_t> {
  auto [code, lng_boxes, lat_boxes] =
      geohash::int64::grid_properties(box, precision * 5);
  return std::make_tuple(
      GeoHash(geohash::int64::decode(code, precision * 5, false), precision),
      lng_boxes, lat_boxes);
}

auto GeoHash::error_with_precision(const uint32_t precision)
    -> std::tuple<double, double> {
  if (precision > 12) {
    throw std::invalid_argument("GeoHash precision must be <= 12");
  }
  return geohash::int64::error_with_precision(precision * 5);
}

auto GeoHash::getstate() const -> std::tuple<double, double, uint32_t> {
  auto point = center();
  return std::make_tuple(point.lon(), point.lat(), precision());
}

GeoHash::GeoHash(const size_t precision) : code_(precision, '\0') {}

GeoHash::GeoHash(const geometry::geographic::Point& point,
                 const uint32_t precision)
    : GeoHash(point.lon(), point.lat(), precision) {}

}  // namespace pyinterp
