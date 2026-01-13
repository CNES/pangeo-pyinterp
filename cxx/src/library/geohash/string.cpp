// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geohash/string.hpp"

#include <algorithm>
#include <boost/geometry.hpp>
#include <ranges>
#include <string>
#include <unordered_set>
#include <utility>

#include "pyinterp/broadcast.hpp"
#include "pyinterp/geohash/base32.hpp"
#include "pyinterp/geometry/geographic/box.hpp"
#include "pyinterp/math.hpp"

namespace pyinterp::geohash {

static Base32 encoder{};

auto encode(const Eigen::Ref<const Eigen::VectorXd>& lon,
            const Eigen::Ref<const Eigen::VectorXd>& lat, uint32_t precision)
    -> EncodedHashes {
  broadcast::check_eigen_shape("lon", lon, "lat", lat);

  auto size = lon.size();
  auto result = EncodedHashes{
      .buffer = std::vector<char>(size * precision),
      .precision = precision,
      .count = static_cast<size_t>(size),
  };

  for (auto [lon_item, lat_item, hash_span] :
       std::views::zip(lon, lat, result)) {
    encode({math::normalize_period(lon_item, -180.0, 360.0), lat_item},
           hash_span);
  }
  return result;
}

// ============================================================================

auto bounding_box(std::span<const char> geohash, uint32_t* precision)
    -> geometry::geographic::Box {
  auto [integer_encoded, chars] = encoder.decode(geohash);
  if (precision != nullptr) {
    *precision = chars;
  }
  return int64::bounding_box(integer_encoded, 5 * chars);
}

// ============================================================================

auto neighbors(std::span<const char> hash) -> EncodedHashes {
  auto [integer_encoded, precision] = encoder.decode(hash);

  const auto integers = int64::neighbors(integer_encoded, precision * 5);
  auto result = EncodedHashes{
      .buffer = std::vector<char>(integers.size() * precision),
      .precision = precision,
      .count = integers.size(),
  };

  for (auto [integer, hash_span] : std::views::zip(integers, result)) {
    encoder.encode(integer, hash_span);
  }
  return result;
}

// ============================================================================

template <typename Geometry>
auto bounding_boxes_impl(const Geometry& geometry, uint32_t precision,
                         size_t num_threads) -> EncodedHashes {
  // Delegate heavy computation to int64 implementation
  auto bits = precision * 5;
  auto int64_hashes = int64::bounding_boxes(geometry, bits, num_threads);

  // Convert int64 hashes to base32 strings
  auto result = EncodedHashes{
      .buffer = std::vector<char>(int64_hashes.size() * precision),
      .precision = precision,
      .count = static_cast<size_t>(int64_hashes.size()),
  };

  for (int64_t ix = 0; ix < int64_hashes.size(); ++ix) {
    Base32::encode(int64_hashes(ix), result.get(static_cast<size_t>(ix)));
  }
  return result;
}

// ============================================================================

auto bounding_boxes(const std::optional<geometry::geographic::Box>& box,
                    const uint32_t precision) -> EncodedHashes {
  return bounding_boxes_impl(
      box.value_or(geometry::geographic::Box::global_bounding_box()), precision,
      1);
}

// ============================================================================

auto bounding_boxes(const geometry::geographic::Polygon& polygon,
                    uint32_t precision, size_t num_threads) -> EncodedHashes {
  return bounding_boxes_impl(polygon, precision, num_threads);
}

// ============================================================================

auto bounding_boxes(const geometry::geographic::MultiPolygon& multipolygon,
                    uint32_t precision, size_t num_threads) -> EncodedHashes {
  return bounding_boxes_impl(multipolygon, precision, num_threads);
}

// ============================================================================

template <typename HashContainer>
auto where_impl(const HashContainer& hash, size_t rows, size_t cols)
    -> HashRegionBounds {
  // Index shifts of neighboring pixels
  static constexpr auto shift_row =
      std::array<int64_t, 8>{-1, -1, -1, 0, 1, 0, 1, 1};
  static constexpr auto shift_col =
      std::array<int64_t, 8>{-1, 1, 0, -1, -1, 1, 0, 1};

  auto result = std::unordered_map<
      std::string,
      std::tuple<std::tuple<int64_t, int64_t>, std::tuple<int64_t, int64_t>>>();

  for (int64_t ix = 0; std::cmp_less(ix, rows); ++ix) {
    for (int64_t jx = 0; std::cmp_less(jx, cols); ++jx) {
      auto current_span = hash.get(ix * cols + jx);
      auto current_code = std::string(current_span.begin(), current_span.end());

      auto it = result.find(current_code);
      if (it == result.end()) {
        result.emplace(current_code, std::make_tuple(std::make_tuple(ix, ix),
                                                     std::make_tuple(jx, jx)));
        continue;
      }

      for (int64_t kx = 0; kx < 8; ++kx) {
        const auto i = ix + shift_row[kx];
        const auto j = jx + shift_col[kx];

        if (i >= 0 && std::cmp_less(i, rows) && j >= 0 &&
            std::cmp_less(j, cols)) {
          auto neighboring_span = hash.get(i * cols + j);
          auto neighboring_code =
              std::string(neighboring_span.begin(), neighboring_span.end());

          if (current_code == neighboring_code) {
            auto& first = std::get<0>(it->second);
            std::get<0>(first) = std::min(std::get<0>(first), i);
            std::get<1>(first) = std::max(std::get<1>(first), i);

            auto& second = std::get<1>(it->second);
            std::get<0>(second) = std::min(std::get<0>(second), j);
            std::get<1>(second) = std::max(std::get<1>(second), j);
          }
        }
      }
    }
  }

  return result;
}

// ============================================================================

// Zoom in from lower to higher precision
template <typename HashContainer>
auto zoom_in(const HashContainer& hash, uint32_t to_precision)
    -> EncodedHashes {
  // Number of bits need to zoom in
  auto bits = to_precision * 5;

  // Calculation of the number of items needed for the result.
  auto size_in = hash.count * (static_cast<size_t>(2)
                               << (5 * (to_precision - hash.precision) - 1));

  // Allocates the result
  auto result = EncodedHashes{
      .buffer = std::vector<char>(size_in * to_precision),
      .precision = to_precision,
      .count = size_in,
  };

  size_t result_ix = 0;
  for (const auto& hash_span : hash) {
    auto bbox = bounding_box(hash_span);
    auto codes = int64::bounding_boxes(bbox, bits);
    for (uint64_t code : codes) {
      Base32::encode(code, result.get(result_ix++));
    }
  }

  return result;
}

// ============================================================================

// Zoom out from higher to lower precision
template <typename HashContainer>
auto zoom_out(const HashContainer& hash, uint32_t to_precision)
    -> EncodedHashes {
  // Use unordered_set for O(1) insertions instead of O(log n) with std::set
  auto zoom_out_codes = std::unordered_set<uint64_t>();
  zoom_out_codes.reserve(hash.count);  // Reserve space to avoid rehashing

  auto to_bits = to_precision * 5;
  auto from_bits = hash.precision * 5;
  auto shift = from_bits - to_bits;

  for (const auto& hash_span : hash) {
    // Decode to integer, shift right to reduce precision, and collect unique
    // values
    auto [integer_hash, chars] = encoder.decode(hash_span);
    auto zoomed_out = integer_hash >> shift;
    zoom_out_codes.insert(zoomed_out);
  }

  auto result = EncodedHashes{
      .buffer = std::vector<char>(zoom_out_codes.size() * to_precision),
      .precision = to_precision,
      .count = zoom_out_codes.size(),
  };

  size_t ix = 0;
  for (auto code : zoom_out_codes) {
    encoder.encode(code, result.get(ix++));
  }

  return result;
}

// ============================================================================

template <typename HashContainer>
auto transform_impl(const HashContainer& hash, uint32_t precision)
    -> EncodedHashes {
  if (hash.precision > precision) {
    return zoom_out(hash, precision);
  }
  return zoom_in(hash, precision);
}

// ============================================================================

auto transform(const EncodedHashesView& hash, uint32_t precision)
    -> EncodedHashes {
  if (hash.precision == precision) {
    // No transformation needed, create a copy
    return EncodedHashes{
        .buffer = std::vector<char>(hash.data,
                                    hash.data + hash.count * hash.precision),
        .precision = hash.precision,
        .count = hash.count,
    };
  }
  return transform_impl(hash, precision);
}

// ============================================================================

auto transform(const EncodedHashes& hash, uint32_t precision) -> EncodedHashes {
  if (hash.precision == precision) {
    // No transformation needed, return the original
    return hash;
  }
  return transform_impl(hash, precision);
}

// ============================================================================

auto where(const EncodedHashes& hash, size_t rows, size_t cols)
    -> HashRegionBounds {
  return where_impl(hash, rows, cols);
}

// ============================================================================

auto where(const EncodedHashesView& hash, size_t rows, size_t cols)
    -> HashRegionBounds {
  return where_impl(hash, rows, cols);
}

}  // namespace pyinterp::geohash
