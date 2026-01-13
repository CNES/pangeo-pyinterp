// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Eigen/Core"
#include "pyinterp/geohash/base32.hpp"
#include "pyinterp/geohash/int64.hpp"
#include "pyinterp/geometry/geographic/box.hpp"
#include "pyinterp/geometry/geographic/multi_polygon.hpp"
#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/geometry/geographic/polygon.hpp"

namespace pyinterp::geohash {

/// @brief Result of encoding: geohash character buffer + precision metadata.
///
/// `EncodedHashes` stores a flat character buffer containing a sequence of
/// fixed-size geohash encoding strings and associated metadata describing
/// the precision (characters per geohash) and number of geohashes encoded.
struct EncodedHashes {
  /// @brief Flat buffer containing `count * precision` geohash characters.
  std::vector<char> buffer;

  /// @brief Number of characters used to encode each geohash (precision).
  uint32_t precision;

  /// @brief Number of geohashes stored in `buffer`.
  size_t count;

  /// @brief Pointer to the underlying geohash character data (mutable).
  [[nodiscard]] constexpr auto data() noexcept -> char* {
    return buffer.data();
  }

  /// @brief Pointer to the underlying geohash character data (const).
  [[nodiscard]] constexpr auto data() const noexcept -> const char* {
    return buffer.data();
  }

  /// @brief Get the `i`-th geohash as a `std::span<const char>`.
  /// @param i Index of the geohash.
  /// @returns A view over the characters representing the `i`-th geohash.
  [[nodiscard]] constexpr auto get(size_t ix) const noexcept
      -> std::span<const char> {
    return {buffer.data() + ix * precision, precision};
  }

  /// @brief Get the `i`-th geohash as a mutable `std::span<char>`.
  [[nodiscard]] constexpr auto get(size_t ix) noexcept -> std::span<char> {
    return {buffer.data() + ix * precision, precision};
  }

  // Iterator support for ranges

  /// @brief Random-access iterator over encoded geohashes.
  ///
  /// Dereferencing returns a `std::span<char>` pointing to the characters
  /// of the current geohash (length = `precision`). Incrementing advances
  /// by `precision` bytes.
  struct iterator {
    char* ptr;
    uint32_t precision;

    using iterator_category = std::random_access_iterator_tag;
    using value_type = std::span<char>;
    using difference_type = std::ptrdiff_t;
    using reference = std::span<char>;

    /// @brief Return a mutable span referring to the current geohash.
    constexpr auto operator*() const noexcept -> std::span<char> {
      return {ptr, precision};
    }

    /// @brief Random access to the `n`-th geohash relative to this iterator.
    constexpr auto operator[](difference_type n) const noexcept
        -> std::span<char> {
      return {ptr + n * precision, precision};
    }

    constexpr auto operator++() noexcept -> iterator& {
      ptr += precision;
      return *this;
    }

    constexpr auto operator++(int) noexcept -> iterator {
      auto tmp = *this;
      ++(*this);
      return tmp;
    }

    constexpr auto operator--() noexcept -> iterator& {
      ptr -= precision;
      return *this;
    }

    constexpr auto operator--(int) noexcept -> iterator {
      auto tmp = *this;
      --(*this);
      return tmp;
    }

    constexpr auto operator+=(difference_type n) noexcept -> iterator& {
      ptr += n * precision;
      return *this;
    }

    constexpr auto operator-=(difference_type n) noexcept -> iterator& {
      ptr -= n * precision;
      return *this;
    }

    friend constexpr auto operator+(iterator it, difference_type n) noexcept
        -> iterator {
      return {.ptr = it.ptr + n * it.precision, .precision = it.precision};
    }

    friend constexpr auto operator+(difference_type n, iterator it) noexcept
        -> iterator {
      return it + n;
    }

    friend constexpr auto operator-(iterator it, difference_type n) noexcept
        -> iterator {
      return {.ptr = it.ptr - n * it.precision, .precision = it.precision};
    }

    friend constexpr auto operator-(iterator a, iterator b) noexcept
        -> difference_type {
      return (a.ptr - b.ptr) / static_cast<difference_type>(a.precision);
    }

    constexpr auto operator<=>(const iterator&) const noexcept = default;
  };

  /// @brief Random-access const iterator over encoded geohashes.
  ///
  /// Behaves like `iterator` but returns `std::span<const char>`.
  struct const_iterator {
    const char* ptr;
    uint32_t precision;

    using iterator_category = std::random_access_iterator_tag;
    using value_type = std::span<const char>;
    using difference_type = std::ptrdiff_t;
    using reference = std::span<const char>;

    /// @brief Return a const span referring to the current geohash.
    constexpr auto operator*() const noexcept -> std::span<const char> {
      return {ptr, precision};
    }

    /// @brief Random access to the `n`-th geohash relative to this iterator.
    constexpr auto operator[](difference_type n) const noexcept
        -> std::span<const char> {
      return {ptr + n * precision, precision};
    }

    constexpr auto operator++() noexcept -> const_iterator& {
      ptr += precision;
      return *this;
    }

    constexpr auto operator++(int) noexcept -> const_iterator {
      auto tmp = *this;
      ++(*this);
      return tmp;
    }

    constexpr auto operator--() noexcept -> const_iterator& {
      ptr -= precision;
      return *this;
    }

    constexpr auto operator--(int) noexcept -> const_iterator {
      auto tmp = *this;
      --(*this);
      return tmp;
    }

    constexpr auto operator+=(difference_type n) noexcept -> const_iterator& {
      ptr += n * precision;
      return *this;
    }

    constexpr auto operator-=(difference_type n) noexcept -> const_iterator& {
      ptr -= n * precision;
      return *this;
    }

    friend constexpr auto operator+(const_iterator it,
                                    difference_type n) noexcept
        -> const_iterator {
      return {.ptr = it.ptr + n * it.precision, .precision = it.precision};
    }

    friend constexpr auto operator+(difference_type n,
                                    const_iterator it) noexcept
        -> const_iterator {
      return it + n;
    }

    friend constexpr auto operator-(const_iterator it,
                                    difference_type n) noexcept
        -> const_iterator {
      return {.ptr = it.ptr - n * it.precision, .precision = it.precision};
    }

    friend constexpr auto operator-(const_iterator a, const_iterator b) noexcept
        -> difference_type {
      return (a.ptr - b.ptr) / static_cast<difference_type>(a.precision);
    }

    constexpr auto operator<=>(const const_iterator&) const noexcept = default;
  };

  /// @brief Return iterator to the first encoded geohash (mutable).
  [[nodiscard]] constexpr auto begin() noexcept -> iterator {
    return {.ptr = buffer.data(), .precision = precision};
  }

  /// @brief Return iterator to past-the-end (mutable).
  [[nodiscard]] constexpr auto end() noexcept -> iterator {
    return {.ptr = buffer.data() + count * precision, .precision = precision};
  }

  /// @brief Return const iterator to the first encoded geohash.
  [[nodiscard]] constexpr auto begin() const noexcept -> const_iterator {
    return {.ptr = buffer.data(), .precision = precision};
  }

  /// @brief Return const iterator to past-the-end.
  [[nodiscard]] constexpr auto end() const noexcept -> const_iterator {
    return {.ptr = buffer.data() + count * precision, .precision = precision};
  }

  /// @brief Return const iterator to the first encoded geohash.
  [[nodiscard]] constexpr auto cbegin() const noexcept -> const_iterator {
    return begin();
  }

  /// @brief Return const iterator to past-the-end.
  [[nodiscard]] constexpr auto cend() const noexcept -> const_iterator {
    return end();
  }
};

/// @brief Non-owning view over encoded geohash data.
///
/// `EncodedHashesView` provides a read-only view into geohash data
/// without owning the underlying buffer. This is useful for zero-copy
/// interoperability with external data sources like numpy arrays.
struct EncodedHashesView {
  /// @brief Pointer to the geohash character data (non-owning).
  const char* data;

  /// @brief Number of characters used to encode each geohash (precision).
  uint32_t precision;

  /// @brief Number of geohashes in the view.
  size_t count;

  /// @brief Get the `i`-th geohash as a `std::span<const char>`.
  /// @param ix Index of the geohash.
  /// @returns A view over the characters representing the `i`-th geohash.
  [[nodiscard]] constexpr auto get(size_t ix) const noexcept
      -> std::span<const char> {
    return {data + ix * precision, precision};
  }

  /// @brief Reuse const_iterator from EncodedHashes for iteration.
  using const_iterator = EncodedHashes::const_iterator;

  /// @brief Return const iterator to the first encoded geohash.
  [[nodiscard]] constexpr auto begin() const noexcept -> const_iterator {
    return {.ptr = data, .precision = precision};
  }

  /// @brief Return const iterator to past-the-end.
  [[nodiscard]] constexpr auto end() const noexcept -> const_iterator {
    return {.ptr = data + count * precision, .precision = precision};
  }

  /// @brief Return const iterator to the first encoded geohash.
  [[nodiscard]] constexpr auto cbegin() const noexcept -> const_iterator {
    return begin();
  }

  /// @brief Return const iterator to past-the-end.
  [[nodiscard]] constexpr auto cend() const noexcept -> const_iterator {
    return end();
  }
};

/// @brief Encode a geographic point into a geohash string.
///
/// Encodes `point` into a geohash string written into `buffer`.
/// The geohash `precision` (characters per hash) is equal to `5 *
/// buffer.size()`.
///
/// @param point Geodetic point (longitude, latitude, altitude).
/// @param buffer Character buffer that will receive the encoded geohash
///               (length equals the desired precision).
inline auto encode(const geometry::geographic::Point& point,
                   std::span<char> buffer) -> void {
  const auto precision = static_cast<uint32_t>(5 * buffer.size());
  Base32::encode(int64::encode(point, precision), buffer);
}

/// @brief Encode a set of coordinates into geohash strings
/// @param lon Vector of longitudes.
/// @param lat Vector of latitudes.
/// @param precision Number of characters for each geohash string.
/// @returns EncodedHashes structure containing the geohash strings and
/// metadata.
[[nodiscard]] auto encode(const Eigen::Ref<const Eigen::VectorXd>& lon,
                          const Eigen::Ref<const Eigen::VectorXd>& lat,
                          uint32_t precision) -> EncodedHashes;

/// @brief Compute the bounding box of a geohash string
/// @param[in] geohash Geohash string as a span of characters.
/// @param[in] precision Optional pointer to store the precision (number of
/// characters) of the geohash.
/// @returns Bounding box corresponding to the geohash.
[[nodiscard]] auto bounding_box(std::span<const char> geohash,
                                uint32_t* precision = nullptr)
    -> geometry::geographic::Box;

/// @brief Decode a geohash string into a geographic point.
/// @param[in] hash Geohash string as a span of characters.
/// @param[in] round If `true`, the decoded point is rounded to the center
/// of the bounding box represented by the geohash.
/// @returns Decoded geographic point (longitude, latitude).
[[nodiscard]] inline auto decode(std::span<const char> hash, bool round)
    -> geometry::geographic::Point {
  auto bbox = bounding_box(hash);
  return round ? bbox.round() : bbox.centroid();
}

/// @brief Decode a set of geohash strings into geographic points.
/// @param[in] hash EncodedHashes structure containing geohash strings.
/// @param[in] round If `true`, the decoded points are rounded to the center
/// of the bounding box represented by each geohash.
/// @returns Tuple of (longitudes, latitudes) as Eigen vectors.
[[nodiscard]] inline auto decode(const EncodedHashes& hash, bool round)
    -> std::tuple<Eigen::VectorXd, Eigen::VectorXd> {
  Eigen::VectorXd lon(hash.count);
  Eigen::VectorXd lat(hash.count);

  for (auto [lon_item, lat_item, hash_span] : std::views::zip(lon, lat, hash)) {
    auto point = decode(hash_span, round);
    lon_item = point.lon();
    lat_item = point.lat();
  }
  return {lon, lat};
}

/// @brief Decode a set of geohash strings into geographic points (view
/// overload).
/// @param[in] hash EncodedHashesView providing a non-owning view over geohash
/// strings.
/// @param[in] round If `true`, the decoded points are rounded to the center
/// of the bounding box represented by each geohash.
/// @returns Tuple of (longitudes, latitudes) as Eigen vectors.
[[nodiscard]] inline auto decode(const EncodedHashesView& hash, bool round)
    -> std::tuple<Eigen::VectorXd, Eigen::VectorXd> {
  Eigen::VectorXd lon(hash.count);
  Eigen::VectorXd lat(hash.count);

  for (auto [lon_item, lat_item, hash_span] : std::views::zip(lon, lat, hash)) {
    auto point = decode(hash_span, round);
    lon_item = point.lon();
    lat_item = point.lat();
  }
  return {lon, lat};
}

/// @brief Returns the 8 neighboring geohashes in clockwise order starting at
/// north.
///
/// The neighbors are ordered as follows (indexes shown):
///   7 0 1
///   6 x 2
///   5 4 3
///
/// @param[in] hash Geohash string as a span of characters.
/// @returns An `EncodedHashes` containing 8 neighbors in the order 0..7
/// (north, northeast, east, southeast, south, southwest, west, northwest).
[[nodiscard]] auto neighbors(std::span<const char> hash) -> EncodedHashes;

/// @brief Returns the area covered by the geohash.
///
/// @param[in] hash Geohash string as a span of characters.
/// @param[in] spheroid Optional spheroid to compute ellipsoidal area; if not
/// provided, a default spheroid is used.
/// @returns Area covered by the geohash in square meters.
[[nodiscard]] inline auto area(
    std::span<const char> hash,
    const std::optional<geometry::geographic::Spheroid>& spheroid) -> double {
  return geometry::geographic::area<
      geometry::geographic::Box,
      geometry::geographic::StrategyMethod::kVincenty>(bounding_box(hash),
                                                       spheroid);
}

/// @brief Compute area covered by each geohash in `hash`.
///
/// @param[in] hash EncodedHashes structure containing geohash strings and
/// metadata (precision and count).
/// @param[in] spheroid Optional spheroid used for ellipsoidal area
/// computation. If not provided, a default spheroid is used.
/// @returns Areas in square meters for each geohash in `hash`.
[[nodiscard]] inline auto area(
    const EncodedHashes& hash,
    const std::optional<geometry::geographic::Spheroid>& spheroid)
    -> Eigen::VectorXd {
  Eigen::VectorXd areas(hash.count);
  for (auto [area_item, hash_span] : std::views::zip(areas, hash)) {
    area_item = area(hash_span, spheroid);
  }
  return areas;
}

/// @brief Compute area covered by each geohash in `hash` (view overload).
///
/// @param[in] hash EncodedHashesView providing a non-owning view over geohash
/// strings.
/// @param[in] spheroid Optional spheroid used for ellipsoidal area
/// computation. If not provided, a default spheroid is used.
/// @returns Areas in square meters for each geohash in `hash`.
[[nodiscard]] inline auto area(
    const EncodedHashesView& hash,
    const std::optional<geometry::geographic::Spheroid>& spheroid)
    -> Eigen::VectorXd {
  Eigen::VectorXd areas(hash.count);
  for (auto [area_item, hash_span] : std::views::zip(areas, hash)) {
    area_item = area(hash_span, spheroid);
  }
  return areas;
}

/// Returns all GeoHash within the given region
/// @brief Generate geohashes representing bounding boxes for a geographic
/// region
/// @param box Optional geodetic bounding box defining the geographic region.
/// If not provided, generates hashes for the entire globe.
/// @param precision Geohash precision level (higher precision yields smaller
/// cells)
/// @return EncodedHashes containing the geohashes of bounding boxes at the
/// specified precision
/// @note The return value should be used, as indicated by [[nodiscard]]
[[nodiscard]] auto bounding_boxes(
    const std::optional<geometry::geographic::Box>& box,
    const uint32_t precision) -> EncodedHashes;

/// @brief Generate geohashes for bounding boxes that intersect with a polygon
/// @param polygon Geodetic polygon defining the region of interest
/// @param precision Geohash precision level (higher precision yields smaller
/// cells)
/// @param num_threads Number of threads to use for parallel computation (0 for
/// hardware concurrency)
/// @return EncodedHashes containing the geohashes that intersect with the
/// polygon
[[nodiscard]] auto bounding_boxes(const geometry::geographic::Polygon& polygon,
                                  uint32_t precision, size_t num_threads = 0)
    -> EncodedHashes;

/// @brief Generate geohashes for bounding boxes that intersect with a
/// multipolygon
/// @param multipolygon Geodetic multipolygon defining the region of interest
/// @param precision Geohash precision level (higher precision yields smaller
/// cells)
/// @param num_threads Number of threads to use for parallel computation (0 for
/// hardware concurrency)
/// @return EncodedHashes containing the geohashes that intersect with the
/// multipolygon
[[nodiscard]] auto bounding_boxes(
    const geometry::geographic::MultiPolygon& multipolygon, uint32_t precision,
    size_t num_threads = 0) -> EncodedHashes;

/// @brief Type alias for bounding region of geohash areas
using HashRegionBounds =
    std::unordered_map<std::string, std::tuple<std::tuple<int64_t, int64_t>,
                                               std::tuple<int64_t, int64_t>>>;

/// @brief Find bounding regions for contiguous geohash areas in a 2D grid
/// @param hash EncodedHashes containing a 2D grid of geohashes with shape
/// (rows, cols)
/// @param rows Number of rows in the grid
/// @param cols Number of columns in the grid
/// @return Map from geohash string to tuple of ((min_row, max_row), (min_col,
/// max_col))
[[nodiscard]] auto where(const EncodedHashes& hash, size_t rows, size_t cols)
    -> HashRegionBounds;

/// @brief Find bounding regions for contiguous geohash areas in a 2D grid
/// (view overload)
/// @param hash EncodedHashesView providing a non-owning view over geohash
/// strings
/// @param rows Number of rows in the grid
/// @param cols Number of columns in the grid
/// @return Map from geohash string to tuple of ((min_row, max_row), (min_col,
/// max_col))
[[nodiscard]] auto where(const EncodedHashesView& hash, size_t rows,
                         size_t cols) -> HashRegionBounds;

/// @brief Transform geohashes to a different precision level
/// @param hash EncodedHashes to transform
/// @param precision Target precision level
/// @return EncodedHashes at the target precision (zoomed in or out as needed)
[[nodiscard]] auto transform(const EncodedHashes& hash, uint32_t precision)
    -> EncodedHashes;

/// @brief Transform geohashes to a different precision level (view overload)
/// @param hash EncodedHashesView providing a non-owning view over geohash
/// strings
/// @param precision Target precision level
/// @return EncodedHashes at the target precision (zoomed in or out as needed)
[[nodiscard]] auto transform(const EncodedHashesView& hash, uint32_t precision)
    -> EncodedHashes;

}  // namespace pyinterp::geohash
