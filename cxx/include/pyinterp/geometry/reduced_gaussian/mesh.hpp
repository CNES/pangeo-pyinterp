// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>

#include "pyinterp/eigen.hpp"

namespace pyinterp::geometry::reduced_gaussian {

/// @brief Degrees in a full turn.
inline constexpr double kFullCircle = 360.0;

/// @brief Degrees in half a turn.
inline constexpr double kHalfCircle = 180.0;

/// @brief Number of corners of a cell.
inline constexpr size_t kCorners = 4;

/// @brief The four grid points around a query and their bilinear weights.
///
/// Corner order: bottom right, bottom left, top right, top left. "Bottom" is
/// the parallel south of the query, "left" the point at or west of it.
struct Cell {
  /// @brief Flat indices of the corners in the global mesh.
  std::array<int64_t, kCorners> index{};
  /// @brief Bilinear weights of the corners, summing to 1.
  std::array<double, kCorners> weight{};
};

/// @brief A reduced Gaussian mesh: parallels of equally spaced points, whose
/// number varies from parallel to parallel.
///
/// The mesh describes geometry only and never holds values. The points are
/// numbered parallel after parallel from the north, and on each parallel
/// eastwards from longitude 0, as GRIB stores a global `reduced_gg` field.
class Mesh {
 public:
  /// @brief Tolerance on coordinates when comparing them inside a cell.
  static constexpr double kEpsilon = 1e-6;

  /// @brief Build the mesh.
  /// @param[in] latitudes Latitudes of the parallels in degrees, strictly
  ///                      decreasing (north to south).
  /// @param[in] points_per_parallel Number of points of each parallel.
  /// @throws std::invalid_argument if the two vectors differ in size, hold
  ///         fewer than two parallels, if the latitudes are not strictly
  ///         decreasing within [-90, 90], or if a parallel has no point.
  Mesh(Vector<double> latitudes, Vector<int64_t> points_per_parallel)
      : lats_(std::move(latitudes)), pl_(std::move(points_per_parallel)) {
    if (lats_.size() != pl_.size()) {
      throw std::invalid_argument(
          "latitudes and points_per_parallel must have the same size");
    }
    if (lats_.size() < 2) {
      throw std::invalid_argument("a mesh needs at least two parallels");
    }
    for (Eigen::Index jx = 0; jx < lats_.size(); ++jx) {
      if (!std::isfinite(lats_[jx]) || lats_[jx] > 90.0 || lats_[jx] < -90.0) {
        throw std::invalid_argument("latitudes must lie within [-90, 90]");
      }
      if (jx != 0 && !(lats_[jx] < lats_[jx - 1])) {
        throw std::invalid_argument(
            "latitudes must be strictly decreasing (north to south)");
      }
      if (pl_[jx] <= 0) {
        throw std::invalid_argument(
            "every parallel must hold at least one point");
      }
    }
    offsets_.resize(pl_.size() + 1);
    offsets_[0] = 0;
    for (Eigen::Index jx = 0; jx < pl_.size(); ++jx) {
      offsets_[jx + 1] = offsets_[jx] + pl_[jx];
    }
    inv_dy_ = static_cast<double>(lats_.size() - 1) /
              (lats_[0] - lats_[lats_.size() - 1]);
    hash_ = compute_hash();
  }

  /// @brief Number of parallels.
  [[nodiscard]] auto ny() const noexcept -> int64_t { return lats_.size(); }

  /// @brief Total number of points.
  [[nodiscard]] auto size() const noexcept -> int64_t {
    return offsets_[offsets_.size() - 1];
  }

  /// @brief Latitudes of the parallels, north to south.
  [[nodiscard]] auto latitudes() const noexcept -> const Vector<double>& {
    return lats_;
  }

  /// @brief Number of points of each parallel.
  [[nodiscard]] auto points_per_parallel() const noexcept
      -> const Vector<int64_t>& {
    return pl_;
  }

  /// @brief Flat index of the first point of each parallel, plus the total
  ///        number of points at the end.
  [[nodiscard]] auto offsets() const noexcept -> const Vector<int64_t>& {
    return offsets_;
  }

  /// @brief A hash of the latitudes and of the points per parallel, so that
  ///        two meshes can be compared cheaply.
  [[nodiscard]] auto hash() const noexcept -> size_t { return hash_; }

  /// @brief Whether two meshes describe the same points.
  [[nodiscard]] auto operator==(const Mesh& other) const -> bool {
    return hash_ == other.hash_ && lats_ == other.lats_ && pl_ == other.pl_;
  }

  /// @brief Longitude of a point of a parallel, in [0, 360).
  [[nodiscard]] auto longitude(int64_t i_lon, int64_t i_parallel) const noexcept
      -> double {
    return static_cast<double>(i_lon) * (kFullCircle / pl_[i_parallel]);
  }

  /// @brief The parallel holding a flat index.
  [[nodiscard]] auto parallel_of(int64_t index) const -> int64_t {
    return static_cast<int64_t>(std::upper_bound(offsets_.data(),
                                                 offsets_.data() + ny(),
                                                 index) -
                                offsets_.data()) -
           1;
  }

  /// @brief The point at or west of a longitude on a parallel.
  ///
  /// The western knot is the last one at or west of the query where
  /// `longitude` places it, at `index * dlon`. The quotient can round up to
  /// the next knot for a query an ulp west of it -- up to the next turn for
  /// one an ulp below 360°, on some parallels and not others -- and the rows
  /// of one cell would then describe two different cells.
  /// @return The index in [0, points of the parallel).
  [[nodiscard]] auto west_of(double lon, int64_t i_parallel) const noexcept
      -> int64_t {
    const auto num_points = pl_[i_parallel];
    const double dlon = kFullCircle / static_cast<double>(num_points);
    // Within a turn, so that the quotient always fits an index: fmod is
    // exact, and a longitude of 1e300 would otherwise overflow it.
    lon = std::fmod(lon, kFullCircle);
    auto index = static_cast<int64_t>(std::floor(lon / dlon));
    if (static_cast<double>(index) * dlon > lon) {
      --index;
    }
    return ((index % num_points) + num_points) % num_points;
  }

  /// @brief The parallel north of a latitude, so that the latitude lies
  ///        between it and the next one.
  /// @return The index in [0, ny - 2], or -1 if the latitude is outside the
  ///         first and the last parallels.
  [[nodiscard]] auto parallel_north_of(double lat) const noexcept -> int64_t {
    const auto last = ny() - 1;
    if (!(lat <= lats_[0] && lat >= lats_[last])) {
      return -1;
    }
    auto ix = static_cast<int64_t>((lats_[0] - lat) * inv_dy_);
    ix = std::clamp(ix, int64_t{0}, last - 1);
    // The first guess assumes evenly spaced parallels: the Gaussian
    // latitudes are only nearly so, hence the walk to the bracketing pair.
    while (ix > 0 && lat > lats_[ix]) {
      --ix;
    }
    while (ix < last - 1 && lat < lats_[ix + 1]) {
      ++ix;
    }
    return ix;
  }

  /// @brief Find the cell around a point and its bilinear weights.
  /// @param[in] lon Longitude in degrees, in any turn.
  /// @param[in] lat Latitude in degrees.
  /// @param[out] cell The cell, valid only if the function returns `true`.
  /// @return `false` if the point lies north of the first parallel, south of
  ///         the last one, or has a non-finite coordinate.
  [[nodiscard]] auto locate(double lon, double lat, Cell& cell) const noexcept
      -> bool {
    if (!std::isfinite(lon)) {
      return false;
    }
    // One turn of the query is enough for `west_of` and `weights`, and keeps
    // the turns they add or remove small.
    lon = std::fmod(lon, kFullCircle);
    const auto y_up = parallel_north_of(lat);
    if (y_up < 0) {
      return false;
    }
    const auto y_lw = y_up + 1;
    const auto x0_bot = west_of(lon, y_lw);
    const auto x1_bot = (x0_bot + 1) % pl_[y_lw];
    const auto x0_top = west_of(lon, y_up);
    const auto x1_top = (x0_top + 1) % pl_[y_up];

    cell.index = {offsets_[y_lw] + x1_bot, offsets_[y_lw] + x0_bot,
                  offsets_[y_up] + x1_top, offsets_[y_up] + x0_top};
    cell.weight = weights(lon, lat, lats_[y_lw], lats_[y_up],
                          longitude(x1_bot, y_lw), longitude(x0_bot, y_lw),
                          longitude(x1_top, y_up), longitude(x0_top, y_up));
    return true;
  }

  /// @brief Bilinear weights on a trapezoidal cell.
  ///
  /// Accounts for the different longitude spacing of the two parallels: the
  /// cell is a trapezoid, not a rectangle.
  ///
  /// @param[in] lon Query longitude, read in the turn of the cell, so 350 and
  ///                -10 are one query.
  /// @param[in] lat Query latitude.
  /// @param[in] lat_bot Latitude of the southern parallel.
  /// @param[in] lat_top Latitude of the northern parallel.
  /// @param[in] lon_br,lon_bl,lon_tr,lon_tl Longitudes of the corners, as
  ///            `longitude` reports them.
  /// @return The weights in corner order {br, bl, tr, tl}.
  [[nodiscard]] static auto weights(double lon, double lat, double lat_bot,
                                    double lat_top, double lon_br,
                                    double lon_bl, double lon_tr,
                                    double lon_tl) noexcept
      -> std::array<double, kCorners> {
    // Degenerate: same latitude row -> equal weights
    if (std::fabs(lat_top - lat_bot) < kEpsilon) {
      constexpr double kEqualWeight = 0.25;
      return {kEqualWeight, kEqualWeight, kEqualWeight, kEqualWeight};
    }

    // Normalised latitude position [0, 1]
    const double normalized_lat =
        std::clamp((lat - lat_bot) / (lat_top - lat_bot), 0.0, 1.0);

    // Bring each bottom knot into the turn of the top knot above it: two
    // knots of one cell are less than half a turn apart once they are in one
    // turn.
    //
    // The eastern pair need it on the cell that closes a parallel, wherever
    // the two rows end at different longitudes: queried at 330° between
    // parallels of 8 and 16 points, the eastern knot of the first is its knot
    // 0, which `longitude` reports at 0°, and that of the second its last
    // knot, at 337.5°.
    //
    // The western pair never need it on a cell `locate` builds: it takes on
    // both rows the last knot at or west of the query, so the two lie within
    // a mesh step of it and of each other.
    if (std::fabs(lon_tl - lon_bl) > kHalfCircle) {
      lon_bl += (lon_bl < lon_tl) ? kFullCircle : -kFullCircle;
    }
    if (std::fabs(lon_tr - lon_br) > kHalfCircle) {
      lon_br += (lon_br < lon_tr) ? kFullCircle : -kFullCircle;
    }

    // Interpolated left/right longitude bounds at query latitude
    const double lon_left = lon_bl + (normalized_lat * (lon_tl - lon_bl));
    double lon_right = lon_br + (normalized_lat * (lon_tr - lon_br));

    // The cell runs eastwards from its left edge by less than a full turn, so
    // the right edge belongs in the turn that starts at the left one. On the
    // cell that joins the last meridian to the prime meridian the right edge
    // comes back from `longitude` as 0, a whole period *below* the left edge
    // rather than one step above it: left over as it arrived, every query in
    // the cell would clamp to a factor of 0, and the cell would return its
    // western corner instead of interpolating across the seam.
    if (lon_right < lon_left) {
      lon_right += kFullCircle;
    }

    constexpr double kDefaultFactor = 0.5;  // degenerate in longitude
    double factor = kDefaultFactor;
    if (std::fabs(lon_right - lon_left) > kEpsilon) {
      // The query in that same turn: within half a turn of the left edge. A
      // query a whole turn away is the same point read in another turn --
      // -10° against a cell starting at 340°, or 360° against a cell starting
      // at 0° -- and is moved by a turn. A query a hair west of the edge is
      // not: it is a query on the edge, which it misses only by rounding, and
      // moving it a turn east would clamp it onto the eastern knot, a mesh
      // step away.
      auto q_lon = lon;
      if (lon_left - q_lon > kHalfCircle) {
        q_lon += kFullCircle *
                 std::ceil((lon_left - q_lon - kHalfCircle) / kFullCircle);
      } else if (q_lon - lon_left > kHalfCircle) {
        q_lon -= kFullCircle *
                 std::ceil((q_lon - lon_left - kHalfCircle) / kFullCircle);
      }
      factor =
          std::clamp((q_lon - lon_left) / (lon_right - lon_left), 0.0, 1.0);
    }

    return {
        factor * (1.0 - normalized_lat),          // br
        (1.0 - factor) * (1.0 - normalized_lat),  // bl
        factor * normalized_lat,                  // tr
        (1.0 - factor) * normalized_lat,          // tl
    };
  }

 private:
  /// Latitudes of the parallels, north to south.
  Vector<double> lats_;
  /// Number of points of each parallel.
  Vector<int64_t> pl_;
  /// Prefix sums of `pl_`, with a leading 0.
  Vector<int64_t> offsets_;
  /// Inverse of the mean spacing between two parallels.
  double inv_dy_{};
  /// Hash of `lats_` and `pl_`.
  size_t hash_{};

  /// @brief Combine the hashes of the latitudes and points per parallel.
  [[nodiscard]] auto compute_hash() const -> size_t {
    auto seed = static_cast<size_t>(lats_.size());
    const auto mix = [&seed](size_t value) {
      // Boost's hash_combine
      seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
    };
    for (Eigen::Index ix = 0; ix < lats_.size(); ++ix) {
      // -0.0 and 0.0 compare equal, so they must hash alike.
      mix(std::hash<double>{}(lats_[ix] == 0.0 ? 0.0 : lats_[ix]));
      mix(std::hash<int64_t>{}(pl_[ix]));
    }
    return seed;
  }
};

}  // namespace pyinterp::geometry::reduced_gaussian
