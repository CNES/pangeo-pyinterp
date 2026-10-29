// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pyinterp/broadcast.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/geometry/reduced_gaussian/mesh.hpp"
#include "pyinterp/parallel_for.hpp"

namespace pyinterp::geometry::reduced_gaussian {

/// @brief The points of a mesh a regional grid holds.
///
/// A zone is a list of sorted, disjoint ranges `[start, stop)` of the flat
/// indices of the mesh. Its values are stored range after range, in the order
/// of the flat index: the order of the points in a global GRIB field, so that
/// a reader walks its file forwards. A parallel may hold several ranges: a
/// satellite pass crosses most parallels twice, on opposite sides of the
/// globe, and a single arc per parallel would then cover half of it.
///
/// The zone is computed from the mesh alone, before any value is read: the
/// caller then reads only the values of the zone.
class Zone {
 public:
  /// @brief Build a zone from its ranges.
  /// @param[in] mesh_size Number of points of the mesh.
  /// @param[in] ranges Ranges `[start, stop)` of flat indices, shape (k, 2).
  ///                   They are sorted and those that touch or overlap are
  ///                   merged.
  /// @throws std::invalid_argument if a range is empty, reversed or outside
  ///         the mesh.
  Zone(int64_t mesh_size,
       const Eigen::Ref<const Eigen::Matrix<int64_t, Eigen::Dynamic, 2,
                                            Eigen::RowMajor>>& ranges)
      : mesh_size_(mesh_size) {
    std::vector<std::pair<int64_t, int64_t>> items;
    items.reserve(static_cast<size_t>(ranges.rows()));
    for (Eigen::Index ix = 0; ix < ranges.rows(); ++ix) {
      const auto start = ranges(ix, 0);
      const auto stop = ranges(ix, 1);
      if (start < 0 || stop > mesh_size || start >= stop) {
        throw std::invalid_argument(
            "a range of the zone is empty or outside the mesh");
      }
      items.emplace_back(start, stop);
    }
    std::ranges::sort(items);
    assign(items);
  }

  /// @brief The zone a box needs: the two parallels that bracket each
  ///        latitude of the box and, on each of them, the two points that
  ///        bracket each longitude of it.
  ///
  /// This is the stencil of `Mesh::locate` counted in points of the mesh
  /// rather than in degrees: on an N640 mesh the points of a parallel are
  /// 0.14° apart at the equator and 20° on the first parallel, so no margin
  /// in degrees is right for both.
  ///
  /// @param[in] mesh The mesh.
  /// @param[in] lon_min,lat_min,lon_max,lat_max The box. Its longitudes may
  ///            lie outside [0, 360): a box across the Greenwich meridian is
  ///            given unwrapped, as [-10, 10].
  [[nodiscard]] static auto from_box(const Mesh& mesh, double lon_min,
                                     double lat_min, double lon_max,
                                     double lat_max) -> Zone {
    if (!std::isfinite(lon_min) || !std::isfinite(lon_max) ||
        !std::isfinite(lat_min) || !std::isfinite(lat_max)) {
      throw std::invalid_argument("the box must have finite coordinates");
    }
    if (!(lon_min <= lon_max) || !(lat_min <= lat_max)) {
      throw std::invalid_argument("the box is empty");
    }
    // Move the box into the first turns, keeping its width (at most a turn),
    // so that the quotients below always fit an index.
    lon_max = std::fmod(lon_min, kFullCircle) +
              std::min(lon_max - lon_min, kFullCircle);
    lon_min = std::fmod(lon_min, kFullCircle);
    const auto& lats = mesh.latitudes();
    const auto& pl = mesh.points_per_parallel();
    const auto& offsets = mesh.offsets();
    const auto count = mesh.ny();
    // The last parallel north of the box, and the first south of it: a
    // latitude that lies on a parallel may be given either of its two cells.
    int64_t top = 0;
    int64_t bottom = count - 1;
    for (int64_t jx = 0; jx < count; ++jx) {
      if (lats[jx] > lat_max) {
        top = jx;
      }
      if (lats[jx] < lat_min) {
        bottom = jx;
        break;
      }
    }
    std::vector<std::pair<int64_t, int64_t>> items;
    for (int64_t jx = top; jx <= bottom; ++jx) {
      const auto points = pl[jx];
      const double step = kFullCircle / static_cast<double>(points);
      // The point at or west of the western edge, as `Mesh::west_of` finds
      // it: the quotient can round up to the next point for an edge an ulp
      // west of it.
      auto first = static_cast<int64_t>(std::floor(lon_min / step));
      if (static_cast<double>(first) * step > lon_min) {
        --first;
      }
      // The point east of the eastern edge.
      const auto last = static_cast<int64_t>(std::floor(lon_max / step)) + 1;
      const auto span = std::min(last - first + 1, points);
      const auto start = ((first % points) + points) % points;
      const auto base = offsets[jx];
      if (start + span <= points) {
        items.emplace_back(base + start, base + start + span);
      } else {
        // The span crosses the last meridian: two ranges.
        items.emplace_back(base, base + start + span - points);
        items.emplace_back(base + start, base + points);
      }
    }
    std::ranges::sort(items);
    return {mesh.size(), items};
  }

  /// @brief The zone holding exactly the corners the points read.
  /// @param[in] mesh The mesh.
  /// @param[in] lon,lat Coordinates of the points, in degrees. Points outside
  ///            the mesh need no corner and are ignored.
  /// @param[in] num_threads Number of threads to locate the points with.
  [[nodiscard]] static auto from_points(const Mesh& mesh,
                                        const ConstVectorRef<double>& lon,
                                        const ConstVectorRef<double>& lat,
                                        size_t num_threads) -> Zone {
    broadcast::check_eigen_shape("lon", lon, "lat", lat);
    const auto n = lon.size();
    std::vector<int64_t> indices(static_cast<size_t>(n) * kCorners, -1);
    parallel_for(
        n,
        [&](int64_t start, int64_t end) {
          Cell cell;
          for (auto ix = start; ix < end; ++ix) {
            if (mesh.locate(lon[ix], lat[ix], cell)) {
              std::ranges::copy(
                  cell.index,
                  indices.begin() + static_cast<std::ptrdiff_t>(ix * kCorners));
            }
          }
        },
        num_threads);
    std::ranges::sort(indices);
    const auto [first, last] = std::ranges::unique(indices);
    indices.erase(first, last);
    // Runs of consecutive indices become ranges; -1 (unlocated) sorts first.
    std::vector<std::pair<int64_t, int64_t>> items;
    for (const auto index : indices) {
      if (index < 0) {
        continue;
      }
      if (!items.empty() && items.back().second == index) {
        ++items.back().second;
      } else {
        items.emplace_back(index, index + 1);
      }
    }
    return {mesh.size(), items};
  }

  /// @brief Number of points of the mesh the zone was built for.
  [[nodiscard]] auto mesh_size() const noexcept -> int64_t {
    return mesh_size_;
  }

  /// @brief Number of points held.
  [[nodiscard]] auto size() const noexcept -> int64_t { return size_; }

  /// @brief Number of ranges.
  [[nodiscard]] auto num_ranges() const noexcept -> int64_t {
    return static_cast<int64_t>(starts_.size());
  }

  /// @brief The ranges `[start, stop)`, shape (k, 2), sorted and disjoint.
  [[nodiscard]] auto ranges() const
      -> Eigen::Matrix<int64_t, Eigen::Dynamic, 2, Eigen::RowMajor> {
    Eigen::Matrix<int64_t, Eigen::Dynamic, 2, Eigen::RowMajor> result(
        num_ranges(), 2);
    for (int64_t ix = 0; ix < num_ranges(); ++ix) {
      result(ix, 0) = starts_[ix];
      result(ix, 1) = stops_[ix];
    }
    return result;
  }

  /// @brief The flat indices of the points held, in storage order.
  [[nodiscard]] auto indices() const -> Vector<int64_t> {
    Vector<int64_t> result(size_);
    int64_t row = 0;
    for (int64_t ix = 0; ix < num_ranges(); ++ix) {
      for (auto index = starts_[ix]; index < stops_[ix]; ++index) {
        result[row++] = index;
      }
    }
    return result;
  }

  /// @brief The row of a point in the values of the zone.
  /// @param[in] index Flat index of the point in the mesh.
  /// @return The row, or -1 if the zone does not hold the point.
  [[nodiscard]] auto row(int64_t index) const noexcept -> int64_t {
    const auto it = std::ranges::upper_bound(starts_, index);
    if (it == starts_.begin()) {
      return -1;
    }
    const auto k = static_cast<size_t>(it - starts_.begin()) - 1;
    return index < stops_[k] ? rows_[k] + (index - starts_[k]) : -1;
  }

  /// @brief Whether two zones hold the same points.
  [[nodiscard]] auto operator==(const Zone& other) const -> bool = default;

 private:
  int64_t mesh_size_{};
  int64_t size_{};
  std::vector<int64_t> starts_;
  std::vector<int64_t> stops_;
  /// Row of the first point of each range.
  std::vector<int64_t> rows_;

  /// @brief Build from sorted ranges, merging those that touch or overlap.
  Zone(int64_t mesh_size,
       const std::vector<std::pair<int64_t, int64_t>>& sorted)
      : mesh_size_(mesh_size) {
    assign(sorted);
  }

  void assign(const std::vector<std::pair<int64_t, int64_t>>& sorted) {
    for (const auto& [start, stop] : sorted) {
      if (!stops_.empty() && start <= stops_.back()) {
        stops_.back() = std::max(stops_.back(), stop);
      } else {
        starts_.push_back(start);
        stops_.push_back(stop);
      }
    }
    rows_.resize(starts_.size());
    size_ = 0;
    for (size_t ix = 0; ix < starts_.size(); ++ix) {
      rows_[ix] = size_;
      size_ += stops_[ix] - starts_[ix];
    }
  }
};

}  // namespace pyinterp::geometry::reduced_gaussian
