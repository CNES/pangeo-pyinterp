// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

#include "pyinterp/eigen.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"
#include "pyinterp/math.hpp"
#include "pyinterp/parallel_for.hpp"

/// First order conservative regridding between rectilinear grids
///
/// The cells of a rectilinear grid are the products of the cells of its two
/// axes. When the measure of a cell is the product of a measure along each
/// axis, which holds for longitude/latitude boxes on the sphere (the area of
/// [λ1, λ2] × [φ1, φ2] is R²·(λ2 − λ1)·(sin φ2 − sin φ1)) and on the ellipsoid
/// (the same formula with the authalic latitude and the authalic radius), the
/// area of the intersection of two cells is the product of the overlaps of
/// their sides. The regridding weights are then the tensor product of two
/// sparse 1D overlap matrices, applied along each axis in turn without
/// building the full 2D weight matrix.
namespace pyinterp::regrid {

/// @brief Sparse matrix, in the CSR format, of the overlaps between the cells
/// of a target axis (rows) and of a source axis (columns)
///
/// A weight is the length of the overlap divided by the length of the target
/// cell, so that the sum of a row is the fraction of the target cell covered
/// by the source axis.
struct OverlapMatrix {
  std::vector<int64_t> indptr{0};  ///< Start of each row, size rows + 1
  std::vector<int64_t> indices;    ///< Index of the source cells
  std::vector<double> weights;     ///< Overlap / length of the target cell

  /// @brief Number of target cells
  [[nodiscard]] auto rows() const noexcept -> int64_t {
    return static_cast<int64_t>(indptr.size()) - 1;
  }

  /// @brief Sum of the weights of each row
  [[nodiscard]] auto row_sums() const -> Vector<double> {
    Vector<double> result(rows());
    for (int64_t ix = 0; ix < rows(); ++ix) {
      result[ix] = std::accumulate(weights.begin() + indptr[ix],
                                   weights.begin() + indptr[ix + 1], 0.0);
    }
    return result;
  }
};

/// @brief Bounds of the cells of an axis, one row (lower, upper) per cell
using Bounds = Eigen::Ref<const RowMajorMatrix<double>>;

namespace detail {

/// @brief Cells of the source axis sorted by increasing coordinates
struct SortedCells {
  std::vector<double> lower;
  std::vector<double> upper;
  std::vector<int64_t> index;  ///< Index of the cell in the source axis
};

/// @brief Sort the cells of an axis and check that they do not overlap
inline auto sort_cells(const Bounds& bounds, const double period)
    -> SortedCells {
  const auto size = bounds.rows();
  std::vector<int64_t> order(static_cast<size_t>(size));
  std::ranges::iota(order, int64_t{0});
  auto lower = [&](const int64_t ix) {
    return std::min(bounds(ix, 0), bounds(ix, 1));
  };
  auto upper = [&](const int64_t ix) {
    return std::max(bounds(ix, 0), bounds(ix, 1));
  };
  std::ranges::sort(order, [&](const int64_t lhs, const int64_t rhs) {
    return lower(lhs) < lower(rhs);
  });

  SortedCells result;
  result.lower.reserve(order.size());
  result.upper.reserve(order.size());
  result.index = order;
  for (const auto ix : order) {
    const auto lo = lower(ix);
    const auto hi = upper(ix);
    if (!std::isfinite(lo) || !std::isfinite(hi)) {
      throw std::invalid_argument("the bounds of the cells must be finite");
    }
    result.lower.push_back(lo);
    result.upper.push_back(hi);
  }
  // A relative tolerance absorbs the rounding of bounds computed from the
  // centers of the cells.
  for (size_t ix = 1; ix < result.lower.size(); ++ix) {
    const auto tolerance = 1e-9 * std::max(1.0, std::abs(result.upper[ix - 1]));
    if (result.lower[ix] < result.upper[ix - 1] - tolerance) {
      throw std::invalid_argument("the cells of the source axis overlap");
    }
  }
  if (period > 0 && !result.lower.empty() &&
      result.upper.back() - result.lower.front() > period * (1 + 1e-9)) {
    throw std::invalid_argument(
        "the cells of the periodic source axis span more than one period");
  }
  return result;
}

/// @brief Add the overlaps of the source cells with an interval to a row
inline auto add_overlaps(const SortedCells& cells, const double lower,
                         const double upper, const double length,
                         OverlapMatrix& matrix, const int64_t row_start)
    -> void {
  // First cell whose upper bound is beyond the lower bound of the interval
  auto first = std::ranges::upper_bound(cells.upper, lower);
  for (auto ix = static_cast<size_t>(first - cells.upper.begin());
       ix < cells.lower.size() && cells.lower[ix] < upper; ++ix) {
    const auto overlap =
        std::min(cells.upper[ix], upper) - std::max(cells.lower[ix], lower);
    if (overlap <= 0) {
      continue;
    }
    const auto index = cells.index[ix];
    const auto weight = overlap / length;
    // With a periodic axis, the same source cell can overlap both images of
    // the target cell.
    auto it = std::find(matrix.indices.begin() + row_start,
                        matrix.indices.end(), index);
    if (it != matrix.indices.end()) {
      matrix.weights[static_cast<size_t>(it - matrix.indices.begin())] +=
          weight;
    } else {
      matrix.indices.push_back(index);
      matrix.weights.push_back(weight);
    }
  }
}

/// @brief Sine of the authalic latitude, the latitude on the sphere of the
/// same area, divided by the radius of that sphere
///
/// The area of the zone of the ellipsoid between the equator and the latitude
/// φ is proportional to q(φ) = (1 - e²) [sin φ / (1 - e² sin² φ) + atanh(e
/// sin φ) / e]. The sine of the authalic latitude is q(φ) / q(90°).
class AuthalicSine {
 public:
  explicit AuthalicSine(
      const std::optional<geometry::geographic::Spheroid>& spheroid) noexcept {
    if (spheroid.has_value()) {
      e2_ = spheroid->first_eccentricity_squared();
      e_ = std::sqrt(e2_);
    }
    qp_ = e_ > 0 ? q(1.0) : 1.0;
  }

  [[nodiscard]] auto operator()(const double latitude) const noexcept
      -> double {
    const auto sin_phi = math::sind(std::clamp(latitude, -90.0, 90.0));
    return e_ > 0 ? q(sin_phi) / qp_ : sin_phi;
  }

 private:
  double e_{0};
  double e2_{0};
  double qp_{1};

  [[nodiscard]] auto q(const double sin_phi) const noexcept -> double {
    return (1 - e2_) * (sin_phi / (1 - e2_ * sin_phi * sin_phi) +
                        std::atanh(e_ * sin_phi) / e_);
  }
};

}  // namespace detail

/// @brief Compute the overlaps between the cells of two axes
///
/// @param[in] source Bounds of the source cells, one row per cell. The cells
/// must not overlap.
/// @param[in] target Bounds of the target cells, one row per cell.
/// @param[in] period Period of the axis (e.g. 360 for longitudes), 0 if the
/// axis is not periodic. The overlaps of a periodic axis are computed modulo
/// the period.
/// @return The overlap matrix
[[nodiscard]] inline auto overlap_matrix(const Bounds& source,
                                         const Bounds& target,
                                         const double period = 0)
    -> OverlapMatrix {
  if (source.cols() != 2 || target.cols() != 2) {
    throw std::invalid_argument("the bounds must have the shape (n, 2)");
  }
  const auto cells = detail::sort_cells(source, period);
  OverlapMatrix result;
  result.indptr.reserve(static_cast<size_t>(target.rows()) + 1);
  for (int64_t jx = 0; jx < target.rows(); ++jx) {
    auto lower = std::min(target(jx, 0), target(jx, 1));
    auto upper = std::max(target(jx, 0), target(jx, 1));
    const auto length = upper - lower;
    const auto row_start = static_cast<int64_t>(result.indices.size());
    if (length > 0 && !cells.lower.empty()) {
      if (period > 0) {
        if (length > period * (1 + 1e-9)) {
          throw std::invalid_argument(
              "a target cell spans more than one period");
        }
        // Move the target cell to [origin, origin + period), then the part of
        // the cell beyond the period wraps around to the origin.
        const auto origin = cells.lower.front();
        const auto shift = std::floor((lower - origin) / period) * period;
        lower -= shift;
        upper -= shift;
        detail::add_overlaps(cells, lower, upper, length, result, row_start);
        detail::add_overlaps(cells, lower - period, upper - period, length,
                             result, row_start);
      } else {
        detail::add_overlaps(cells, lower, upper, length, result, row_start);
      }
    }
    result.indptr.push_back(static_cast<int64_t>(result.indices.size()));
  }
  return result;
}

/// @brief Transform latitude bounds so that the measure along the axis is
/// proportional to the area of the cells
///
/// @param[in] bounds Latitude bounds, in degrees
/// @param[in] spheroid Ellipsoid of the coordinates, a sphere if not set
/// @return The sines of the authalic latitudes of the bounds
[[nodiscard]] inline auto latitude_measure(
    const Bounds& bounds,
    const std::optional<geometry::geographic::Spheroid>& spheroid)
    -> RowMajorMatrix<double> {
  const auto sine = detail::AuthalicSine(spheroid);
  RowMajorMatrix<double> result(bounds.rows(), bounds.cols());
  for (int64_t ix = 0; ix < bounds.rows(); ++ix) {
    for (int64_t jx = 0; jx < bounds.cols(); ++jx) {
      result(ix, jx) = sine(bounds(ix, jx));
    }
  }
  return result;
}

namespace detail {

/// @brief Minimum number of multiply-adds processed by a thread: below,
/// starting the thread costs more than it saves.
constexpr int64_t kMinWorkPerThread = 200'000;

/// @brief Number of threads to use for a given amount of work
[[nodiscard]] inline auto threads_for(const int64_t work,
                                      const int64_t num_threads) -> int64_t {
  return std::clamp(work / kMinWorkPerThread, int64_t{1}, num_threads);
}

/// @brief Regrid one field stored as a (nx, ny) row-major matrix
///
/// The second axis is processed first, row by row of the source, then the
/// rows of the result are linear combinations of the rows of this
/// intermediate matrix.
template <typename T>
auto regrid_field(const OverlapMatrix& wx, const OverlapMatrix& wy,
                  const Vector<double>& coverage_x,
                  const Vector<double>& coverage_y, const T* source,
                  const int64_t nx, const int64_t ny, T* target,
                  const double min_valid_fraction, const bool normalize,
                  const int64_t num_threads) -> void {
  const auto mx = wx.rows();
  const auto my = wy.rows();

  // Sums along the second axis, for each row of the source: values and
  // fraction of the target cell covered by defined values. The fractions are
  // only computed for the rows holding undefined values; the others are
  // covered as much as the source axis covers the target cell.
  RowMajorMatrix<double> values(nx, my);
  RowMajorMatrix<double> valid(nx, my);
  std::vector<uint8_t> row_has_nan(static_cast<size_t>(nx), 0);
  parallel_for(
      nx,
      [&](const int64_t start, const int64_t end) {
        for (auto ix = start; ix < end; ++ix) {
          const auto* row = source + ix * ny;
          const auto has_nan =
              std::any_of(row, row + ny, [](const T x) { return x != x; });
          row_has_nan[static_cast<size_t>(ix)] = has_nan ? 1 : 0;
          if (!has_nan) {
            for (int64_t kx = 0; kx < my; ++kx) {
              double sum = 0;
              for (auto nz = wy.indptr[kx]; nz < wy.indptr[kx + 1]; ++nz) {
                sum +=
                    wy.weights[nz] * static_cast<double>(row[wy.indices[nz]]);
              }
              values(ix, kx) = sum;
            }
            valid.row(ix) = coverage_y.transpose();
            continue;
          }
          for (int64_t kx = 0; kx < my; ++kx) {
            double sum = 0;
            double weight = 0;
            for (auto nz = wy.indptr[kx]; nz < wy.indptr[kx + 1]; ++nz) {
              const auto value = static_cast<double>(row[wy.indices[nz]]);
              if (!std::isnan(value)) {
                sum += wy.weights[nz] * value;
                weight += wy.weights[nz];
              }
            }
            values(ix, kx) = sum;
            valid(ix, kx) = weight;
          }
        }
      },
      threads_for(nx * static_cast<int64_t>(wy.indices.size()), num_threads));
  const auto has_nan = std::ranges::any_of(
      row_has_nan, [](const uint8_t flag) { return flag != 0; });

  // Combinations of the rows along the first axis
  parallel_for(
      mx,
      [&](const int64_t start, const int64_t end) {
        Vector<double> sum(my);
        Vector<double> weight(my);
        for (auto jx = start; jx < end; ++jx) {
          sum.setZero();
          if (has_nan) {
            weight.setZero();
            for (auto nz = wx.indptr[jx]; nz < wx.indptr[jx + 1]; ++nz) {
              const auto ix = wx.indices[nz];
              sum += wx.weights[nz] * values.row(ix).transpose();
              weight += wx.weights[nz] * valid.row(ix).transpose();
            }
          } else {
            for (auto nz = wx.indptr[jx]; nz < wx.indptr[jx + 1]; ++nz) {
              sum += wx.weights[nz] * values.row(wx.indices[nz]).transpose();
            }
            weight = coverage_x[jx] * coverage_y;
          }
          auto* row = target + jx * my;
          for (int64_t kx = 0; kx < my; ++kx) {
            const auto fraction = weight[kx];
            if (fraction <= 0 || fraction < min_valid_fraction) {
              row[kx] = std::numeric_limits<T>::quiet_NaN();
            } else {
              row[kx] =
                  static_cast<T>(normalize ? sum[kx] / fraction : sum[kx]);
            }
          }
        }
      },
      threads_for(static_cast<int64_t>(wx.indices.size()) * my, num_threads));
}

}  // namespace detail

/// @brief First order conservative regridding of fields defined on a
/// rectilinear grid
///
/// @tparam T Type of the values
/// @param[in] wx Overlaps along the first axis (see overlap_matrix)
/// @param[in] wy Overlaps along the second axis
/// @param[in] data Fields to regrid, stored as a row-major matrix of shape
/// (fields × nx, ny): the fields follow each other, each one being a (nx, ny)
/// matrix.
/// @param[in] nx Number of source cells along the first axis
/// @param[in] min_valid_fraction Minimum fraction of the target cell covered
/// by defined source values; below, the result is undefined
/// @param[in] normalize If true, the result is the mean of the defined source
/// values weighted by their overlap with the target cell. Otherwise, the
/// weighted sum is divided by the area of the whole target cell, as if the
/// undefined values were zero.
/// @param[in] num_threads Number of threads, 0 for all CPUs
/// @return The regridded fields, a matrix of shape (fields × mx, my)
template <typename T>
[[nodiscard]] auto conservative(const OverlapMatrix& wx,
                                const OverlapMatrix& wy,
                                const Eigen::Ref<const RowMajorMatrix<T>>& data,
                                const int64_t nx,
                                const double min_valid_fraction,
                                const bool normalize, int64_t num_threads)
    -> RowMajorMatrix<T> {
  const auto ny = data.cols();
  if (nx <= 0 || data.rows() % nx != 0) {
    throw std::invalid_argument(
        "the number of rows of the data must be a multiple of the number of "
        "cells of the first axis");
  }
  const auto fields = data.rows() / nx;
  const auto mx = wx.rows();
  const auto my = wy.rows();
  const auto max_x =
      wx.indices.empty() ? int64_t{-1} : *std::ranges::max_element(wx.indices);
  const auto max_y =
      wy.indices.empty() ? int64_t{-1} : *std::ranges::max_element(wy.indices);
  if (max_x >= nx || max_y >= ny) {
    throw std::invalid_argument(
        "the shape of the data does not match the source axes");
  }
  const auto coverage_x = wx.row_sums();
  const auto coverage_y = wy.row_sums();

  RowMajorMatrix<T> result(fields * mx, my);
  if (num_threads == 0) {
    num_threads = static_cast<int64_t>(std::thread::hardware_concurrency());
  }
  // Many fields: one field per thread; few fields: the threads share the
  // rows of each field.
  const auto per_field = fields > 1 && 2 * fields >= num_threads;
  auto regrid = [&](const int64_t field, const int64_t threads) {
    detail::regrid_field(wx, wy, coverage_x, coverage_y,
                         data.data() + field * nx * ny, nx, ny,
                         result.data() + field * mx * my, min_valid_fraction,
                         normalize, threads);
  };
  if (per_field) {
    parallel_for(
        fields,
        [&](const int64_t start, const int64_t end) {
          for (auto field = start; field < end; ++field) {
            regrid(field, 1);
          }
        },
        num_threads);
  } else {
    for (int64_t field = 0; field < fields; ++field) {
      regrid(field, num_threads);
    }
  }
  return result;
}

}  // namespace pyinterp::regrid
