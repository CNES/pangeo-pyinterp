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
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pyinterp/eigen.hpp"
#include "pyinterp/geometry/reduced_gaussian/grid.hpp"
#include "pyinterp/geometry/reduced_gaussian/mesh.hpp"

namespace pyinterp::geometry::reduced_gaussian {

/// @brief Smallest sum of the weights of the valid corners for which a value
///        is still interpolated rather than reported missing.
inline constexpr double kMinWeight = 1e-6;

/// @brief Outcome of the interpolation of a point.
enum class Status : uint8_t {
  kOk,           ///< Interpolated.
  kOutsideMesh,  ///< North of the first parallel, south of the last, or NaN.
  kOutsideTime,  ///< Outside the series, or a step it needs is missing.
  kNotInZone,    ///< A corner is not in the zone of a step it reads.
};

/// @brief Combine the corners of one field into the column of a point.
///
/// Each value is checked: a NaN corner is left out and the weights of the
/// others renormalised, so that a level under the orography of one corner
/// does not void the whole column.
///
/// @param[in] field The field.
/// @param[in] rows Rows of the four corners in the values.
/// @param[in] weight Weights of the four corners.
/// @param[out] out The column, `field.width` values.
/// @param[out] weight_sum Scratch space of `field.width` values.
inline void combine(const Field& field,
                    const std::array<int64_t, kCorners>& rows,
                    const std::array<double, kCorners>& weight, double* out,
                    double* weight_sum) noexcept {
  const auto width = field.width;
  std::fill_n(out, width, 0.0);
  std::fill_n(weight_sum, width, 0.0);
  const auto accumulate = [&]<typename T>(const T* values) {
    for (size_t k = 0; k < kCorners; ++k) {
      // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
      const auto w = weight[k];
      if (w == 0.0) {
        continue;
      }
      const T* column = values + (rows[k] * width);
      // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
      for (int64_t jx = 0; jx < width; ++jx) {
        const auto value = static_cast<double>(column[jx]);
        if (!std::isnan(value)) {
          out[jx] += w * value;
          weight_sum[jx] += w;
        }
      }
    }
  };
  if (field.dtype == DType::kFloat32) {
    accumulate(static_cast<const float*>(field.data));
  } else {
    accumulate(static_cast<const double*>(field.data));
  }
  for (int64_t jx = 0; jx < width; ++jx) {
    out[jx] = weight_sum[jx] < kMinWeight
                  ? std::numeric_limits<double>::quiet_NaN()
                  : out[jx] / weight_sum[jx];
  }
}

/// @brief Interpolates the fields of a series at points.
///
/// The evaluator is shared by the threads; each thread brings its own
/// `Scratch`.
class Evaluator {
 public:
  /// @brief Per-thread working space.
  struct Scratch {
    /// @brief Sum of the weights, one per value of the widest field.
    std::vector<double> weight_sum;
    /// @brief The columns of the later step, before the blend.
    std::vector<double> later;
  };

  /// @brief Build the evaluator.
  ///
  /// The series is copied -- its dates and pointers to its grids, not their
  /// values -- so that a step appended or dropped while the evaluator runs
  /// without the GIL does not reallocate the steps under it.
  ///
  /// @param[in] series The series; a single grid is a series of one step
  ///                   read without dates.
  /// @param[in] fields Indices of the fields to interpolate.
  /// @param[in] times Dates of the points, or null to read the first step.
  /// @param[in] rule How the steps of a date are chosen.
  Evaluator(Series series, std::vector<size_t> fields, const int64_t* times,
            BracketRule rule)
      : series_(std::move(series)),
        fields_(std::move(fields)),
        times_(times),
        rule_(rule) {
    if (series_.size() == 0) {
      throw std::invalid_argument("the series has no step");
    }
    const auto& all = series_.grid(0).fields();
    offsets_.push_back(0);
    for (const auto ix : fields_) {
      if (ix >= all.size()) {
        throw std::out_of_range("field index out of range");
      }
      const auto width = all[ix].width;
      max_width_ = std::max(max_width_, width);
      offsets_.push_back(offsets_.back() + width);
    }
  }

  /// @brief Number of fields interpolated.
  [[nodiscard]] auto num_fields() const noexcept -> size_t {
    return fields_.size();
  }

  /// @brief The field interpolated in position `ix`.
  [[nodiscard]] auto field(size_t ix) const -> const Field& {
    return series_.grid(0).fields()[fields_[ix]];
  }

  /// @brief Offset of each field in a concatenated column, plus the total
  ///        width at the end.
  [[nodiscard]] auto offsets() const noexcept -> const std::vector<int64_t>& {
    return offsets_;
  }

  /// @brief Total number of values of a concatenated column.
  [[nodiscard]] auto total_width() const noexcept -> int64_t {
    return offsets_.back();
  }

  /// @brief Allocate the working space of a thread.
  [[nodiscard]] auto make_scratch() const -> Scratch {
    return {.weight_sum = std::vector<double>(static_cast<size_t>(max_width_)),
            .later = std::vector<double>(static_cast<size_t>(total_width()))};
  }

  /// @brief The cell and the steps a point reads.
  /// @param[in] ix Index of the point.
  /// @param[in] lon,lat Coordinates of the point.
  /// @param[out] cell The cell.
  /// @param[out] bracket The steps.
  [[nodiscard]] auto prepare(int64_t ix, double lon, double lat, Cell& cell,
                             Bracket& bracket) const -> Status {
    if (!series_.mesh().locate(lon, lat, cell)) {
      return Status::kOutsideMesh;
    }
    if (times_ == nullptr) {
      bracket = Bracket{};
      return Status::kOk;
    }
    const auto found = series_.bracket(times_[ix], rule_);
    if (!found) {
      return Status::kOutsideTime;
    }
    bracket = *found;
    return Status::kOk;
  }

  /// @brief Interpolate in space the fields of one step.
  /// @param[in] step The step.
  /// @param[in] cell The cell of the point.
  /// @param[out] out The concatenated columns, `total_width()` values.
  /// @param[in,out] scratch Working space.
  [[nodiscard]] auto step(int64_t step, const Cell& cell, double* out,
                          Scratch& scratch) const -> Status {
    const auto& grid = series_.grid(step);
    std::array<int64_t, kCorners> rows{};
    for (size_t k = 0; k < kCorners; ++k) {
      // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
      rows[k] = grid.row(cell.index[k]);
      if (rows[k] < 0) {
        return Status::kNotInZone;
      }
      // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
    }
    const auto& fields = grid.fields();
    for (size_t ix = 0; ix < fields_.size(); ++ix) {
      combine(fields[fields_[ix]], rows, cell.weight, out + offsets_[ix],
              scratch.weight_sum.data());
    }
    return Status::kOk;
  }

  /// @brief Interpolate in space the fields of each step a point reads,
  ///        without blending them.
  /// @param[in] ix Index of the point.
  /// @param[in] lon,lat Coordinates of the point.
  /// @param[out] early The concatenated columns of the earlier step.
  /// @param[out] later The concatenated columns of the later step, written
  ///             only if the point reads two steps.
  /// @param[out] bracket The steps read.
  /// @param[in,out] scratch Working space.
  /// @param[out] failed_step The step whose zone misses the point, when the
  ///             status is `kNotInZone`.
  [[nodiscard]] auto evaluate_steps(int64_t ix, double lon, double lat,
                                    double* early, double* later,
                                    Bracket& bracket, Scratch& scratch,
                                    int64_t& failed_step) const -> Status {
    Cell cell;
    if (const auto status = prepare(ix, lon, lat, cell, bracket);
        status != Status::kOk) {
      return status;
    }
    failed_step = bracket.i0;
    if (const auto status = step(bracket.i0, cell, early, scratch);
        status != Status::kOk || bracket.single()) {
      return status;
    }
    failed_step = bracket.i1;
    return step(bracket.i1, cell, later, scratch);
  }

  /// @brief Interpolate in space and time the fields at a point.
  /// @param[in] ix Index of the point.
  /// @param[in] lon,lat Coordinates of the point.
  /// @param[out] out The concatenated columns, `total_width()` values.
  /// @param[in,out] scratch Working space.
  /// @param[out] failed_step The step whose zone misses the point, when the
  ///             status is `kNotInZone`.
  [[nodiscard]] auto evaluate(int64_t ix, double lon, double lat, double* out,
                              Scratch& scratch, int64_t& failed_step) const
      -> Status {
    Bracket bracket;
    auto* later = scratch.later.data();
    if (const auto status = evaluate_steps(ix, lon, lat, out, later, bracket,
                                           scratch, failed_step);
        status != Status::kOk || bracket.single()) {
      return status;
    }
    const auto w = bracket.weight;
    for (int64_t jx = 0; jx < total_width(); ++jx) {
      out[jx] = ((1.0 - w) * out[jx]) + (w * later[jx]);
    }
    return Status::kOk;
  }

  /// @brief Throw the error of a status, for `bounds_error`.
  /// @param[in] status The status (not `kOk`).
  /// @param[in] ix Index of the point.
  /// @param[in] lon,lat Coordinates of the point.
  /// @param[in] failed_step The step, for `kNotInZone`.
  [[noreturn]] void raise(Status status, int64_t ix, double lon, double lat,
                          int64_t failed_step) const {
    const auto where = "point " + std::to_string(ix) +
                       " (lon=" + std::to_string(lon) +
                       ", lat=" + std::to_string(lat) + ")";
    switch (status) {
      case Status::kOutsideMesh:
        throw std::out_of_range(where + " is outside the mesh");
      case Status::kOutsideTime:
        throw std::out_of_range(
            where +
            " has a date outside the series, or a step it needs is "
            "missing");
      case Status::kNotInZone:
        throw std::out_of_range(
            where + " reads a corner that the zone of step " +
            std::to_string(failed_step) +
            " does not hold: the zone read for that step is too small");
      default:
        throw std::logic_error("no error to raise");
    }
  }

 private:
  Series series_;
  std::vector<size_t> fields_;
  const int64_t* times_;
  BracketRule rule_;
  std::vector<int64_t> offsets_;
  int64_t max_width_{1};
};

}  // namespace pyinterp::geometry::reduced_gaussian
