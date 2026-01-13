// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <cmath>
#include <concepts>

#include "pyinterp/eigen.hpp"

namespace pyinterp::fill {
namespace detail {

/// Fills in the gaps between defined values in a line with interpolated
/// values.
///
/// @tparam T The type of the coordinates.
/// @param[in,out] x The values of the points defining the line.
/// @param[in,out] is_undefined A boolean vector indicating which points are
/// undefined.
template <std::floating_point T>
void fill_line(EigenRefBlock<T> x, EigenRefBlock<bool> is_undefined) {
  const auto size = x.size();
  Eigen::Index first_valid = -1;
  Eigen::Index last_valid = -1;

  // Find first and last valid points, interpolate interior gaps
  for (int64_t ix = 0; ix < size; ++ix) {
    if (!is_undefined[ix]) {
      if (first_valid == -1) {
        first_valid = ix;
      }
      // Interpolate gap if there's one
      if (last_valid != -1 && (ix - last_valid) > 1) {
        const T x0 = x[last_valid];
        const T x1 = x[ix];
        const auto di = ix - last_valid;
        const T dx = (x1 - x0) / di;
        for (int64_t jx = last_valid + 1; jx < ix; ++jx) {
          x[jx] = x0 + dx * (jx - last_valid);
        }
      }
      last_valid = ix;
    }
  }

  // No valid points at all
  if (first_valid == -1) {
    return;  // Leave is_undefined as-is
  }

  // Single valid point: fill everything with that constant
  if (first_valid == last_valid) {
    x.setConstant(x[first_valid]);
    is_undefined.setZero();
    return;
  }

  // Extrapolate edges using overall slope
  const T x0 = x[first_valid];
  const T x1 = x[last_valid];
  const T dx = (x1 - x0) / (last_valid - first_valid);

  // Extrapolate beyond last valid point
  for (int64_t jx = last_valid + 1; jx < size; ++jx) {
    x[jx] = x1 + dx * (jx - last_valid);
  }

  // Extrapolate before first valid point
  for (int64_t jx = 0; jx < first_valid; ++jx) {
    x[jx] = x0 - dx * (first_valid - jx);
  }

  is_undefined.setZero();
}

}  // namespace detail

/// Fills in the gaps between defined values in a matrix with interpolated
/// values.
///
/// @param[in,out] x The data to be processed.
/// @param[in] fill_value Value to use for missing data.
template <std::floating_point T>
void matrix(EigenDRef<Matrix<T>> x, const T &fill_value) {
  Matrix<bool> mask;
  if (std::isnan(fill_value)) {
    mask = Eigen::isnan(x.array());
  } else {
    mask = x.array() == fill_value;
  }
  auto num_rows = x.rows();
  auto num_cols = x.cols();
  // Fill in the rows.
  for (int ix = 0; ix < num_rows; ix++) {
    auto m = mask.row(ix);
    if (m.all()) {
      continue;
    }
    detail::fill_line<T>(x.row(ix), m);
  }
  // Fill in the columns.
  for (int ix = 0; ix < num_cols; ix++) {
    detail::fill_line<T>(x.col(ix), mask.col(ix));
  }
}

/// Fill gaps between defined values in a vector with interpolated values.
///
/// The data is assumed to be monotonically increasing or decreasing.
///
/// @param[in,out] array Array of dates.
/// @param[in] fill_value Value to use for missing data.
template <std::floating_point T>
inline auto vector(Eigen::Ref<Vector<T>> array, const T &fill_value) {
  matrix(EigenDRef<Matrix<T>>(array), fill_value);
}

}  // namespace pyinterp::fill
