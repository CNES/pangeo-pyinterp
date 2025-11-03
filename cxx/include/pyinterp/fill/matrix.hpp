#pragma once

#include "pyinterp/detail/math.hpp"
#include "pyinterp/eigen.hpp"

namespace pyinterp {
namespace detail {

/// Fills in the gaps between defined values in a line with interpolated
/// values.
///
/// @tparam T The type of the coordinates.
/// @param x The values of the points defining the line.
/// @param is_undefined A boolean vector indicating which points are undefined.
template <typename T>
void fill_line(EigenRefBlock<T> x, EigenRefBlock<bool> is_undefined) {
  T x0;
  T x1;
  T dx;
  Eigen::Index di;
  Eigen::Index last_valid = -1;
  Eigen::Index first_valid = -1;
  const auto size = x.size();

  for (Eigen::Index ix = 0; ix < size; ++ix) {
    // If the point is undefined, then we can't interpolate it.
    if (!is_undefined[ix]) {
      // If there is a gap between the last valid point and this one, then
      // interpolate the gap.
      if (last_valid != -1 && (ix - last_valid) > 1) {
        x0 = x[last_valid];
        x1 = x[ix];
        di = ix - last_valid;
        dx = (x1 - x0) / di;
        for (Eigen::Index jx = last_valid + 1; jx < ix; ++jx) {
          di = jx - last_valid;
          x[jx] = dx * di + x0;
        }
      } else if (first_valid == -1) {
        // If this is the first valid point, then we can't interpolate the
        // undefined points before it.
        first_valid = ix;
      }
      // Update the last valid point.
      last_valid = ix;
    }
  }

  // If there are no valid points, then we can't interpolate anything.
  if (last_valid == first_valid) {
    is_undefined.setOnes();
    return;
  }

  // If the last valid point is not the last point, then we can't interpolate
  x0 = x[first_valid];
  x1 = x[last_valid];
  dx = (x1 - x0) / (last_valid - first_valid);

  // If there is a gap between the last valid point and the end of the line,
  // then interpolate the gap.
  if (last_valid < (size - 1)) {
    for (Eigen::Index jx = last_valid + 1; jx < size; ++jx) {
      di = jx - last_valid;
      x[jx] = dx * di + x1;
    }
  }
  // If there is a gap between the first valid point and the beginning of the
  // line, then interpolate the gap.
  if (first_valid > 0) {
    for (Eigen::Index jx = 0; jx < first_valid; ++jx) {
      di = first_valid - jx;
      x[jx] = x0 - dx * di;
    }
  }
  // Mark all points as defined.
  is_undefined.setZero();
}

}  // namespace detail

namespace fill {

/// Fills in the gaps between defined values in a matrix with interpolated
/// values.
///
/// @param x The data to be processed.
template <typename T>
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
/// @param array Array of dates.
/// @param fill_value Value to use for missing data.
template <typename T>
auto vector(Eigen::Ref<Vector<T>> array, const T &fill_value) {
  Vector<bool> mask;
  if (detail::math::Fill<T>::is(fill_value)) {
    mask = Eigen::isnan(array.array());
  } else {
    mask = array.array() == fill_value;
  }
  detail::fill_line<T>(array, mask);
}

}  // namespace fill
}  // namespace pyinterp
