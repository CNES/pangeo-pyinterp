// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <algorithm>
#include <limits>
#include <string>
#include <tuple>

#include "pyinterp/axis.hpp"
#include "pyinterp/detail/gsl/error_handler.hpp"
#include "pyinterp/detail/gsl/interpolate1d.hpp"
#include "pyinterp/eigen.hpp"

namespace pyinterp {
namespace detail {

/// Calculate the start, end and size of a window
auto calculate_start_end_size(const int64_t index, const int64_t size,
                              const int64_t half_window_size)
    -> std::tuple<int64_t, int64_t, int64_t> {
  auto start = std::max(index - half_window_size, int64_t(0));
  auto end = std::min(index + half_window_size, size - 1);
  return std::make_tuple(start, end, end - start + 1);
}

}  // namespace detail

/// Interpolation of a 1D function
///
/// @param x The x-coordinates of the data points, must be increasing
/// @param y The y-coordinates of the data points, same length as x
/// @param xi The x-coordinates of the interpolated values
/// @param half_window_size Size of the half window.
/// @param bounds_error If true, an exception is raised if the value to be
///     interpolated is out of the range of the axis.
/// @param kind Type of interpolation to be used.
/// @return The interpolated values
auto interpolate_1d(const pyinterp::Axis<double>& x,
                    const Eigen::Ref<const Vector<double>>& y,
                    const Eigen::Ref<const Vector<double>>& xi,
                    const int64_t half_window_size, const bool bounds_error,
                    const std::string& kind) -> Vector<double> {
  if (x.size() != y.size()) {
    throw std::invalid_argument("x and y must have the same size");
  }

  // Note: if the window size is invalid, GSL will raise an exception.

  // GSL Interpolation type
  const auto* interp_type = detail::gsl::Interpolate1D::parse_interp_type(kind);

  // Full window size
  const auto window_size = half_window_size * 2 + 1;

  // Downcast the axis to the raw C++ type
  auto* axis = dynamic_cast<const pyinterp::detail::Axis<double>*>(&x);

  // Allocate the vector storing the result.
  auto result = Vector<double>(xi.size());

  for (int64_t ix = 0; ix < xi.size(); ++ix) {
    // Nearest index of the current value
    auto index = axis->find_index(xi(ix), true);

    if (index == -1) {
      if (bounds_error) {
        throw std::out_of_range("The value is out of the range of the axis");
      } else {
        result(ix) = std::numeric_limits<double>::quiet_NaN();
        continue;
      }
    }

    // Calculate the start, end and size of the current window
    const auto [start, end, n] =
        detail::calculate_start_end_size(index, axis->size(), half_window_size);

    // Select the data in the current window
    const Vector<double> yw = y.segment(start, n);

    // Interpolation of the current value
    auto interpolator =
        detail::gsl::Interpolate1D(n, interp_type, detail::gsl::Accelerator());
    result(ix) = interpolator.interpolate(axis->slice(start, n), yw, xi(ix));
  }
  return result;
}

}  // namespace pyinterp
