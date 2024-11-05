// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Core>
#include <cstdint>
#include <string>

#include "pyinterp/axis.hpp"
#include "pyinterp/detail/math/frame.hpp"
#include "pyinterp/grid.hpp"

namespace pyinterp {

// Error thrown if it' s not possible to frame the value on the specified axis.
inline auto index_error(const std::string &axis, const std::string &value,
                        size_t n) -> void {
  throw std::invalid_argument("Unable to frame the value " + value + " with " +
                              std::to_string(n) + " items of the " + axis +
                              " axis");
}

/// Loads the interpolation frame into memory
template <typename DataType>
auto load_frame(const Grid2D<DataType> &grid, const double x, const double y,
                const axis::Boundary boundary, const bool bounds_error,
                detail::math::Frame2D &frame) -> bool {
  const auto &x_axis = *grid.x();
  const auto &y_axis = *grid.y();
  const auto y_indexes =
      y_axis.find_indexes(y, static_cast<uint32_t>(frame.ny()), boundary);
  const auto x_indexes =
      x_axis.find_indexes(x, static_cast<uint32_t>(frame.nx()), boundary);

  if (x_indexes.empty() || y_indexes.empty()) {
    if (bounds_error) {
      if (x_indexes.empty()) {
        index_error("x", x_axis.coordinate_repr(x), frame.nx());
      }
      index_error("y", y_axis.coordinate_repr(y), frame.ny());
    }
    return false;
  }

  auto x0 = x_axis(x_indexes[0]);

  for (Eigen::Index jx = 0; jx < frame.y()->size(); ++jx) {
    frame.y(jx) = y_axis(y_indexes[jx]);
  }

  for (Eigen::Index ix = 0; ix < frame.x()->size(); ++ix) {
    const auto index = x_indexes[ix];

    frame.x(ix) = x_axis.is_angle()
                      ? detail::math::normalize_angle(x_axis(index), x0, 360.0)
                      : x_axis(index);

    for (Eigen::Index jx = 0; jx < frame.y()->size(); ++jx) {
      frame.q(ix, jx) = static_cast<double>(grid.value(index, y_indexes[jx]));
    }
  }
  return frame.is_valid();
}

/// Loads the interpolation frame into memory
template <typename DataType, typename AxisType>
auto load_frame(const Grid3D<DataType, AxisType> &grid, const double x,
                const double y, const AxisType z, const axis::Boundary boundary,
                const bool bounds_error, detail::math::Frame3D<AxisType> &frame)
    -> bool {
  const auto &x_axis = *grid.x();
  const auto &y_axis = *grid.y();
  const auto &z_axis = *grid.z();
  const auto z_indexes =
      z_axis.find_indexes(z, static_cast<uint32_t>(frame.nz()), boundary);
  const auto y_indexes =
      y_axis.find_indexes(y, static_cast<uint32_t>(frame.ny()), boundary);
  const auto x_indexes =
      x_axis.find_indexes(x, static_cast<uint32_t>(frame.nx()), boundary);

  if (x_indexes.empty() || y_indexes.empty() || z_indexes.empty()) {
    if (bounds_error) {
      if (x_indexes.empty()) {
        index_error("x", x_axis.coordinate_repr(x), frame.nx());
      } else if (y_indexes.empty()) {
        index_error("y", y_axis.coordinate_repr(y), frame.ny());
      }
      index_error("z", z_axis.coordinate_repr(z), frame.nz());
    }
    return false;
  }

  auto x0 = x_axis(x_indexes[0]);

  for (Eigen::Index jx = 0; jx < frame.y()->size(); ++jx) {
    frame.y(jx) = y_axis(y_indexes[jx]);
  }

  for (Eigen::Index kx = 0; kx < frame.z().size(); ++kx) {
    frame.z(kx) = z_axis(z_indexes[kx]);
  }

  for (Eigen::Index ix = 0; ix < frame.x()->size(); ++ix) {
    const auto x_index = x_indexes[ix];

    frame.x(ix) = x_axis.is_angle() ? detail::math::normalize_angle(
                                          x_axis(x_index), x0, 360.0)
                                    : x_axis(x_index);

    for (Eigen::Index jx = 0; jx < frame.y()->size(); ++jx) {
      const auto y_index = y_indexes[jx];

      for (Eigen::Index kx = 0; kx < frame.z().size(); ++kx) {
        frame.q(ix, jx, kx) =
            static_cast<double>(grid.value(x_index, y_index, z_indexes[kx]));
      }
    }
  }
  return frame.is_valid();
}

/// Loads the interpolation frame into memory
template <typename DataType, typename AxisType>
auto load_frame(const Grid4D<DataType, AxisType> &grid, const double x,
                const double y, const AxisType z, const double u,
                const axis::Boundary boundary, const bool bounds_error,
                detail::math::Frame4D<AxisType> &frame) -> bool {
  const auto &x_axis = *grid.x();
  const auto &y_axis = *grid.y();
  const auto &z_axis = *grid.z();
  const auto &u_axis = *grid.u();
  const auto u_indexes =
      u_axis.find_indexes(u, static_cast<uint32_t>(frame.nu()), boundary);
  const auto z_indexes =
      z_axis.find_indexes(z, static_cast<uint32_t>(frame.nz()), boundary);
  const auto y_indexes =
      y_axis.find_indexes(y, static_cast<uint32_t>(frame.ny()), boundary);
  const auto x_indexes =
      x_axis.find_indexes(x, static_cast<uint32_t>(frame.nx()), boundary);

  if (x_indexes.empty() || y_indexes.empty() || z_indexes.empty() ||
      u_indexes.empty()) {
    if (bounds_error) {
      if (x_indexes.empty()) {
        index_error("x", x_axis.coordinate_repr(x), frame.nx());
      } else if (y_indexes.empty()) {
        index_error("y", y_axis.coordinate_repr(y), frame.ny());
      } else if (z_indexes.empty()) {
        index_error("z", z_axis.coordinate_repr(z), frame.nz());
      }
      index_error("u", u_axis.coordinate_repr(u), frame.nu());
    }
    return false;
  }

  auto x0 = x_axis(x_indexes[0]);

  for (Eigen::Index jx = 0; jx < frame.y()->size(); ++jx) {
    frame.y(jx) = y_axis(y_indexes[jx]);
  }

  for (Eigen::Index kx = 0; kx < frame.z().size(); ++kx) {
    frame.z(kx) = z_axis(z_indexes[kx]);
  }

  for (Eigen::Index lx = 0; lx < frame.u().size(); ++lx) {
    frame.u(lx) = u_axis(u_indexes[lx]);
  }

  for (Eigen::Index ix = 0; ix < frame.x()->size(); ++ix) {
    const auto x_index = x_indexes[ix];

    frame.x(ix) = x_axis.is_angle() ? detail::math::normalize_angle(
                                          x_axis(x_index), x0, 360.0)
                                    : x_axis(x_index);

    for (Eigen::Index jx = 0; jx < frame.y()->size(); ++jx) {
      const auto y_index = y_indexes[jx];

      for (Eigen::Index kx = 0; kx < frame.z().size(); ++kx) {
        const auto z_index = z_indexes[kx];

        for (Eigen::Index lx = 0; lx < frame.u().size(); ++lx) {
          frame.q(ix, jx, kx, lx) = static_cast<double>(
              grid.value(x_index, y_index, z_index, u_indexes[lx]));
        }
      }
    }
  }
  return frame.is_valid();
}

}  // namespace pyinterp
