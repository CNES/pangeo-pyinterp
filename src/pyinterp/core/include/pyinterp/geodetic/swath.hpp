// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/eigen.h>

#include <Eigen/Core>
#include <tuple>

#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/detail/geodetic/coordinates.hpp"
#include "pyinterp/detail/geodetic/spheroid.hpp"
#include "pyinterp/detail/math/orbit.hpp"

namespace pyinterp::geodetic {

/// Convert geodetic coordinates to ECEF coordinates.
template <typename T>
inline auto lla_to_ecef(const Eigen::Ref<const Vector<T>>& lon_nadir,
                        const Eigen::Ref<const Vector<T>>& lat_nadir,
                        const detail::geodetic::Coordinates& coordinates)
    -> Eigen::Matrix<T, Eigen::Dynamic, 3> {
  auto result = Eigen::Matrix<T, Eigen::Dynamic, 3>(lon_nadir.rows(), 3);
  for (auto ix = 0; ix < lon_nadir.rows(); ++ix) {
    auto ecef =
        coordinates.lla_to_ecef<T>({lon_nadir(ix), lat_nadir(ix), T(0)});
    result(ix, 0) = boost::geometry::get<0>(ecef);
    result(ix, 1) = boost::geometry::get<1>(ecef);
    result(ix, 2) = boost::geometry::get<2>(ecef);
  }
  return result;
}

/// Calculate the swath coordinates from the swath center coordinates.
template <typename T>
[[nodiscard]] auto calculate_swath(
    const Eigen::Ref<const Vector<T>>& lon_nadir,
    const Eigen::Ref<const Vector<T>>& lat_nadir, const T& delta_ac,
    const T& half_gap, const int half_swath,
    const std::optional<detail::geodetic::Spheroid>& spheroid)
    -> std::tuple<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> {
  detail::check_eigen_shape("lon_nadir", lon_nadir, "lat_nadir", lat_nadir);
  auto spheroid_ = spheroid.value_or(detail::geodetic::Spheroid());
  auto coordinates = detail::geodetic::Coordinates(spheroid_);
  auto location = lla_to_ecef(lon_nadir, lat_nadir, coordinates);
  auto direction = detail::math::satellite_direction<T>(location);
  auto lon = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(lon_nadir.rows(),
                                                              2 * half_swath);
  auto lat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(lon_nadir.rows(),
                                                              2 * half_swath);
  auto loc = Eigen::Matrix<T, 3, 1>();

  for (auto ix = 0; ix < location.rows(); ++ix) {
    const auto loc_row = location.row(ix).transpose();
    const auto dir_row = direction.row(ix);
    const auto radius = spheroid_.geocentric_radius(lat_nadir(ix));

    for (auto jx = 0; jx < half_swath; ++jx) {
      auto rotation = detail::math::rotation_3d_matrix<T>(
          (-(jx * delta_ac + half_gap) / radius), dir_row);
      auto kx = half_swath + jx;

      loc = rotation * loc_row;
      auto lla = coordinates.ecef_to_lla<T>({loc[0], loc[1], loc[2]});
      lon(ix, kx) = boost::geometry::get<0>(lla);
      lat(ix, kx) = boost::geometry::get<1>(lla);

      loc = rotation.transpose() * loc_row;
      kx = half_swath - jx - 1;
      lla = coordinates.ecef_to_lla<T>({loc[0], loc[1], loc[2]});
      lon(ix, kx) = boost::geometry::get<0>(lla);
      lat(ix, kx) = boost::geometry::get<1>(lla);
    }
  }
  return std::make_tuple(lon, lat);
}

}  // namespace pyinterp::geodetic
