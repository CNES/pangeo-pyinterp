// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <Eigen/Core>
#include <concepts>
#include <cstdint>
#include <ranges>

#include "pyinterp/broadcast.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/geometry/geographic/coordinates.hpp"
#include "pyinterp/geometry/satellite/rotation.hpp"

namespace pyinterp::geometry::satellite::swath {

/// @brief Convert LLA coordinates to ECEF coordinates
/// @tparam T Numeric type
/// @param[in] lon_nadir Longitude of nadir
/// @param[in] lat_nadir Latitude of nadir
/// @param[in] coordinates Transformation object
/// @return ECEF coordinates
template <std::floating_point T>
[[nodiscard]] inline auto lla_to_ecef(
    const Eigen::Ref<const Vector<T>>& lon_nadir,
    const Eigen::Ref<const Vector<T>>& lat_nadir,
    const geographic::Coordinates& coordinates)
    -> Eigen::Matrix<T, Eigen::Dynamic, 3> {
  constexpr T kSealLevelAltitude = T(0);
  auto result = Eigen::Matrix<T, Eigen::Dynamic, 3>(lon_nadir.rows(), 3);
  int64_t row_index = 0;
  for (auto [lon, lat] : std::ranges::views::zip(lon_nadir, lat_nadir)) {
    auto ecef = coordinates.lla_to_ecef<T>({lon, lat, kSealLevelAltitude});
    result(row_index, 0) = boost::geometry::get<0>(ecef);
    result(row_index, 1) = boost::geometry::get<1>(ecef);
    result(row_index, 2) = boost::geometry::get<2>(ecef);
    ++row_index;
  }
  return result;
}

/// @brief Calculate the swath coordinates from the nadir coordinates
/// @tparam T Numeric type
/// @param[in] lon_nadir Longitude of nadir
/// @param[in] lat_nadir Latitude of nadir
/// @param[in] delta_ac Across-track distance between two consecutive pixels
/// (meters)
/// @param[in] half_gap Half of the gap between the nadir and the first pixel
/// (meters)
/// @param[in] half_swath Half of the swath width (pixels)
/// @param[in] spheroid Optional spheroid model
/// @return Tuple of longitude and latitude matrices of the swath
template <std::floating_point T>
[[nodiscard]] auto calculate(
    const Eigen::Ref<const Vector<T>>& lon_nadir,
    const Eigen::Ref<const Vector<T>>& lat_nadir, const T& delta_ac,
    const T& half_gap, const int64_t half_swath,
    const std::optional<geographic::Spheroid>& spheroid)
    -> std::tuple<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> {
  broadcast::check_eigen_shape("lon_nadir", lon_nadir, "lat_nadir", lat_nadir);
  auto spheroid_model = spheroid.value_or(geographic::Spheroid());
  auto coordinates = geographic::Coordinates(spheroid_model);
  auto location = lla_to_ecef(lon_nadir, lat_nadir, coordinates);
  auto direction = rotation::satellite_direction<T>(location);
  auto lon = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(lon_nadir.rows(),
                                                              2 * half_swath);
  auto lat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(lon_nadir.rows(),
                                                              2 * half_swath);

  for (int64_t ix = 0; ix < location.rows(); ++ix) {
    const auto loc_row = location.row(ix).transpose();
    const auto dir_row = direction.row(ix);
    const auto radius = spheroid_model.geocentric_radius(lat_nadir(ix));

    for (auto jx = 0; jx < half_swath; ++jx) {
      auto rotation = rotation::matrix_3d<T>(
          (-(jx * delta_ac + half_gap) / radius), dir_row);
      auto kx = half_swath + jx;

      Eigen::Matrix<T, 3, 1> loc = rotation * loc_row;
      auto lla = coordinates.ecef_to_lla<T>({loc[0], loc[1], loc[2]});
      lon(ix, kx) = lla.template get<0>();
      lat(ix, kx) = lla.template get<1>();

      loc = rotation.transpose() * loc_row;
      kx = half_swath - jx - 1;
      lla = coordinates.ecef_to_lla<T>({loc[0], loc[1], loc[2]});
      lon(ix, kx) = lla.template get<0>();
      lat(ix, kx) = lla.template get<1>();
    }
  }
  return std::make_tuple(lon, lat);
}

}  // namespace pyinterp::geometry::satellite::swath
