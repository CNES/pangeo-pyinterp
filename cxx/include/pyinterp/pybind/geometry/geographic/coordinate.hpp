// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <nanobind/nanobind.h>

#include <optional>
#include <ranges>
#include <tuple>

#include "pyinterp/broadcast.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/geometry/geographic/coordinates.hpp"
#include "pyinterp/parallel_for.hpp"

namespace pyinterp::geometry::geographic::pybind {

/// @brief Python wrapper for geometry::geographic::Coordinates
class Coordinates : public geographic::Coordinates {
 public:
  using geographic::Coordinates::Coordinates;

  /// Convert Cartesian ECEF coordinates to geographic coordinates
  /// @param[in] x X-coordinates in meters
  /// @param[in] y Y-coordinates in meters
  /// @param[in] z Z-coordinates in meters
  /// @param[in] num_threads Number of threads to use (0 = auto)
  /// @return Tuple of (longitudes, latitudes, altitudes)
  template <std::floating_point T>
  [[nodiscard]] auto ecef_to_lla(const Eigen::Ref<const Vector<T>> &x,
                                 const Eigen::Ref<const Vector<T>> &y,
                                 const Eigen::Ref<const Vector<T>> &z,
                                 const size_t num_threads) const
      -> std::tuple<Vector<T>, Vector<T>, Vector<T>> {
    broadcast::check_eigen_shape("x", x, "y", y, "z", z);

    const auto n = x.size();

    Vector<T> lon(n);
    Vector<T> lat(n);
    Vector<T> alt(n);

    parallel_for(
        n,
        [&](size_t start, size_t end) {
          auto make_segment = [start, end](auto &vec) {
            return vec.segment(static_cast<Eigen::Index>(start),
                               static_cast<Eigen::Index>(end - start));
          };

          auto x_segment = make_segment(x);
          auto y_segment = make_segment(y);
          auto z_segment = make_segment(z);
          auto lon_segment = make_segment(lon);
          auto lat_segment = make_segment(lat);
          auto alt_segment = make_segment(alt);

          for (const auto &[x_i, y_i, z_i, lon_i, lat_i, alt_i] :
               std::views::zip(x_segment, y_segment, z_segment, lon_segment,
                               lat_segment, alt_segment)) {
            auto lla = geographic::Coordinates::ecef_to_lla<T>({x_i, y_i, z_i});
            lon_i = boost::geometry::get<0>(lla);
            lat_i = boost::geometry::get<1>(lla);
            alt_i = boost::geometry::get<2>(lla);
          }
        },
        num_threads);
    return std::make_tuple(lon, lat, alt);
  }

  /// Convert geographic coordinates to Cartesian ECEF coordinates
  /// @param[in] lon Longitudes in degrees
  /// @param[in] lat Latitudes in degrees
  /// @param[in] alt Altitudes in meters
  /// @param[in] num_threads Number of threads to use (0 = auto)
  /// @return Tuple of (X, Y, Z) ECEF coordinates in meters
  template <std::floating_point T>
  [[nodiscard]] auto lla_to_ecef(const Eigen::Ref<const Vector<T>> &lon,
                                 const Eigen::Ref<const Vector<T>> &lat,
                                 const Eigen::Ref<const Vector<T>> &alt,
                                 const size_t num_threads) const
      -> std::tuple<Vector<T>, Vector<T>, Vector<T>> {
    broadcast::check_eigen_shape("lon", lon, "lat", lat, "alt", alt);

    const auto n = lon.size();

    Vector<T> x(n);
    Vector<T> y(n);
    Vector<T> z(n);

    parallel_for(
        n,
        [&](size_t start, size_t end) {
          auto make_segment = [start, end](auto &vec) {
            return vec.segment(static_cast<Eigen::Index>(start),
                               static_cast<Eigen::Index>(end - start));
          };

          auto lon_segment = make_segment(lon);
          auto lat_segment = make_segment(lat);
          auto alt_segment = make_segment(alt);
          auto x_segment = make_segment(x);
          auto y_segment = make_segment(y);
          auto z_segment = make_segment(z);

          for (const auto &[lon_i, lat_i, alt_i, x_i, y_i, z_i] :
               std::views::zip(lon_segment, lat_segment, alt_segment, x_segment,
                               y_segment, z_segment)) {
            auto ecef =
                geographic::Coordinates::lla_to_ecef<T>({lon_i, lat_i, alt_i});
            x_i = boost::geometry::get<0>(ecef);
            y_i = boost::geometry::get<1>(ecef);
            z_i = boost::geometry::get<2>(ecef);
          }
        },
        num_threads);
    return std::make_tuple(x, y, z);
  }

  /// Transform positions from one coordinate system to another
  /// @param[in] target Target coordinate system
  /// @param[in] lon Longitudes in degrees
  /// @param[in] lat Latitudes in degrees
  /// @param[in] alt Altitudes in meters
  /// @param[in] num_threads Number of threads to use (0 = auto)
  /// @return Tuple of (longitudes, latitudes, altitudes) in the new coordinate
  /// system
  template <std::floating_point T>
  [[nodiscard]] auto transform(const geographic::Coordinates &target,
                               const Eigen::Ref<const Vector<T>> &lon,
                               const Eigen::Ref<const Vector<T>> &lat,
                               const Eigen::Ref<const Vector<T>> &alt,
                               const size_t num_threads) const
      -> std::tuple<Vector<T>, Vector<T>, Vector<T>> {
    broadcast::check_eigen_shape("lon", lon, "lat", lat, "alt", alt);

    const auto n = lon.size();

    Vector<T> lon_out(n);
    Vector<T> lat_out(n);
    Vector<T> alt_out(n);

    parallel_for(
        n,
        [&](size_t start, size_t end) {
          auto make_segment = [start, end](auto &vec) {
            return vec.segment(static_cast<Eigen::Index>(start),
                               static_cast<Eigen::Index>(end - start));
          };

          auto lon_segment = make_segment(lon);
          auto lat_segment = make_segment(lat);
          auto alt_segment = make_segment(alt);
          auto lon_out_segment = make_segment(lon_out);
          auto lat_out_segment = make_segment(lat_out);
          auto alt_out_segment = make_segment(alt_out);

          for (const auto &[lon_i, lat_i, alt_i, lon_o_i, lat_o_i, alt_o_i] :
               std::views::zip(lon_segment, lat_segment, alt_segment,
                               lon_out_segment, lat_out_segment,
                               alt_out_segment)) {
            auto lla = geographic::Coordinates::transform<T>(
                target, {lon_i, lat_i, alt_i});
            lon_o_i = boost::geometry::get<0>(lla);
            lat_o_i = boost::geometry::get<1>(lla);
            alt_o_i = boost::geometry::get<2>(lla);
          }
        },
        num_threads);
    return std::make_tuple(lon_out, lat_out, alt_out);
  }

  /// Get a tuple that fully encodes the state of this instance.
  [[nodiscard]] auto getstate() const -> nanobind::tuple {
    auto spheroid = this->spheroid();
    return nanobind::make_tuple(spheroid.semi_major_axis(),
                                spheroid.flattening());
  }

  /// Restore the state of this instance from the given tuple.
  /// @param[in] state The state to restore.
  /// @return A pointer to the restored instance.
  static auto setstate(nanobind::tuple &state) -> Coordinates {
    if (state.size() != 2) {
      throw std::invalid_argument("Invalid state");
    }
    auto semi_major_axis = nanobind::cast<double>(state[0]);
    auto flattening = nanobind::cast<double>(state[1]);
    return Coordinates{geographic::Spheroid(semi_major_axis, flattening)};
  }
};

}  // namespace pyinterp::geometry::geographic::pybind
