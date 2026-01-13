// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <cmath>
#include <concepts>
#include <optional>

#include "pyinterp/geometry/geographic/spheroid.hpp"
#include "pyinterp/geometry/point.hpp"
#include "pyinterp/math.hpp"

namespace pyinterp::geometry::geographic {

/// World Geodetic Coordinates System for transformations between
/// geodetic (lat/lon/alt) and Earth-Centered Earth-Fixed (ECEF) Cartesian
/// coordinates
class Coordinates {
 public:
  /// Constructor with optional spheroid (defaults to WGS84)
  /// @param[in] spheroid Optional spheroid definition
  explicit constexpr Coordinates(
      const std::optional<Spheroid>& spheroid = std::nullopt)
      : a_{spheroid.value_or(Spheroid{}).semi_major_axis()},
        f_{spheroid.value_or(Spheroid{}).flattening()},
        e2_{spheroid.value_or(Spheroid{}).first_eccentricity_squared()},
        reciprocal_a_squared_{1.0 / (a_ * a_)},
        e2_squared_{e2_ * e2_} {}

  /// Virtual destructor for polymorphic use
  virtual ~Coordinates() = default;

  /// Copy operations
  Coordinates(const Coordinates&) = default;
  auto operator=(const Coordinates&) -> Coordinates& = default;

  /// Move operations
  Coordinates(Coordinates&&) noexcept = default;
  auto operator=(Coordinates&&) noexcept -> Coordinates& = default;

  /// Gets the spheroid model used by this coordinate system
  /// @return Spheroid with semi-major axis and flattening
  [[nodiscard]] constexpr auto spheroid() const noexcept -> Spheroid {
    return {a_, f_};
  }

  /// Converts ECEF Cartesian coordinates to geodetic (LLA) coordinates
  /// Uses Vermeille's method (2002) for improved accuracy and speed
  /// @tparam T Floating-point type
  /// @param[in] ecef Cartesian coordinates (x, y, z) in meters
  /// @return Geodetic coordinates (longitude, latitude in degrees, altitude in
  /// meters)
  template <std::floating_point T>
  [[nodiscard]] constexpr auto ecef_to_lla(
      const geometry::ECEF<T>& ecef) const noexcept -> geometry::LLA<T> {
    const double x = boost::geometry::get<0>(ecef);
    const double y = boost::geometry::get<1>(ecef);
    const double z = boost::geometry::get<2>(ecef);

    // Vermeille's method (2002) - accurate and efficient
    const double p = (x * x + y * y) * reciprocal_a_squared_;
    const double q = ((1.0 - e2_) * (z * z)) * reciprocal_a_squared_;
    const double r = (p + q - e2_squared_) / 6.0;

    const double s = (e2_squared_ * p * q) / (4.0 * r * r * r);

    // Use cbrt for cube root (more stable than pow(x, 1/3))
    const double t = std::cbrt(1.0 + s + std::sqrt(s * (2.0 + s)));
    const double u = r * (1.0 + t + 1.0 / t);
    const double v = std::sqrt(u * u + e2_squared_ * q);
    const double w = e2_ * (u + v - q) / (2.0 * v);
    const double k = std::sqrt(u + v + w * w) - w;
    const double d = k * std::sqrt(x * x + y * y) / (k + e2_);

    // Compute geodetic coordinates
    const T lon = static_cast<T>(std::atan2(y, x));
    const T lat = static_cast<T>(std::atan2(z, d));
    const T alt =
        static_cast<T>((k + e2_ - 1.0) / k * std::sqrt(d * d + z * z));

    // Convert from radians to degrees
    return {math::degrees(lon), math::degrees(lat), alt};
  }

  /// Converts geodetic (LLA) coordinates to ECEF Cartesian coordinates
  /// @tparam T Floating-point type
  /// @param[in] lla Geodetic coordinates (longitude, latitude in degrees,
  /// altitude in meters)
  /// @return Cartesian ECEF coordinates (x, y, z) in meters
  template <std::floating_point T>
  [[nodiscard]] constexpr auto lla_to_ecef(
      const geometry::LLA<T>& lla) const noexcept -> geometry::ECEF<T> {
    const auto lon = boost::geometry::get<0>(lla);
    const auto lat = boost::geometry::get<1>(lla);
    const auto alt = boost::geometry::get<2>(lla);

    // Compute sin/cos for longitude and latitude
    const auto [sin_lon, cos_lon] = math::sincosd(lon);
    const auto [sin_lat, cos_lat] = math::sincosd(lat);

    // Prime vertical radius of curvature
    const double n = a_ / std::sqrt(1.0 - e2_ * sin_lat * sin_lat);

    // Compute ECEF coordinates
    return {static_cast<T>((n + alt) * cos_lat * cos_lon),
            static_cast<T>((n + alt) * cos_lat * sin_lon),
            static_cast<T>((n * (1.0 - e2_) + alt) * sin_lat)};
  }

  /// Transform geodetic coordinates from this coordinate system to another
  /// @tparam T Floating-point type
  /// @param[in] target Target coordinate system
  /// @param[in] lla Geodetic coordinates in this system
  /// @return Geodetic coordinates in target system
  template <std::floating_point T>
  [[nodiscard]] constexpr auto transform(
      const Coordinates& target, const geometry::LLA<T>& lla) const noexcept
      -> geometry::LLA<T> {
    // Two-step transformation: this LLA → ECEF → target LLA
    return target.ecef_to_lla(lla_to_ecef(lla));
  }

 private:
  double a_{};                     // Semi-major axis (meters)
  double f_{};                     // Flattening (for reconstruction)
  double e2_{};                    // First eccentricity squared
  double reciprocal_a_squared_{};  // 1/(a²) precomputed
  double e2_squared_{};            // e² × e² precomputed
};

}  // namespace pyinterp::geometry::geographic
