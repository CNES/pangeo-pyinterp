// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <boost/geometry/srs/spheroid.hpp>
#include <cmath>
#include <format>
#include <numbers>
#include <string>

#include "pyinterp/math.hpp"

namespace pyinterp::geometry::geographic {

/// @brief World Geodetic System (WGS) ellipsoid representation
class Spheroid {
 public:
  /// @brief Default constructor - initializes to WGS84 parameters
  constexpr Spheroid() noexcept = default;

  /// @brief Virtual destructor for polymorphic use
  virtual ~Spheroid() = default;

  /// Copy operations
  Spheroid(const Spheroid&) = default;
  auto operator=(const Spheroid&) -> Spheroid& = default;

  /// Move operations
  Spheroid(Spheroid&&) noexcept = default;
  auto operator=(Spheroid&&) noexcept -> Spheroid& = default;

  /// @brief Construct spheroid with given ellipsoid parameters
  /// @param[in] semi_major_axis Semi-major axis of ellipsoid, in meters (a)
  /// @param[in] flattening Flattening of ellipsoid (f)
  constexpr Spheroid(const double semi_major_axis,
                     const double flattening) noexcept
      : semi_major_axis_{semi_major_axis}, flattening_{flattening} {}

  /// @brief Gets the semi-major axis of the ellipsoid
  /// @return a (in meters)
  [[nodiscard]] constexpr auto semi_major_axis() const noexcept -> double {
    return semi_major_axis_;
  }

  /// @brief Gets the flattening of the ellipsoid
  /// @return f = (a - b) / a
  [[nodiscard]] constexpr auto flattening() const noexcept -> double {
    return flattening_;
  }

  /// @brief Gets the semi-minor axis (polar radius)
  /// @return b = a(1 - f)
  [[nodiscard]] constexpr auto semi_minor_axis() const noexcept -> double {
    return semi_major_axis_ * (1.0 - flattening_);
  }

  /// @brief Gets the first eccentricity squared
  /// @return e² = (a² - b²) / a²
  [[nodiscard]] constexpr auto first_eccentricity_squared() const noexcept
      -> double {
    // Numerically stable formula: e² = f(2 - f)
    return flattening_ * (2.0 - flattening_);
  }

  /// @brief Gets the second eccentricity squared
  /// @return e'² = (a² - b²) / b²
  [[nodiscard]] constexpr auto second_eccentricity_squared() const noexcept
      -> double {
    const double b = semi_minor_axis();
    const double b2 = b * b;
    return (semi_major_axis_ * semi_major_axis_ - b2) / b2;
  }

  /// @brief Gets the equatorial circumference
  /// @param[in] semi_major_axis If true, returns 2πa; otherwise returns 2πb
  /// @return Circumference in meters
  [[nodiscard]] constexpr auto equatorial_circumference(
      const bool semi_major_axis = true) const noexcept -> double {
    constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;
    return two_pi * (semi_major_axis ? semi_major_axis_ : semi_minor_axis());
  }

  /// @brief Gets the polar radius of curvature
  /// @return c = a² / b
  [[nodiscard]] constexpr auto polar_radius_of_curvature() const noexcept
      -> double {
    return (semi_major_axis_ * semi_major_axis_) / semi_minor_axis();
  }

  /// @brief Gets the equatorial radius of curvature for a meridian
  /// @return b² / a
  [[nodiscard]] constexpr auto equatorial_radius_of_curvature() const noexcept
      -> double {
    const double b = semi_minor_axis();
    return (b * b) / semi_major_axis_;
  }

  /// @brief Gets the axis ratio
  /// @return b / a
  [[nodiscard]] constexpr auto axis_ratio() const noexcept -> double {
    return semi_minor_axis() / semi_major_axis_;
  }

  /// @brief Gets the linear eccentricity
  /// @return E = √(a² - b²)
  [[nodiscard]] __CONSTEXPR auto linear_eccentricity() const noexcept
      -> double {
    const double b = semi_minor_axis();
    return std::sqrt(semi_major_axis_ * semi_major_axis_ - b * b);
  }

  /// @brief Gets the mean radius
  /// @return R₁ = (2a + b) / 3
  [[nodiscard]] constexpr auto mean_radius() const noexcept -> double {
    return (2.0 * semi_major_axis_ + semi_minor_axis()) / 3.0;
  }

  /// @brief Gets the authalic radius (radius of sphere with same surface area)
  /// @return R₂ = √[(a² + ab²/E × ln((a+E)/b)) / 2]
  [[nodiscard]] __CONSTEXPR auto authalic_radius() const noexcept -> double {
    const double b = semi_minor_axis();
    const double E = linear_eccentricity();
    const double a2 = semi_major_axis_ * semi_major_axis_;
    const double b2 = b * b;

    const double term =
        a2 + (semi_major_axis_ * b2 / E) * std::log((semi_major_axis_ + E) / b);
    return std::sqrt(term * 0.5);
  }

  /// @brief Gets the volumetric radius (radius of sphere with same volume)
  /// @return R₃ = ∛(a²b)
  [[nodiscard]] __CONSTEXPR auto volumetric_radius() const noexcept -> double {
    return std::cbrt(semi_major_axis_ * semi_major_axis_ * semi_minor_axis());
  }

  /// @brief Gets the geocentric radius at the given latitude
  /// @param latitude Latitude in degrees
  /// @return R(φ) = √[((a²cos(φ))² + (b²sin(φ))²) / ((a·cos(φ))² +
  /// (b·sin(φ))²)]
  [[nodiscard]] __CONSTEXPR auto geocentric_radius(
      const double latitude) const noexcept -> double {
    const double cos_phi = math::cosd(latitude);
    const double sin_phi = math::sind(latitude);
    const double b = semi_minor_axis();

    const double a2 = semi_major_axis_ * semi_major_axis_;
    const double b2 = b * b;

    const double numerator =
        (a2 * cos_phi) * (a2 * cos_phi) + (b2 * sin_phi) * (b2 * sin_phi);
    const double denominator =
        (semi_major_axis_ * cos_phi) * (semi_major_axis_ * cos_phi) +
        (b * sin_phi) * (b * sin_phi);

    return std::sqrt(numerator / denominator);
  }

  /// @brief Spaceship operator for three-way comparison
  [[nodiscard]] constexpr auto operator<=>(const Spheroid& rhs) const noexcept =
      default;

  /// @brief Converts to boost::geometry::srs::spheroid
  [[nodiscard]] explicit operator boost::geometry::srs::spheroid<double>()
      const {
    return {semi_major_axis_, semi_minor_axis()};
  }

  /// @brief Get a string representation of this instance
  [[nodiscard]] explicit operator std::string() const {
    return std::format("Spheroid(a={:.9f}, b={:.9f}, f={:.9f})",
                       semi_major_axis_, semi_minor_axis(), flattening_);
  }

 private:
  // WGS84 parameters (default values)
  double semi_major_axis_{6'378'137.0};       // meters
  double flattening_{1.0 / 298.257'223'563};  // dimensionless
};

}  // namespace pyinterp::geometry::geographic
