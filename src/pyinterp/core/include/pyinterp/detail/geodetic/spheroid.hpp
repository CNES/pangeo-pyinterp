// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <boost/geometry/srs/spheroid.hpp>
#include <iomanip>
#include <sstream>
#include <string>

#include "pyinterp/detail/math.hpp"

namespace pyinterp::detail::geodetic {

/// World Geodetic System (WGS)
class Spheroid {
 public:
  /// The constructor defaults the ellipsoid parameters to WGS84.
  Spheroid() noexcept = default;

  /// Destructor
  virtual ~Spheroid() = default;

  /// Copy constructor
  ///
  /// @param rhs right value
  Spheroid(const Spheroid &rhs) = default;

  /// Move constructor
  ///
  /// @param rhs right value
  Spheroid(Spheroid &&rhs) = default;

  /// Copy assignment operator
  ///
  /// @param rhs right value
  auto operator=(const Spheroid &rhs) -> Spheroid & = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  auto operator=(Spheroid &&rhs) -> Spheroid & = default;

  /// Obtains an instance of System with the given ellipsoid
  /// parameters
  ///
  /// @param semi_major_axis  Semi-major axis of ellipsoid, in meters
  /// @param flattening  Flattening of ellipsoid
  constexpr Spheroid(double semi_major_axis, double flattening) noexcept
      : semi_major_axis_(semi_major_axis), flattening_(flattening) {}

  /// Gets the Semi-major axis of the defined ellipsoid
  ///
  /// @return \f$a\f$
  [[nodiscard]] constexpr auto semi_major_axis() const noexcept -> double {
    return semi_major_axis_;
  }

  /// Gets the flattening of the defined ellipsoid
  ///
  /// @return \f$f=\frac{a-b}{a}\f$
  [[nodiscard]] constexpr auto flattening() const noexcept -> double {
    return flattening_;
  }

  /// Gets the semi-minor axis (also named polar radius)
  ///
  /// @return \f$b=a(1-f)\f$
  [[nodiscard]] constexpr auto semi_minor_axis() const noexcept -> double {
    return semi_major_axis_ * (1 - flattening_);
  }

  /// Gets the first eccentricity squared
  ///
  /// @return \f$e^2=\frac{a^2-b^2}{a^2}\f$
  [[nodiscard]] constexpr auto first_eccentricity_squared() const noexcept
      -> double {
    double a2 = math::sqr(semi_major_axis_);
    return (a2 - math::sqr(semi_minor_axis())) / a2;
  }

  /// Gets the second eccentricity squared
  ///
  /// @return \f$e^2=\frac{a^2-b^2}{b^2}\f$
  [[nodiscard]] constexpr auto second_eccentricity_squared() const noexcept
      -> double {
    double b2 = math::sqr(semi_minor_axis());
    return (math::sqr(semi_major_axis_) - b2) / b2;
  }

  /// Gets the equatorial circumference
  ///
  /// @return \f$2\pi \times a\f$ if semi_major_axis is true otherwise
  ///   \f$2\pi \times b\f$
  [[nodiscard]] constexpr auto equatorial_circumference(
      const bool semi_major_axis) const noexcept -> double {
    return math::two_pi<double>() *
           (semi_major_axis ? semi_major_axis_ : semi_minor_axis());
  }

  /// Gets the polar radius of curvature
  ///
  /// @return \f$\frac{a^2}{b}\f$
  [[nodiscard]] constexpr auto polar_radius_of_curvature() const noexcept
      -> double {
    return math::sqr(semi_major_axis_) / semi_minor_axis();
  }

  /// Gets the equatorial radius of curvature for a meridian
  ///
  /// @return \f$\frac{b^2}{a}\f$
  [[nodiscard]] constexpr auto equatorial_radius_of_curvature() const noexcept
      -> double {
    return math::sqr(semi_minor_axis()) / semi_major_axis_;
  }

  /// Gets the axis ratio
  ///
  /// @return \f$\frac{b}{a}\f$
  [[nodiscard]] constexpr auto axis_ratio() const noexcept -> double {
    return semi_minor_axis() / semi_major_axis_;
  }

  /// Gets the linear eccentricity
  ///
  /// @return \f$E=\sqrt{{a^2}-{b^2}}\f$
  [[nodiscard]] inline auto linear_eccentricity() const noexcept -> double {
    return std::sqrt(math::sqr(semi_major_axis_) -
                     math::sqr(semi_minor_axis()));
  }

  /// Gets the mean radius
  ///
  /// @return \f$R_1=\frac{2a+b}{3}\f$
  [[nodiscard]] constexpr auto mean_radius() const noexcept -> double {
    return (2 * semi_major_axis_ + semi_minor_axis()) / 3.0;
  }

  /// Gets the authalic radius
  ///
  /// @return \f$R_2=\sqrt{\frac{a^2+\frac{ab^2}{E}ln(\frac{a + E}{b})}{2}}\f$
  [[nodiscard]] inline auto authalic_radius() const noexcept -> double {
    return std::sqrt((math::sqr(semi_major_axis_) +
                      ((semi_major_axis_ * math::sqr(semi_minor_axis())) /
                       linear_eccentricity()) *
                          std::log((semi_major_axis_ + linear_eccentricity()) /
                                   semi_minor_axis())) *
                     0.5);
  }

  /// Gets the volumetric radius
  ///
  /// @return \f$R_3=\sqrt[3]{a^{2}b}\f$
  [[nodiscard]] inline auto volumetric_radius() const noexcept -> double {
    return std::pow(math::sqr(semi_major_axis_) * semi_minor_axis(), 1.0 / 3.0);
  }

  /// Gets the geocentric radius at the given latitude
  ///
  /// @param latitude  Latitude, in degrees
  /// @return \f$R(\phi)=\sqrt{\frac{{(a^{2}\cos(\phi))}^{2} +
  /// (b^{2}\sin(\phi))^{2}}{(a\cos(\phi))^{2} + (b\cos(\phi))^{2}}}\f$
  [[nodiscard]] inline auto geocentric_radius(double latitude) const noexcept
      -> double {
    const auto cos_phi = math::cosd(latitude);
    const auto sin_phi = math::sind(latitude);
    return std::sqrt((math::sqr(math::sqr(semi_major_axis_) * cos_phi) +
                      math::sqr(math::sqr(semi_minor_axis()) * sin_phi)) /
                     (math::sqr(semi_major_axis_ * cos_phi) +
                      math::sqr(semi_minor_axis() * sin_phi)));
  }

  /// Tests if two spheroid objects are equal
  ///
  /// @param rhs Other spheroid
  /// @return rue if the two instances are equal
  constexpr auto operator==(const Spheroid &rhs) const noexcept -> bool {
    return semi_major_axis_ == rhs.semi_major_axis_ &&
           flattening_ == rhs.flattening_;
  }

  /// Tests if two spheroid objects are not equal
  ///
  /// @param rhs Other spheroid
  /// @return rue if the two instances are different
  constexpr auto operator!=(const Spheroid &rhs) const noexcept -> bool {
    return semi_major_axis_ != rhs.semi_major_axis_ ||
           flattening_ != rhs.flattening_;
  }

  /// Converts this instance to a boost::geometry::srs::spheroid equivalent.
  ///
  /// @return A boost::geometry::srs::spheroid equivalent
  explicit inline operator boost::geometry::srs::spheroid<double>() const {
    return {semi_major_axis_, semi_minor_axis()};
  }

  /// Get a string representing this instance.
  ///
  /// @return a string holding the converted instance.
  explicit operator std::string() const {
    auto ss = std::stringstream();
    ss << std::setprecision(9) << "Spheroid(a=" << semi_major_axis_
       << ", b=" << semi_minor_axis() << ", f=" << flattening_ << ")";
    return ss.str();
  }

 private:
  // WGS84 parameters
  double semi_major_axis_{6'378'137.0};
  double flattening_{1 / 298.257'223'563};
};

}  // namespace pyinterp::detail::geodetic
