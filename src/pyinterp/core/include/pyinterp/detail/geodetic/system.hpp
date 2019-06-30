// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include "pyinterp/detail/math.hpp"
#include <iomanip>
#include <sstream>

namespace pyinterp {
namespace detail {
namespace geodetic {

/// World Geodetic System (WGS)
class System {
 public:
  /// The constructor defaults the ellipsoid parameters to WGS84.
  System() noexcept = default;

  /// Destructor
  virtual ~System() = default;

  /// Copy constructor
  ///
  /// @param rhs right value
  System(const System& rhs) = default;

  /// Move constructor
  ///
  /// @param rhs right value
  System(System&& rhs) = default;

  /// Copy assignment operator
  ///
  /// @param rhs right value
  System& operator=(const System& rhs) = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  System& operator=(System&& rhs) = default;

  /// Obtains an instance of System with the given ellipsoid
  /// parameters
  ///
  /// @param semi_major_axis  Semi-major axis of ellipsoid, in meters
  /// @param flattening  Flattening of ellipsoid
  inline System(double semi_major_axis, double flattening) noexcept
      : semi_major_axis_(semi_major_axis), flattening_(flattening) {}

  /// Gets the Semi-major axis of the defined ellipsoid
  ///
  /// @return \f$a\f$
  constexpr double semi_major_axis() const noexcept { return semi_major_axis_; }

  /// Gets the flattening of the defined ellipsoid
  ///
  /// @return \f$f=\frac{a-b}{a}\f$
  constexpr double flattening() const noexcept { return flattening_; }

  /// Gets the semi-minor axis (also named polar radius)
  ///
  /// @return \f$b=a(1-f)\f$
  constexpr double semi_minor_axis() const noexcept {
    return semi_major_axis_ * (1 - flattening_);
  }

  /// Gets the first eccentricity squared
  ///
  /// @return \f$e^2=\frac{a^2-b^2}{a^2}\f$
  constexpr double first_eccentricity_squared() const noexcept {
    double a2 = math::sqr(semi_major_axis_);
    return (a2 - math::sqr(semi_minor_axis())) / a2;
  }

  /// Gets the second eccentricity squared
  ///
  /// @return \f$e^2=\frac{a^2-b^2}{b^2}\f$
  constexpr double second_eccentricity_squared() const noexcept {
    double b2 = math::sqr(semi_minor_axis());
    return (math::sqr(semi_major_axis_) - b2) / b2;
  }

  /// Gets the equatorial circumference
  ///
  /// @return \f$2\pi \times a\f$ if semi_major_axis is true otherwise
  ///   \f$2\pi \times b\f$
  constexpr double equatorial_circumference(const bool semi_major_axis) const
      noexcept {
    return math::two_pi<double>() *
           (semi_major_axis ? semi_major_axis_ : semi_minor_axis());
  }

  /// Gets the polar radius of curvature
  ///
  /// @return \f$\frac{a^2}{b}\f$
  constexpr double polar_radius_of_curvature() const noexcept {
    return math::sqr(semi_major_axis_) / semi_minor_axis();
  }

  /// Gets the equatorial radius of curvature for a meridian
  ///
  /// @return \f$\frac{b^2}{a}\f$
  constexpr double equatorial_radius_of_curvature() const noexcept {
    return math::sqr(semi_minor_axis()) / semi_major_axis_;
  }

  /// Gets the axis ratio
  ///
  /// @return \f$\frac{b}{a}\f$
  constexpr double axis_ratio() const noexcept {
    return semi_minor_axis() / semi_major_axis_;
  }

  /// Gets the linear eccentricity
  ///
  /// @return \f$E=\sqrt{{a^2}-{b^2}}\f$
  inline double linear_eccentricity() const noexcept {
    return std::sqrt(math::sqr(semi_major_axis_) -
                     math::sqr(semi_minor_axis()));
  }

  /// Gets the mean radius
  ///
  /// @return \f$R_1=\frac{2a+b}{3}\f$
  constexpr double mean_radius() const noexcept {
    return (2 * semi_major_axis_ + semi_minor_axis()) / 3.0;
  }

  /// Gets the authalic radius
  ///
  /// @return \f$R_2=\sqrt{\frac{a^2+\frac{ab^2}{E}ln(\frac{a + E}{b})}{2}}\f$
  inline double authalic_radius() const noexcept {
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
  inline double volumetric_radius() const noexcept {
    return std::pow(math::sqr(semi_major_axis_) * semi_minor_axis(), 1.0 / 3.0);
  }

  /// Tests if two Geodetic system are equal
  ///
  /// @param rhs Other geodetic system to compare
  /// @return rue if the two instances are equal
  constexpr bool operator==(const System& rhs) const noexcept {
    return semi_major_axis_ == rhs.semi_major_axis_ &&
           flattening_ == rhs.flattening_;
  }

  /// Tests if two Geodetic system are different
  ///
  /// @param rhs Other geodetic system to compare
  /// @return rue if the two instances are different
  constexpr bool operator!=(const System& rhs) const noexcept {
    return semi_major_axis_ != rhs.semi_major_axis_ ||
           flattening_ != rhs.flattening_;
  }

  /// Get a string representing this instance.
  ///
  /// @return a string holding the converted instance.
  explicit operator std::string() const {
    auto ss = std::stringstream();
    ss << std::setprecision(9) << "System(a=" << semi_major_axis_
       << ", b=" << semi_minor_axis() << ", f=" << flattening_ << ")";
    return ss.str();
  }

 private:
  // WGS84 parameters
  double semi_major_axis_{6'378'137.0};
  double flattening_{1 / 298.257'223'563};
};

}  // namespace geodetic
}  // namespace detail
}  // namespace pyinterp
