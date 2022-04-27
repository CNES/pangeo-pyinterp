// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <algorithm>
#include <optional>

#include "pyinterp/detail/geodetic/spheroid.hpp"
#include "pyinterp/detail/geometry/point.hpp"
#include "pyinterp/detail/math.hpp"

namespace pyinterp::detail::geodetic {

/// World Geodetic Coordinates System
class Coordinates {
 public:
  /// Default constructor.
  explicit Coordinates(const std::optional<Spheroid> &spheroid) {
    auto _system = spheroid.value_or(Spheroid());
    // semi-major axis
    a_ = _system.semi_major_axis();
    // flatenning (is only necessary for serialization/deserialization)
    f_ = _system.flattening();
    // first eccentricity squared
    e2_ = _system.first_eccentricity_squared();
    // a1 = a*e2
    a1_ = a_ * e2_;
    // a2 = a1*a1
    a2_ = a1_ * a1_;
    // a3 = a1*e2/2
    a3_ = a1_ * (e2_ * 0.5);
    // a4 = 2.5*a2
    a4_ = 2.5 * a2_;
    // a5 = a1+a3
    a5_ = a1_ + a3_;
    // a6 = 1-e2
    a6_ = 1 - e2_;
  }

  // Default destructor
  virtual ~Coordinates() = default;

  /// Default copy constructor
  Coordinates(const Coordinates &) = default;

  /// Default copy assignment operator
  auto operator=(const Coordinates &) -> Coordinates & = default;

  /// Move constructor
  Coordinates(Coordinates &&) noexcept = default;

  /// Move assignment operator
  auto operator=(Coordinates &&) noexcept -> Coordinates & = default;

  /// Gets the spheroid model used by this coordinate system
  [[nodiscard]] inline auto spheroid() const noexcept -> Spheroid {
    return {a_, f_};
  }

  /// Converts Cartesian coordinates to Geographic latitude, longitude, and
  /// altitude. Cartesian coordinates should be in meters. The returned latitude
  /// and longitude are in degrees, and the altitude will be in meters.
  template <typename T>
  auto ecef_to_lla(const geometry::Point3D<T> &ecef) const noexcept
      -> geometry::EquatorialPoint3D<T> {
    const double x = boost::geometry::get<0>(ecef);
    const double y = boost::geometry::get<1>(ecef);
    const double z = boost::geometry::get<2>(ecef);
    const double zp = std::abs(z);
    const double w2 = math::sqr(x) + math::sqr(y);
    const double w = std::sqrt(w2);
    const double inv_r2 = 1 / (w2 + math::sqr(z));
    const double inv_r = std::sqrt(inv_r2);
    const double s2 = math::sqr(z) * inv_r2;
    const double c2 = w2 * inv_r2;

    double u = a2_ * inv_r;
    double v = a3_ - a4_ * inv_r;
    double s;
    double c;
    double ss;
    double lat;

    if (c2 > 0.3) {
      s = (zp * inv_r) * (1.0 + c2 * (a1_ + u + s2 * v) * inv_r);
      lat = std::asin(s);
      ss = s * s;
      c = std::sqrt(1.0 - ss);
    } else {
      c = (w * inv_r) * (1.0 - s2 * (a5_ - u - c2 * v) * inv_r);
      lat = std::acos(c);
      ss = 1.0 - c * c;
      s = std::sqrt(ss);
    }

    const double g = 1.0 - e2_ * ss;
    const double rg = a_ / std::sqrt(g);
    const double rf = a6_ * rg;
    u = w - rg * c;
    v = zp - rf * s;
    const double f = c * u + s * v;
    const double m = c * v - s * u;
    const double p = m / (rf / g + f);
    lat += p;
    if (z < 0.0) {
      lat = -lat;
    }
    return {T(math::atan2d(y, x)), T(math::degrees(lat)), T(f + m * p * 0.5)};
  }

  /// Converts Geographic coordinates latitude, longitude, and altitude to
  /// Cartesian coordinates. The latitude and longitude should be in degrees and
  /// the altitude in meters. The returned ECEF coordinates will be in meters.
  template <typename T>
  inline auto lla_to_ecef(const geometry::EquatorialPoint3D<T> &lla)
      const noexcept -> geometry::Point3D<T> {
    auto [sinx, cosx] = math::sincosd(boost::geometry::get<0>(lla));
    auto [siny, cosy] = math::sincosd(boost::geometry::get<1>(lla));
    auto n = a_ / std::sqrt(1.0 - e2_ * math::sqr(siny));
    auto alt = boost::geometry::get<2>(lla);
    return {T((n + alt) * cosy * cosx), T((n + alt) * cosy * sinx),
            T((n * (1.0 - e2_) + alt) * siny)};
  }

  /// Transform points between two coordinate systems defined by the
  /// Coordinates instances this and target.
  template <typename T>
  inline auto transform(const Coordinates &target,
                        const geometry::EquatorialPoint3D<T> &lla)
      const noexcept -> geometry::EquatorialPoint3D<T> {
    return target.ecef_to_lla(lla_to_ecef(lla));
  }

 private:
  double a_, f_, e2_, a1_, a2_, a3_, a4_, a5_, a6_;
};

}  // namespace pyinterp::detail::geodetic
