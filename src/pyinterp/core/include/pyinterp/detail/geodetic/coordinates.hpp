// Copyright (c) 2025 CNES
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
    // flattening (is only necessary for serialization/deserialization)
    f_ = _system.flattening();
    // first eccentricity squared
    e2_ = _system.first_eccentricity_squared();
    // inv_a2 = 1/(a*a)
    reciprocal_a_squared_ = 1 / (a_ * a_);
    // e2_squared = e2 * e2
    e2_squared_ = e2_ * e2_;
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
    const T x = boost::geometry::get<0>(ecef);
    const T y = boost::geometry::get<1>(ecef);
    const T z = boost::geometry::get<2>(ecef);

    const T lon = std::atan2(y, x);

    // Vermeille's method (2002)
    const T p = (x * x + y * y) * reciprocal_a_squared_;
    const T q = ((1.0 - e2_) * (z * z)) * reciprocal_a_squared_;
    const T r = (p + q - e2_squared_) / 6.0;

    const T s = (e2_squared_ * p * q) / (4.0 * r * r * r);
    /// The paper uses `(1+s+sqrt(s*(2+s)))^(1/3)`.
    /// std::cbrt is used for the cube root, which is often faster and more
    /// stable. The expression is algebraically equivalent to
    /// `cbrt(1 + s + sqrt(s * (2 + s)))`.
    const T t = std::cbrt(1.0 + s + std::sqrt(s * (2.0 + s)));
    const T u = r * (1.0 + t + 1.0 / t);
    const T v = std::sqrt((u * u) + (e2_squared_ * q));
    const T w = e2_ * (u + v - q) / (2.0 * v);
    const T k = std::sqrt(u + v + (w * w)) - w;
    const T d = k * std::sqrt((x * x) + (y * y)) / (k + e2_);

    const T lat = std::atan2(z, d);
    const T alt = (k + e2_ - 1.0) / k * std::sqrt((d * d) + (z * z));

    // Convert radians to degrees for your Geographic struct
    return {math::degrees(lon), math::degrees(lat), alt};
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
  double a_, e2_, f_, reciprocal_a_squared_, e2_squared_;
};

}  // namespace pyinterp::detail::geodetic
