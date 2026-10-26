// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <boost/geometry.hpp>
#include <cmath>
#include <concepts>
#include <numbers>

#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"
#include "pyinterp/geometry/point.hpp"

namespace pyinterp::geometry {

/// @brief Geodesic distance between ECEF points lying on the ellipsoid.
///
/// Passed as the distance strategy of an R-tree of ECEF points, it makes the
/// tree rank the neighbors by geodesic distance (see RTree::nearest_k). The
/// tree keeps searching with Euclidean distances: the chord between two
/// points of the ellipsoid is never longer than the geodesic joining them,
/// so once the chord to the next candidate exceeds the k-th smallest
/// geodesic distance found, no remaining point can be closer.
///
/// The points must lie on the ellipsoid (zero altitude): their geodetic
/// coordinates are then recovered exactly from ECEF in closed form, and the
/// chord bound holds.
///
/// @tparam T Floating-point type of the ECEF coordinates
template <std::floating_point T>
class EcefGeodesic {
 public:
  /// Marks the strategies whose distance is bounded below by the Euclidean
  /// distance between ECEF points
  using chord_bounded_tag = void;

  /// Relative margin applied to the chord bound. It covers the error of the
  /// geodesic formula: Andoyer's is accurate to about 1e-5 in relative terms.
  static constexpr double kMargin = 1e-3;

  /// @param[in] spheroid Ellipsoid of the ECEF coordinates
  explicit EcefGeodesic(const geographic::Spheroid& spheroid)
      : one_minus_e2_{1 - spheroid.first_eccentricity_squared()},
        andoyer_{
            static_cast<boost::geometry::srs::spheroid<double>>(spheroid)} {}

  /// @brief Geodesic distance between two points of the ellipsoid, in meters
  [[nodiscard]] auto operator()(const ECEF<T>& lhs, const ECEF<T>& rhs) const
      -> double {
    return boost::geometry::distance(to_geographic(lhs), to_geographic(rhs),
                                     andoyer_);
  }

 private:
  /// 1 - e², with e the first eccentricity
  double one_minus_e2_;
  /// Andoyer's formula on the ellipsoid
  boost::geometry::strategy::distance::andoyer<
      boost::geometry::srs::spheroid<double>>
      andoyer_;

  /// @brief Geodetic longitude and latitude of a point of the ellipsoid
  ///
  /// For a point at zero altitude, the geodetic latitude φ satisfies
  /// tan(φ) = z / ((1 - e²) p), with p the distance to the polar axis.
  [[nodiscard]] auto to_geographic(const ECEF<T>& point) const
      -> geographic::Point {
    constexpr auto kDegrees = 180.0 / std::numbers::pi;
    const auto x = static_cast<double>(boost::geometry::get<0>(point));
    const auto y = static_cast<double>(boost::geometry::get<1>(point));
    const auto z = static_cast<double>(boost::geometry::get<2>(point));
    return {std::atan2(y, x) * kDegrees,
            std::atan2(z, one_minus_e2_ * std::hypot(x, y)) * kDegrees};
  }
};

}  // namespace pyinterp::geometry
