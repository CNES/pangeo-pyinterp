// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <optional>
#include <tuple>
#include <utility>

#include "pyinterp/geodetic/algorithm.hpp"
#include "pyinterp/geodetic/line_string.hpp"
#include "pyinterp/geodetic/point.hpp"
#include "pyinterp/geodetic/spheroid.hpp"

namespace pyinterp::geodetic {

/// Calculate the crossover between two half-orbits.
class Crossover {
 public:
  /// Default constructor
  ///
  /// @param[in] half_orbit_1 first half-orbit.
  /// @param[in] half_orbit_2 second half-orbit.
  Crossover(LineString half_orbit_1, LineString half_orbit_2);

  /// Get the first half-orbit.
  [[nodiscard]] constexpr auto get_half_orbit_1() const -> LineString const& {
    return half_orbit_1_;
  }

  /// Get the second half_orbit.
  [[nodiscard]] constexpr auto get_half_orbit_2() const -> LineString const& {
    return half_orbit_2_;
  }

  /// Get the crossover point between the two passes.
  ///
  /// @return the crossover location.
  [[nodiscard]] auto search(const std::optional<Spheroid>& wgs) const
      -> std::optional<Point>;

  /// Test if there is a crossover point between the two passes.
  ///
  /// @return true if there is a crossover point.
  [[nodiscard]] inline auto exists(const std::optional<Spheroid>& wgs) const
      -> bool {
    return half_orbit_1_.intersects(half_orbit_2_, wgs);
  }

  /// Search nearest indexes from a given point.
  ///
  /// @param point the point to search.
  /// @param predicate the distance predicate, in meters.
  /// @return the indices found on the half-orbits #1 & #2 or none if no
  ///         crossover point is found.
  [[nodiscard]] auto nearest(const Point& point, double predicate,
                             DistanceStrategy strategy,
                             const std::optional<Spheroid>& wgs) const
      -> std::optional<std::tuple<size_t, size_t>>;

  /// Get a tuple that fully encodes the state of this instance
  [[nodiscard]] auto getstate() const -> pybind11::tuple {
    return pybind11::make_tuple(half_orbit_1_.getstate(),
                                half_orbit_2_.getstate());
  }

  /// Create a new instance from a registered state of an instance of this
  /// object.
  static auto setstate(const pybind11::tuple& state) -> Crossover {
    if (state.size() != 2) {
      throw std::runtime_error("invalid state");
    }
    auto half_orbit_1 = LineString::setstate(state[0]);
    auto half_orbit_2 = LineString::setstate(state[1]);
    return {std::move(half_orbit_1), std::move(half_orbit_2)};
  }

 private:
  LineString half_orbit_1_;
  LineString half_orbit_2_;
};

/// Calculate the crossover between two half-orbits.
///
/// @param[in] lon1 Longitude, in degrees, of the first half-orbit.
/// @param[in] lat1 Latitude, in degrees, of the first half-orbit.
/// @param[in] lon2 Longitude, in degrees, of the second half-orbit.
/// @param[in] lat2 Latitude, in degrees, of the second half-orbit.
/// @param[in] predicate the maximal distance predicate, in meters, to discard
/// the crossover point if it is too far.
/// @param[in] strategy the distance strategy.
/// @param[in] wgs the spheroid to use.
/// @param[in] cartesian_plane if true, the crossing point is calculated in the
/// Cartesian plane, otherwise in the geodesic plane. Warning, the calculation
/// in a Cartesian plane is valid if the entry positions in the half-orbits are
/// not too far from each other (less than 10 km).
/// @return A tuple containing the crossover point and the indexes closest to
/// the crossover point on both half orbits, or None if the half orbits do not
/// intersect.
auto crossover(const Eigen::Ref<const Eigen::VectorXd>& lon1,
               const Eigen::Ref<const Eigen::VectorXd>& lat1,
               const Eigen::Ref<const Eigen::VectorXd>& lon2,
               const Eigen::Ref<const Eigen::VectorXd>& lat2, double predicate,
               const DistanceStrategy strategy,
               const std::optional<Spheroid>& wgs, bool cartesian_plane)
    -> std::optional<std::tuple<Point, std::tuple<size_t, size_t>>>;

}  // namespace pyinterp::geodetic
