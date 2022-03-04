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
#include "pyinterp/geodetic/system.hpp"

namespace pyinterp::geodetic {

/// Calculate the crossover between two half-orbits.
class Crossover {
 public:
  /// Defaut constructor
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
  [[nodiscard]] inline auto search() const -> std::optional<Point> {
    return half_orbit_1_.intersection(half_orbit_2_);
  }

  /// Test if there is a crossover point between the two passes.
  ///
  /// @return true if there is a crossover point.
  [[nodiscard]] inline auto exists() const -> bool {
    return half_orbit_1_.intersects(half_orbit_2_);
  }

  /// Search nearest indexes from a given point.
  ///
  /// @param point the point to search.
  /// @param predicate the distance predicate, in meters.
  /// @return the indices found on the half-orbits #1 & #2 or none if no
  ///         crossover point is found.
  [[nodiscard]] auto nearest(const Point& point, double predicate,
                             DistanceStrategy strategy,
                             const std::optional<System>& wgs) const
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

}  // namespace pyinterp::geodetic
