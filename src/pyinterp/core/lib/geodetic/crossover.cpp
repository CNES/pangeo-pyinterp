// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geodetic/crossover.hpp"

namespace pyinterp::geodetic {

Crossover::Crossover(LineString half_orbit_1, LineString half_orbit_2)
    : half_orbit_1_(std::move(half_orbit_1)),
      half_orbit_2_(std::move(half_orbit_2)) {}

auto Crossover::nearest(const Point& point, const double predicate,
                        const DistanceStrategy strategy,
                        const std::optional<System>& wgs) const
    -> std::optional<std::tuple<size_t, size_t>> {
  auto ix1 = half_orbit_1_.nearest(point);
  if (half_orbit_1_[ix1].distance(point, strategy, wgs) > predicate) {
    return {};
  }

  auto ix2 = half_orbit_2_.nearest(point);
  if (half_orbit_2_[ix2].distance(point, strategy, wgs) > predicate) {
    return {};
  }

  return std::make_tuple(ix1, ix2);
}

}  // namespace pyinterp::geodetic
