// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <boost/geometry/srs/spheroid.hpp>
#include <cstdint>
#include <optional>

#include "pyinterp/geometry/geographic/spheroid.hpp"

namespace pyinterp::geometry::geographic {

/// Strategy for geodetic calculations
enum class StrategyMethod : std::int8_t {
  kAndoyer,  ///< Andoyer
  kKarney,   ///< Karney
  kThomas,   ///< Thomas
  kVincenty  ///< Vincenty
};

/// @brief Create spheroid from optional Spheroid
/// @param[in] wgs Optional Spheroid
/// @return Boost geometry spheroid
[[nodiscard]] inline auto make_spheroid(const std::optional<Spheroid> &wgs)
    -> boost::geometry::srs::spheroid<double> {
  return wgs.has_value() ? boost::geometry::srs::spheroid<double>(*wgs)
                         : boost::geometry::srs::spheroid<double>{};
}

}  // namespace pyinterp::geometry::geographic
