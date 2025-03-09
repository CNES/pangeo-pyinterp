// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geodetic/point.hpp"

namespace pyinterp::geodetic {

auto Point::azimuth(const Point& other,
                    const std::optional<Spheroid>& wgs) const -> double {
  double result;
  if (wgs) {
    result = boost::geometry::azimuth(
        *this, other,
        boost::geometry::strategy::azimuth::geographic<>(
            static_cast<boost::geometry::srs::spheroid<double>>(*wgs)));
  } else {
    result = boost::geometry::azimuth(*this, other);
  }
  return detail::math::degrees(result);
}

}  // namespace pyinterp::geodetic
