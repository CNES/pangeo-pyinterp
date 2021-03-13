#pragma once
#include <boost/geometry/algorithms/area.hpp>
#include <boost/geometry/srs/spheroid.hpp>
#if BOOST_VERSION >= 107500
#include <boost/geometry/strategy/area.hpp>
#include <boost/geometry/strategy/geographic/area.hpp>
#else
#include <boost/geometry/strategies/area.hpp>
#include <boost/geometry/strategies/geographic/area.hpp>
#endif
#include <optional>

#include "pyinterp/geodetic/system.hpp"

namespace pyinterp::geodetic {

/// Calculate the area
template <typename Geometry>
[[nodiscard]] inline auto area(const Geometry& geometry,
                               const std::optional<System>& wgs) -> double {
  auto spheroid = wgs.has_value()
                      ? boost::geometry::srs::spheroid(wgs->semi_major_axis(),
                                                       wgs->semi_minor_axis())
                      : boost::geometry::srs::spheroid<double>();
  auto strategy = boost::geometry::strategy::area::geographic<
      boost::geometry::strategy::vincenty, 5>(spheroid);
  return boost::geometry::area(geometry, strategy);
}

}  // namespace pyinterp::geodetic