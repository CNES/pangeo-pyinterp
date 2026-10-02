// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <boost/geometry.hpp>
#include <boost/geometry/index/parameters.hpp>
#include <boost/geometry/strategies/distance/detail.hpp>
#include <boost/geometry/strategies/index/geographic.hpp>
#include <cmath>
#include <type_traits>

namespace pyinterp::geometry {
namespace detail {

/// @brief Distance between a point and a box on the spheroid.
///
/// Replacement for Boost's @c geographic_cross_track_point_box. When the box
/// crosses the antimeridian (its maximum longitude exceeds 180°), Boost picks
/// the wrong meridian edge of the box for points located west of the box: the
/// "midway" meridian is computed as `(lon_min - lon_max) / 2 + π` instead of
/// `(lon_min + lon_max) / 2 - π`. The returned distance then largely
/// overestimates the true one, and the R-tree nearest-neighbour search prunes
/// nodes that contain the actual nearest neighbours (see GitHub issue #38).
///
/// This strategy selects the closest edge from the longitudinal gaps between
/// the point and both edges of the box, which is valid whatever the
/// normalization of the box or of the point.
template <typename FormulaPolicy = boost::geometry::strategy::andoyer,
          typename Spheroid = boost::geometry::srs::spheroid<double>,
          typename CalculationType = void>
class GeographicPointBoxDistance {
 public:
  /// Strategy used to compute the distance between a point and a meridian
  /// segment of the box.
  using ps_strategy_t =
      boost::geometry::strategy::distance::geographic_cross_track<
          FormulaPolicy, Spheroid, CalculationType>;

  /// Type of the computed distance
  template <typename Point, typename Box>
  struct return_type
      : boost::geometry::strategy::distance::services::return_type<
            ps_strategy_t, Point, boost::geometry::point_type_t<Box>> {};

  /// Constructor
  /// @param[in] spheroid Spheroid used for the calculations
  explicit GeographicPointBoxDistance(const Spheroid& spheroid = Spheroid())
      : spheroid_(spheroid) {}

  /// Calculate the distance between a point and a box
  /// @param[in] point Point
  /// @param[in] box Box
  /// @return Distance between the point and the box
  template <typename Point, typename Box>
  [[nodiscard]] auto apply(const Point& point, const Box& box) const ->
      typename return_type<Point, Box>::type {
    using result_t = typename return_type<Point, Box>::type;
    using box_point_t = boost::geometry::point_type_t<Box>;

    box_point_t bottom_left;
    box_point_t bottom_right;
    box_point_t top_left;
    box_point_t top_right;
    boost::geometry::detail::assign_box_corners(box, bottom_left, bottom_right,
                                                top_left, top_right);

    const auto plon = boost::geometry::get_as_radian<0>(point);
    const auto plat = boost::geometry::get_as_radian<1>(point);
    const auto lon_min = boost::geometry::get_as_radian<0>(bottom_left);
    const auto lat_min = boost::geometry::get_as_radian<1>(bottom_left);
    const auto lon_max = boost::geometry::get_as_radian<0>(top_right);
    const auto lat_max = boost::geometry::get_as_radian<1>(top_right);

    const auto two_pi = boost::geometry::math::two_pi<result_t>();
    const auto ps_strategy = ps_strategy_t(spheroid_);

    // Longitudinal offset of the point east of the western edge of the box,
    // reduced to [0, 2π[.
    auto offset = static_cast<result_t>(std::fmod(plon - lon_min, two_pi));
    if (offset < 0) {
      offset += two_pi;
    }
    const auto width = static_cast<result_t>(lon_max - lon_min);

    // The point lies within the longitude band of the box: the distance is
    // measured along the meridian.
    if (offset <= width) {
      if (plat > lat_max) {
        return ps_strategy.vertical_or_meridian(plat, lat_max);
      }
      if (plat < lat_min) {
        return ps_strategy.vertical_or_meridian(lat_min, plat);
      }
      return result_t(0);
    }

    // Otherwise, the closest edge of the box is the one with the smallest
    // longitudinal gap with the point.
    const auto east_gap = offset - width;
    const auto west_gap = two_pi - offset;
    return west_gap < east_gap
               ? ps_strategy.apply(point, bottom_left, top_left)
               : ps_strategy.apply(point, bottom_right, top_right);
  }

  /// Get the spheroid used for the calculations
  [[nodiscard]] auto model() const -> Spheroid { return spheroid_; }

 private:
  Spheroid spheroid_;
};

}  // namespace detail

/// @brief R-tree strategies for geographic coordinates.
///
/// Identical to Boost's default index strategy for the geographic coordinate
/// system, except for the point/box distance used to prune the nodes during
/// nearest-neighbour searches (see @ref detail::GeographicPointBoxDistance).
template <typename FormulaPolicy = boost::geometry::strategy::andoyer,
          typename Spheroid = boost::geometry::srs::spheroid<double>,
          typename CalculationType = void>
class GeographicIndexStrategy
    : public boost::geometry::strategies::index::geographic<
          FormulaPolicy, Spheroid, CalculationType> {
  using base_t =
      boost::geometry::strategies::index::geographic<FormulaPolicy, Spheroid,
                                                     CalculationType>;

 public:
  /// Default constructor
  GeographicIndexStrategy() = default;

  /// Constructor
  /// @param[in] spheroid Spheroid used for the calculations
  explicit GeographicIndexStrategy(const Spheroid& spheroid)
      : base_t(spheroid) {}

  using base_t::distance;

  /// Get the strategy computing the distance between a point and a box
  template <typename Geometry1, typename Geometry2>
  [[nodiscard]] auto distance(
      const Geometry1& /*unused*/, const Geometry2& /*unused*/,
      boost::geometry::strategies::distance::detail::enable_if_pb_t<
          Geometry1, Geometry2>* /*unused*/
      = nullptr) const {
    return detail::GeographicPointBoxDistance<FormulaPolicy, Spheroid,
                                              CalculationType>(
        base_t::m_spheroid);
  }
};

/// @brief Parameters of the R-tree used to index points of type @c Point.
///
/// Points expressed in a geographic coordinate system use
/// @ref GeographicIndexStrategy; the other coordinate systems use Boost's
/// default strategies.
template <typename Point, typename Parameters>
struct rtree_parameters {
  using type = Parameters;
};

/// @brief Specialization for geographic coordinates
template <typename Point, typename Parameters>
  requires std::is_same_v<boost::geometry::cs_tag_t<Point>,
                          boost::geometry::geographic_tag>
struct rtree_parameters<Point, Parameters> {
  using type =
      boost::geometry::index::parameters<Parameters, GeographicIndexStrategy<>>;
};

}  // namespace pyinterp::geometry

// Boost.Geometry traits
namespace boost::geometry::strategy::distance::services {

template <typename FormulaPolicy, typename Spheroid, typename CalculationType>
struct tag<pyinterp::geometry::detail::GeographicPointBoxDistance<
    FormulaPolicy, Spheroid, CalculationType>> {
  using type = strategy_tag_distance_point_box;
};

template <typename FormulaPolicy, typename Spheroid, typename CalculationType,
          typename Point, typename Box>
struct return_type<pyinterp::geometry::detail::GeographicPointBoxDistance<
                       FormulaPolicy, Spheroid, CalculationType>,
                   Point, Box>
    : pyinterp::geometry::detail::GeographicPointBoxDistance<
          FormulaPolicy, Spheroid,
          CalculationType>::template return_type<Point, Box> {};

template <typename FormulaPolicy, typename Spheroid, typename CalculationType>
struct comparable_type<pyinterp::geometry::detail::GeographicPointBoxDistance<
    FormulaPolicy, Spheroid, CalculationType>> {
  using type = pyinterp::geometry::detail::GeographicPointBoxDistance<
      FormulaPolicy, Spheroid, CalculationType>;
};

template <typename FormulaPolicy, typename Spheroid, typename CalculationType>
struct get_comparable<pyinterp::geometry::detail::GeographicPointBoxDistance<
    FormulaPolicy, Spheroid, CalculationType>> {
  using strategy_t = pyinterp::geometry::detail::GeographicPointBoxDistance<
      FormulaPolicy, Spheroid, CalculationType>;

  static auto apply(const strategy_t& strategy) -> strategy_t {
    return strategy;
  }
};

}  // namespace boost::geometry::strategy::distance::services
