// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <Eigen/Core>
#include <boost/geometry/algorithms/area.hpp>
#include <boost/geometry/algorithms/covered_by.hpp>
#include <boost/geometry/srs/spheroid.hpp>
#if BOOST_VERSION >= 107500
#include <boost/geometry/strategy/area.hpp>
#include <boost/geometry/strategy/geographic/area.hpp>
#else
#include <boost/geometry/strategies/area.hpp>
#include <boost/geometry/strategies/geographic/area.hpp>
#endif
#include <boost/geometry/strategies/geographic/distance_andoyer.hpp>
#include <boost/geometry/strategies/geographic/distance_thomas.hpp>
#include <boost/geometry/strategies/geographic/distance_vincenty.hpp>
#include <optional>

#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/detail/thread.hpp"
#include "pyinterp/geodetic/spheroid.hpp"

namespace pyinterp::geodetic {

/// Distance calculation strategy.
enum DistanceStrategy { kAndoyer = 0x0, kThomas = 0x1, kVincenty = 0x2 };

using Andoyer = boost::geometry::strategy::distance::andoyer<
    boost::geometry::srs::spheroid<double>>;
using Thomas = boost::geometry::strategy::distance::thomas<
    boost::geometry::srs::spheroid<double>>;
using Vincenty = boost::geometry::strategy::distance::vincenty<
    boost::geometry::srs::spheroid<double>>;

/// Calculate the area
template <typename Geometry>
[[nodiscard]] inline auto area(const Geometry &geometry,
                               const std::optional<Spheroid> &wgs) -> double {
  auto spheroid =
      wgs.has_value()
          ? static_cast<boost::geometry::srs::spheroid<double>>(*wgs)
          : boost::geometry::srs::spheroid<double>();
  auto strategy = boost::geometry::strategy::area::geographic<
      boost::geometry::strategy::vincenty, 5>(spheroid);
  return boost::geometry::area(geometry, strategy);
}

/// Checks if the first geometry is inside or on border the second geometry
/// using the specified strategy.
template <typename Geometry1, typename Geometry2>
[[nodiscard]] inline auto covered_by(
    const Geometry2 &geometry2, const Eigen::Ref<const Eigen::VectorXd> &lon,
    const Eigen::Ref<const Eigen::VectorXd> &lat, const size_t num_threads)
    -> pybind11::array_t<bool> {
  detail::check_eigen_shape("lon", lon, "lat", lat);
  auto size = lon.size();
  auto result =
      pybind11::array_t<bool>(pybind11::array::ShapeContainer{{size}});
  auto _result = result.template mutable_unchecked<1>();

  {
    pybind11::gil_scoped_release release;

    // Captures the detected exceptions in the calculation function
    // (only the last exception captured is kept)
    auto except = std::exception_ptr(nullptr);

    detail::dispatch(
        [&](size_t start, size_t end) {
          try {
            for (auto ix = static_cast<int64_t>(start);
                 ix < static_cast<int64_t>(end); ++ix) {
              _result(ix) = boost::geometry::covered_by(
                  Geometry1(lon(ix), lat(ix)), geometry2);
            }
          } catch (...) {
            except = std::current_exception();
          }
        },
        size, num_threads);

    if (except != nullptr) {
      std::rethrow_exception(except);
    }
  }
  return result;
}

/// Calculate the distance between two geometries.
template <typename Geometry1, typename Geometry2>
[[nodiscard]] inline auto distance(const Geometry1 &geometry1,
                                   const Geometry2 &geometry2,
                                   const DistanceStrategy strategy,
                                   const std::optional<Spheroid> &wgs)
    -> double {
  auto spheroid =
      wgs.has_value()
          ? static_cast<boost::geometry::srs::spheroid<double>>(*wgs)
          : boost::geometry::srs::spheroid<double>();
  switch (strategy) {
    case kAndoyer:
      return boost::geometry::distance(geometry1, geometry2, Andoyer(spheroid));
      break;
    case kThomas:
      return boost::geometry::distance(geometry1, geometry2, Thomas(spheroid));
      break;
    case kVincenty:
      return boost::geometry::distance(geometry1, geometry2,
                                       Vincenty(spheroid));
      break;
  }
  throw std::invalid_argument("unknown strategy: " +
                              std::to_string(static_cast<int>(strategy)));
}

/// Calculate the distance between two geometries.
template <typename Geometry1, typename Geometry2>
[[nodiscard]] inline auto distance(const Geometry1 &geometry1,
                                   const Geometry2 &geometry2) -> double {
  return boost::geometry::distance(geometry1, geometry2);
}

/// Calculate the distance between coordinates.
template <typename Geometry, typename Strategy>
[[nodiscard]] inline auto coordinate_distances(
    const Eigen::Ref<const Eigen::VectorXd> &lon1,
    const Eigen::Ref<const Eigen::VectorXd> &lat1,
    const Eigen::Ref<const Eigen::VectorXd> &lon2,
    const Eigen::Ref<const Eigen::VectorXd> &lat2, const Strategy &strategy,
    const size_t num_threads) -> pybind11::array_t<double> {
  auto size = lon1.size();
  auto result =
      pybind11::array_t<double>(pybind11::array::ShapeContainer{{size}});
  auto _result = result.template mutable_unchecked<1>();

  {
    pybind11::gil_scoped_release release;

    // Captures the detected exceptions in the calculation function
    // (only the last exception captured is kept)
    auto except = std::exception_ptr(nullptr);

    detail::dispatch(
        [&](size_t start, size_t end) {
          try {
            for (auto ix = static_cast<int64_t>(start);
                 ix < static_cast<int64_t>(end); ++ix) {
              _result(ix) = boost::geometry::distance(
                  Geometry(lon1(ix), lat1(ix)), Geometry(lon2(ix), lat2(ix)),
                  strategy);
            }
          } catch (...) {
            except = std::current_exception();
          }
        },
        size, num_threads);

    if (except != nullptr) {
      std::rethrow_exception(except);
    }
  }
  return result;
}

/// Calculate the distance between coordinates.
template <typename Geometry>
[[nodiscard]] inline auto coordinate_distances(
    const Eigen::Ref<const Eigen::VectorXd> &lon1,
    const Eigen::Ref<const Eigen::VectorXd> &lat1,
    const Eigen::Ref<const Eigen::VectorXd> &lon2,
    const Eigen::Ref<const Eigen::VectorXd> &lat2,
    const DistanceStrategy strategy, const std::optional<Spheroid> &wgs,
    const size_t num_threads) -> pybind11::array_t<double> {
  detail::check_eigen_shape("lon1", lon1, "lat1", lat1, "lon2", lon2, "lat2",
                            lat2);
  auto spheroid =
      wgs.has_value()
          ? static_cast<boost::geometry::srs::spheroid<double>>(*wgs)
          : boost::geometry::srs::spheroid<double>();
  switch (strategy) {
    case kAndoyer:
      return coordinate_distances<Geometry, Andoyer>(
          lon1, lat1, lon2, lat2, Andoyer(spheroid), num_threads);
      break;
    case kThomas:
      return coordinate_distances<Geometry, Thomas>(
          lon1, lat1, lon2, lat2, Thomas(spheroid), num_threads);
      break;
    case kVincenty:
      return coordinate_distances<Geometry, Vincenty>(
          lon1, lat1, lon2, lat2, Vincenty(spheroid), num_threads);
      break;
  }
  throw std::invalid_argument("unknown strategy: " +
                              std::to_string(static_cast<int>(strategy)));
}

}  // namespace pyinterp::geodetic
