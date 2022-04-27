// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/numpy.h>

#include <Eigen/Core>
#include <algorithm>

#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/detail/geodetic/coordinates.hpp"
#include "pyinterp/detail/thread.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/geodetic/spheroid.hpp"

namespace pyinterp::geodetic {

/// Wrapper
class Coordinates : public detail::geodetic::Coordinates {
 public:
  /// The constructor defaults the ellipsoid parameters to WGS84.
  explicit Coordinates(const std::optional<Spheroid> &spheroid)
      : detail::geodetic::Coordinates(spheroid) {}

  /// Gets the spheroid model used by this coordinate system
  [[nodiscard]] inline auto spheroid() const noexcept -> Spheroid {
    return Spheroid(detail::geodetic::Coordinates::spheroid());
  }

  /// Converts Cartesian coordinates to Geographic latitude, longitude, and
  /// altitude. Cartesian coordinates should be in meters. The returned latitude
  /// and longitude are in degrees, and the altitude will be in meters.
  template <typename T>
  auto ecef_to_lla(const Eigen::Ref<const Vector<T>> &x,
                   const Eigen::Ref<const Vector<T>> &y,
                   const Eigen::Ref<const Vector<T>> &z,
                   const size_t num_threads) const -> pybind11::tuple {
    detail::check_eigen_shape("x", x, "y", y, "z", z);
    auto size = x.size();
    auto lon = pybind11::array_t<T>(pybind11::array::ShapeContainer{{size}});
    auto lat = pybind11::array_t<T>(pybind11::array::ShapeContainer{{size}});
    auto alt = pybind11::array_t<T>(pybind11::array::ShapeContainer{{size}});
    auto _lon = lon.template mutable_unchecked<1>();
    auto _lat = lat.template mutable_unchecked<1>();
    auto _alt = alt.template mutable_unchecked<1>();

    {
      pybind11::gil_scoped_release release;

      // Captures the detected exceptions in the calculation function
      // (only the last exception captured is kept)
      auto except = std::exception_ptr(nullptr);

      detail::dispatch(
          [&](size_t start, size_t end) {
            try {
              for (size_t ix = start; ix < end; ++ix) {
                auto lla = detail::geodetic::Coordinates::ecef_to_lla(
                    detail::geometry::Point3D<T>{x(ix), y(ix), z(ix)});
                _lon(ix) = boost::geometry::get<0>(lla);
                _lat(ix) = boost::geometry::get<1>(lla);
                _alt(ix) = boost::geometry::get<2>(lla);
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
    return pybind11::make_tuple(lon, lat, alt);
  }

  /// Converts Geographic coordinates latitude, longitude, and altitude to
  /// Cartesian coordinates. The latitude and longitude should be in degrees and
  /// the altitude in meters. The returned ECEF coordinates will be in meters.
  template <typename T>
  auto lla_to_ecef(const Eigen::Ref<const Vector<T>> &lon,
                   const Eigen::Ref<const Vector<T>> &lat,
                   const Eigen::Ref<const Vector<T>> &alt,
                   const size_t num_threads) const -> pybind11::tuple {
    detail::check_eigen_shape("lon", lon, "lat", lat, "alt", alt);
    auto size = lon.size();
    auto x = pybind11::array_t<T>(pybind11::array::ShapeContainer{{size}});
    auto y = pybind11::array_t<T>(pybind11::array::ShapeContainer{{size}});
    auto z = pybind11::array_t<T>(pybind11::array::ShapeContainer{{size}});
    auto x_ = x.template mutable_unchecked<1>();
    auto y_ = y.template mutable_unchecked<1>();
    auto z_ = z.template mutable_unchecked<1>();

    {
      pybind11::gil_scoped_release release;

      // Captures the detected exceptions in the calculation function
      // (only the last exception captured is kept)
      auto except = std::exception_ptr(nullptr);

      detail::dispatch(
          [&](size_t start, size_t end) {
            try {
              for (size_t ix = start; ix < end; ++ix) {
                auto ecef = detail::geodetic::Coordinates::lla_to_ecef(
                    detail::geometry::EquatorialPoint3D<T>{lon(ix), lat(ix),
                                                           alt(ix)});
                x_(ix) = boost::geometry::get<0>(ecef);
                y_(ix) = boost::geometry::get<1>(ecef);
                z_(ix) = boost::geometry::get<2>(ecef);
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
    return pybind11::make_tuple(x, y, z);
  }

  /// Transform points between two coordinate systems defined by the
  /// Coordinates instances this and target.
  template <typename T>
  auto transform(const Coordinates &target,
                 const Eigen::Ref<const Vector<T>> &lon1,
                 const Eigen::Ref<const Vector<T>> &lat1,
                 const Eigen::Ref<const Vector<T>> &alt1,
                 const size_t num_threads) const -> pybind11::tuple {
    detail::check_eigen_shape("lon1", lon1, "lat1", lat1, "alt1", alt1);
    auto size = lon1.size();
    auto lon2 = pybind11::array_t<T>(pybind11::array::ShapeContainer{{size}});
    auto lat2 = pybind11::array_t<T>(pybind11::array::ShapeContainer{{size}});
    auto alt2 = pybind11::array_t<T>(pybind11::array::ShapeContainer{{size}});
    auto _lon2 = lon2.template mutable_unchecked<1>();
    auto _lat2 = lat2.template mutable_unchecked<1>();
    auto _alt2 = alt2.template mutable_unchecked<1>();

    {
      pybind11::gil_scoped_release release;

      // Captures the detected exceptions in the calculation function
      // (only the last exception captured is kept)
      auto except = std::exception_ptr(nullptr);

      detail::dispatch(
          [&](size_t start, size_t end) {
            try {
              for (size_t ix = start; ix < end; ++ix) {
                auto lla = detail::geodetic::Coordinates::transform(
                    target, detail::geometry::EquatorialPoint3D<T>{
                                lon1(ix), lat1(ix), alt1(ix)});
                _lon2(ix) = boost::geometry::get<0>(lla);
                _lat2(ix) = boost::geometry::get<1>(lla);
                _alt2(ix) = boost::geometry::get<2>(lla);
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
    return pybind11::make_tuple(lon2, lat2, alt2);
  }

  /// Get a tuple that fully encodes the state of this instance
  [[nodiscard]] auto getstate() const -> pybind11::tuple {
    return spheroid().getstate();
  }

  /// Create a new instance from a registered state of an instance of this
  /// object.
  static auto setstate(const pybind11::tuple &state) -> Coordinates {
    return Coordinates(Spheroid::setstate(state));
  }
};

}  // namespace pyinterp::geodetic
