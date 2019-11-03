// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Core>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/weighted_kurtosis.hpp>
#include <boost/accumulators/statistics/weighted_mean.hpp>
#include <boost/accumulators/statistics/weighted_median.hpp>
#include <boost/accumulators/statistics/weighted_skewness.hpp>
#include <boost/accumulators/statistics/weighted_sum.hpp>
#include <boost/accumulators/statistics/weighted_variance.hpp>
#include <boost/geometry.hpp>
#include <optional>
#include <pybind11/numpy.h>
#include "pyinterp/axis.hpp"
#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/detail/geometry/point.hpp"
#include "pyinterp/detail/math/binning.hpp"
#include "pyinterp/geodetic/system.hpp"

namespace pyinterp {

/// Group a number of more or less continuous values into a smaller number of
/// "bins" located on a grid.
template <typename T>
class Binning2D {
 public:
  /// Default constructor
  ///
  /// @param x Definition of the bin edges for the X axis of the grid.
  /// @param y Definition of the bin edges for the Y axis of the grid.
  /// @param wgs WGS of the coordinate system used to manipulate geographic
  /// coordinates. If this parameter is not set, the handled coordinates will be
  /// considered as Cartesian coordinates. Otherwise, "x" and "y" are considered
  /// to represents the longitudes and latitudes on a grid.
  Binning2D(std::shared_ptr<Axis> x, std::shared_ptr<Axis> y,
            std::optional<geodetic::System> wgs)
      : x_(std::move(x)),
        y_(std::move(y)),
        acc_(x_->size(), y_->size()),
        wgs_(std::move(wgs)) {}

  /// Inserts new values in the grid from Z values for X, Y data coordinates.
  void push(const pybind11::array_t<T>& x, const pybind11::array_t<T>& y,
            const pybind11::array_t<T>& z, const bool simple) {
    detail::check_array_ndim("x", 1, x, "y", 1, y, "z", 1, z);
    detail::check_ndarray_shape("x", x, "y", y, "z", z);

    if (simple) {
      // Nearest
      push_nearest(x, y, z);
    } else if (!wgs_) {
      // Cartesian linear
      push_linear<detail::geometry::Point2D,
                  boost::geometry::strategy::area::cartesian<>>(
          x, y, z, boost::geometry::strategy::area::cartesian<>());
    } else {
      // Geographic linear
      auto strategy = boost::geometry::strategy::area::geographic<>(
          boost::geometry::srs::spheroid(wgs_->semi_major_axis(),
                                         wgs_->semi_minor_axis()));
      push_linear<detail::geometry::SpheriodPoint2D,
                  boost::geometry::strategy::area::geographic<>>(x, y, z,
                                                                 strategy);
    }
  }

  /// Reset the statistics.
  void clear() {
    acc_ =
        std::move(Eigen::Matrix<Accumulators, Eigen::Dynamic, Eigen::Dynamic>(
            x_->size(), y_->size()));
  }

  /// Compute the count of points within each bin.
  [[nodiscard]] auto count() const -> pybind11::array_t<T> {
    return calculate_statistics(boost::accumulators::count);
  }

  /// Compute the minimum of values for points within each bin.
  [[nodiscard]] auto min() const -> pybind11::array_t<T> {
    return calculate_statistics(boost::accumulators::min);
  }

  /// Compute the maximum of values for points within each bin.
  [[nodiscard]] auto max() const -> pybind11::array_t<T> {
    return calculate_statistics(boost::accumulators::max);
  }

  /// Compute the mean of values for points within each bin.
  [[nodiscard]] auto mean() const -> pybind11::array_t<T> {
    return calculate_statistics(boost::accumulators::weighted_mean);
  }

  /// Compute the median of values for points within each bin.
  [[nodiscard]] auto median() const -> pybind11::array_t<T> {
    return calculate_statistics(boost::accumulators::weighted_median);
  }

  /// Compute the variance of values for points within each bin.
  [[nodiscard]] auto variance() const -> pybind11::array_t<T> {
    return calculate_statistics(boost::accumulators::weighted_variance);
  }

  /// Compute the kurtosis of values for points within each bin.
  [[nodiscard]] auto kurtosis() const -> pybind11::array_t<T> {
    return calculate_statistics(boost::accumulators::weighted_kurtosis);
  }

  /// Compute the skewness of values for points within each bin.
  [[nodiscard]] auto skewness() const -> pybind11::array_t<T> {
    return calculate_statistics(boost::accumulators::weighted_skewness);
  }

  /// Compute the sum of values for points within each bin.
  [[nodiscard]] auto sum() const -> pybind11::array_t<T> {
    return calculate_statistics(boost::accumulators::weighted_sum);
  }

  /// Compute the sum of weights within each bin.
  [[nodiscard]] auto sum_of_weights() const -> pybind11::array_t<T> {
    return calculate_statistics(boost::accumulators::sum_of_weights);
  }

  /// Gets the X-Axis
  [[nodiscard]] inline auto x() const -> std::shared_ptr<Axis> { return x_; }

  /// Gets the Y-Axis
  [[nodiscard]] inline auto y() const -> std::shared_ptr<Axis> { return y_; }

 private:
  /// Statistics handled by this object.
  using Accumulators = boost::accumulators::accumulator_set<
      T,
      boost::accumulators::stats<
          boost::accumulators::tag::count, boost::accumulators::tag::max,
          boost::accumulators::tag::min,
          boost::accumulators::tag::sum_of_weights,
          boost::accumulators::tag::weighted_kurtosis,
          boost::accumulators::tag::weighted_mean,
          boost::accumulators::tag::weighted_median(
              boost::accumulators::with_p_square_quantile),
          boost::accumulators::tag::weighted_skewness,
          boost::accumulators::tag::weighted_sum,
          boost::accumulators::tag::weighted_variance(
              boost::accumulators::lazy)>,
      T>;

  /// Grid axis
  std::shared_ptr<Axis> x_;
  std::shared_ptr<Axis> y_;

  /// Statistics grid
  Eigen::Matrix<Accumulators, Eigen::Dynamic, Eigen::Dynamic> acc_;

  /// Geodetic coordinate system required to calculate areas (optional if the
  /// user wishes to handle Cartesian coordinates).
  std::optional<geodetic::System> wgs_;

  /// Calculation of a given statistical variable.
  template <typename Func>
  [[nodiscard]] auto calculate_statistics(const Func& func) const
      -> pybind11::array_t<T> {
    pybind11::array_t<T> z({x_->size(), y_->size()});
    auto _z = z.template mutable_unchecked<2>();
    {
      pybind11::gil_scoped_release release;

      for (Eigen::Index ix = 0; ix < acc_.rows(); ++ix) {
        for (Eigen::Index iy = 0; iy < acc_.cols(); ++iy) {
          _z(ix, iy) = func(acc_(ix, iy));
        }
      }
    }
    return z;
  }

  /// Insertion of data on the nearest bin.
  void push_nearest(const pybind11::array_t<T>& x,
                    const pybind11::array_t<T>& y,
                    const pybind11::array_t<T>& z) {
    auto _x = x.template unchecked<1>();
    auto _y = y.template unchecked<1>();
    auto _z = z.template unchecked<1>();

    {
      pybind11::gil_scoped_release release;

      const auto& x_axis = static_cast<pyinterp::detail::Axis&>(*x_);
      const auto& y_axis = static_cast<pyinterp::detail::Axis&>(*y_);

      for (pybind11::ssize_t idx = 0; idx < x.size(); ++idx) {
        auto value = _z(idx);

        if (!std::isnan(value)) {
          auto ix = x_axis.find_index(_x(idx), true);
          auto iy = y_axis.find_index(_y(idx), true);

          if (ix != -1 && iy != -1) {
            acc_(ix, iy)(value, boost::accumulators::weight = 1);
          }
        }
      }
    }
  }

  /// Update statistics for the linear binning (ignore zero weights).
  void update_acc(const int64_t ix, const int64_t iy, const T& value,
                  const T& weight) {
    if (!detail::math::is_almost_zero(weight,
                                      std::numeric_limits<T>::epsilon())) {
      acc_(ix, iy)(value, boost::accumulators::weight = weight);
    }
  }

  /// Set bins with nearest binning.
  template <template <class> class Point, typename Strategy>
  void push_linear(const pybind11::array_t<T>& x, const pybind11::array_t<T>& y,
                   const pybind11::array_t<T>& z, const Strategy& strategy) {
    auto _x = x.template unchecked<1>();
    auto _y = y.template unchecked<1>();
    auto _z = z.template unchecked<1>();

    {
      pybind11::gil_scoped_release release;

      const auto& x_axis = static_cast<pyinterp::detail::Axis&>(*x_);
      const auto& y_axis = static_cast<pyinterp::detail::Axis&>(*y_);

      for (pybind11::ssize_t idx = 0; idx < x.size(); ++idx) {
        auto value = _z(idx);
        if (std::isnan(value)) {
          continue;
        }

        auto x_indexes = x_axis.find_indexes(_x(idx));
        auto y_indexes = y_axis.find_indexes(_y(idx));

        if (x_indexes.has_value() && y_indexes.has_value()) {
          int64_t ix0;
          int64_t ix1;
          int64_t iy0;
          int64_t iy1;

          std::tie(ix0, ix1) = *x_indexes;
          std::tie(iy0, iy1) = *y_indexes;

          auto x0 = x_axis(ix0);

          auto weights = detail::math::binning_2d<Point, Strategy, double>(
              Point<double>(x_axis.is_angle()
                                ? detail::math::normalize_angle<double>(
                                      _x(idx), x0, 360.0)
                                : _x(idx),
                            _y(idx)),
              Point<double>(x0, y_axis(iy0)),
              Point<double>(x_axis(ix1), y_axis(iy1)), strategy);

          update_acc(ix0, iy0, value, static_cast<T>(std::get<0>(weights)));
          update_acc(ix0, iy1, value, static_cast<T>(std::get<1>(weights)));
          update_acc(ix1, iy1, value, static_cast<T>(std::get<2>(weights)));
          update_acc(ix1, iy0, value, static_cast<T>(std::get<3>(weights)));
        }
      }
    }
  }
};

}  // namespace pyinterp
