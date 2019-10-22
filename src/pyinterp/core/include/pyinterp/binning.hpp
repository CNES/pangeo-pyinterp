// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include "pyinterp/axis.hpp"
#include "pyinterp/detail/broadcast.hpp"
#include <Eigen/Core>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/kurtosis.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/skewness.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <pybind11/numpy.h>

namespace pyinterp::binning {

/// Discretizes the data into a regular grid (computes a binned approximation)
/// using the nearest binning technique.
template <typename T>
class Nearest {
 public:
  /// Default constructor
  ///
  /// @param x Definition of the bin edges for the X axis of the grid.
  /// @param y Definition of the bin edges for the Y axis of the grid.
  Nearest(std::shared_ptr<Axis> x, std::shared_ptr<Axis> y)
      : x_(std::move(x)), y_(std::move(y)), acc_(x_->size(), y_->size()) {}

  /// Inserts new values in the grid from Z values for X, Y data coordinates.
  void push(const pybind11::array_t<T>& x, const pybind11::array_t<T>& y,
            const pybind11::array_t<T>& z) {
    detail::check_array_ndim("x", 1, x, "y", 1, y, "z", 1, z);
    detail::check_ndarray_shape("x", x, "y", y, "z", z);
    auto _x = x.template unchecked<1>();
    auto _y = y.template unchecked<1>();
    auto _z = z.template unchecked<1>();

    {
      pybind11::gil_scoped_release release;

      const auto& x_axis = static_cast<pyinterp::detail::Axis&>(*x_);
      const auto& y_axis = static_cast<pyinterp::detail::Axis&>(*y_);

      for (pybind11::ssize_t idx = 0; idx < x.size(); ++idx) {
        auto ix = x_axis.find_index(_x(idx), true);
        auto iy = y_axis.find_index(_y(idx), true);

        if (ix != -1 && iy != -1) {
          auto value = _z(idx);
          if (!std::isnan(value)) {
            acc_(ix, iy)(_z(idx));
          }
        }
      }
    }
  }

  /// Reset the statistics.
  void clear() {
    for (Eigen::Index ix = 0; ix < acc_.rows(); ++ix) {
      for (Eigen::Index iy = 0; iy < acc_.cols(); ++iy) {
        acc_(ix, iy) = {};
      }
    }
  }

  /// Gets the number of samples pushed into the bin.
  [[nodiscard]] pybind11::array_t<T> count() const {
    return calculate_statistics(boost::accumulators::count);
  }

      /// Gets the minimum value of all the samples pushed into the bin.
      [[nodiscard]] pybind11::array_t<T> min() const {
    return calculate_statistics(boost::accumulators::min);
  }

  /// Gets the maximum value of all the samples pushed into the bin.
  [[nodiscard]] pybind11::array_t<T> max() const {
    return calculate_statistics(boost::accumulators::max);
  }

      /// Gets the mean of all the samples pushed into the bin.
      [[nodiscard]] pybind11::array_t<T> mean() const {
    return calculate_statistics(boost::accumulators::mean);
  }

  /// Gets the median of all the samples pushed into the bin.
  [[nodiscard]] pybind11::array_t<T> median() const {
    return calculate_statistics(boost::accumulators::median);
  }

      /// Gets the variance of all the samples pushed into the bin.
      [[nodiscard]] pybind11::array_t<T> variance() const {
    return calculate_statistics(boost::accumulators::variance);
  }

  /// Gets the kurtosis of all the samples pushed into the bin.
  [[nodiscard]] pybind11::array_t<T> kurtosis() const {
    return calculate_statistics(boost::accumulators::kurtosis);
  }

  /// Gets the skewness of all the samples pushed into the bin.
  [[nodiscard]] pybind11::array_t<T> skewness() const {
    return calculate_statistics(boost::accumulators::skewness);
  }

  /// Gets the X-Axis
  [[nodiscard]] inline std::shared_ptr<Axis> x() const { return x_; }

  /// Gets the Y-Axis
  [[nodiscard]] inline std::shared_ptr<Axis> y() const { return y_; }

 private:
  using Accumulators = boost::accumulators::accumulator_set<
      T,
      boost::accumulators::stats<
          boost::accumulators::tag::count, boost::accumulators::tag::min,
          boost::accumulators::tag::max, boost::accumulators::tag::mean,
          boost::accumulators::tag::median, boost::accumulators::tag::variance,
          boost::accumulators::tag::kurtosis,
          boost::accumulators::tag::skewness>>;
  std::shared_ptr<Axis> x_;
  std::shared_ptr<Axis> y_;
  Eigen::Matrix<Accumulators, Eigen::Dynamic, Eigen::Dynamic> acc_;

  /// Calculation of a given statistical variable.
  template <typename Func>
  [[nodiscard]] pybind11::array_t<T> calculate_statistics(
      const Func& func) const {
    pybind11::array_t<T> z({x_->size(), y_->size()});
    auto _z = z.template mutable_unchecked<2>();
    for (Eigen::Index ix = 0; ix < acc_.rows(); ++ix) {
      for (Eigen::Index iy = 0; iy < acc_.cols(); ++iy) {
        _z(ix, iy) = func(acc_(ix, iy));
      }
    }
    return z;
  }
};

}  // namespace pyinterp::binning
