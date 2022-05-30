// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <algorithm>
#include <boost/geometry.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "pyinterp/axis.hpp"
#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/detail/isviewstream.hpp"
#include "pyinterp/detail/math/binning.hpp"
#include "pyinterp/detail/math/streaming_histogram.hpp"
#include "pyinterp/eigen.hpp"

namespace pyinterp {

/// Group a number of more or less continuous values into a smaller number of
/// "bins" located on a grid.
template <typename T>
class Histogram2D {
 public:
  /// Statistics handled by this object.
  using StreamingHistogram = detail::math::StreamingHistogram<T>;

  /// Default constructor
  ///
  /// @param x Definition of the bin centers for the X axis of the grid.
  /// @param y Definition of the bin centers for the Y axis of the grid.
  Histogram2D(std::shared_ptr<Axis<double>> x, std::shared_ptr<Axis<double>> y,
              const std::optional<size_t> &bin_count)
      : x_(std::move(x)), y_(std::move(y)), histogram_(x_->size(), y_->size()) {
    if (bin_count) {
      for (int ix = 0; ix < histogram_.rows(); ++ix) {
        for (int jx = 0; jx < histogram_.cols(); ++jx) {
          histogram_(ix, jx).resize(*bin_count);
        }
      }
    }
  }

  /// Default destructor
  virtual ~Histogram2D() = default;

  /// Copy constructor
  ///
  /// @param rhs right value
  Histogram2D(const Histogram2D &rhs) = default;

  /// Move constructor
  ///
  /// @param rhs right value
  Histogram2D(Histogram2D &&rhs) noexcept = default;

  /// Copy assignment operator
  ///
  /// @param rhs right value
  auto operator=(const Histogram2D &rhs) -> Histogram2D & = delete;

  /// Move assignment operator
  ///
  /// @param rhs right value
  auto operator=(Histogram2D &&rhs) noexcept -> Histogram2D & = delete;

  /// Inserts new values in the grid from Z values for X, Y data
  /// coordinates.
  void push(const pybind11::array_t<T> &x, const pybind11::array_t<T> &y,
            const pybind11::array_t<T> &z) {
    detail::check_array_ndim("x", 1, x, "y", 1, y, "z", 1, z);
    detail::check_ndarray_shape("x", x, "y", y, "z", z);

    auto _x = x.template unchecked<1>();
    auto _y = y.template unchecked<1>();
    auto _z = z.template unchecked<1>();

    {
      pybind11::gil_scoped_release release;

      const auto &x_axis = static_cast<pyinterp::detail::Axis<double> &>(*x_);
      const auto &y_axis = static_cast<pyinterp::detail::Axis<double> &>(*y_);

      for (pybind11::ssize_t idx = 0; idx < x.size(); ++idx) {
        auto value = _z(idx);

        if (!std::isnan(value)) {
          auto ix = x_axis.find_index(_x(idx), true);
          auto iy = y_axis.find_index(_y(idx), true);

          if (ix != -1 && iy != -1) {
            histogram_(ix, iy)(value);
          }
        }
      }
    }
  }

  /// Reset the statistics.
  void clear() {
    for (Eigen::Index ix = 0; ix < histogram_.rows(); ++ix) {
      for (Eigen::Index jx = 0; jx < histogram_.cols(); ++jx) {
        histogram_(ix, jx).clear();
      }
    }
  }

  /// Compute the count of points within each bin.
  [[nodiscard]] auto count() const -> pybind11::array_t<uint64_t> {
    return calculate_statistics<decltype(&StreamingHistogram::count), uint64_t>(
        &StreamingHistogram::count);
  }

  /// Compute the minimum of values for points within each bin.
  [[nodiscard]] auto min() const -> pybind11::array_t<T> {
    return calculate_statistics(&StreamingHistogram::min);
  }

  /// Compute the maximum of values for points within each bin.
  [[nodiscard]] auto max() const -> pybind11::array_t<T> {
    return calculate_statistics(&StreamingHistogram::max);
  }

  /// Compute the mean of values for points within each bin.
  [[nodiscard]] auto mean() const -> pybind11::array_t<T> {
    return calculate_statistics(&StreamingHistogram::mean);
  }

  /// Compute the variance of values for points within each bin.
  [[nodiscard]] auto variance() const -> pybind11::array_t<T> {
    return calculate_statistics(&StreamingHistogram::variance);
  }

  /// Compute the kurtosis of values for points within each bin.
  [[nodiscard]] auto kurtosis() const -> pybind11::array_t<T> {
    return calculate_statistics(&StreamingHistogram::kurtosis);
  }

  /// Compute the quantile of values for points within each bin.
  [[nodiscard]] auto quantile(const T &q) const -> pybind11::array_t<T> {
    return calculate_statistics(&StreamingHistogram::quantile, q);
  }

  /// Compute the skewness of values for points within each bin.
  [[nodiscard]] auto skewness() const -> pybind11::array_t<T> {
    return calculate_statistics(&StreamingHistogram::skewness);
  }
  /// Compute the sum of weights within each bin.
  [[nodiscard]] auto sum_of_weights() const -> pybind11::array_t<T> {
    return calculate_statistics(&StreamingHistogram::sum_of_weights);
  }

  /// Gets the X-Axis
  [[nodiscard]] inline auto x() const -> std::shared_ptr<Axis<double>> {
    return x_;
  }

  /// Gets the Y-Axis
  [[nodiscard]] inline auto y() const -> std::shared_ptr<Axis<double>> {
    return y_;
  }

  /// Pickle support: get state of this instance
  [[nodiscard]] auto getstate() const -> pybind11::tuple {
    return pybind11::make_tuple(x_->getstate(), y_->getstate(), marshal());
  }

  /// Pickle support: set state of this instance
  static auto setstate(const pybind11::tuple &state)
      -> std::unique_ptr<Histogram2D<T>> {
    if (state.size() != 3) {
      throw std::invalid_argument("invalid state");
    }

    // Unmarshalling X-Axis
    auto x = std::make_shared<Axis<double>>();
    *x = Axis<double>::setstate(state[0].cast<pybind11::tuple>());

    // Unmarshalling Y-Axis
    auto y = std::make_shared<Axis<double>>();
    *y = Axis<double>::setstate(state[1].cast<pybind11::tuple>());

    // Unmarshalling instance
    auto result = std::make_unique<Histogram2D<T>>(x, y, 40);
    auto marshal_data = state[2].cast<pybind11::bytes>();
    Histogram2D::unmarshal(marshal_data.cast<std::string_view>(),
                           result->histogram_);
    return result;
  }

  /// Aggregation of statistics
  auto operator+=(const Histogram2D &other) -> Histogram2D & {
    if (*x_ != *(other.x_) || *y_ != *(other.y_)) {
      throw std::invalid_argument("Unable to combine different grids");
    }
    for (Eigen::Index ix = 0; ix < histogram_.rows(); ++ix) {
      for (Eigen::Index iy = 0; iy < histogram_.cols(); ++iy) {
        auto &lhs = histogram_(ix, iy);
        auto &rhs = other.histogram_(ix, iy);

        // Statistics are defined only in the other instance.
        if (lhs.size() == 0 && rhs.size() != 0) {
          lhs = rhs;
          // If the statistics are defined in both instances they can be
          // combined.
        } else if (lhs.size() != 0 && rhs.size() != 0) {
          lhs += rhs;
        }
      }
    }
    return *this;
  }

  /// Returns the histogram for each bin.
  auto histograms() const -> pybind11::array_t<detail::math::Bin<T>> {
    auto bins_count = static_cast<size_t>(0);
    for (Eigen::Index ix = 0; ix < histogram_.rows(); ++ix) {
      for (Eigen::Index iy = 0; iy < histogram_.cols(); ++iy) {
        bins_count = std::max(bins_count, histogram_(ix, iy).size());
      }
    }
    auto result =
        pybind11::array_t<detail::math::Bin<T>>(pybind11::array::ShapeContainer(
            {x_->size(), y_->size(),
             static_cast<pybind11::ssize_t>(bins_count)}));
    auto _result = result.template mutable_unchecked<3>();
    {
      auto gil = pybind11::gil_scoped_release();

      for (Eigen::Index ix = 0; ix < histogram_.rows(); ++ix) {
        for (Eigen::Index iy = 0; iy < histogram_.cols(); ++iy) {
          auto iz = static_cast<size_t>(0);
          auto &bins = histogram_(ix, iy).bins();
          for (iz = 0; iz < bins.size(); ++iz) {
            _result(ix, iy, iz) = bins[iz];
          }
          for (; iz < bins_count; ++iz) {
            _result(ix, iy, iz) =
                detail::math::Bin<T>{std::numeric_limits<T>::quiet_NaN(), T(0)};
          }
        }
      }
    }
    return result;
  }

 private:
  /// Grid axis
  std::shared_ptr<Axis<double>> x_;
  std::shared_ptr<Axis<double>> y_;

  /// Statistics grid
  Matrix<StreamingHistogram> histogram_;

  /// Calculation of a given statistical variable.
  template <typename Func, typename Type = T, typename... Args>
  [[nodiscard]] auto calculate_statistics(const Func &func, Args... args) const
      -> pybind11::array_t<Type> {
    pybind11::array_t<Type> z({x_->size(), y_->size()});
    auto _z = z.template mutable_unchecked<2>();
    {
      pybind11::gil_scoped_release release;

      for (Eigen::Index ix = 0; ix < histogram_.rows(); ++ix) {
        for (Eigen::Index iy = 0; iy < histogram_.cols(); ++iy) {
          _z(ix, iy) = (histogram_(ix, iy).*func)(args...);
        }
      }
    }
    return z;
  }

  [[nodiscard]] auto marshal() const -> pybind11::bytes {
    auto gil = pybind11::gil_scoped_release();
    auto ss = std::stringstream();
    ss.exceptions(std::stringstream::failbit);
    auto rows = histogram_.rows();
    ss.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    auto cols = histogram_.cols();
    ss.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
    for (int ix = 0; ix < histogram_.rows(); ++ix) {
      for (int jx = 0; jx < histogram_.cols(); ++jx) {
        auto marshal_hist = static_cast<std::string>(histogram_(ix, jx));
        auto size = marshal_hist.size();
        ss.write(reinterpret_cast<const char *>(&size), sizeof(size));
        ss.write(marshal_hist.c_str(), static_cast<std::streamsize>(size));
      }
    }
    return ss.str();
  }

  static auto unmarshal(const std::string_view &data,
                        Matrix<StreamingHistogram> &histogram) -> void {
    auto gil = pybind11::gil_scoped_release();
    auto ss = detail::isviewstream(data);
    ss.exceptions(std::stringstream::failbit);

    try {
      auto rows = static_cast<Eigen::Index>(0);
      auto cols = static_cast<Eigen::Index>(0);
      ss.read(reinterpret_cast<char *>(&rows), sizeof(rows));
      ss.read(reinterpret_cast<char *>(&cols), sizeof(cols));
      if (rows != histogram.rows() || cols != histogram.cols()) {
        throw std::invalid_argument("invalid state");
      }
      for (int ix = 0; ix < rows; ++ix) {
        for (int jx = 0; jx < cols; ++jx) {
          auto size = static_cast<size_t>(0);
          ss.read(reinterpret_cast<char *>(&size), sizeof(size));
          histogram(ix, jx) = std::move(StreamingHistogram(
              ss.readview(static_cast<std::streamsize>(size))));
        }
      }
    } catch (std::ios_base::failure &) {
      throw std::invalid_argument("invalid state");
    }
  }
};

}  // namespace pyinterp
