// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/detail/math/streaming_histogram.hpp"
#include "pyinterp/detail/numpy.hpp"
#include "pyinterp/eigen.hpp"

namespace pyinterp {

template <typename T>
class StreamingHistogram {
 public:
  using Accumulators = detail::math::StreamingHistogram<T>;
  using Bin = detail::math::Bin<T>;

  /// Constructucts a new DescriptiveStatistics object from serialized state.
  StreamingHistogram(Vector<Accumulators> accumulators,
                     std::vector<pybind11::ssize_t> shape)
      : accumulators_(std::move(accumulators)), shape_(std::move(shape)) {}

  /// Constructor.
  StreamingHistogram(
      pybind11::array_t<T, pybind11::array::c_style> &values,
      std::optional<pybind11::array_t<T, pybind11::array::c_style>> &weights,
      std::optional<std::list<pybind11::ssize_t>> &axis,
      std::optional<size_t> &bin_count) {
    // Default number of bins.
    bin_count = bin_count.value_or(100);

    // Check if the given axis is valid.
    if (axis) {
      detail::numpy::check_axis_bounds(values, *axis);
    }

    // If weights are given, values and weights must have the same shape.
    if (weights) {
      detail::check_ndarray_shape("values", values, "weights", *weights);
    }

    if (!axis) {
      // Compute the statistics for the whole array.
      shape_ = {1};
      accumulators_ = std::move(
          StreamingHistogram::allocate_accumulators(*bin_count, shape_));
      weights ? push(values, *weights) : push(values);
    } else {
      // Compute the statistics on a reduced dimension.
      if (!weights) {
        // If no weights are specified, the weights are set to ones.
        weights = std::move(detail::numpy::ones_like(values));
      }
      auto [shape, strides, adjusted_strides] =
          detail::numpy::reduced_properties(values, *axis);
      shape_ = std::move(shape);
      accumulators_ = std::move(
          StreamingHistogram::allocate_accumulators(*bin_count, shape_));
      push(values, *weights, strides, adjusted_strides);
    }
  }

  /// Resizes the maximum number of bins (0 truncate all values)
  auto resize(const size_t bin_count) {
    std::for_each(accumulators_.data(),
                  accumulators_.data() + accumulators_.size(),
                  [bin_count](auto &item) { item.resize(bin_count); });
  }

  /// Returns the number of bins.
  [[nodiscard]] auto size() const -> pybind11::array_t<size_t> {
    return calculate_statistics<decltype(&Accumulators::size), size_t>(
        &Accumulators::size);
  }

  /// Returns the count of samples.
  [[nodiscard]] auto count() const -> pybind11::array_t<uint64_t> {
    return calculate_statistics<decltype(&Accumulators::count), uint64_t>(
        &Accumulators::count);
  }

  /// Returns the minimum of samples.
  [[nodiscard]] auto min() const -> pybind11::array_t<T> {
    return calculate_statistics(&Accumulators::min);
  }

  /// Returns the maximum of samples.
  [[nodiscard]] auto max() const -> pybind11::array_t<T> {
    return calculate_statistics(&Accumulators::max);
  }

  /// Returns the mean of samples.
  [[nodiscard]] auto mean() const -> pybind11::array_t<T> {
    return calculate_statistics(&Accumulators::mean);
  }

  /// Returns the quantile of samples.
  [[nodiscard]] auto quantile(const T &q) const -> pybind11::array_t<T> {
    return calculate_statistics(&Accumulators::quantile, q);
  }

  /// Returns the variance of samples.
  [[nodiscard]] auto variance() const -> pybind11::array_t<T> {
    return calculate_statistics(&Accumulators::variance);
  }

  /// Returns the kurtosis of samples.
  [[nodiscard]] auto kurtosis() const -> pybind11::array_t<T> {
    return calculate_statistics(&Accumulators::kurtosis);
  }

  /// Returns the skewness of samples.
  [[nodiscard]] auto skewness() const -> pybind11::array_t<T> {
    return calculate_statistics(&Accumulators::skewness);
  }

  /// Returns the sum of weights.
  [[nodiscard]] auto sum_of_weights() const -> pybind11::array_t<T> {
    return calculate_statistics(&Accumulators::sum_of_weights);
  }

  /// Aggregation of statistics
  auto operator+=(const StreamingHistogram<T> &other) -> StreamingHistogram & {
    if (shape_ != other.shape_) {
      throw std::invalid_argument("incompatible shapes");
    }
    accumulators_ += other.accumulators_;
    return *this;
  }

  /// Returns the histogram bins
  [[nodiscard]] auto bins() const -> pybind11::array_t<Bin> {
    auto max_size = std::numeric_limits<size_t>::min();
    std::for_each(accumulators_.data(),
                  accumulators_.data() + accumulators_.size(),
                  [&max_size](const auto &item) {
                    max_size = std::max(max_size, item.size());
                  });
    auto shape = std::vector<size_t>(shape_.begin(), shape_.end());
    shape.push_back(max_size);
    auto result = pybind11::array_t<Bin>(shape);
    auto ptr_result = reinterpret_cast<Bin *>(
        pybind11::detail::array_proxy(result.ptr())->data);
    {
      pybind11::gil_scoped_release release;

      for (auto ix = 0; ix < accumulators_.size(); ++ix) {
        auto &item = accumulators_[ix];
        auto &bins = item.bins();

        auto jx = static_cast<size_t>(0);
        auto shift = ix * max_size;
        for (; jx < bins.size(); ++jx) {
          ptr_result[shift + jx] = bins[jx];
        }
        for (; jx < max_size; ++jx) {
          ptr_result[shift + jx] = {std::numeric_limits<T>::quiet_NaN(), 0};
        }
      }
    }
    return result;
  }

  /// Pickle support: get state of this instance
  [[nodiscard]] auto getstate() const -> pybind11::tuple {
    return pybind11::make_tuple(shape_, marshal());
  }

  /// Pickle support: set state of this instance
  static auto setstate(const pybind11::tuple &state)
      -> std::unique_ptr<StreamingHistogram<T>> {
    if (state.size() != 2) {
      throw std::invalid_argument("invalid state");
    }
    auto shape = state[0].cast<std::vector<pybind11::ssize_t>>();
    auto marshal_data = state[1].cast<pybind11::bytes>();
    auto accumulators =
        StreamingHistogram::unmarshal(marshal_data.cast<std::string_view>());
    return std::make_unique<StreamingHistogram<T>>(std::move(accumulators),
                                                   std::move(shape));
  }

 private:
  Vector<Accumulators> accumulators_{};
  std::vector<pybind11::ssize_t> shape_{};

  /// Returns the total number of elements in the array.
  [[nodiscard]] static constexpr auto shape_size(
      const std::vector<pybind11::ssize_t> &shape) -> pybind11::ssize_t {
    return std::accumulate(shape.begin(), shape.end(),
                           static_cast<pybind11::ssize_t>(1),
                           std::multiplies<>());
  }

  /// Allocates the accumulators needed for the result statistics.
  [[nodiscard]] static auto allocate_accumulators(
      const size_t bin_count, const std::vector<pybind11::ssize_t> &shape)
      -> Vector<Accumulators> {
    auto result = Vector<Accumulators>(StreamingHistogram::shape_size(shape));
    std::for_each(result.data(), result.data() + result.size(),
                  [bin_count](auto &acc) { acc.resize(bin_count); });
    return result;
  }

  /// Push values to the accumulators when the user wants to
  /// calculate statistics on the whole array. NaNs are ignored.
  auto push(pybind11::array_t<T, pybind11::array::c_style> &arr) -> void {
    auto *ptr_arr = detail::numpy::get_data_pointer<T>(arr.ptr());
    auto &item = accumulators_[0];

    {
      pybind11::gil_scoped_release release;

      std::for_each(ptr_arr, ptr_arr + arr.size(), [&item](const auto &value) {
        if (!std::isnan(value)) {
          item(value);
        }
      });
    }
  }

  /// Push values and weights to the accumulators when the user wants to
  /// calculate statistics on the whole array. NaNs are ignored.
  auto push(pybind11::array_t<T, pybind11::array::c_style> &arr,
            pybind11::array_t<T, pybind11::array::c_style> &weights) -> void {
    auto *ptr_arr = detail::numpy::get_data_pointer<T>(arr.ptr());
    auto *ptr_weights = detail::numpy::get_data_pointer<T>(weights.ptr());
    auto &item = accumulators_[0];

    {
      pybind11::gil_scoped_release release;

      for (auto ix = 0; ix < arr.size(); ++ix) {
        const auto xi = ptr_arr[ix];
        if (!std::isnan(xi)) {
          item(xi, ptr_weights[ix]);
        }
      }
    }
  }

  /// Push values and weights to the accumulators when the user wants to
  /// calculate statistics on a reduced array. NaNs are ignored.
  auto push(pybind11::array_t<T, pybind11::array::c_style> &arr,
            pybind11::array_t<T, pybind11::array::c_style> &weights,
            const Vector<pybind11::ssize_t> &strides,
            const Vector<pybind11::ssize_t> &adjusted_strides) -> void {
    auto *ptr_arr = detail::numpy::get_data_pointer<T>(arr.ptr());
    auto *ptr_weights = detail::numpy::get_data_pointer<T>(weights.ptr());
    auto indexes = Eigen::Matrix<pybind11::ssize_t, -1, 1>(arr.ndim());
    {
      pybind11::gil_scoped_release release;

      for (auto ix = 0; ix < arr.size(); ++ix) {
        const auto xi = ptr_arr[ix];
        if (!std::isnan(xi)) {
          detail::numpy::unravel(ix, strides, indexes);
          auto jx = (indexes.array() * adjusted_strides.array()).sum();
          accumulators_[jx](xi, ptr_weights[ix]);
        }
      }
    }
  }

  /// Calculation of a given statistical variable.
  template <typename Func, typename Type = T, typename... Args>
  [[nodiscard]] auto calculate_statistics(const Func &func, Args... args) const
      -> pybind11::array_t<Type> {
    auto result = pybind11::array_t<Type>(shape_);
    auto ptr_result = reinterpret_cast<Type *>(
        pybind11::detail::array_proxy(result.ptr())->data);
    {
      pybind11::gil_scoped_release release;

      for (auto ix = 0; ix < result.size(); ++ix) {
        ptr_result[ix] = (accumulators_[ix].*func)(args...);
      }
    }
    return result;
  }

  [[nodiscard]] auto marshal() const -> pybind11::bytes {
    auto gil = pybind11::gil_scoped_release();
    auto ss = std::stringstream();
    ss.exceptions(std::stringstream::failbit);
    auto size = accumulators_.size();
    ss.write(reinterpret_cast<const char *>(&size), sizeof(size));
    for (int ix = 0; ix < size; ++ix) {
      auto marshal_hist = static_cast<std::string>(accumulators_(ix));
      auto size = marshal_hist.size();
      ss.write(reinterpret_cast<const char *>(&size), sizeof(size));
      ss.write(marshal_hist.c_str(), static_cast<std::streamsize>(size));
    }
    return ss.str();
  }

  static auto unmarshal(const std::string_view &data) -> Vector<Accumulators> {
    auto gil = pybind11::gil_scoped_release();
    auto ss = detail::isviewstream(data);
    ss.exceptions(std::stringstream::failbit);

    try {
      auto size = static_cast<Eigen::Index>(0);
      ss.read(reinterpret_cast<char *>(&size), sizeof(size));
      auto accumulators = Vector<Accumulators>(size);
      for (int ix = 0; ix < size; ++ix) {
        auto size = static_cast<size_t>(0);
        ss.read(reinterpret_cast<char *>(&size), sizeof(size));
        accumulators(ix) = std::move(
            Accumulators(ss.readview(static_cast<std::streamsize>(size))));
      }
      return accumulators;
    } catch (std::ios_base::failure &) {
      throw std::invalid_argument("invalid state");
    }
  }
};

}  // namespace pyinterp
