// Copyright (c) 2021 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <memory>
#include <optional>

#include "pyinterp/detail/math/descriptive_statistics.hpp"
#include "pyinterp/detail/numpy.hpp"
#include "pyinterp/eigen.hpp"

namespace pyinterp {

/// Univariate descriptive statistics
/// Reference: Numerically stable, scalable formulas for parallel and online
/// computation of higher-order multivariate central moments with arbitrary
/// weights
/// https://doi.org/10.1007/s00180-015-0637-z
template <typename T>
class DescriptiveStatistics {
 public:
  using Accumulators = detail::math::DescriptiveStatistics<T>;

  /// Constructucts a new DescriptiveStatistics object from serialized state.
  DescriptiveStatistics(Vector<Accumulators> accumulators,
                        std::vector<pybind11::ssize_t> shape)
      : accumulators_(std::move(accumulators)), shape_(std::move(shape)) {}

  /// Constructor.
  DescriptiveStatistics(
      pybind11::array_t<T, pybind11::array::c_style> values,
      std::optional<pybind11::array_t<T, pybind11::array::c_style>> weights,
      std::optional<std::list<pybind11::ssize_t>>& axis) {
    // If no axes are specified, we compute the statistics on the whole array.
    if (!axis) {
      auto range = std::list<pybind11::ssize_t>(values.ndim());
      std::iota(range.begin(), range.end(), 0);
      axis = std::move(range);
    }
    detail::numpy::check_bounds(values, *axis);

    // If no weights are specified, we use ones.
    if (!weights) {
      auto ones = pybind11::array_t<T, pybind11::array::c_style>(
          pybind11::array::ShapeContainer{values.size()});
      auto ptr_ones =
          reinterpret_cast<T*>(pybind11::detail::array_proxy(ones.ptr())->data);
      std::fill(ptr_ones, ptr_ones + values.size(), 1);
      weights = std::move(ones);
    }
  
    auto [shape, strides, adjusted_strides] =
        detail::numpy::reduced_properties(values, *axis);
    shape_ = std::move(shape);
    accumulators_ =
        std::move(push(values, *weights, strides, adjusted_strides));
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

  /// Returns the variance of samples.
  [[nodiscard]] auto variance(const int ddof = 0) const
      -> pybind11::array_t<T> {
    return calculate_statistics(&Accumulators::variance, ddof);
  }

  /// Returns the kurtosis of samples.
  [[nodiscard]] auto kurtosis() const -> pybind11::array_t<T> {
    return calculate_statistics(&Accumulators::kurtosis);
  }

  /// Returns the skewness of samples.
  [[nodiscard]] auto skewness() const -> pybind11::array_t<T> {
    return calculate_statistics(&Accumulators::skewness);
  }

  /// Returns the sum of samples.
  [[nodiscard]] auto sum() const -> pybind11::array_t<T> {
    return calculate_statistics(&Accumulators::sum);
  }

  /// Returns the sum of weights.
  [[nodiscard]] auto sum_of_weights() const -> pybind11::array_t<T> {
    return calculate_statistics(&Accumulators::sum_of_weights);
  }

  /// Aggregation of statistics
  auto operator+=(const DescriptiveStatistics<T>& other)
      -> DescriptiveStatistics& {
    if (shape_ != other.shape_) {
      throw std::invalid_argument("incompatible shapes");
    }
    accumulators_ += other.accumulators_;
    return *this;
  }

  /// Pickle support: get state of this instance
  [[nodiscard]] auto getstate() const -> pybind11::tuple {
    return pybind11::make_tuple(
        accumulators_.template cast<detail::math::Accumulators<T>>(), shape_);
  }

  /// Pickle support: set state of this instance
  static auto setstate(pybind11::tuple state)
      -> std::unique_ptr<DescriptiveStatistics<T>> {
    if (state.size() != 2) {
      throw std::invalid_argument("invalid state");
    }
    auto accumulators = state[0].cast<Vector<detail::math::Accumulators<T>>>();
    auto shape = state[1].cast<std::vector<pybind11::ssize_t>>();

    return std::make_unique<DescriptiveStatistics<T>>(
        std::move(accumulators.template cast<Accumulators>()),
        std::move(shape));
  }

 private:
  Vector<Accumulators> accumulators_;
  std::vector<pybind11::ssize_t> shape_{};

  /// Returns the total number of elements in the array.
  [[nodiscard]] inline auto size() const -> pybind11::ssize_t {
    return std::accumulate(shape_.begin(), shape_.end(),
                           static_cast<pybind11::ssize_t>(1),
                           std::multiplies<pybind11::ssize_t>());
  }

  /// Push values and weights to the accumulators.
  template <typename T>
  auto push(pybind11::array_t<T, pybind11::array::c_style>& arr,
            pybind11::array_t<T, pybind11::array::c_style>& weights,
            const Vector<pybind11::ssize_t>& strides,
            const Vector<pybind11::ssize_t>& adjusted_strides)
      -> Vector<Accumulators> {
    auto ptr_arr = reinterpret_cast<T*>(  //
        pybind11::detail::array_proxy(arr.ptr())->data);
    auto ptr_weights = reinterpret_cast<T*>(
        pybind11::detail::array_proxy(weights.ptr())->data);
    auto result = Vector<Accumulators>(size());
    auto indexes = Eigen::Matrix<pybind11::ssize_t, -1, 1>(arr.ndim());

    {
      pybind11::gil_scoped_release release;

      for (auto ix = 0; ix < arr.size(); ++ix) {
        detail::numpy::unravel(ix, strides, indexes);
        auto jx = (indexes.array() * adjusted_strides.array()).sum();
        result[jx](ptr_arr[ix], ptr_weights[ix]);
      }
    }
    return result;
  }

  /// Calculation of a given statistical variable.
  template <typename Func, typename Type = T, typename... Args>
  [[nodiscard]] auto calculate_statistics(const Func& func, Args... args) const
      -> pybind11::array_t<Type> {
    auto result = pybind11::array_t<Type>(shape_);
    auto ptr_result = reinterpret_cast<Type*>(
        pybind11::detail::array_proxy(result.ptr())->data);
    {
      pybind11::gil_scoped_release release;

      for (auto ix = 0; ix < result.size(); ++ix) {
        ptr_result[ix] = (accumulators_[ix].*func)(args...);
      }
    }
    return result;
  }
};

}  // namespace pyinterp