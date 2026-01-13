// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <boost/geometry.hpp>
#include <boost/geometry/srs/spheroid.hpp>
#include <boost/geometry/strategy/spherical/area.hpp>
#include <concepts>
#include <cstdint>
#include <optional>
#include <ranges>
#include <tuple>
#include <utility>

#include "pyinterp/broadcast.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"
#include "pyinterp/geometry/point.hpp"
#include "pyinterp/math.hpp"
#include "pyinterp/math/axis.hpp"
#include "pyinterp/math/descriptive_statistics.hpp"
#include "pyinterp/math/interpolate/bilinear_weights.hpp"
#include "pyinterp/pybind/axis.hpp"

namespace pyinterp::pybind {

/// Concept for valid binning value types
template <typename T>
concept BinningValueType = std::floating_point<T> && requires(T a, T b) {
  { std::isnan(a) } -> std::convertible_to<bool>;
  { a + b } -> std::convertible_to<T>;
  { a * b } -> std::convertible_to<T>;
};

/// Binning strategy tag types
struct NearestBinning {};
struct LinearBinning {};

/// Group a number of more or less continuous values into a smaller number of
/// "bins" located on a grid.
/// @tparam T Binning value type
template <BinningValueType T>
class Binning2D {
 public:
  /// Type aliases
  using value_type = T;
  using Accumulators = math::Accumulators<T>;
  using DescriptiveStatistics = math::DescriptiveStatistics<T>;

  /// @brief Constructor
  ///
  /// @param[in] x Definition of the bin centers for the X axis of the grid.
  /// @param[in] y Definition of the bin centers for the Y axis of the grid.
  /// @param[in] spheroid Optional WGS system for geographic coordinates
  Binning2D(
      Axis<double> x, Axis<double> y,
      std::optional<geometry::geographic::Spheroid> spheroid = std::nullopt)
      : x_(std::move(x)),
        y_(std::move(y)),
        acc_(x_.size(), y_.size()),
        spheroid_(std::move(spheroid)) {}

  /// Deleted special members (proper value semantics)
  Binning2D(const Binning2D&) = default;
  Binning2D(Binning2D&&) noexcept = default;
  auto operator=(const Binning2D&) -> Binning2D& = delete;
  auto operator=(Binning2D&&) noexcept -> Binning2D& = delete;
  virtual ~Binning2D() = default;

  /// @brief Generic push interface with strategy selection
  /// @param[in] x X coordinates of the input values
  /// @param[in] y Y coordinates of the input values
  /// @param[in] z Input values to be binned
  /// @param[in] use_nearest If true, use nearest binning strategy; otherwise
  /// use linear binning
  void push(const Eigen::Ref<const Vector<T>>& x,
            const Eigen::Ref<const Vector<T>>& y,
            const Eigen::Ref<const Vector<T>>& z,
            const bool use_nearest = false) {
    broadcast::check_eigen_shape("x", x, "y", y, "z", z);

    if (use_nearest) {
      push_nearest(x, y, z);
    } else if (!spheroid_) {
      push_linear_cartesian(x, y, z);
    } else {
      push_linear_geographic(x, y, z);
    }
  }

  /// @brief Reset all statistics
  void clear() noexcept {
    acc_ = Matrix<DescriptiveStatistics>(x_.size(), y_.size());
  }

  /// @brief Statistical computation methods
  /// @return Matrix of counts
  [[nodiscard]] auto count() const -> Matrix<uint64_t> {
    return generate_statistical_matrix<uint64_t>(
        [](const auto& stats) { return stats.count(); });
  }

  /// @brief Minimum value per bin
  /// @return Matrix of minimum values
  [[nodiscard]] auto min() const -> Matrix<T> {
    return generate_statistical_matrix<T>(
        [](const auto& stats) { return stats.min(); });
  }

  /// @brief Maximum value per bin
  /// @return Matrix of maximum values
  [[nodiscard]] auto max() const -> Matrix<T> {
    return generate_statistical_matrix<T>(
        [](const auto& stats) { return stats.max(); });
  }

  /// @brief Mean value per bin
  /// @return Matrix of mean values
  [[nodiscard]] auto mean() const -> Matrix<T> {
    return generate_statistical_matrix<T>(
        [](const auto& stats) { return stats.mean(); });
  }

  /// @brief Variance per bin
  /// @return Matrix of variance values
  [[nodiscard]] auto variance(const int ddof = 0) const -> Matrix<T> {
    return generate_statistical_matrix<T>(
        [ddof](const auto& stats) { return stats.variance(ddof); });
  }

  /// @brief Kurtosis per bin
  /// @return Matrix of kurtosis values
  [[nodiscard]] auto kurtosis() const -> Matrix<T> {
    return generate_statistical_matrix<T>(
        [](const auto& stats) { return stats.kurtosis(); });
  }

  /// @brief Skewness per bin
  /// @return Matrix of skewness values
  [[nodiscard]] auto skewness() const -> Matrix<T> {
    return generate_statistical_matrix<T>(
        [](const auto& stats) { return stats.skewness(); });
  }

  /// @brief Sum per bin
  /// @return Matrix of sum values
  [[nodiscard]] auto sum() const -> Matrix<T> {
    return generate_statistical_matrix<T>(
        [](const auto& stats) { return stats.sum(); });
  }

  /// @brief Sum of weights per bin
  /// @return Matrix of sum of weights values
  [[nodiscard]] auto sum_of_weights() const -> Matrix<T> {
    return generate_statistical_matrix<T>(
        [](const auto& stats) { return stats.sum_of_weights(); });
  }

  /// @brief Get the X axis
  /// @return X axis
  [[nodiscard]] constexpr auto x() const noexcept -> const Axis<double>& {
    return x_;
  }

  /// @brief Get the Y axis
  /// @return Y axis
  [[nodiscard]] constexpr auto y() const noexcept -> const Axis<double>& {
    return y_;
  }

  /// @brief Get the WGS spheroid
  /// @return Optional WGS spheroid
  [[nodiscard]] constexpr auto spheroid() const noexcept
      -> const std::optional<geometry::geographic::Spheroid>& {
    return spheroid_;
  }

  /// @brief Get the statistics matrix
  /// @return Matrix of descriptive statistics
  [[nodiscard]] constexpr auto statistics() const noexcept
      -> const Matrix<DescriptiveStatistics>& {
    return acc_;
  }

  /// @brief Aggregate another Binning2D into this one
  /// @param[in] other Another Binning2D to aggregate
  /// @return Reference to this Binning2D after aggregation
  auto operator+=(const Binning2D& other) -> Binning2D& {
    validate_compatibility(other);

    for (auto [lhs, rhs] :
         std::ranges::views::zip(acc_.reshaped(), other.acc_.reshaped())) {
      if (rhs.count() != 0) {
        if (lhs.count() == 0) {
          lhs = rhs;
        } else {
          lhs += rhs;
        }
      }
    }
    return *this;
  }

  /// @brief Get a tuple that fully encodes the state of this instance.
  /// @return Tuple representing the state
  [[nodiscard]] auto getstate() const
      -> std::tuple<Axis<double>, Axis<double>, Vector<int8_t>,
                    std::optional<geometry::geographic::Spheroid>> {
    Vector<int8_t> acc_view(x_.size() * y_.size() *
                            sizeof(DescriptiveStatistics));
    std::memcpy(acc_view.data(), acc_.data(),
                x_.size() * y_.size() * sizeof(DescriptiveStatistics));
    return std::make_tuple(x_, y_, acc_view, spheroid_);
  }

  /// @brief Create an instance from a serialized state.
  /// @param[in] state Tuple representing the serialized state
  /// @return New Binning2D instance
  /// @throw std::invalid_argument If the state is invalid
  static auto setstate(
      const std::tuple<Axis<double>, Axis<double>, Vector<int8_t>,
                       std::optional<geometry::geographic::Spheroid>>& state)
      -> Binning2D {
    auto x_axis = std::get<0>(state);
    auto y_axis = std::get<1>(state);
    auto acc_view = std::get<2>(state);
    auto expected_size =
        x_axis.size() * y_axis.size() * sizeof(DescriptiveStatistics);
    if (std::cmp_not_equal(acc_view.size(), expected_size)) {
      throw std::invalid_argument("Invalid state.");
    }
    Binning2D binning(std::get<0>(state), std::get<1>(state),
                      std::get<3>(state));
    auto* dst = binning.acc_.data();
    auto* src = reinterpret_cast<const DescriptiveStatistics*>(acc_view.data());
    std::uninitialized_copy_n(src, x_axis.size() * y_axis.size(), dst);
    return binning;
  }

 protected:
  Axis<double> x_;
  Axis<double> y_;
  Matrix<DescriptiveStatistics> acc_;
  std::optional<geometry::geographic::Spheroid> spheroid_;

 private:
  /// @brief Insert values using nearest binning strategy
  /// @param[in] x X coordinates of the input values
  /// @param[in] y Y coordinates of the input values
  /// @param[in] z Input values to be binned
  void push_nearest(const Eigen::Ref<const Vector<T>>& x,
                    const Eigen::Ref<const Vector<T>>& y,
                    const Eigen::Ref<const Vector<T>>& z) {
    const auto& x_axis = static_cast<const math::Axis<double>&>(x_);
    const auto& y_axis = static_cast<const math::Axis<double>&>(y_);

    for (int64_t idx = 0; idx < x.size(); ++idx) {
      const auto value = z(idx);

      if (!std::isnan(value)) {
        const auto ix = x_axis.find_index(x(idx), true);
        const auto iy = y_axis.find_index(y(idx), true);
        if (ix != -1 && iy != -1) {
          acc_(ix, iy)(value);
        }
      }
    }
  }

  /// @brief Insert values using linear binning strategy (Cartesian)
  /// @param[in] x X coordinates of the input values
  /// @param[in] y Y coordinates of the input values
  /// @param[in] z Input values to be binned
  void push_linear_cartesian(const Eigen::Ref<const Vector<T>>& x,
                             const Eigen::Ref<const Vector<T>>& y,
                             const Eigen::Ref<const Vector<T>>& z) {
    using Strategy = boost::geometry::strategy::area::cartesian<>;
    push_linear_impl<geometry::SphericalPoint, Strategy>(x, y, z, Strategy{});
  }

  /// @brief Insert values using linear binning strategy (Geographic)
  /// @param[in] x X coordinates of the input values
  /// @param[in] y Y coordinates of the input values
  /// @param[in] z Input values to be binned
  void push_linear_geographic(const Eigen::Ref<const Vector<T>>& x,
                              const Eigen::Ref<const Vector<T>>& y,
                              const Eigen::Ref<const Vector<T>>& z) {
    if (!spheroid_) {
      throw std::invalid_argument(
          "Geographic binning requires a WGS system to be set");
    }

    using Strategy = boost::geometry::strategy::area::geographic<
        boost::geometry::strategy::vincenty, 5>;
    using Spheroid = boost::geometry::srs::spheroid<double>;
    push_linear_impl<geometry::SphericalPoint, Strategy>(
        x, y, z, Strategy(static_cast<Spheroid>(*spheroid_)));
  }

  /// @brief Validate compatibility for aggregation
  /// @param[in] other Another Binning2D to compare
  void validate_compatibility(const Binning2D& other) const {
    if (x_ != other.x_ || y_ != other.y_) {
      throw std::invalid_argument("Cannot combine grids with different axes");
    }

    const bool wgs_mismatch =
        (spheroid_.has_value() != other.spheroid_.has_value()) ||
        (spheroid_.has_value() && other.spheroid_.has_value() &&
         *spheroid_ != *other.spheroid_);

    if (wgs_mismatch) {
      throw std::invalid_argument(
          "Cannot combine grids with different geodetic systems");
    }
  }

  /// @brief Calculate statistics using a generic callable
  /// @tparam ResultType The type of the result matrix elements
  /// @tparam Func Callable type that takes a DescriptiveStatistics object and
  /// returns a ResultType
  /// @param[in] func Callable to apply to each accumulator
  /// @return Matrix of results
  template <typename ResultType, typename Func>
  [[nodiscard]] auto generate_statistical_matrix(Func&& func) const
      -> Matrix<ResultType> {
    Matrix<ResultType> result(x_.size(), y_.size());

    for (auto [lhs, rhs] :
         std::ranges::views::zip(result.reshaped(), acc_.reshaped())) {
      lhs = static_cast<ResultType>(func(rhs));
    }
    return result;
  }

  /// @brief Update accumulator if weight is significant
  /// @param[in] ix X index
  /// @param[in] iy Y index
  /// @param[in] value Value to accumulate
  /// @param[in] weight Weight of the value
  void update_acc(const int64_t ix, const int64_t iy, const T value,
                  const T weight) {
    if (!math::is_almost_zero(weight)) {
      acc_(ix, iy)(value, weight);
    }
  }

  /// @brief Linear binning implementation
  /// @tparam Point Template template parameter for point type
  /// @tparam Strategy Binning strategy type
  /// @param[in] x X coordinates of the input values
  /// @param[in] y Y coordinates of the input values
  /// @param[in] z Input values to be binned
  /// @param[in] strategy Binning strategy instance
  template <template <class> class Point, typename Strategy>
  void push_linear_impl(const Eigen::Ref<const Vector<T>>& x,
                        const Eigen::Ref<const Vector<T>>& y,
                        const Eigen::Ref<const Vector<T>>& z,
                        const Strategy& strategy) {
    const auto& x_axis = static_cast<const math::Axis<double>&>(x_);
    const auto& y_axis = static_cast<const math::Axis<double>&>(y_);

    for (int64_t idx = 0; idx < x.size(); ++idx) {
      const auto value = z(idx);
      if (std::isnan(value)) {
        continue;
      }

      const auto x_indexes = x_axis.find_indexes(x(idx));
      const auto y_indexes = y_axis.find_indexes(y(idx));

      if (x_indexes && y_indexes) {
        const auto [ix0, ix1] = *x_indexes;
        const auto [iy0, iy1] = *y_indexes;

        const auto x0 = x_axis(ix0);

        // Normalize angle if needed
        const auto x_coord =
            x_axis.is_periodic()
                ? math::normalize_period<double>(x(idx), x0, 360.0)
                : x(idx);

        const auto weights =
            math::interpolate::bilinear_weights<Point, Strategy, double>(
                Point<double>(x_coord, y(idx)), Point<double>(x0, y_axis(iy0)),
                Point<double>(x_axis(ix1), y_axis(iy1)), strategy);

        update_acc(ix0, iy0, value, static_cast<T>(std::get<0>(weights)));
        update_acc(ix0, iy1, value, static_cast<T>(std::get<1>(weights)));
        update_acc(ix1, iy1, value, static_cast<T>(std::get<2>(weights)));
        update_acc(ix1, iy0, value, static_cast<T>(std::get<3>(weights)));
      }
    }
  }
};

/// @brief Specialized 1D binning class
/// @tparam T Binning value type
template <BinningValueType T>
class Binning1D : public Binning2D<T> {
 public:
  using typename Binning2D<T>::value_type;
  using typename Binning2D<T>::DescriptiveStatistics;

  /// @brief Constructor
  /// @param[in] x Definition of the bin centers for the X axis of the grid.
  /// @param[in] range Optional defined range for binning. If not provided,
  /// the full range of the X axis is used.
  explicit Binning1D(
      Axis<double> x,
      const std::optional<std::tuple<double, double>>& range = std::nullopt)
      : Binning2D<T>(std::move(x), Axis<double>(0, 1, 1, 0), std::nullopt) {
    if (range) {
      std::tie(x_min_, x_max_) = *range;
    } else {
      x_min_ = this->x_.min_value();
      x_max_ = this->x_.max_value();
    }
  }

  /// @brief Push data to bins with optional weights
  /// @param[in] x X coordinates of the input values
  /// @param[in] z Input values to be binned
  /// @param[in] weights Optional weights for the input values
  void push(const Eigen::Ref<const Vector<T>>& x,
            const Eigen::Ref<const Vector<T>>& z,
            const std::optional<Eigen::Ref<const Vector<T>>>& weights =
                std::nullopt) {
    broadcast::check_eigen_shape("x", x, "z", z);
    if (weights) {
      broadcast::check_eigen_shape("x", x, "weights", *weights);
    }

    const auto& x_axis = static_cast<const math::Axis<double>&>(this->x_);

    for (Eigen::Index idx = 0; idx < x.size(); ++idx) {
      const auto value = z(idx);
      if (!std::isnan(value)) {
        const auto xi = x(idx);
        if (xi >= x_min_ && xi <= x_max_) {
          const auto ix = x_axis.find_index(xi, true);
          if (ix != -1) {
            const auto weight = weights ? (*weights)(idx) : T{1};
            this->acc_(ix, 0)(value, weight);
          }
        }
      }
    }
  }

  /// @brief Statistical computation methods returning vectors
  /// @return Vector of counts
  [[nodiscard]] auto count() const -> Vector<uint64_t> {
    return generate_statistical_vector<uint64_t>(
        [](const auto& stats) { return stats.count(); });
  }

  /// @brief Minimum value per bin
  /// @return Vector of minimum values
  [[nodiscard]] auto min() const -> Vector<T> {
    return generate_statistical_vector<T>(
        [](const auto& stats) { return stats.min(); });
  }

  /// @brief Maximum value per bin
  /// @return Vector of maximum values
  [[nodiscard]] auto max() const -> Vector<T> {
    return generate_statistical_vector<T>(
        [](const auto& stats) { return stats.max(); });
  }

  /// @brief Mean value per bin
  /// @return Vector of mean values
  [[nodiscard]] auto mean() const -> Vector<T> {
    return generate_statistical_vector<T>(
        [](const auto& stats) { return stats.mean(); });
  }

  /// @brief Variance per bin
  /// @return Vector of variance values
  [[nodiscard]] auto variance(const int ddof = 0) const -> Vector<T> {
    return generate_statistical_vector<T>(
        [ddof](const auto& stats) { return stats.variance(ddof); });
  }

  /// @brief Kurtosis per bin
  /// @return Vector of kurtosis values
  [[nodiscard]] auto kurtosis() const -> Vector<T> {
    return generate_statistical_vector<T>(
        [](const auto& stats) { return stats.kurtosis(); });
  }

  /// @brief Skewness per bin
  /// @return Vector of skewness values
  [[nodiscard]] auto skewness() const -> Vector<T> {
    return generate_statistical_vector<T>(
        [](const auto& stats) { return stats.skewness(); });
  }

  /// @brief Sum per bin
  /// @return Vector of sum values
  [[nodiscard]] auto sum() const -> Vector<T> {
    return generate_statistical_vector<T>(
        [](const auto& stats) { return stats.sum(); });
  }

  /// @brief Sum of weights per bin
  /// @return Vector of sum of weights values
  [[nodiscard]] auto sum_of_weights() const -> Vector<T> {
    return generate_statistical_vector<T>(
        [](const auto& stats) { return stats.sum_of_weights(); });
  }

  /// @brief Get the defined range
  /// @return Tuple of (min, max) defining the range
  [[nodiscard]] auto range() const noexcept -> std::tuple<double, double> {
    return {x_min_, x_max_};
  }

  /// @brief Get a tuple that fully encodes the state of this instance.
  /// @return Tuple representing the state
  [[nodiscard]] auto getstate() const
      -> std::tuple<Axis<double>, Vector<int8_t>,
                    std::optional<geometry::geographic::Spheroid>, double,
                    double> {
    Vector<int8_t> acc_view(this->x_.size() * sizeof(DescriptiveStatistics));
    std::memcpy(acc_view.data(), this->acc_.data(),
                this->x_.size() * sizeof(DescriptiveStatistics));
    return std::make_tuple(this->x_, acc_view, this->spheroid_, x_min_, x_max_);
  }

  /// @brief Create an instance from a serialized state.
  /// @param[in] state Tuple representing the state
  static auto setstate(
      const std::tuple<Axis<double>, Vector<int8_t>,
                       std::optional<geometry::geographic::Spheroid>, double,
                       double>& state) -> Binning1D {
    auto x_axis = std::get<0>(state);
    auto acc_view = std::get<1>(state);
    auto expected_size = x_axis.size() * sizeof(DescriptiveStatistics);
    if (std::cmp_not_equal(acc_view.size(), expected_size)) {
      throw std::invalid_argument("Invalid state.");
    }
    Binning1D binning(x_axis, std::make_optional<std::tuple<double, double>>(
                                  std::get<3>(state), std::get<4>(state)));
    auto* dst = binning.acc_.data();
    auto* src = reinterpret_cast<const DescriptiveStatistics*>(acc_view.data());
    std::uninitialized_copy_n(src, x_axis.size(), dst);
    return binning;
  }

 private:
  double x_min_;
  double x_max_;

  /// @brief Calculate statistics using a generic callable (1D version)
  /// @tparam ResultType The type of the result vector elements
  /// @tparam Func Callable type that takes a DescriptiveStatistics object and
  /// returns a ResultType
  /// @param[in] func Callable to apply to each accumulator
  /// @return Vector of results
  template <typename ResultType, typename Func>
  [[nodiscard]] auto generate_statistical_vector(Func&& func) const
      -> Vector<ResultType> {
    Vector<ResultType> result(this->x_.size());

    for (Eigen::Index ix = 0; ix < this->acc_.rows(); ++ix) {
      result(ix) = static_cast<ResultType>(func(this->acc_(ix, 0)));
    }

    return result;
  }
};

/// Bindings for the binning module
auto init_binning(nanobind::module_& m) -> void;

}  // namespace pyinterp::pybind
