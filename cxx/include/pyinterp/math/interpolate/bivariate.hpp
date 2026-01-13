// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <ranges>
#include <stdexcept>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math/fill.hpp"
#include "pyinterp/math/interpolate/bracket_finder.hpp"

namespace pyinterp::math::interpolate {

/// @brief Abstract interface for bivariate interpolation
///
/// This base class defines the common interface for all bivariate
/// interpolation methods, enabling polymorphic usage of different
/// interpolation algorithms.
///
/// @tparam T type of the data (must be floating point)
template <std::floating_point T>
class BivariateBase {
 public:
  /// @brief Virtual destructor
  virtual ~BivariateBase() = default;

  // Prevent slicing
  BivariateBase() = default;
  BivariateBase(const BivariateBase&) = default;
  BivariateBase(BivariateBase&&) = default;
  auto operator=(const BivariateBase&) -> BivariateBase& = default;
  auto operator=(BivariateBase&&) -> BivariateBase& = default;

  /// @brief Interpolate the value at a single point (x, y)
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @param[in] za Z-values of the data points (2D grid).
  /// @param[in] x The x-coordinate where the interpolation must be calculated.
  /// @param[in] y The y-coordinate where the interpolation must be calculated.
  /// @return The interpolated value at point (x, y).
  [[nodiscard]] virtual auto operator()(const Eigen::Ref<const Vector<T>>& xa,
                                        const Eigen::Ref<const Vector<T>>& ya,
                                        const Eigen::Ref<const Matrix<T>>& za,
                                        const T& x, const T& y) -> T = 0;

  /// @brief Interpolate the values at multiple points
  ///
  /// Default implementation calls the single-point operator for each point.
  /// Derived classes may override for optimized batch processing.
  ///
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @param[in] za Z-values of the data points (2D grid).
  /// @param[in] x X-coordinates where the interpolation must be calculated.
  /// @param[in] y Y-coordinates where the interpolation must be calculated.
  /// @return The interpolated values at the given points.
  [[nodiscard]] virtual auto operator()(const Eigen::Ref<const Vector<T>>& xa,
                                        const Eigen::Ref<const Vector<T>>& ya,
                                        const Eigen::Ref<const Matrix<T>>& za,
                                        const Eigen::Ref<const Vector<T>>& x,
                                        const Eigen::Ref<const Vector<T>>& y)
      -> Vector<T> {
    auto z = Vector<T>(x.size());
    auto indices = std::views::iota(int64_t{0}, x.size());
    std::ranges::for_each(
        indices, [&](auto i) { z(i) = (*this)(xa, ya, za, x(i), y(i)); });
    return z;
  }
};

/// @brief Bivariate interpolation base class with coefficient computation
///
/// This class extends BivariateBase with a coefficient computation pattern,
/// where derived classes can precompute values (like derivatives) before
/// performing the actual interpolation.
///
/// @tparam T type of the data (must be floating point)
template <std::floating_point T>
class Bivariate : public BivariateBase<T>, public BracketFinder<T> {
 public:
  /// @brief The minimum size of the arrays to be interpolated.
  [[nodiscard]] virtual constexpr auto min_size() const -> int64_t = 0;

  /// @brief Interpolate the value at point (x, y).
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @param[in] za Z-values of the data points.
  /// @param[in] x The x-coordinate where the interpolation must be calculated.
  /// @param[in] y The y-coordinate where the interpolation must be calculated.
  /// @return The interpolated value at the point (x, y).
  [[nodiscard]] auto operator()(const Eigen::Ref<const Vector<T>>& xa,
                                const Eigen::Ref<const Vector<T>>& ya,
                                const Eigen::Ref<const Matrix<T>>& za,
                                const T& x, const T& y) -> T final {
    if (!compute_coefficients(xa, ya, za)) {
      return Fill<T>::value();
    }
    return interpolate_(xa, ya, za, x, y);
  }

  /// @brief Interpolate the values at multiple points.
  ///
  /// Optimized batch version that computes coefficients once for all points.
  ///
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @param[in] za Z-values of the data points.
  /// @param[in] x X-coordinates where the interpolation must be calculated.
  /// @param[in] y Y-coordinates where the interpolation must be calculated.
  /// @return The interpolated values at the given points.
  [[nodiscard]] auto operator()(const Eigen::Ref<const Vector<T>>& xa,
                                const Eigen::Ref<const Vector<T>>& ya,
                                const Eigen::Ref<const Matrix<T>>& za,
                                const Eigen::Ref<const Vector<T>>& x,
                                const Eigen::Ref<const Vector<T>>& y)
      -> Vector<T> override {
    if (!compute_coefficients(xa, ya, za)) {
      return Vector<T>::Constant(x.size(), Fill<T>::value());
    }

    auto z = Vector<T>(x.size());
    auto indices = std::views::iota(int64_t{0}, x.size());

    std::ranges::for_each(
        indices, [&](auto i) { z(i) = interpolate_(xa, ya, za, x(i), y(i)); });

    return z;
  }

 protected:
  /// @brief Interpolate the value at (x, y) using precomputed coefficients.
  [[nodiscard]] virtual auto interpolate_(const Eigen::Ref<const Vector<T>>& xa,
                                          const Eigen::Ref<const Vector<T>>& ya,
                                          const Eigen::Ref<const Matrix<T>>& za,
                                          const T& x, const T& y) const
      -> T = 0;

  /// @brief Check if the arrays are valid and compute any necessary
  /// coefficients.
  /// @return true if coefficients computed successfully, false otherwise.
  [[nodiscard]] virtual auto compute_coefficients(
      const Eigen::Ref<const Vector<T>>& xa,
      const Eigen::Ref<const Vector<T>>& ya,
      const Eigen::Ref<const Matrix<T>>& za) -> bool {
    if (xa.size() != za.rows()) [[unlikely]] {
      throw std::invalid_argument(
          "xa and za must have the same number of rows");
    }
    if (ya.size() != za.cols()) [[unlikely]] {
      throw std::invalid_argument(
          "ya and za must have the same number of columns");
    }
    return (xa.size() >= min_size() && ya.size() >= min_size());
  }
};

}  // namespace pyinterp::math::interpolate
