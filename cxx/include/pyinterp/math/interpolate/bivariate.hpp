// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
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

  /// @brief The minimum size of the arrays to be interpolated.
  [[nodiscard]] virtual constexpr auto min_size() const -> int64_t = 0;

  /// @brief Prepare interpolator with data and compute coefficients
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @param[in] za Z-values of the data points (2D grid).
  auto prepare(const Eigen::Ref<const Vector<T>>& xa,
               const Eigen::Ref<const Vector<T>>& ya,
               const Eigen::Ref<const Matrix<T>>& za) -> void {
    if (xa.size() != za.rows()) [[unlikely]] {
      throw std::invalid_argument(
          "xa and za must have the same number of rows");
    }
    if (ya.size() != za.cols()) [[unlikely]] {
      throw std::invalid_argument(
          "ya and za must have the same number of columns");
    }
    if (xa.size() < min_size() || ya.size() < min_size()) {
      is_valid_ = false;
      return;
    }

    // Store copies of the data to ensure persistence.
    // Note: This incurs a one-time memory allocation, but avoids
    // lifetime issues with temporary Eigen conversions and enables
    // the prepare-once, interpolate-many pattern.
    xa_ = xa;
    ya_ = ya;
    za_ = za;

    is_valid_ = compute_coefficients();
  }

  /// @brief Interpolate the value at a single point (x, y)
  /// @param[in] x The x-coordinate where the interpolation must be calculated.
  /// @param[in] y The y-coordinate where the interpolation must be calculated.
  /// @return The interpolated value at point (x, y).
  /// @warning Must call prepare() first
  [[nodiscard]] virtual auto operator()(const T& x, const T& y) -> T = 0;

  /// @brief Interpolate the values at multiple points
  /// @param[in] x X-coordinates where the interpolation must be calculated.
  /// @param[in] y Y-coordinates where the interpolation must be calculated.
  /// @return The interpolated values at the given points.
  /// @warning Must call prepare() first
  [[nodiscard]] virtual auto operator()(const Eigen::Ref<const Vector<T>>& x,
                                        const Eigen::Ref<const Vector<T>>& y)
      -> Vector<T> {
    auto z = Vector<T>(x.size());
    auto indices = std::views::iota(int64_t{0}, x.size());
    std::ranges::for_each(indices, [&](auto i) { z(i) = (*this)(x(i), y(i)); });
    return z;
  }

 protected:
  /// @brief Compute coefficients from stored data
  /// @return true if successful, false otherwise.
  [[nodiscard]] virtual auto compute_coefficients() -> bool { return true; }

  /// @brief Return the X-coordinates of the data points.
  /// @return X-coordinates of the data points.
  [[nodiscard]] constexpr auto xa() const -> const Vector<T>& { return xa_; }

  /// @brief Return the Y-coordinates of the data points.
  /// @return Y-coordinates of the data points.
  [[nodiscard]] constexpr auto ya() const -> const Vector<T>& { return ya_; }

  /// @brief Return the Z-values of the data points.
  /// @return Z-values of the data points.
  [[nodiscard]] constexpr auto za() const -> const Matrix<T>& { return za_; }

  /// @brief Check if the interpolator is valid (coefficients computed)
  /// @return True if the interpolator is valid, false otherwise.
  [[nodiscard]] constexpr auto is_valid() const noexcept -> bool {
    return is_valid_;
  }

 private:
  /// Stored X-coordinates of the data points
  Vector<T> xa_;
  /// Stored Y-coordinates of the data points
  Vector<T> ya_;
  /// Stored Z-values of the data points
  Matrix<T> za_;

  /// True if the coefficients have been computed successfully
  bool is_valid_{false};
};

/// @brief Bivariate interpolation base class with coefficient computation
///
/// This class extends BivariateBase with a two-phase pattern:
/// 1. prepare() - computes and stores coefficients once
/// 2. operator() - performs fast interpolation using stored coefficients
///
/// @tparam T type of the data (must be floating point)
template <std::floating_point T>
class Bivariate : public BivariateBase<T>, public BracketFinder<T> {
 public:
  /// @brief Interpolate at a single point using pre-computed coefficients
  /// @param[in] x The x-coordinate where the interpolation must be calculated.
  /// @param[in] y The y-coordinate where the interpolation must be calculated.
  /// @return The interpolated value at point (x, y).
  /// @warning Must call prepare() first
  [[nodiscard]] auto operator()(const T& x, const T& y) -> T final {
    if (!this->is_valid()) [[unlikely]] {
      return Fill<T>::value();
    }
    return interpolate_(x, y);
  }

  /// @brief Batch interpolation using pre-computed coefficients
  /// @param[in] x X-coordinates where the interpolation must be calculated.
  /// @param[in] y Y-coordinates where the interpolation must be calculated.
  /// @return The interpolated values at the given points.
  /// @warning Must call prepare() first
  [[nodiscard]] auto operator()(const Eigen::Ref<const Vector<T>>& x,
                                const Eigen::Ref<const Vector<T>>& y)
      -> Vector<T> final {
    auto z = Vector<T>(x.size());
    if (!this->is_valid()) [[unlikely]] {
      z.fill(Fill<T>::value());
      return z;
    }
    auto indices = std::views::iota(int64_t{0}, x.size());
    std::ranges::for_each(indices,
                          [&](auto i) { z(i) = interpolate_(x(i), y(i)); });
    return z;
  }

 protected:
  /// @brief Interpolate the value at (x, y) using precomputed coefficients
  /// @param[in] x The x-coordinate where the interpolation must be calculated.
  /// @param[in] y The y-coordinate where the interpolation must be calculated.
  /// @return The interpolated value at point (x, y).
  [[nodiscard]] virtual auto interpolate_(const T& x, const T& y) const
      -> T = 0;

  /// @brief Compute coefficients from stored data
  /// @return true if successful, false otherwise.
  [[nodiscard]] virtual auto compute_coefficients() -> bool { return true; }
};

}  // namespace pyinterp::math::interpolate
