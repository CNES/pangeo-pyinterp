// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <Eigen/Core>
#include <concepts>
#include <cstddef>
#include <stdexcept>

#include "pyinterp/math/interpolate/kriging.hpp"

namespace pyinterp::math::interpolate {

/// @brief Pointer to a radial covariance function of squared distance.
///
/// Signature: `T (T r2, T sigma, T lambda) noexcept`.
///
/// The `_from_r2` family in `kriging.hpp` matches this signature. When used
/// from the anisotropic OI path, `r2` already carries per-axis length scales
/// (Lₓ, Lᵧ, Lₜ, …) and the caller passes `lambda = 1`.
template <std::floating_point T>
using RadialCovariancePtr = T (*)(T, T, T) noexcept;

/// @brief Return a pointer to the squared-distance form of a covariance.
///
/// @tparam T Floating-point type (float / double).
/// @param[in] func Covariance kernel identifier.
/// @return Function pointer to the matching `*_covariance_from_r2` template
///   instantiation.
/// @throws std::invalid_argument when @p func does not name a known kernel
///   (unreachable with a well-formed enum value).
template <std::floating_point T>
[[nodiscard]] inline auto select_radial_covariance(CovarianceFunction func)
    -> RadialCovariancePtr<T> {
  switch (func) {
    case CovarianceFunction::kMatern_12:
      return &matern_covariance_12_from_r2<T>;
    case CovarianceFunction::kMatern_32:
      return &matern_covariance_32_from_r2<T>;
    case CovarianceFunction::kMatern_52:
      return &matern_covariance_52_from_r2<T>;
    case CovarianceFunction::kCauchy:
      return &cauchy_covariance_from_r2<T>;
    case CovarianceFunction::kSpherical:
      return &spherical_covariance_from_r2<T>;
    case CovarianceFunction::kGaussian:
      return &gaussian_covariance_from_r2<T>;
    case CovarianceFunction::kWendland:
      return &wendland_covariance_from_r2<T>;
  }
  throw std::invalid_argument("Invalid covariance function");
}

/// @brief Compute the anisotropic squared distance between two points.
///
/// @f[
///   r^2 = \sum_{i=0}^{\mathrm{Dim}-1} \left( \frac{p_1^{(i)} - p_2^{(i)}}
///                                                  {L^{(i)}} \right)^2
/// @f]
///
/// @tparam T Floating-point type (float / double).
/// @tparam Dim Compile-time dimensionality of the points.
/// @param[in] p1 First point (length @p Dim).
/// @param[in] p2 Second point (length @p Dim).
/// @param[in] inv_L Element-wise inverse of the decorrelation length scales
///   (i.e. `1 / Lᵢ`). Pre-inverting once per query saves @p Dim divisions per
///   matrix entry.
/// @return Scaled squared distance, always non-negative.
template <std::floating_point T, int Dim>
[[nodiscard]] inline auto anisotropic_distance_squared(
    const Eigen::Matrix<T, Dim, 1>& p1, const Eigen::Matrix<T, Dim, 1>& p2,
    const Eigen::Matrix<T, Dim, 1>& inv_L) noexcept -> T {
  return ((p1 - p2).cwiseProduct(inv_L)).squaredNorm();
}

}  // namespace pyinterp::math::interpolate
