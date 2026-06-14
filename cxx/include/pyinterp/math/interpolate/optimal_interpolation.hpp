// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <concepts>
#include <stdexcept>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math.hpp"
#include "pyinterp/math/interpolate/anisotropic.hpp"
#include "pyinterp/math/interpolate/kriging.hpp"

namespace pyinterp::math::interpolate {

/// @brief 4D Optimal Interpolation (BLUE) estimator with an anisotropic
/// covariance kernel and per-observation error variance.
///
/// For a single query point this class assembles
/// @f$(C_{oo} + R)\, w = c_{og}@f$ where
/// @f$C_{oo}[i,j] = \sigma^2 K(r_{ij})@f$,
/// @f$c_{og}[i] = \sigma^2 K(r_{i\bullet})@f$,
/// @f$r^2 = \sum_d ((p_1^{(d)} - p_2^{(d)}) / L^{(d)})^2@f$
/// and @f$R = \mathrm{diag}(\sigma^2_\mathrm{obs})@f$. It then returns the
/// analysis @f$f = w^\top \cdot \mathrm{obs}@f$ and the formal error
/// variance @f$e^2 = \sigma^2 - c_{og}^\top w@f$.
///
/// This is @b simple @b kriging: the BLUE about a known, zero mean. The
/// weights are not constrained to sum to one, so @c obs must be anomalies
/// about a zero background mean; where @f$c_{og} \to 0@f$ (no correlated
/// neighbour) the analysis relaxes to @f$0@f$ and @f$e^2 \to \sigma^2@f$.
/// @f$\sigma@f$ is taken at the query point and assumed constant across its
/// neighbourhood.
///
/// The class is stateless besides its compiled-in radial covariance choice,
/// so a single instance is safe to share across worker threads.
///
/// @tparam T Floating-point type used for matrix arithmetic.
template <std::floating_point T>
class OptimalInterpolation {
 public:
  /// Result returned by @ref solve.
  struct Result {
    /// Analysed value at the query point.
    T value;
    /// Formal error variance (always non-negative; numerical negatives are
    /// clamped to zero).
    T error2;
  };

  /// @brief Construct an estimator with a given anisotropic radial kernel.
  ///
  /// @param[in] covariance Radial covariance kernel name.
  /// @throws std::invalid_argument when @p covariance is not a known
  ///   value of @ref CovarianceFunction.
  explicit OptimalInterpolation(CovarianceFunction covariance)
      : kernel_(select_radial_covariance<T>(covariance)) {}

  /// @brief Solve the OI system at a single query point.
  ///
  /// @param[in] obs_coords Observations, shape @c (n, 4).
  /// @param[in] obs_values Observed values, shape @c (n,).
  /// @param[in] obs_sigma2 Per-observation error variance @f$\sigma^2_\mathrm{obs}@f$,
  ///   shape @c (n,). Must be strictly positive.
  /// @param[in] query Query point, shape @c (4,).
  /// @param[in] inv_L Element-wise inverse of the anisotropic decorrelation
  ///   length scales @f$1/L_d@f$, shape @c (4,). Pre-inverting avoids
  ///   per-cell divisions in the @c (n, n) matrix build.
  /// @param[in] sigma Field standard deviation @f$\sigma = \sqrt{C(0)}@f$ at
  ///   the query point.
  /// @return Analysed value and formal error variance.
  [[nodiscard]] auto solve(
      const Eigen::Ref<const Eigen::Matrix<T, -1, 4, Eigen::RowMajor>>&
          obs_coords,
      const Eigen::Ref<const Vector<T>>& obs_values,
      const Eigen::Ref<const Vector<T>>& obs_sigma2,
      const Eigen::Vector<T, 4>& query, const Eigen::Vector<T, 4>& inv_L,
      const T sigma) const -> Result {
    const Eigen::Index n = obs_coords.rows();
    if (n == 0) {
      throw std::invalid_argument("OI requires at least one observation");
    }
    if (obs_values.size() != n || obs_sigma2.size() != n) {
      throw std::invalid_argument(
          "obs_coords, obs_values and obs_sigma2 must share the same length");
    }

    const T sigma2_field = math::sqr(sigma);

    // Anisotropic squared distances obs <-> obs.
    Matrix<T> c_oo(n, n);
    for (Eigen::Index i = 0; i < n; ++i) {
      const Eigen::Vector<T, 4> pi = obs_coords.row(i).transpose();
      c_oo(i, i) = sigma2_field + obs_sigma2[i];  // diag = σ² + R
      for (Eigen::Index j = i + 1; j < n; ++j) {
        const Eigen::Vector<T, 4> pj = obs_coords.row(j).transpose();
        const T r2 = anisotropic_distance_squared<T, 4>(pi, pj, inv_L);
        // λ = 1: per-axis scaling is already baked into r².
        const T cov = sigma2_field * kernel_(r2, T{1}, T{1});
        c_oo(i, j) = cov;
        c_oo(j, i) = cov;
      }
    }

    // Anisotropic distances obs <-> query and the covariance vector c_og.
    Vector<T> c_og(n);
    for (Eigen::Index i = 0; i < n; ++i) {
      const Eigen::Vector<T, 4> pi = obs_coords.row(i).transpose();
      const T r2 = anisotropic_distance_squared<T, 4>(pi, query, inv_L);
      c_og[i] = sigma2_field * kernel_(r2, T{1}, T{1});
    }

    // Solve (C_oo + R) w = c_og — Cholesky first, LDLT fallback.
    Vector<T> w;
    {
      const Eigen::LLT<Matrix<T>> llt(c_oo);
      if (llt.info() == Eigen::Success) {
        w = llt.solve(c_og);
      } else {
        w = c_oo.ldlt().solve(c_og);
      }
    }

    const T prediction = obs_values.dot(w);
    const T error2 = std::max(sigma2_field - c_og.dot(w), T{0});
    return {prediction, error2};
  }

 private:
  /// Pointer to the radial covariance form (output of
  /// @ref select_radial_covariance).
  RadialCovariancePtr<T> kernel_;
};

}  // namespace pyinterp::math::interpolate
