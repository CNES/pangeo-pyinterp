// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <concepts>
#include <cstdint>
#include <format>
#include <numbers>
#include <optional>
#include <ranges>
#include <stdexcept>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math.hpp"

namespace pyinterp::math::interpolate {

// ===========================================================================
// Radial covariance forms.
//
// Each `*_covariance_from_r2(r2, sigma, lambda)` evaluates the covariance
// kernel given the SQUARED distance between two points. These factored forms
// serve two purposes:
//
// 1. The existing 3D isotropic `*_covariance(p1, p2, sigma, lambda)` functions
//    are now thin wrappers that compute `(p1 - p2).squaredNorm()` once and
//    delegate to the `_from_r2` form. Results are numerically equivalent to
//    the pre-refactor implementation, and bit-identical for the ν = 1/2 and
//    ν = 3/2 Matérn kernels (Eigen's `norm()` is `sqrt(squaredNorm())`). For
//    the others the algebra is re-associated (`sqr(r/λ)` becomes `r²/sqr(λ)`,
//    and the compact-support cutoff `r >= λ` becomes `r² >= λ²`), so a
//    handful of last-ULP differences — and a boundary reclassification within
//    one ULP of `r == λ` for the compact kernels — are possible.
//
// 2. The anisotropic Optimal Interpolation path (see anisotropic.hpp) computes
//    `r2 = Σ((Δᵢ / Lᵢ)²)` with per-axis decorrelation scales, then evaluates
//    the kernel with `lambda = 1` — the length scales are already absorbed
//    into `r2`.
// ===========================================================================

/// Matérn ν = 1/2 (exponential) covariance from squared distance.
template <std::floating_point T>
[[nodiscard]] auto matern_covariance_12_from_r2(const T r2, const T sigma,
                                                const T lambda) noexcept -> T {
  return math::sqr(sigma) * std::exp(-std::sqrt(r2) / lambda);
}

/// Matérn ν = 3/2 covariance from squared distance.
template <std::floating_point T>
[[nodiscard]] auto matern_covariance_32_from_r2(const T r2, const T sigma,
                                                const T lambda) noexcept -> T {
  const T d = std::sqrt(r2) / lambda;
  constexpr T sqrt3 = std::numbers::sqrt3_v<T>;
  return math::sqr(sigma) * std::fma(sqrt3, d, T{1}) * std::exp(-sqrt3 * d);
}

/// Matérn ν = 5/2 covariance from squared distance.
template <std::floating_point T>
[[nodiscard]] auto matern_covariance_52_from_r2(const T r2, const T sigma,
                                                const T lambda) noexcept -> T {
  const T d2 = r2 / math::sqr(lambda);
  const T d = std::sqrt(d2);
  // sqrt(5) [https://oeis.org/A002163]
  constexpr T sqrt5 =
      T{2.23606797749978969640917366873127623544061835961152572427089724};
  const T term = std::fma(T{5} / T{3}, d2, std::fma(sqrt5, d, T{1}));
  return math::sqr(sigma) * term * std::exp(-sqrt5 * d);
}

/// Cauchy covariance (heavy-tailed) from squared distance.
template <std::floating_point T>
[[nodiscard]] auto cauchy_covariance_from_r2(const T r2, const T sigma,
                                             const T lambda) noexcept -> T {
  return math::sqr(sigma) / (T{1} + r2 / math::sqr(lambda));
}

/// Spherical covariance (compact support, C⁰) from squared distance.
template <std::floating_point T>
[[nodiscard]] auto spherical_covariance_from_r2(const T r2, const T sigma,
                                                const T lambda) noexcept -> T {
  if (r2 >= math::sqr(lambda)) [[unlikely]] {
    return T{0};
  }
  const T t = std::sqrt(r2) / lambda;
  return math::sqr(sigma) *
         std::fma(T{0.5}, t * t * t, std::fma(T{-1.5}, t, T{1}));
}

/// Gaussian covariance (C∞) from squared distance.
template <std::floating_point T>
[[nodiscard]] auto gaussian_covariance_from_r2(const T r2, const T sigma,
                                               const T lambda) noexcept -> T {
  return math::sqr(sigma) * std::exp(-r2 / math::sqr(lambda));
}

/// Wendland φ_{3,0} covariance (compact support) from squared distance.
template <std::floating_point T>
[[nodiscard]] auto wendland_covariance_from_r2(const T r2, const T sigma,
                                               const T lambda) noexcept -> T {
  if (r2 >= math::sqr(lambda)) [[unlikely]] {
    return T{0};
  }
  const T t = T{1} - std::sqrt(r2) / lambda;
  return math::sqr(sigma) * math::sqr(t);
}

// ===========================================================================
// Isotropic 3D wrappers — preserved for backward compatibility.
// ===========================================================================

/// Matérn covariance function for ν = 0.5 (exponential covariance)
template <std::floating_point T>
[[nodiscard]] auto matern_covariance_12(
    const Eigen::Ref<const Eigen::Vector3<T>>& p1,
    const Eigen::Ref<const Eigen::Vector3<T>>& p2, const T sigma,
    const T lambda) noexcept -> T {
  return matern_covariance_12_from_r2<T>((p1 - p2).squaredNorm(), sigma,
                                         lambda);
}

/// Matérn covariance function for ν = 1.5
template <std::floating_point T>
[[nodiscard]] auto matern_covariance_32(
    const Eigen::Ref<const Eigen::Vector3<T>>& p1,
    const Eigen::Ref<const Eigen::Vector3<T>>& p2, const T sigma,
    const T lambda) noexcept -> T {
  return matern_covariance_32_from_r2<T>((p1 - p2).squaredNorm(), sigma,
                                         lambda);
}

/// Matérn covariance function for ν = 2.5
template <std::floating_point T>
[[nodiscard]] auto matern_covariance_52(
    const Eigen::Ref<const Eigen::Vector3<T>>& p1,
    const Eigen::Ref<const Eigen::Vector3<T>>& p2, const T sigma,
    const T lambda) noexcept -> T {
  return matern_covariance_52_from_r2<T>((p1 - p2).squaredNorm(), sigma,
                                         lambda);
}

/// Cauchy covariance function (heavy-tailed, long-range correlations)
template <std::floating_point T>
[[nodiscard]] auto cauchy_covariance(
    const Eigen::Ref<const Eigen::Vector3<T>>& p1,
    const Eigen::Ref<const Eigen::Vector3<T>>& p2, const T sigma,
    const T lambda) noexcept -> T {
  return cauchy_covariance_from_r2<T>((p1 - p2).squaredNorm(), sigma, lambda);
}

/// Spherical covariance function (compact support, C⁰ continuous)
template <std::floating_point T>
[[nodiscard]] auto spherical_covariance(
    const Eigen::Ref<const Eigen::Vector3<T>>& p1,
    const Eigen::Ref<const Eigen::Vector3<T>>& p2, const T sigma,
    const T lambda) noexcept -> T {
  return spherical_covariance_from_r2<T>((p1 - p2).squaredNorm(), sigma,
                                         lambda);
}

/// Gaussian covariance function (infinitely smooth, C∞)
template <std::floating_point T>
[[nodiscard]] auto gaussian_covariance(
    const Eigen::Ref<const Eigen::Vector3<T>>& p1,
    const Eigen::Ref<const Eigen::Vector3<T>>& p2, const T sigma,
    const T lambda) noexcept -> T {
  return gaussian_covariance_from_r2<T>((p1 - p2).squaredNorm(), sigma, lambda);
}

/// Wendland φ_{3,0} covariance function (compact support, positive definite in
/// ℝ³) C(r) = σ² (1 - r/λ)²₊
template <std::floating_point T>
[[nodiscard]] auto wendland_covariance(
    const Eigen::Ref<const Eigen::Vector3<T>>& p1,
    const Eigen::Ref<const Eigen::Vector3<T>>& p2, const T sigma,
    const T lambda) noexcept -> T {
  return wendland_covariance_from_r2<T>((p1 - p2).squaredNorm(), sigma, lambda);
}

/// Known covariance functions
enum class CovarianceFunction : uint8_t {
  kMatern_12,  ///< Matérn ν = 0.5 (exponential, C⁰)
  kMatern_32,  ///< Matérn ν = 1.5 (C¹)
  kMatern_52,  ///< Matérn ν = 2.5 (C²)
  kCauchy,     ///< Cauchy (heavy-tailed)
  kSpherical,  ///< Spherical (compact support)
  kGaussian,   ///< Gaussian (C∞, can cause numerical issues)
  kWendland,   ///< Wendland φ_{3,0} (compact support, sparse matrices)
};

/// Known drift functions for universal kriging
enum class DriftFunction : uint8_t {
  kLinear,     ///< Constant + linear terms (4 parameters)
  kQuadratic,  ///< Constant + linear + quadratic terms (10 parameters)
};

/// @brief Kriging interpolation for spatial data in 3D
///
/// Supports both simple kriging (assumes known mean, typically zero) and
/// universal kriging (with polynomial drift for non-stationary data).
///
/// @tparam T Floating-point type for calculations (float or double)
///
/// The class template parameter is intentionally unconstrained at the
/// declaration level so that an unrelated container (e.g. an R-tree storing
/// composite values) can mention `Kriging<promotion_t>` in a function
/// signature it never instantiates. The actual constraint is enforced via a
/// `static_assert` in the class body, which fires only when the class is
/// instantiated.
template <typename T>
class Kriging {
  static_assert(std::floating_point<T>,
                "Kriging requires a floating-point template parameter");

 public:
  /// Pointer to the covariance function
  using CovarianceFunctionPtr =
      T (*)(const Eigen::Ref<const Eigen::Vector3<T>>&,
            const Eigen::Ref<const Eigen::Vector3<T>>&, T, T) noexcept;

  /// @brief Result of a kriging query
  struct Result {
    T value;     ///< Estimated value at query point
    T variance;  ///< Estimation variance (kriging variance)

    /// @brief Standard error (square root of variance)
    [[nodiscard]] auto std_error() const noexcept -> T {
      return std::sqrt(std::max(variance, T{0}));
    }
  };

  /// @brief Constructor
  /// @param[in] sigma Magnitude parameter (sill - nugget, must be > 0)
  /// @param[in] lambda Correlation length parameter (range, must be > 0)
  /// @param[in] nugget Nugget effect (measurement error variance, must be >= 0)
  /// @param[in] function Covariance function to use
  /// @param[in] drift_function Optional drift function for universal kriging
  /// @throws std::invalid_argument if parameters are invalid
  Kriging(const T sigma, const T lambda, const T nugget,
          const CovarianceFunction function,
          const std::optional<DriftFunction> drift_function = std::nullopt)
      : sigma_{sigma},
        lambda_{lambda},
        nugget_{nugget},
        drift_function_{drift_function.value_or(DriftFunction::kLinear)},
        use_universal_{drift_function.has_value()} {
    validate_parameters();
    function_ = select_covariance_function(function);
  }

  /// @brief Estimate the value at a query point
  /// @param[in] coordinates Coordinates of known points (3 × n matrix)
  /// @param[in] values Values at known points (n-vector)
  /// @param[in] query Coordinates of query point (3-vector)
  /// @return Estimated value at query point
  /// @throws std::invalid_argument if dimensions mismatch
  [[nodiscard]] auto operator()(const Eigen::Matrix<T, 3, -1>& coordinates,
                                const Eigen::Matrix<T, -1, 1>& values,
                                const Eigen::Vector3<T>& query) const -> T {
    return solve(coordinates, values, query).value;
  }

  /// @brief Estimate the value and variance at a query point
  /// @param[in] coordinates Coordinates of known points (3 × n matrix)
  /// @param[in] values Values at known points (n-vector)
  /// @param[in] query Coordinates of query point (3-vector)
  /// @return Result structure containing value and variance
  /// @throws std::invalid_argument if dimensions mismatch
  /// @throws std::runtime_error for universal kriging with insufficient points
  [[nodiscard]] auto solve(const Eigen::Matrix<T, 3, -1>& coordinates,
                           const Eigen::Matrix<T, -1, 1>& values,
                           const Eigen::Vector3<T>& query) const -> Result {
    validate_input(coordinates, values);
    return use_universal_ ? universal_kriging(coordinates, values, query)
                          : simple_kriging(coordinates, values, query);
  }

  /// @brief Get the sigma (sill) parameter
  [[nodiscard]] auto sigma() const noexcept -> T { return sigma_; }

  /// @brief Get the lambda (range) parameter
  [[nodiscard]] auto lambda() const noexcept -> T { return lambda_; }

  /// @brief Get the nugget parameter
  [[nodiscard]] auto nugget() const noexcept -> T { return nugget_; }

  /// @brief Check if universal kriging is enabled
  [[nodiscard]] auto is_universal() const noexcept -> bool {
    return use_universal_;
  }

 private:
  T sigma_;
  T lambda_;
  T nugget_;
  DriftFunction drift_function_;
  bool use_universal_;
  CovarianceFunctionPtr function_;

  /// @brief Validate constructor parameters
  void validate_parameters() const {
    if (sigma_ <= T{0}) {
      throw std::invalid_argument("sigma must be greater than 0");
    }
    if (lambda_ <= T{0}) {
      throw std::invalid_argument("lambda must be greater than 0");
    }
    if (nugget_ < T{0}) {
      throw std::invalid_argument("nugget must be >= 0");
    }
  }

  /// @brief Validate input data
  /// @param[in] coordinates Coordinates of known points (3 × n matrix)
  /// @param[in] values Values at known points (n-vector)
  void validate_input(const Eigen::Matrix<T, 3, -1>& coordinates,
                      const Eigen::Matrix<T, -1, 1>& values) const {
    if (coordinates.cols() != values.rows()) {
      throw std::invalid_argument(
          std::format("coordinates.cols() ({}) != values.size() ({})",
                      coordinates.cols(), values.rows()));
    }
    if (coordinates.cols() == 0) {
      throw std::invalid_argument("At least one data point is required");
    }
  }

  /// @brief Select covariance function based on enum
  /// @param[in] func Covariance function enum
  /// @return Pointer to the selected covariance function
  [[nodiscard]] static auto select_covariance_function(CovarianceFunction func)
      -> CovarianceFunctionPtr {
    switch (func) {
      case CovarianceFunction::kMatern_12:
        return matern_covariance_12<T>;
      case CovarianceFunction::kMatern_32:
        return matern_covariance_32<T>;
      case CovarianceFunction::kMatern_52:
        return matern_covariance_52<T>;
      case CovarianceFunction::kCauchy:
        return cauchy_covariance<T>;
      case CovarianceFunction::kSpherical:
        return spherical_covariance<T>;
      case CovarianceFunction::kGaussian:
        return gaussian_covariance<T>;
      case CovarianceFunction::kWendland:
        return wendland_covariance<T>;
    }
    // Unreachable with valid enum, but satisfies compiler
    throw std::invalid_argument("Invalid covariance function");
  }

  /// @brief Get size of drift basis
  /// @param[in] func Drift function enum
  /// @return Size of drift basis
  [[nodiscard]] static constexpr auto drift_basis_size(DriftFunction func)
      -> Eigen::Index {
    return func == DriftFunction::kLinear ? 4 : 10;
  }

  /// @brief Build covariance matrix C (n × n)
  /// @param[in] coordinates Coordinates of known points (3 × n matrix)
  /// @return Covariance matrix C
  [[nodiscard]] auto build_covariance_matrix(
      const Eigen::Matrix<T, 3, -1>& coordinates) const -> Matrix<T> {
    const auto n = coordinates.cols();
    Matrix<T> C(n, n);

    for (auto i : std::views::iota(Eigen::Index{0}, n)) {
      // Diagonal: covariance at zero distance + nugget
      C(i, i) = math::sqr(sigma_) + nugget_;

      // Off-diagonal: symmetric
      for (auto j : std::views::iota(i + 1, n)) {
        const T cov =
            function_(coordinates.col(i), coordinates.col(j), sigma_, lambda_);
        C(i, j) = cov;
        C(j, i) = cov;
      }
    }
    return C;
  }

  /// @brief Build covariance vector c (n-vector) between query and data points
  /// @param[in] coordinates Coordinates of known points (3 × n matrix)
  /// @param[in] query Coordinates of query point (3-vector)
  /// @return Covariance vector c
  [[nodiscard]] auto build_covariance_vector(
      const Eigen::Matrix<T, 3, -1>& coordinates,
      const Eigen::Vector3<T>& query) const -> Vector<T> {
    const auto n = coordinates.cols();
    Vector<T> c(n);

    for (auto i : std::views::iota(Eigen::Index{0}, n)) {
      c(i) = function_(query, coordinates.col(i), sigma_, lambda_);
    }
    return c;
  }

  /// @brief Evaluate drift basis at a point
  /// @param[in] point Coordinates of the point (3-vector)
  /// @return Drift basis vector
  [[nodiscard]] auto evaluate_drift(const Eigen::Vector3<T>& point) const
      -> Vector<T> {
    const auto p = drift_basis_size(drift_function_);
    Vector<T> f(p);

    f(0) = T{1};
    f(1) = point(0);
    f(2) = point(1);
    f(3) = point(2);

    if (drift_function_ == DriftFunction::kQuadratic) {
      f(4) = math::sqr(point(0));
      f(5) = math::sqr(point(1));
      f(6) = math::sqr(point(2));
      f(7) = point(0) * point(1);
      f(8) = point(0) * point(2);
      f(9) = point(1) * point(2);
    }
    return f;
  }

  /// @brief Build drift matrix F (n × p)
  /// @param[in] coordinates Coordinates of known points (3 × n matrix)
  /// @return Drift matrix F
  [[nodiscard]] auto build_drift_matrix(
      const Eigen::Matrix<T, 3, -1>& coordinates) const -> Matrix<T> {
    const auto n = coordinates.cols();
    const auto p = drift_basis_size(drift_function_);
    Matrix<T> F(n, p);

    for (auto i : std::views::iota(Eigen::Index{0}, n)) {
      F.row(i) = evaluate_drift(coordinates.col(i)).transpose();
    }
    return F;
  }

  /// @brief Simple kriging (assumes zero mean)
  /// @param[in] coordinates Coordinates of known points (3 × n matrix)
  /// @param[in] values Values at known points (n-vector)
  /// @param[in] query Coordinates of query point (3-vector)
  /// @return Result structure containing value and variance
  [[nodiscard]] auto simple_kriging(const Eigen::Matrix<T, 3, -1>& coordinates,
                                    const Eigen::Matrix<T, -1, 1>& values,
                                    const Eigen::Vector3<T>& query) const
      -> Result {
    const auto C = build_covariance_matrix(coordinates);
    const auto c = build_covariance_vector(coordinates, query);

    // Solve C w = c using Cholesky decomposition (C is SPD)
    const Eigen::LLT<Matrix<T>> llt(C);
    if (llt.info() != Eigen::Success) {
      // Fall back to more robust solver if Cholesky fails
      const Vector<T> w = C.ldlt().solve(c);
      const T variance = std::max(math::sqr(sigma_) + nugget_ - c.dot(w), T{0});
      return {values.dot(w), variance};
    }

    const Vector<T> w = llt.solve(c);

    // Variance: σ² + nugget - c'C⁻¹c
    // Clamped to zero to handle numerical errors
    const T total_variance = math::sqr(sigma_) + nugget_;
    const T variance = std::max(total_variance - c.dot(w), T{0});

    return {values.dot(w), variance};
  }

  /// @brief Universal kriging (with polynomial drift)
  /// @param[in] coordinates Coordinates of known points (3 × n matrix)
  /// @param[in] values Values at known points (n-vector)
  /// @param[in] query Coordinates of query point (3-vector)
  /// @return Result structure containing value and variance
  [[nodiscard]] auto universal_kriging(
      const Eigen::Matrix<T, 3, -1>& coordinates,
      const Eigen::Matrix<T, -1, 1>& values,
      const Eigen::Vector3<T>& query) const -> Result {
    const auto n = coordinates.cols();
    const auto p = drift_basis_size(drift_function_);

    if (n < p) {
      throw std::runtime_error(std::format(
          "Universal kriging requires at least {} points for {} drift, got {}",
          p, drift_function_ == DriftFunction::kLinear ? "linear" : "quadratic",
          n));
    }

    // Build system components
    const auto C = build_covariance_matrix(coordinates);
    const auto F = build_drift_matrix(coordinates);
    const auto c = build_covariance_vector(coordinates, query);
    const auto f = evaluate_drift(query);

    // Build augmented system: [C  F ] [w] = [c]
    //                         [F' 0 ] [β]   [f]
    Matrix<T> A(n + p, n + p);
    A.topLeftCorner(n, n) = C;
    A.topRightCorner(n, p) = F;
    A.bottomLeftCorner(p, n) = F.transpose();
    A.bottomRightCorner(p, p).setZero();

    // Build RHS
    Vector<T> b(n + p);
    b.head(n) = c;
    b.tail(p) = f;

    // Solve using QR decomposition (robust for indefinite augmented system)
    const Vector<T> x = A.colPivHouseholderQr().solve(b);

    // Prediction: w · values (first n components are weights)
    const T prediction = values.dot(x.head(n));

    // Variance: σ² + nugget - b'A⁻¹b
    const T total_variance = math::sqr(sigma_) + nugget_;
    const T variance = std::max(total_variance - b.dot(x), T{0});

    return {prediction, variance};
  }
};

}  // namespace pyinterp::math::interpolate
