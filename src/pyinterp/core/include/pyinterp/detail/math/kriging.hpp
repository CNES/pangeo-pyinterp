// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "pyinterp/detail/math.hpp"
#include "pyinterp/eigen.hpp"

namespace pyinterp::detail::math {

/// Matern covariance function for nu = 0.5 (i.e., exponential covariance)
template <typename T>
inline auto matern_covariance_12(const Eigen::Ref<const Eigen::Vector3<T>>& p1,
                                 const Eigen::Ref<const Eigen::Vector3<T>>& p2,
                                 const T& sigma, const T& lambda) -> T {
  auto r = (p1 - p2).norm();
  return math::sqr(sigma) * std::exp(-r / lambda);
}

/// Matern covariance function for nu = 1.5
template <typename T>
inline auto matern_covariance_32(const Eigen::Ref<const Eigen::Vector3<T>>& p1,
                                 const Eigen::Ref<const Eigen::Vector3<T>>& p2,
                                 const T& sigma, const T& lambda) -> T {
  auto r = (p1 - p2).norm();
  auto d = r / lambda;
  auto result = math::sqr(sigma);
  result *= (1 + std::sqrt(T(3)) * d) * std::exp(-std::sqrt(T(3)) * d);
  return result;
}

/// Matern covariance function for nu = 2.5
template <typename T>
inline auto matern_covariance_52(const Eigen::Ref<const Eigen::Vector3<T>>& p1,
                                 const Eigen::Ref<const Eigen::Vector3<T>>& p2,
                                 const T& sigma, const T& lambda) -> T {
  auto r = (p1 - p2).norm();
  auto d = r / lambda;
  auto result = math::sqr(sigma);
  result *= (1 + std::sqrt(T(5)) * d + T(5) / T(3) * math::sqr(d)) *
            std::exp(-std::sqrt(T(5)) * d);
  return result;
}

/// Cauchy covariance function
template <typename T>
inline auto cauchy_covariance(const Eigen::Ref<const Eigen::Vector3<T>>& p1,
                              const Eigen::Ref<const Eigen::Vector3<T>>& p2,
                              const T& sigma, const T& lambda) -> T {
  auto r = (p1 - p2).norm();
  return math::sqr(sigma) / (1 + math::sqr(r / lambda));
}

/// Spherical covariance function
template <typename T>
inline auto spherical_covariance(const Eigen::Ref<const Eigen::Vector3<T>>& p1,
                                 const Eigen::Ref<const Eigen::Vector3<T>>& p2,
                                 const T& sigma, const T& lambda) -> T {
  auto r = (p1 - p2).norm();
  if (r > lambda) {
    return T(0);
  }
  return math::sqr(sigma) *
         (T(1) - T(1.5) * r / lambda + T(0.5) * std::pow(r / lambda, T(3)));
}

/// Gaussian covariance function
template <typename T>
inline auto gaussian_covariance(const Eigen::Ref<const Eigen::Vector3<T>>& p1,
                                const Eigen::Ref<const Eigen::Vector3<T>>& p2,
                                const T& sigma, const T& lambda) -> T {
  auto r = (p1 - p2).norm();
  return math::sqr(sigma) * std::exp(-math::sqr(r) / math::sqr(lambda));
}

/// Linear covariance function
/// Uses the Wendland phi_{3,0}(r) = (1 - r/λ)^2_+ which is positive definite in
/// R^3. Keeps the enum name kLinear for backward compatibility.
template <typename T>
inline auto linear_covariance(const Eigen::Ref<const Eigen::Vector3<T>>& p1,
                              const Eigen::Ref<const Eigen::Vector3<T>>& p2,
                              const T& sigma, const T& lambda) -> T {
  auto r = (p1 - p2).norm();
  if (r >= lambda) {
    return T(0);
  }
  auto t = T(1) - r / lambda;
  return math::sqr(sigma) * math::sqr(t);
}

/// Known Covariance functions.
enum CovarianceFunction : uint8_t {
  kMatern_12 = 0,
  kMatern_32 = 1,
  kMatern_52 = 2,
  kCauchy = 3,
  kSpherical = 4,
  kGaussian = 5,
  kLinear = 6,
};

/// Known drift functions.
enum DriftFunction : uint8_t {
  kLinearDrift = 0,
  kQuadraticDrift = 1,
};

/// @brief Krige the value of a point.
/// @tparam T Type of the input.
template <typename T>
class Kriging {
 public:
  /// Pointer to the covariance function used to estimate the value of a point.
  using PtrCovarianceFunction =
      T (*)(const Eigen::Ref<const Eigen::Vector3<T>>&,
            const Eigen::Ref<const Eigen::Vector3<T>>&, const T&, const T&);

  /// @brief Default constructor.
  /// @param sigma The magnitude parameter. Determines the overall scale of the
  /// covariance function. It represents the maximum possible covariance between
  /// two points.
  /// @param lambda Decay rate parameter. Determines the rate at which the
  /// covariance decreases. It represents the spatial scale of the covariance
  /// function and can be used to control the smoothness of the spatial
  /// dependence structure.
  /// @param nugget Nugget effect term. A small positive value added to the
  /// diagonal of the covariance matrix for numerical stability.
  /// @param function The covariance function used to estimate the value of a
  /// point.
  /// @param drift_function The drift function to use for universal kriging. If
  /// not provided, simple kriging will be used.
  /// @note If no ``drift_function`` is specified, simple kriging with a known
  /// (zero) mean is used.
  Kriging(const T& sigma, const T& lambda, const T& nugget,
          const CovarianceFunction& function,
          const std::optional<DriftFunction>& drift_function = std::nullopt)
      : sigma_(sigma),
        lambda_(lambda),
        nugget_(nugget),
        drift_function_(drift_function.value_or(DriftFunction::kLinearDrift)) {
    switch (function) {
      case CovarianceFunction::kMatern_12:
        function_ = matern_covariance_12<T>;
        break;
      case CovarianceFunction::kMatern_32:
        function_ = matern_covariance_32<T>;
        break;
      case CovarianceFunction::kMatern_52:
        function_ = matern_covariance_52<T>;
        break;
      case CovarianceFunction::kCauchy:
        function_ = cauchy_covariance<T>;
        break;
      case CovarianceFunction::kSpherical:
        function_ = spherical_covariance<T>;
        break;
      case CovarianceFunction::kGaussian:
        function_ = gaussian_covariance<T>;
        break;
      case CovarianceFunction::kLinear:
        function_ = linear_covariance<T>;
        break;
      default:
        throw std::invalid_argument("Invalid covariance function");
    }
    if (sigma_ <= 0) {
      throw std::invalid_argument("sigma must be greater than 0");
    }
    if (lambda_ <= 0) {
      throw std::invalid_argument("lambda must be greater than 0");
    }
    if (nugget_ < 0) {
      throw std::invalid_argument("nugget must be >= 0");
    }
    method_ptr_ = drift_function.has_value() ? &Kriging::universal_kriging
                                             : &Kriging::simple_kriging;
  }

  /// @brief Estimate the value of a point.
  /// @param coordinates Coordinates of the points used to estimate the value.
  /// @param values Values of the points used to estimate the value.
  /// @param query Coordinates of the point to estimate.
  /// @return The estimated value of the point.
  auto operator()(const Eigen::Matrix<T, 3, -1>& coordinates,
                  const Eigen::Matrix<T, -1, 1>& values,
                  const Eigen::Vector3<T>& query) const -> T {
    return (this->*method_ptr_)(coordinates, values, query);
  }

 private:
  using MethodPtr = T (Kriging::*)(const Eigen::Matrix<T, 3, -1>&,
                                   const Eigen::Matrix<T, -1, 1>&,
                                   const Eigen::Vector3<T>&) const;

  const T sigma_;
  const T lambda_;
  const T nugget_;
  DriftFunction drift_function_;
  PtrCovarianceFunction function_;
  MethodPtr method_ptr_;

  /// @brief Get the drift terms for a given point.
  /// @param point The point to get the drift terms for.
  /// @param function The drift function to use.
  /// @return The drift terms for the given point.
  static auto get_drift_terms(const Eigen::Vector3<T>& point,
                              const DriftFunction& function) -> Vector<T> {
    switch (function) {
      case DriftFunction::kLinearDrift: {
        Vector<T> terms(4);
        terms << 1, point(0), point(1), point(2);
        return terms;
      }
      case DriftFunction::kQuadraticDrift: {
        Vector<T> terms(10);
        terms << 1, point(0), point(1), point(2), math::sqr(point(0)),
            math::sqr(point(1)), math::sqr(point(2)), point(0) * point(1),
            point(0) * point(2), point(1) * point(2);
        return terms;
      }
      default:
        throw std::invalid_argument("Invalid drift function");
    }
  }

  /// @brief Estimate the value of a point using simple kriging.
  /// @param coordinates Coordinates of the points used to estimate the value.
  /// @param values Values of the points used to estimate the value.
  /// @param query Coordinates of the point to estimate.
  /// @return The estimated value of the point.
  auto simple_kriging(const Eigen::Matrix<T, 3, -1>& coordinates,
                      const Eigen::Matrix<T, -1, 1>& values,
                      const Eigen::Vector3<T>& query) const -> T {
    auto k = coordinates.cols();
    Matrix<T> C(k, k);
    for (int i = 0; i < k; ++i) {
      for (int j = i; j < k; ++j) {
        C(i, j) =
            function_(coordinates.col(i), coordinates.col(j), sigma_, lambda_);
        if (i != j) {
          C(j, i) = C(i, j);
        }
      }
      C(i, i) += nugget_;  // add nugget on diagonal
    }
    Vector<T> c(k);
    for (int i = 0; i < k; i++) {
      c[i] = function_(query, coordinates.col(i), sigma_, lambda_);
    }
    Vector<T> w = C.ldlt().solve(c);
    return values.dot(w);
  }

  /// @brief Estimate the value of a point using universal kriging.
  /// @param coordinates Coordinates of the points used to estimate the value.
  /// @param values Values of the points used to estimate the value.
  /// @param query Coordinates of the point to estimate.
  /// @return The estimated value of the point.
  auto universal_kriging(const Eigen::Matrix<T, 3, -1>& coordinates,
                         const Eigen::Matrix<T, -1, 1>& values,
                         const Eigen::Vector3<T>& query) const -> T {
    auto k = coordinates.cols();
    auto f = get_drift_terms(query, drift_function_);
    auto p = f.size();

    Matrix<T> C(k, k);
    for (auto i = 0; i < k; ++i) {
      for (auto j = i; j < k; ++j) {
        C(i, j) =
            function_(coordinates.col(i), coordinates.col(j), sigma_, lambda_);
        if (i != j) {
          C(j, i) = C(i, j);
        }
      }
      C(i, i) += nugget_;
    }

    Matrix<T> F(k, p);
    for (auto i = 0; i < k; ++i) {
      F.row(i) = get_drift_terms(coordinates.col(i), drift_function_);
    }

    Matrix<T> A(k + p, k + p);
    A.topLeftCorner(k, k) = C;
    A.topRightCorner(k, p) = F;
    A.bottomLeftCorner(p, k) = F.transpose();
    A.bottomRightCorner(p, p).setZero();

    Vector<T> c(k);
    for (int i = 0; i < k; i++) {
      c[i] = function_(query, coordinates.col(i), sigma_, lambda_);
    }

    Vector<T> b(k + p);
    b.head(k) = c;
    b.tail(p) = f;

    Vector<T> x = A.colPivHouseholderQr().solve(b);
    return values.dot(x.head(k));
  }
};

}  // namespace pyinterp::detail::math
