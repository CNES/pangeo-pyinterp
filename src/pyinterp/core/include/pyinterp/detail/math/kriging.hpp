#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "pyinterp/detail/math.hpp"
#include "pyinterp/eigen.hpp"

namespace pyinterp::detail::math {

/// Matern covariance function
// template <typename T>
// inline auto matern_covariance(const Eigen::Ref<const Eigen::Vector3<T>>& p1,
//                               const Eigen::Ref<const Eigen::Vector3<T>>& p2,
//                               const T& sigma, const T& lambda, const T& nu)
//                               -> T {
//   auto r = (p1 - p2).norm();
//   auto result = sigma * sigma;
//   auto d = r / lambda;
//   auto bessel = boost::math::cyl_bessel_k(nu, d);
//   result *= std::pow(2, 1 - nu) / boost::math::tgamma(nu);
//   result *= std::pow(d, nu) * bessel;
//   return result;
// }

/// Matern covariance function for nu = 0.5
template <typename T>
inline auto matern_covariance_12(const Eigen::Ref<const Eigen::Vector3<T>>& p1,
                                 const Eigen::Ref<const Eigen::Vector3<T>>& p2,
                                 const T& sigma, const T& lambda) -> T {
  auto r = (p1 - p2).norm();
  return math::sqr(sigma) * std::exp(-r / lambda);
}

/// Matern covariance function for nu = 1.5
template <typename T>
inline auto mattern_covariance_32(const Eigen::Ref<const Eigen::Vector3<T>>& p1,
                                  const Eigen::Ref<const Eigen::Vector3<T>>& p2,
                                  const T& sigma, const T& lambda) -> T {
  auto r = (p1 - p2).norm();
  auto d = r / lambda;
  auto result = math::sqr(sigma);
  result *= (1 + std::sqrt(3.0) * d) * std::exp(-std::sqrt(3.0) * d);
  return result;
}

/// Matern covariance function for nu = 2.5
template <typename T>
inline auto mattern_covariance_52(const Eigen::Ref<const Eigen::Vector3<T>>& p1,
                                  const Eigen::Ref<const Eigen::Vector3<T>>& p2,
                                  const T& sigma, const T& lambda) -> T {
  auto r = (p1 - p2).norm();
  auto d = r / lambda;
  auto result = math::sqr(sigma);
  result *= (1.0 + std::sqrt(5.0) * d + 5.0 / 3.0 * math::sqr(d)) *
            std::exp(-std::sqrt(5.0) * d);
  return result;
}

/// Whittle-Matern covariance function
template <typename T>
inline auto whittle_matern_covariance(
    const Eigen::Ref<const Eigen::Vector3<T>>& p1,
    const Eigen::Ref<const Eigen::Vector3<T>>& p2, const T& sigma,
    const T& lambda) -> T {
  auto r = (p1 - p2).norm();
  return sigma * std::pow(1 + std::sqrt(3) * r / lambda, -1.5) *
         std::exp(-std::sqrt(3) * r / lambda);
}

/// Cauchy covariance function
template <typename T>
auto cauchy_covariance(const Eigen::Ref<const Eigen::Vector3<T>>& p1,
                       const Eigen::Ref<const Eigen::Vector3<T>>& p2,
                       const T& sigma, const T& lambda) -> T {
  auto r = (p1 - p2).norm();
  return sigma / (1 + r * r / (lambda * lambda));
}

/// Known Covariance functions.
enum CovarianceFunction : uint8_t {
  kMatern_12 = 0,
  kMatern_32 = 1,
  kMatern_52 = 2,
  kWhittleMatern = 3,
  kCauchy = 4,
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
  /// @param function The covariance function used to estimate the value of a
  /// point.
  Kriging(const T& sigma, const T& lambda, const CovarianceFunction& function)
      : sigma_(sigma), lambda_(lambda), function_(nullptr) {
    switch (function) {
      case CovarianceFunction::kMatern_12:
        function_ = matern_covariance_12<T>;
        break;
      case CovarianceFunction::kMatern_32:
        function_ = mattern_covariance_32<T>;
        break;
      case CovarianceFunction::kWhittleMatern:
        function_ = whittle_matern_covariance<T>;
        break;
      case CovarianceFunction::kCauchy:
        function_ = cauchy_covariance<T>;
        break;
      case CovarianceFunction::kMatern_52:
      default:
        function_ = mattern_covariance_52<T>;
        break;
    }
    if (sigma_ <= 0) {
      throw std::invalid_argument("sigma must be greater than 0");
    }
    if (lambda_ <= 0) {
      throw std::invalid_argument("alpha must be greater than 0");
    }
  }

  /// @brief Estimate the value of a point.
  /// @param coordinates Coordinates of the points used to estimate the value.
  /// @param values Values of the points used to estimate the value.
  /// @param query Coordinates of the point to estimate.
  /// @return The estimated value of the point.
  auto universal_kriging(const Eigen::Matrix<T, 3, -1>& coordinates,
                         const Eigen::Matrix<T, -1, 1>& values,
                         const Eigen::Vector3<T>& query) const -> T {
    auto k = coordinates.cols();
    Matrix<T> C(k, k);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < k; j++) {
        C(i, j) =
            function_(coordinates.col(i), coordinates.col(j), sigma_, lambda_);
      }
    }
    Vector<T> c(k);
    for (int i = 0; i < k; i++) {
      c[i] = function_(query, coordinates.col(i), sigma_, lambda_);
    }
    Vector<T> w = C.ldlt().solve(c);
    auto result = T(0);
    for (int i = 0; i < k; i++) {
      result += w[i] * values[i];
    }
    return result;
  }

 private:
  const T sigma_;
  const T lambda_;
  PtrCovarianceFunction function_;
};

}  // namespace pyinterp::detail::math
