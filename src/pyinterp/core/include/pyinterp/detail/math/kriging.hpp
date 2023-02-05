#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "pyinterp/eigen.hpp"

namespace pyinterp::detail::math {

template <typename T>
inline auto covariance(const Eigen::Ref<const Eigen::Vector3<T>>& p1,
                       const Eigen::Ref<const Eigen::Vector3<T>>& p2,
                       const T& sigma, const T& alpha) -> T {
  auto r = (p1 - p2).norm();
  return sigma * std::pow(1 - alpha * r, 2) * std::exp(-alpha * r);
}

template <typename T>
auto universal_kriging(const Eigen::Matrix<T, 3, -1>& coordinates,
                       const Eigen::Matrix<T, -1, 1>& values,
                       const Eigen::Vector3<T>& query, const T& sigma,
                       const T& alpha) -> T {
  auto k = coordinates.cols();
  Matrix<T> C(k, k);
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < k; j++) {
      C(i, j) =
          covariance<T>(coordinates.col(i), coordinates.col(j), sigma, alpha);
    }
  }
  Vector<T> c(k);
  for (int i = 0; i < k; i++) {
    c[i] = covariance<T>(query, coordinates.col(i), sigma, alpha);
  }
  Vector<T> w = C.ldlt().solve(c);
  auto result = T(0);
  for (int i = 0; i < k; i++) {
    result += w[i] * values[i];
  }
  return result;
}

}  // namespace pyinterp::detail::math
