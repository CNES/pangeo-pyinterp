// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <string>

#include "pyinterp/eigen.hpp"

namespace pyinterp::math::interpolate {

/// Known radial basis functions.
enum class RBFKernel : uint8_t {
  kCubic,
  kGaussian,
  kInverseMultiquadric,
  kLinear,
  kMultiquadric,
  kThinPlate
};

/// @brief Radial basis function (RBF) interpolator
///
/// A radial basis function (RBF) is a real-valued function φ whose value
/// depends only on the distance between the input and some fixed point,
/// either the origin, so that φ(x) = φ(‖x‖), or some other fixed point
/// c, called a center, so that φ(x) = φ(‖x - c‖). Any function φ that
/// satisfies the property φ(x) = φ(‖x‖) is a radial function.
/// @tparam T Floating point type
template <std::floating_point T>
class RBF {
 public:
  /// Pointer to the Radial basis function
  using PtrRadialBasisFunction =
      Matrix<T> (*)(const Eigen::Ref<const Matrix<T>>&, T);

  /// @brief Constructor
  /// @param[in] epsilon Adjustable constant for gaussian or multiquadrics
  /// functions - defaults to the reciprocal of the approximate average
  /// distance between nodes (which is a good start). Use NaN for automatic
  /// computation.
  /// @param[in] smooth Values greater than zero increase the smoothness of the
  /// approximation. 0 is for interpolation (default), the function will always
  /// go through the nodal points in this case.
  /// @param[in] rbf The radial basis function, based on the radius, r, given by
  /// the norm (Euclidean distance)
  constexpr RBF(const T epsilon, const T smooth, const RBFKernel rbf)
      : epsilon_reciprocal_{T{1} / epsilon}, smooth_{smooth} {
    switch (rbf) {
      case RBFKernel::kCubic:
        function_ = &RBF::cubic;
        break;
      case RBFKernel::kGaussian:
        function_ = &RBF::gaussian;
        break;
      case RBFKernel::kInverseMultiquadric:
        function_ = &RBF::inverse_multiquadric;
        break;
      case RBFKernel::kLinear:
        function_ = &RBF::linear;
        break;
      case RBFKernel::kMultiquadric:
        function_ = &RBF::multiquadric;
        break;
      case RBFKernel::kThinPlate:
        function_ = &RBF::thin_plate;
        break;
      [[unlikely]] default:
        throw std::invalid_argument("Radial function unknown: " +
                                    std::to_string(static_cast<int>(rbf)));
    }
  }

  /// Default destructor
  virtual ~RBF() = default;

  /// @brief Calculates the interpolated values
  /// @param[in] xk Coordinates of the nodes (dimension × n_nodes)
  /// @param[in] yk Values of the nodes (n_nodes)
  /// @param[in] xi Coordinates to evaluate the interpolant at (dimension ×
  /// n_points)
  /// @return Interpolated values for each coordinate provided (n_points)
  [[nodiscard]] auto interpolate(const Eigen::Ref<const Matrix<T>>& xk,
                                 const Eigen::Ref<const Vector<T>>& yk,
                                 const Eigen::Ref<const Matrix<T>>& xi) const
      -> Vector<T> {
    // Matrix of distances between node coordinates
    const auto r = distance_matrix(xk, xk);

    // Default epsilon to approximate average distance between nodes
    const T epsilon = std::isnan(epsilon_reciprocal_) ? T{1} / average(r)
                                                      : epsilon_reciprocal_;

    // Build the interpolation matrix
    auto A = function_(r, epsilon);

    // Apply smoothing factor if needed (Tikhonov regularization)
    if (smooth_ != T{0}) {
      A.diagonal().array() += smooth_;
    }

    // Solve the linear system and evaluate at query points
    const auto weights = solve_linear_system(A, yk);
    return function_(distance_matrix(xk, xi), epsilon) * weights;
  }

 protected:
  /// @brief Return the distance between two points in a Euclidean space
  /// @param[in] x First point (dimension vector)
  /// @param[in] y Second point (dimension vector)
  /// @return Distance between the two points
  [[nodiscard]] virtual auto calculate_distance(
      const Eigen::Ref<const Vector<T>>& x,
      const Eigen::Ref<const Vector<T>>& y) const -> T {
    return (x - y).norm();
  }

 private:
  /// Adjustable constant for gaussian or multiquadrics functions. We store
  /// here the reciprocal value as it is often used in calculations.
  T epsilon_reciprocal_;

  /// Smoothing factor (Tikhonov regularization parameter)
  T smooth_;

  /// Radial basis function pointer
  PtrRadialBasisFunction function_;

  /// @brief Calculate the distance average excluding the diagonal
  /// @param[in] distance Symmetric distance matrix (n × n)
  /// @return Average of off-diagonal elements
  [[nodiscard]] static auto average(const Eigen::Ref<const Matrix<T>>& distance)
      -> T {
    assert(distance.cols() == distance.rows());

    const auto n = distance.cols();
    if (n < 2) {
      return T{1};  // Avoid division by zero; return sensible default
    }

    T sum{0};

    // Sum upper triangle (excluding diagonal)
    for (const auto ix : std::views::iota(int64_t{0}, n - 1)) {
      for (const auto jx : std::views::iota(ix + 1, n)) {
        sum += distance(ix, jx);
      }
    }

    // Number of upper triangle elements: n*(n-1)/2
    // Multiply by 2 instead of dividing by 0.5 for better precision
    return (T{2} * sum) / static_cast<T>(n * (n - 1));
  }

  /// Multiquadric: √(1 + (εr)²)
  /// @param[in] r Distance matrix
  /// @param[in] epsilon Adjustable constant
  /// @return Matrix after applying the multiquadric function
  [[nodiscard]] static auto multiquadric(const Eigen::Ref<const Matrix<T>>& r,
                                         const T epsilon) -> Matrix<T> {
    return ((epsilon * r).array().square() + T{1}).sqrt();
  }

  /// Inverse multiquadric: 1/√(1 + (εr)²)
  /// @param[in] r Distance matrix
  /// @param[in] epsilon Adjustable constant
  /// @return Matrix after applying the inverse multiquadric function
  [[nodiscard]] static auto inverse_multiquadric(
      const Eigen::Ref<const Matrix<T>>& r, const T epsilon) -> Matrix<T> {
    return T{1} / multiquadric(r, epsilon).array();
  }

  /// Gaussian: exp(-(εr)²)
  /// @param[in] r Distance matrix
  /// @param[in] epsilon Adjustable constant
  /// @return Matrix after applying the gaussian function
  [[nodiscard]] static auto gaussian(const Eigen::Ref<const Matrix<T>>& r,
                                     const T epsilon) -> Matrix<T> {
    return (-(epsilon * r).array().square()).exp();
  }

  /// Linear: r
  /// @param[in] r Distance matrix
  /// @param[in] epsilon Adjustable constant (unused)
  /// @return Matrix after applying the linear function
  [[nodiscard]] static auto linear(const Eigen::Ref<const Matrix<T>>& r,
                                   [[maybe_unused]] const T epsilon)
      -> Matrix<T> {
    return r;
  }

  /// Cubic: r³
  /// @param[in] r Distance matrix
  /// @param[in] epsilon Adjustable constant (unused)
  /// @return Matrix after applying the cubic function
  [[nodiscard]] static auto cubic(const Eigen::Ref<const Matrix<T>>& r,
                                  [[maybe_unused]] const T epsilon)
      -> Matrix<T> {
    return r.array().cube();
  }

  /// Thin plate spline: r² log(r)
  /// @note lim_{r→0} r² log(r) = 0
  /// @param[in] r Distance matrix
  /// @param[in] epsilon Adjustable constant (unused)
  /// @return Matrix after applying the thin plate spline function
  [[nodiscard]] static auto thin_plate(const Eigen::Ref<const Matrix<T>>& r,
                                       [[maybe_unused]] const T epsilon)
      -> Matrix<T> {
    // Use a small threshold instead of exact zero comparison for robustness
    constexpr T threshold = std::numeric_limits<T>::epsilon() * T{100};
    return (r.array().abs() < threshold)
        .select(T{0}, r.array().square() * r.array().log());
  }

  /// @brief Calculation of distances between the coordinates provided.
  /// @param[in] xk Reference coordinates (dimension × n_ref)
  /// @param[in] xi Query coordinates (dimension × n_query)
  /// @return Distance matrix (n_query × n_ref)
  /// @note When xk and xi point to the same data, the result is symmetric
  ///       and this method exploits that symmetry for efficiency.
  [[nodiscard]] auto distance_matrix(
      const Eigen::Ref<const Matrix<T>>& xk,
      const Eigen::Ref<const Matrix<T>>& xi) const -> Matrix<T> {
    assert(xk.rows() == xi.rows());

    const auto n_xi = xi.cols();
    const auto n_xk = xk.cols();
    Matrix<T> result(n_xi, n_xk);

    // Check if computing self-distance matrix (symmetric case)
    const bool is_symmetric = (n_xi == n_xk) && (xi.data() == xk.data());

    if (is_symmetric) {
      // Exploit symmetry: only compute upper triangle
      for (auto i0 = int64_t{0}; i0 < n_xi; ++i0) {
        result(i0, i0) = T{0};  // Diagonal is always zero
        for (auto i1 = i0 + 1; i1 < n_xk; ++i1) {
          const T dist = calculate_distance(xi.col(i0), xk.col(i1));
          result(i0, i1) = dist;
          result(i1, i0) = dist;
        }
      }
    } else {
      // General case: compute all elements
      for (auto i0 = int64_t{0}; i0 < n_xi; ++i0) {
        for (auto i1 = int64_t{0}; i1 < n_xk; ++i1) {
          result(i0, i1) = calculate_distance(xi.col(i0), xk.col(i1));
        }
      }
    }

    return result;
  }

  /// @brief Resolution of the linear system using LU decomposition with partial
  /// pivoting
  /// @param[in] A Coefficient matrix
  /// @param[in] b Right-hand side vector
  /// @return Solution vector
  [[nodiscard]] static auto solve_linear_system(const Matrix<T>& A,
                                                const Vector<T>& b)
      -> Vector<T> {
    Eigen::PartialPivLU<Matrix<T>> lu(A);
    return lu.solve(b);
  }
};

}  // namespace pyinterp::math::interpolate
