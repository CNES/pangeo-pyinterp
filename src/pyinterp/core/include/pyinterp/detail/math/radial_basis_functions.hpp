// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Core>
#include <Eigen/LU>
#include <cmath>
#include <functional>

#include "pyinterp/eigen.hpp"

namespace pyinterp::detail::math {

/// Known radial functions.
enum RadialBasisFunction : uint8_t {
  Cubic,
  Gaussian,
  InverseMultiquadric,
  Linear,
  Multiquadric,
  ThinPlate
};

//// A radial basis function (RBF) is a real-valued function φ whose value
/// depends only on the distance between the input and some fixed point,
/// either the origin, so that φ(x) = φ(║x║), or some other fixed point
/// c, called a center, so that φ(x) = φ(║x - c║). Any function φ that
/// satisfies the property φ(x) = φ(║x║) is a radial function.
template <typename T>
class RBF {
 public:
  /// Pointer to the Radial function used
  using PtrRadialBasisFunction =
      Matrix<T> (*)(const Eigen::Ref<const Matrix<T>> &r, const T);

  /// Default constructor
  ///
  /// @param epsilon Adjustable constant for gaussian or multiquadrics
  /// functions - defaults to approximate average distance between nodes
  /// (which is a good start).
  /// @param smooth Values greater than zero increase the smoothness of the
  /// approximation. 0 is for interpolation (default), the function will always
  /// go through the nodal points in this case.
  /// @param rbf The radial basis function, based on the radius, r, given by the
  /// norm (Euclidean distance)
  RBF(const T &epsilon, const T &smooth, const RadialBasisFunction rbf)
      : epsilon_(T(1) / epsilon), smooth_(smooth) {
    switch (rbf) {
      case RadialBasisFunction::Cubic:
        function_ = &RBF::cubic;
        break;
      case RadialBasisFunction::Gaussian:
        function_ = &RBF::gaussian;
        break;
      case RadialBasisFunction::InverseMultiquadric:
        function_ = &RBF::inverse_multiquadric;
        break;
      case RadialBasisFunction::Linear:
        function_ = &RBF::linear;
        break;
      case RadialBasisFunction::Multiquadric:
        function_ = &RBF::multiquadric;
        break;
      case RadialBasisFunction::ThinPlate:
        function_ = &RBF::thin_plate;
        break;
      default:
        throw std::invalid_argument("Radial function unknown: " +
                                    std::to_string(static_cast<int>(rbf)));
    }
  }

  /// Calculates the interpolated values
  ///
  /// @param xk Coordinates of the nodes
  /// @param yk Values of the nodes
  /// @param xi Coordinates to evaluate the interpolant at.
  /// @return interpolated values for each coordinates provided.
  [[nodiscard]] auto interpolate(const Eigen::Ref<const Matrix<T>> &xk,
                                 const Eigen::Ref<const Vector<T>> &yk,
                                 const Eigen::Ref<const Matrix<T>> &xi) const
      -> Vector<T> {
    // Matrix of distances between the coordinates provided.
    const auto r = distance_matrix(xk, xk);

    // Default epsilon to approximate average distance between nodes
    const auto epsilon =
        std::isnan(epsilon_) ? 1 / RBF<T>::average(r) : epsilon_;

    // TODO(fbriol)
    auto A = function_(r, epsilon);

    // Apply smoothing factor if needed
    if (smooth_) {
      A -= Matrix<T>::Identity(xk.cols(), xk.cols()) * smooth_;
    }

    return function_(distance_matrix(xk, xi), epsilon) *
           RBF<T>::solve_linear_system(A, yk);
  }

 private:
  /// Adjustable constant for gaussian or multiquadrics functions
  T epsilon_;

  /// Smooth factor
  T smooth_;

  /// Radial bassis function, based on the radius
  PtrRadialBasisFunction function_;

  // Calculates the distance average excluding the diagonal
  static auto average(const Eigen::Ref<const Matrix<T>> &distance) -> T {
    assert(distance.cols() == distance.rows());
    auto sum = T(0);
    auto n = distance.cols();
    for (Eigen::Index ix = 0; ix < distance.rows() - 1; ++ix) {
      for (Eigen::Index jx = ix + 1; jx < distance.cols(); ++jx) {
        sum += distance(ix, jx);
      }
    }
    return static_cast<T>(sum / (n * (n - 1) * 0.5));
  };

  /// Returns the distance between two points in a euclidean space
  virtual auto calculate_distance(const Eigen::Ref<const Vector<T>> &x,
                                  const Eigen::Ref<const Vector<T>> &y) const
      -> T {
    return std::sqrt((x - y).array().pow(2).sum());
  }

  /// Multiquadric
  static auto multiquadric(const Eigen::Ref<const Matrix<T>> &r,
                           const T epsilon) -> Matrix<T> {
    return ((epsilon * r).array().pow(2) + 1).sqrt();
  }

  /// Inverse multiquadric
  static auto inverse_multiquadric(const Eigen::Ref<const Matrix<T>> &r,
                                   const T epsilon) -> Matrix<T> {
    return 1.0 / multiquadric(r, epsilon).array();
  }

  /// Gauss
  static auto gaussian(const Eigen::Ref<const Matrix<T>> &r, const T epsilon)
      -> Matrix<T> {
    return (-(epsilon * r).array().pow(2)).exp();
  }

  /// Linear spline
  static auto linear(const Eigen::Ref<const Matrix<T>> &r, const T /*epsilon*/)
      -> Matrix<T> {
    return r;
  }

  /// Cubic spline
  static auto cubic(const Eigen::Ref<const Matrix<T>> &r, const T /*epsilon*/)
      -> Matrix<T> {
    return r.array().pow(3);
  }

  /// Thin plate spline
  static auto thin_plate(const Eigen::Ref<const Matrix<T>> &r,
                         const T /*epsilon*/) -> Matrix<T> {
    return (r.array() == 0).select(0, r.array().pow(2) * r.array().log());
  }

  /// Calculation of distances between the coordinates provided.
  auto distance_matrix(const Eigen::Ref<const Matrix<T>> &xk,
                       const Eigen::Ref<const Matrix<T>> &xi) const
      -> Matrix<T> {
    assert(xk.rows() == xi.rows());

    auto result = Matrix<T>(xi.cols(), xk.cols());

    for (Eigen::Index i0 = 0; i0 < xi.cols(); ++i0) {
      for (Eigen::Index i1 = 0; i1 < xk.cols(); ++i1) {
        result(i0, i1) = calculate_distance(xi.col(i0), xk.col(i1));
      }
    }
    return result;
  }

  /// Resolution of the linear system
  static auto solve_linear_system(const Matrix<T> &A, const Vector<T> &di)
      -> Vector<T> {
    Eigen::FullPivLU<Matrix<T>> lu(A);
    return lu.solve(di);
  }
};

}  // namespace pyinterp::detail::math
