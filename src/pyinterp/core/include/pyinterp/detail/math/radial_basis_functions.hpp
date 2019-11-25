// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Core>
#include <Eigen/LU>
#include <cmath>
#include <functional>

namespace pyinterp::detail::math {

/// Known radial functions.
enum RadialBasisFunction : uint8_t {
  Multiquadric,
  Inverse,
  Gaussian,
  Linear,
  Cubic,
  Quintic,
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
  using PtrRadialBasisFunction = Eigen::Matrix<T, -1, -1> (*)(
      const Eigen::Ref<const Eigen::Matrix<T, -1, -1>>& r, const double);

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
  RBF(const T& epsilon, const T& smooth, const RadialBasisFunction rbf)
      : epsilon_(epsilon), smooth_(smooth) {
    switch (rbf) {
      case RadialBasisFunction::Multiquadric:
        function_ = &RBF::multiquadric;
        break;
      case RadialBasisFunction::Inverse:
        function_ = &RBF::inverse_multiquadric;
        break;
      case RadialBasisFunction::Gaussian:
        function_ = &RBF::gaussian;
        break;
      case RadialBasisFunction::Linear:
        function_ = &RBF::linear;
        break;
      case RadialBasisFunction::Cubic:
        function_ = &RBF::cubic;
        break;
      case RadialBasisFunction::Quintic:
        function_ = &RBF::quintic;
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
  [[nodiscard]] auto interpolate(
      const Eigen::Ref<const Eigen::Matrix<T, -1, -1>>& xk,
      const Eigen::Ref<const Eigen::Matrix<T, -1, 1>>& yk,
      const Eigen::Ref<const Eigen::Matrix<T, -1, -1>>& xi) const
      -> Eigen::Matrix<T, -1, 1> {
    // Matrix of distances between the coordinates provided.
    auto r = RBF::distance_matrix(xk, xk);

    // Default epsilon to approximate average distance between nodes
    auto epsilon = std::isnan(epsilon_) ? r.mean() : epsilon_;

    // TODO(fbriol)
    auto A = function_(r, epsilon);

    // Apply smoothing factor if needed
    if (smooth_) {
      A -= Eigen::Matrix<T, -1, -1>::Identity(xk.cols(), xk.cols()) * smooth_;
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

  /// Returns the distance between two points in a euclidean space
  static auto euclidean_distance(
      const Eigen::Ref<const Eigen::Matrix<T, -1, 1>>& x,
      const Eigen::Ref<const Eigen::Matrix<T, -1, 1>>& y) -> T {
    return std::sqrt((x - y).array().pow(2).sum());
  }

  /// Multiquadric
  static auto multiquadric(const Eigen::Ref<const Eigen::Matrix<T, -1, -1>>& r,
                           const double epsilon) -> Eigen::Matrix<T, -1, -1> {
    return ((1.0 / epsilon * r).array().pow(2) + 1).sqrt();
  }

  /// Inverse multiquadric
  static auto inverse_multiquadric(
      const Eigen::Ref<const Eigen::Matrix<T, -1, -1>>& r, const double epsilon)
      -> Eigen::Matrix<T, -1, -1> {
    return 1.0 / multiquadric(r, epsilon).array();
  }

  /// Gauss
  static auto gaussian(const Eigen::Ref<const Eigen::Matrix<T, -1, -1>>& r,
                       const double epsilon) -> Eigen::Matrix<T, -1, -1> {
    return (-(1.0 / epsilon * r).array().pow(2)).exp();
  }

  /// Linear spline
  static auto linear(const Eigen::Ref<const Eigen::Matrix<T, -1, -1>>& r,
                     const double /*epsilon*/) -> Eigen::Matrix<T, -1, -1> {
    return r;
  }

  /// Cubic spline
  static auto cubic(const Eigen::Ref<const Eigen::Matrix<T, -1, -1>>& r,
                    const double /*epsilon*/) -> Eigen::Matrix<T, -1, -1> {
    return r.array().pow(3);
  }

  /// Quintic spline
  static auto quintic(const Eigen::Ref<const Eigen::Matrix<T, -1, -1>>& r,
                      const double /*epsilon*/) -> Eigen::Matrix<T, -1, -1> {
    return r.array().pow(5);
  }

  /// Thin plate spline
  static auto thin_plate(const Eigen::Ref<const Eigen::Matrix<T, -1, -1>>& r,
                         const double /*epsilon*/) -> Eigen::Matrix<T, -1, -1> {
    return (r.array() == 0).select(0, r.array().pow(2) * r.array().log());
  }

  /// Calculation of distances between the coordinates provided.
  static auto distance_matrix(
      const Eigen::Ref<const Eigen::Matrix<T, -1, -1>>& xk,
      const Eigen::Ref<const Eigen::Matrix<T, -1, -1>>& xi)
      -> Eigen::Matrix<T, -1, -1> {
    assert(xk.rows() == xi.rows());

    auto result = Eigen::Matrix<T, -1, -1>(xi.cols(), xk.cols());

    for (Eigen::Index i0 = 0; i0 < xi.cols(); ++i0) {
      for (Eigen::Index i1 = 0; i1 < xk.cols(); ++i1) {
        result(i0, i1) = euclidean_distance(xi.col(i0), xk.col(i1));
      }
    }
    return result;
  }

  /// Resolution of the linear system
  static auto solve_linear_system(const Eigen::Matrix<T, -1, -1>& A,
                                  const Eigen::Matrix<T, -1, 1>& di)
      -> Eigen::Matrix<T, -1, 1> {
    Eigen::FullPivLU<Eigen::Matrix<T, -1, -1>> lu(A);
    return lu.solve(di);
  }
};

}  // namespace pyinterp::detail::math
