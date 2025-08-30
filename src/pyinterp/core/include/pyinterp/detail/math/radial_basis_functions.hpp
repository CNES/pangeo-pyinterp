// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <stdexcept>

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
  /// @param epsilon Characteristic length scale parameter provided by user.
  /// Internally the reciprocal (1/epsilon) is stored because the implemented
  /// formulas use (r / epsilon). For Gaussian & (Inverse)Multiquadric kernels
  /// we compute exp(-(r/epsilon)^2) and sqrt(1 + (r/epsilon)^2) etc. Thus the
  /// internal variable passed to the kernel is 1/epsilon.
  /// If the provided epsilon is NaN, an automatic value is chosen equal to the
  /// average distance between distinct nodes (NOT its reciprocal). Because we
  /// store 1/epsilon, when epsilon is auto-computed we actually store its
  /// reciprocal directly during interpolation.
  /// @param smooth Values greater than zero increase the smoothness of the
  /// approximation. 0 is for interpolation (default), the function will always
  /// go through the nodal points in this case.
  /// @param rbf The radial basis function, based on the radius, r, given by the
  /// norm (Euclidean distance)
  RBF(const T &epsilon, const T &smooth, const RadialBasisFunction rbf)
      : epsilon_reciprocal_(T(1) / epsilon), smooth_(smooth), rbf_(rbf) {
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

  /// Default destructor
  virtual ~RBF() = default;

  /// Calculates the interpolated values
  ///
  /// @param xk Coordinates of the nodes (shape: dim x N)
  /// @param yk Values of the nodes (size: N)
  /// @param xi Coordinates to evaluate the interpolant at (shape: dim x M)
  /// @return interpolated values for each coordinates provided.
  [[nodiscard]] auto interpolate(const Eigen::Ref<const Matrix<T>> &xk,
                                 const Eigen::Ref<const Vector<T>> &yk,
                                 const Eigen::Ref<const Matrix<T>> &xi) const
      -> Vector<T> {
    // Matrix of distances between the coordinates provided.
    const auto r = distance_matrix(xk, xk);

    // Default epsilon to approximate average distance between nodes
    const auto epsilon = std::isnan(epsilon_reciprocal_)
                             ? 1 / RBF<T>::average(r)
                             : epsilon_reciprocal_;

    // Build the interpolation matrix Φ
    auto A = function_(r, epsilon);

    // Apply smoothing factor if needed (Tikhonov diagonal term)
    if (smooth_) {
      A += Matrix<T>::Identity(xk.cols(), xk.cols()) * smooth_;
    }

    // Determine if we must augment with polynomial terms to satisfy the
    // conditional positive definiteness constraints. For Linear, Cubic and
    // ThinPlate kernels, we include a polynomial basis of degree 1: {1, x_i}
    // for each spatial dimension i.
    const bool augment = (rbf_ == RadialBasisFunction::Linear ||
                          rbf_ == RadialBasisFunction::Cubic ||
                          rbf_ == RadialBasisFunction::ThinPlate);

    Vector<T> weights;      // w
    Vector<T> poly_coeffs;  // λ (if augment)

    if (augment) {
      const Eigen::Index dim = xk.rows();
      const Eigen::Index N = xk.cols();
      const Eigen::Index q = dim + 1;  // constant + linear terms
      if (N < q) {
        // Not enough points to build augmented system; inform caller to
        // request more neighbors (k) instead of silently falling back.
        throw std::invalid_argument(
            "Not enough points to perform polynomial augmentation for the "
            "selected RBF. Need at least " +
            std::to_string(q) + " points, got " + std::to_string(N) +
            ". Increase the number of neighbors (k).");
      }
      // Build polynomial matrix P (N x q)
      Matrix<T> P(N, q);
      P.col(0).setOnes();
      for (Eigen::Index d = 0; d < dim; ++d) {
        P.col(d + 1) = xk.row(d).transpose();
      }

      // Assemble block matrix K = [ Φ  P ]
      //                            [ Pᵀ 0 ]
      Matrix<T> K(N + q, N + q);
      K.setZero();
      K.topLeftCorner(N, N) = A;
      K.topRightCorner(N, q) = P;
      K.bottomLeftCorner(q, N) = P.transpose();
      // bottom-right already zero

      Vector<T> rhs(N + q);
      rhs.head(N) = yk;
      rhs.tail(q).setZero();

      Eigen::LDLT<Matrix<T>> ldlt(K);
      auto sol = ldlt.solve(rhs);
      weights = sol.head(N);
      poly_coeffs = sol.tail(q);

      // Interpolate: f(x) = Σ w_i φ(||x - x_i||) + p(x)
      const auto r_xi = distance_matrix(xi, xk);
      const auto B = function_(r_xi, epsilon);  // M x N

      // Polynomial part P_x (M x q)
      Matrix<T> P_x(xi.cols(), q);
      P_x.col(0).setOnes();
      for (Eigen::Index d = 0; d < dim; ++d) {
        P_x.col(d + 1) = xi.row(d).transpose();
      }
      return B * weights + P_x * poly_coeffs;
    }

    // Fallback: classic RBF without polynomial augmentation
    const auto weights_fallback = RBF<T>::solve_linear_system(A, yk);
    const auto r_xi = distance_matrix(xi, xk);
    const auto B = function_(r_xi, epsilon);
    return B * weights_fallback;
  }

 private:
  /// Adjustable constant for gaussian or multiquadric family. We store the
  /// reciprocal value as it is directly used in the implemented formulas.
  T epsilon_reciprocal_;

  /// Smooth factor (>=0 expected, validated at higher layer)
  T smooth_;

  /// Radial basis function enum used (needed to decide augmentation)
  RadialBasisFunction rbf_;

  /// Radial basis function, based on the radius
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
    return (x - y).norm();
  }

  /// Multiquadric ( ( (r/ε)^2 + 1 )^{1/2} ) where internal epsilon is 1/ε
  static auto multiquadric(const Eigen::Ref<const Matrix<T>> &r,
                           const T epsilon) -> Matrix<T> {
    return ((epsilon * r).array().pow(2) + 1).sqrt();
  }

  /// Inverse multiquadric ( 1 / multiquadric )
  static auto inverse_multiquadric(const Eigen::Ref<const Matrix<T>> &r,
                                   const T epsilon) -> Matrix<T> {
    return 1.0 / multiquadric(r, epsilon).array();
  }

  /// Gaussian: exp(-(r/ε)^2) where internal epsilon is 1/ε
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

  /// Thin plate spline φ(r)=r^2 log(r)
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
    Eigen::LDLT<Matrix<T>> ldlt(A);
    return ldlt.solve(di);
  }
};

}  // namespace pyinterp::detail::math
