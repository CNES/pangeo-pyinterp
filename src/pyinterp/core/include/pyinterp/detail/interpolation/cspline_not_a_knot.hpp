// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <limits>
#include <vector>

#include "pyinterp/detail/interpolation/cspline_base.hpp"

namespace pyinterp::detail::interpolation {

/// Cubic spline interpolation with not‑a‑knot end conditions.
template <typename T>
class CSplineNotAKnot : public CSplineBase<T> {
 public:
  using CSplineBase<T>::operator();

  /// @brief Triplet type for sparse matrix construction
  using Triplet = Eigen::Triplet<T>;

  /// @brief Default constructor.
  CSplineNotAKnot() : CSplineBase<T>(), triplets_{} {}

  /// Minimum number of data points required is 4.
  auto min_size() const -> Eigen::Index override { return 4; }

 protected:
  /// Compute the spline coefficients (i.e. the first derivatives at the data
  /// points) by solving an \f$n\times n\f$ system with not‑a‑knot boundary
  /// conditions.
  auto compute_coefficients(const Vector<T>& xa, const Vector<T>& ya)
      -> bool override;

 private:
  /// @brief Member variable to avoid reallocations in repeated calls to
  /// compute_coefficients
  std::vector<Triplet> triplets_;
};

template <typename T>
auto CSplineNotAKnot<T>::compute_coefficients(const Vector<T>& xa,
                                              const Vector<T>& ya) -> bool {
  if (!Interpolator1D<T>::compute_coefficients(xa, ya)) {
    return false;
  }
  const auto size = xa.size();
  const auto size_m1 = size - 1;
  const auto size_m2 = size - 2;

  // Use tolerance for numerical stability
  constexpr T epsilon = std::numeric_limits<T>::epsilon() * T(10);

  // Compute step sizes: h_i = x[i+1] - x[i] for i = 0 ... n-2.
  // Vectorized computation for better performance
  Eigen::Matrix<T, Eigen::Dynamic, 1> h = xa.tail(size_m1) - xa.head(size_m1);

  // Check for duplicate x values
  if ((h.array().abs() < epsilon).any()) {
    return false;  // Two consecutive x values are too close.
  }

  // The not-a-knot boundary conditions create a banded matrix with bandwidth 3
  this->b_.resize(size);
  this->x_.resize(size);

  // Not-a-knot boundary conditions result in b[0] = b[n-1] = 0
  this->b_(0) = T(0);
  this->b_(size_m1) = T(0);

  // Compute divided differences: delta_i = (y[i+1] - y[i]) / h[i]
  auto delta = (ya.tail(size_m1) - ya.head(size_m1)).array() / h.array();

  // Compute interior values: b_i = 3 * (delta_i - delta_{i-1})
  this->b_.segment(1, size_m2) =
      T(3) * (delta.tail(size_m2) - delta.head(size_m2));

  // Build the sparse system matrix A
  triplets_.clear();
  triplets_.reserve(3 * size);

  // Row 0: Left not-a-knot condition
  triplets_.emplace_back(0, 0, -h(1));
  triplets_.emplace_back(0, 1, h(0) + h(1));
  triplets_.emplace_back(0, 2, -h(0));

  // Rows 1 to n-2: Standard cubic spline tridiagonal equations
  for (int64_t i = 1; i <= size_m2; ++i) {
    triplets_.emplace_back(i, i - 1, h(i - 1));
    triplets_.emplace_back(i, i, T(2) * (h(i - 1) + h(i)));
    triplets_.emplace_back(i, i + 1, h(i));
  }

  // Row n-1: Right not-a-knot condition
  const auto h_nm2 = h(size_m2);
  const auto h_nm3 = h(size - 3);
  triplets_.emplace_back(size_m1, size - 3, h_nm2);
  triplets_.emplace_back(size_m1, size_m2, -(h_nm2 + h_nm3));
  triplets_.emplace_back(size_m1, size_m1, h_nm3);

  // Solve the sparse linear system Ax = b
  Eigen::SparseMatrix<T> A_sparse(size, size);
  A_sparse.setFromTriplets(triplets_.begin(), triplets_.end());

  Eigen::SparseLU<Eigen::SparseMatrix<T>> solver;
  solver.compute(A_sparse);

  if (solver.info() != Eigen::Success) {
    return false;
  }

  this->x_ = solver.solve(this->b_);

  return solver.info() == Eigen::Success;
}

}  // namespace pyinterp::detail::interpolation
