// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <Eigen/Core>

namespace pyinterp {

/// @brief Dynamic vector of type T
/// @tparam T The data type of the vector elements.
template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

/// @brief Dynamic matrix of type T
/// @tparam T The data type of the matrix elements.
template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

/// @brief Row major dynamic matrix of type T
/// @tparam T The data type of the matrix elements.
template <typename T>
using RowMajorMatrix =
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// @brief Alias for a RowMajor Eigen Complex Matrix.
/// @tparam T The data type of the matrix elements.
template <typename T>
using RowMajorComplexMatrix = Eigen::Matrix<std::complex<T>, Eigen::Dynamic,
                                            Eigen::Dynamic, Eigen::RowMajor>;

/// @brief Eigen reference block with dynamic inner stride
/// @tparam T The data type of the vector elements.
template <typename T>
using EigenRefBlock = Eigen::Ref<Vector<T>, 0, Eigen::InnerStride<>>;

/// @brief Read-only reference to a contiguous vector.
///
/// Binds to vectors, maps and contiguous blocks without copying them, unlike
/// `const Vector<T>&`, which materializes a temporary for anything but a
/// Vector. A strided argument is copied. The contiguous layout is kept on
/// purpose: with a run-time stride, the loops of the spline solvers are no
/// longer vectorized.
/// @tparam T The data type of the vector elements.
template <typename T>
using ConstVectorRef = Eigen::Ref<const Vector<T>>;

/// @brief Dynamic stride type
using EigenDStride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

/// @brief Eigen reference with dynamic stride
/// @tparam MatrixType The type of the matrix.
template <typename T>
using EigenDRef = Eigen::Ref<T, 0, EigenDStride>;

}  // namespace pyinterp
