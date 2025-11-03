// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include "pyinterp/detail/math/dct2d/pocketfft.hpp"
#include "pyinterp/eigen.hpp"

namespace pyinterp::detail::math::dct2d {

/// @brief The DCT implementation using the pocketfft library.
template <typename T>
struct Impl : public pocket::Impl<T> {
  using pocket::Impl<T>::Impl;
};

}  // namespace pyinterp::detail::math::dct2d

namespace pyinterp::detail::math {

/// @brief 2D Discrete Cosine Transform (DCT-II) and Inverse (DCT-III)
///
/// @tparam T The data type of the input and output arrays.
template <typename T>
class DCT2D {
 public:
  /// @brief Creates a new 2D DCT plan.
  /// @param rows The number of rows in the grid.
  /// @param cols The number of columns in the grid.
  DCT2D(int64_t rows, int64_t cols) : implementation_(rows, cols) {}

  /// @brief Performs a forward 2D DCT (DCT-II) on the input grid.
  /// @param grid The grid to be transformed (in-place).
  /// @param num_threads The number of threads to use.
  inline void forward(Eigen::Ref<RowMajorMatrix<T>> grid,
                      const size_t num_threads = 1) {
    implementation_.forward(grid, num_threads);
  }

  /// @brief Performs an inverse 2D DCT (DCT-III) on the input grid.
  /// @param grid The grid to be transformed (in-place).
  /// @param num_threads The number of threads to use.
  inline void inverse(Eigen::Ref<RowMajorMatrix<T>> grid,
                      const size_t num_threads = 1) {
    implementation_.inverse(grid, num_threads);
  }

 private:
  dct2d::Impl<T> implementation_;
};

}  // namespace pyinterp::detail::math
