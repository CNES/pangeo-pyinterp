// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pocketfft_hdronly.h>

#include "pyinterp/eigen.hpp"

namespace pyinterp::detail::math::dct2d::pocket {

/// @brief 2D DCT implementation using PocketFFT.
template <typename T>
class Impl {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "Type must be float or double");

 public:
  /// @brief Creates a new 2D DCT plan.
  /// @param rows The number of rows in the grid.
  /// @param cols The number of columns in the grid.
  Impl(int64_t rows, int64_t cols)
      : shape_{static_cast<size_t>(rows), static_cast<size_t>(cols)},
        // PocketFFT DCT requires row-major ("C") strides.
        stride_row_major_{
            static_cast<long>(sizeof(T) * static_cast<size_t>(cols)),
            static_cast<long>(sizeof(T))},
        axes_{0, 1} {}  // Transform over both axes

  void forward(Eigen::Ref<pyinterp::RowMajorMatrix<T>> grid,
               const size_t num_threads) {
    // DCT type 2 is the standard DCT (what FFTW calls DCT-II)
    pocketfft::dct(shape_, stride_row_major_, stride_row_major_, axes_, 2,
                   grid.data(), grid.data(), static_cast<T>(1), false,
                   num_threads);
  }

  void inverse(Eigen::Ref<pyinterp::RowMajorMatrix<T>> grid,
               const size_t num_threads) {
    // DCT type 3 is the inverse of DCT type 2 (what FFTW calls DCT-III)
    pocketfft::dct(shape_, stride_row_major_, stride_row_major_, axes_, 3,
                   grid.data(), grid.data(), static_cast<T>(1), false,
                   num_threads);

    // Normalization for DCT-II/DCT-III pair
    grid /= (T(4) * static_cast<T>(shape_[0]) * static_cast<T>(shape_[1]));
  }

  constexpr auto rows() const noexcept -> int64_t { return shape_[0]; }
  constexpr auto cols() const noexcept -> int64_t { return shape_[1]; }

 private:
  /// @brief Shape of the input grid
  pocketfft::shape_t shape_;
  /// @brief Row-major strides for PocketFFT
  pocketfft::stride_t stride_row_major_;
  /// @brief Axes over which to perform the DCT
  pocketfft::shape_t axes_;
};

}  // namespace pyinterp::detail::math::dct2d::pocket
