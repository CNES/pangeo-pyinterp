// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <pocketfft_hdronly.h>

#include <cassert>
#include <cstdint>

#include "pyinterp/eigen.hpp"

namespace pyinterp::detail::math::fft2d::pocket {

/// @brief 2D FFT (R2C/C2R) implementation using PocketFFT.
/// @tparam T The data type of the input and output arrays.
template <typename T>
class Impl {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "Type must be float or double");

 public:
  /// @brief Creates a new 2D FFT plan.
  /// @param rows The number of rows in the grid.
  /// @param cols The number of columns in the grid.
  Impl(int64_t rows, int64_t cols)
      : rows_(rows),
        cols_(cols),
        c_cols_(cols / 2 + 1),
        shape_{static_cast<size_t>(rows), static_cast<size_t>(cols)},
        // Strides for RowMajor real matrix
        stride_r_{static_cast<long>(sizeof(T) * static_cast<size_t>(cols)),
                  static_cast<long>(sizeof(T))},
        // Strides for RowMajor complex matrix
        stride_c_{static_cast<long>(sizeof(std::complex<T>) *
                                    static_cast<size_t>(c_cols_)),
                  static_cast<long>(sizeof(std::complex<T>))},
        axes_{0, 1} {}  // Transform over both axes

  /// @brief Performs the forward FFT.
  /// @param grid The input grid (spatial domain).
  /// @param c_grid The output grid (frequency domain).
  /// @param num_threads The number of threads to use.
  void forward(const Eigen::Ref<const RowMajorMatrix<T>>& grid,
               Eigen::Ref<RowMajorComplexMatrix<T>>& c_grid,
               const size_t num_threads) {
    assert(grid.rows() == rows_ && grid.cols() == cols_);
    assert(c_grid.rows() == rows_ && c_grid.cols() == c_cols_);

    pocketfft::r2c(shape_, stride_r_, stride_c_, axes_, pocketfft::FORWARD,
                   grid.data(), c_grid.data(), static_cast<T>(1), num_threads);
  }

  /// @brief Performs the inverse FFT.
  /// @param c_grid The input grid (frequency domain).
  /// @param grid The output grid (spatial domain).
  /// @param num_threads The number of threads to use.
  void inverse(const Eigen::Ref<const RowMajorComplexMatrix<T>>& c_grid,
               Eigen::Ref<RowMajorMatrix<T>>& grid, const size_t num_threads) {
    assert(c_grid.rows() == rows_ && c_grid.cols() == c_cols_);
    assert(grid.rows() == rows_ && grid.cols() == cols_);

    pocketfft::c2r(shape_, stride_c_, stride_r_, axes_, pocketfft::BACKWARD,
                   c_grid.data(), grid.data(), static_cast<T>(1), num_threads);

    // PocketFFT's C2R is unnormalized.
    grid /= (static_cast<T>(rows_) * static_cast<T>(cols_));
  }

  /// @brief Returns the number of rows in the grid.
  constexpr auto rows() const noexcept -> int64_t { return rows_; }

  /// @brief Returns the number of columns in the grid.
  constexpr auto cols() const noexcept -> int64_t { return cols_; }

  /// @brief Returns the number of complex columns in the grid.
  constexpr auto c_cols() const noexcept -> int64_t { return c_cols_; }

 private:
  /// @brief Number of rows of the grid
  int64_t rows_;
  /// @brief Number of columns of the grid
  int64_t cols_;
  /// @brief Number of complex columns of the grid
  int64_t c_cols_;
  /// @brief Shape of the input grid
  pocketfft::shape_t shape_;
  /// @brief Strides for RowMajor real matrix
  pocketfft::stride_t stride_r_;
  /// @brief Strides for RowMajor complex matrix
  pocketfft::stride_t stride_c_;
  /// @brief Axes over which to perform the FFT
  pocketfft::shape_t axes_;
};

}  // namespace pyinterp::detail::math::fft2d::pocket
