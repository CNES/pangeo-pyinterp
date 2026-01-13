// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <pocketfft_hdronly.h>

#include <cassert>
#include <concepts>
#include <cstdint>

#include "pyinterp/eigen.hpp"

namespace pyinterp::math::fft2d::pocket {

/// Supported floating-point types for FFT operations.
template <typename T>
concept FFTScalar = std::same_as<T, float> || std::same_as<T, double>;

/// 2D FFT (R2C/C2R) implementation using PocketFFT.
///
/// PocketFFT is a header-only, dependency-free FFT library that provides
/// good performance without requiring external libraries like FFTW or MKL.
///
/// @tparam T Floating-point type (float or double)
template <FFTScalar T>
class Impl {
 public:
  using Scalar = T;
  using Complex = std::complex<T>;
  using RealMatrix = RowMajorMatrix<T>;
  using ComplexMatrix = RowMajorComplexMatrix<T>;

  /// Creates a new 2D FFT plan.
  /// @param[in] rows Number of rows in the spatial domain grid
  /// @param[in] cols Number of columns in the spatial domain grid
  Impl(std::int64_t rows, std::int64_t cols)
      : rows_{rows},
        cols_{cols},
        c_cols_{cols / 2 + 1},
        normalization_factor_{Scalar{1} / static_cast<Scalar>(rows * cols)},
        shape_{static_cast<std::size_t>(rows), static_cast<std::size_t>(cols)},
        stride_real_{compute_stride<Scalar>(cols)},
        stride_complex_{compute_stride<Complex>(c_cols_)},
        axes_{0, 1} {}

  /// Performs forward FFT (real to complex).
  /// @param[in] grid Input grid in spatial domain
  /// @param[out] c_grid Output grid in frequency domain
  /// @param[in] num_threads Number of threads for computation (default: 1)
  void forward(const Eigen::Ref<const RealMatrix>& grid,
               Eigen::Ref<ComplexMatrix> c_grid,
               std::size_t num_threads = 1) const {
    assert(grid.rows() == rows_ && grid.cols() == cols_);
    assert(c_grid.rows() == rows_ && c_grid.cols() == c_cols_);

    pocketfft::r2c(shape_, stride_real_, stride_complex_, axes_,
                   pocketfft::FORWARD, grid.data(), c_grid.data(), Scalar{1},
                   num_threads);
  }

  /// Performs inverse FFT (complex to real).
  /// @param[in] c_grid Input grid in frequency domain
  /// @param[out] grid Output grid in spatial domain
  /// @param[in] num_threads Number of threads for computation (default: 1)
  void inverse(const Eigen::Ref<const ComplexMatrix>& c_grid,
               Eigen::Ref<RealMatrix> grid, std::size_t num_threads = 1) const {
    assert(c_grid.rows() == rows_ && c_grid.cols() == c_cols_);
    assert(grid.rows() == rows_ && grid.cols() == cols_);

    // PocketFFT's C2R transform is unnormalized - apply scaling directly
    // in the transform call for better efficiency
    pocketfft::c2r(shape_, stride_complex_, stride_real_, axes_,
                   pocketfft::BACKWARD, c_grid.data(), grid.data(),
                   normalization_factor_, num_threads);
  }

  /// @return Number of rows in the spatial domain grid
  [[nodiscard]] constexpr auto rows() const noexcept -> std::int64_t {
    return rows_;
  }

  /// @return Number of columns in the spatial domain grid
  [[nodiscard]] constexpr auto cols() const noexcept -> std::int64_t {
    return cols_;
  }

  /// @return Number of columns in the frequency domain grid
  [[nodiscard]] constexpr auto c_cols() const noexcept -> std::int64_t {
    return c_cols_;
  }

  /// @return Total number of elements in spatial domain
  [[nodiscard]] constexpr auto size() const noexcept -> std::int64_t {
    return rows_ * cols_;
  }

 private:
  std::int64_t rows_;
  std::int64_t cols_;
  std::int64_t c_cols_;
  Scalar normalization_factor_;

  // PocketFFT configuration (immutable after construction)
  pocketfft::shape_t shape_;
  pocketfft::stride_t stride_real_;
  pocketfft::stride_t stride_complex_;
  pocketfft::shape_t axes_;

  /// Computes row-major strides for a matrix with given element type and
  /// column count.
  /// @tparam Element The element type (affects stride size)
  /// @param[in] num_cols Number of columns in the matrix
  /// @return PocketFFT stride specification for row-major layout
  template <typename Element>
  [[nodiscard]] static constexpr auto compute_stride(std::int64_t num_cols)
      -> pocketfft::stride_t {
    // Row-major layout: row stride = cols * element_size, col stride =
    // element_size
    constexpr auto element_size = static_cast<std::ptrdiff_t>(sizeof(Element));
    return {element_size * static_cast<std::ptrdiff_t>(num_cols), element_size};
  }
};

}  // namespace pyinterp::math::fft2d::pocket
