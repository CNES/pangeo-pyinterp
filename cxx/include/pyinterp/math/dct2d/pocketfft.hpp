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

namespace pyinterp::math::dct2d::pocket {

/// Supported floating-point types for DCT operations.
template <typename T>
concept DCTScalar = std::same_as<T, float> || std::same_as<T, double>;

/// DCT type constants for clarity.
enum class DCTType : uint8_t {
  II = 2,   ///< Forward DCT (DCT-II, "the" DCT)
  III = 3,  ///< Inverse DCT (DCT-III, "the" IDCT)
};

/// 2D Discrete Cosine Transform implementation using PocketFFT.
///
/// Provides in-place DCT-II (forward) and DCT-III (inverse) transforms.
/// The inverse transform includes normalization to make the round-trip
/// forward→inverse return the original data.
///
/// @tparam T Floating-point type (float or double)
template <DCTScalar T>
class Impl {
 public:
  using Scalar = T;
  using Matrix = RowMajorMatrix<T>;

  /// Creates a new 2D DCT plan.
  /// @param[in] rows Number of rows in the grid
  /// @param[in] cols Number of columns in the grid
  Impl(std::int64_t rows, std::int64_t cols)
      : rows_{rows},
        cols_{cols},
        normalization_factor_{Scalar{1} /
                              (Scalar{4} * static_cast<Scalar>(rows * cols))},
        shape_{static_cast<std::size_t>(rows), static_cast<std::size_t>(cols)},
        stride_{compute_row_major_stride(cols)},
        axes_{0, 1} {}

  /// Performs forward DCT (DCT-II) in-place.
  /// @param[in,out] grid Grid to transform (modified in-place)
  /// @param[in] num_threads Number of threads for computation (default: 1)
  void forward(Eigen::Ref<Matrix> grid, std::size_t num_threads = 1) const {
    assert(grid.rows() == rows_ && grid.cols() == cols_);

    pocketfft::dct(shape_, stride_, stride_, axes_,
                   static_cast<int>(DCTType::II), grid.data(), grid.data(),
                   Scalar{1}, /*ortho=*/false, num_threads);
  }

  /// Performs inverse DCT (DCT-III) in-place with normalization.
  /// @param[in,out] grid Grid to transform (modified in-place)
  /// @param[in] num_threads Number of threads for computation (default: 1)
  void inverse(Eigen::Ref<Matrix> grid, std::size_t num_threads = 1) const {
    assert(grid.rows() == rows_ && grid.cols() == cols_);

    // DCT-III with integrated normalization for efficiency
    pocketfft::dct(shape_, stride_, stride_, axes_,
                   static_cast<int>(DCTType::III), grid.data(), grid.data(),
                   normalization_factor_, /*ortho=*/false, num_threads);
  }

  /// @return Number of rows in the grid
  [[nodiscard]] constexpr auto rows() const noexcept -> std::int64_t {
    return rows_;
  }

  /// @return Number of columns in the grid
  [[nodiscard]] constexpr auto cols() const noexcept -> std::int64_t {
    return cols_;
  }

  /// @return Total number of elements (rows × cols)
  [[nodiscard]] constexpr auto size() const noexcept -> std::int64_t {
    return rows_ * cols_;
  }

 private:
  std::int64_t rows_;
  std::int64_t cols_;
  Scalar normalization_factor_;

  // PocketFFT configuration
  pocketfft::shape_t shape_;
  pocketfft::stride_t stride_;
  pocketfft::shape_t axes_;

  /// Computes row-major strides for the given column count.
  [[nodiscard]] static constexpr auto compute_row_major_stride(
      std::int64_t num_cols) -> pocketfft::stride_t {
    constexpr auto element_size = static_cast<std::ptrdiff_t>(sizeof(Scalar));
    return {element_size * static_cast<std::ptrdiff_t>(num_cols), element_size};
  }
};

}  // namespace pyinterp::math::dct2d::pocket
