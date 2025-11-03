// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include "pyinterp/eigen.hpp"

#if defined(USE_MKL_DTFI)
#include "pyinterp/detail/math/fft2d/mkl.hpp"
namespace pyinterp::detail::math::fft2d {

/// @brief The FFT implementation using the MKL library.
/// @tparam T The data type of the input and output arrays.
template <typename T>
struct Impl : public mkl::Impl<T> {
  using mkl::Impl<T>::Impl;
};

}  // namespace pyinterp::detail::math::fft2d
#else
#include "pyinterp/detail/math/fft2d/pocketfft.hpp"
namespace pyinterp::detail::math::fft2d {

/// @brief The FFT implementation using the pocketfft library.
/// @tparam T The data type of the input and output arrays.
template <typename T>
struct Impl : public pocket::Impl<T> {
  using pocket::Impl<T>::Impl;
};

}  // namespace pyinterp::detail::math::fft2d
#endif

namespace pyinterp::detail::math {

/// @brief The FFT class provides forward and backward Fast Fourier Transform
/// operations.
///
/// @tparam T The data type of the input and output arrays.
template <typename T>
class FFT2D {
 public:
  /// @brief Creates a new 2D FFT plan.
  /// @param rows The number of rows in the grid.
  /// @param cols The number of columns in the grid.
  FFT2D(int64_t rows, int64_t cols) : implementation_(rows, cols) {}

  /// @brief Performs a forward 2D FFT (Real-to-Complex).
  /// @param grid The real input grid.
  /// @param c_grid The complex output grid.
  /// @param size_t num_threads The number of threads to use.
  inline void forward(const Eigen::Ref<const RowMajorMatrix<T>>& grid,
                      Eigen::Ref<RowMajorComplexMatrix<T>> c_grid,
                      const size_t num_threads = 1) {
    implementation_.forward(grid, c_grid, num_threads);
  }

  /// @brief Performs an inverse 2D FFT (Complex-to-Real).
  /// @param c_grid The complex input grid.
  /// @param grid The real output grid.
  /// @param size_t num_threads The number of threads to use.
  inline void inverse(const Eigen::Ref<const RowMajorComplexMatrix<T>>& c_grid,
                      Eigen::Ref<RowMajorMatrix<T>> grid,
                      const size_t num_threads = 1) {
    implementation_.inverse(c_grid, grid, num_threads);
  }

  /// @brief Returns the number of rows in the grid.
  constexpr auto rows() const noexcept -> int64_t {
    return implementation_.rows();
  }
  /// @brief Returns the number of columns in the grid.
  constexpr auto cols() const noexcept -> int64_t {
    return implementation_.cols();
  }
  /// @brief Returns the number of complex columns in the grid.
  constexpr auto c_cols() const noexcept -> int64_t {
    return implementation_.c_cols();
  }

 private:
  fft2d::Impl<T> implementation_;
};

}  // namespace pyinterp::detail::math
