// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <cstdint>
#include <string_view>

#include "pyinterp/eigen.hpp"

#if defined(USE_MKL_DFTI)
#include "pyinterp/math/fft2d/mkl.hpp"
#else
#include "pyinterp/math/fft2d/pocketfft.hpp"
#endif

namespace pyinterp::math {

/// FFT backend identifier for runtime queries.
enum class FFTBackend : std::uint8_t { PocketFFT, MKL };

/// @return String representation of the FFT backend
[[nodiscard]] constexpr auto to_string(FFTBackend backend) noexcept
    -> std::string_view {
  switch (backend) {
    case FFTBackend::MKL:
      return "MKL";
    case FFTBackend::PocketFFT:
      return "PocketFFT";
  }
  return "Unknown";  // Unreachable, but silences warnings
}

namespace fft2d {

/// Backend-specific implementation type alias.
/// Selects MKL or PocketFFT based on compile-time configuration.
template <typename T>
#if defined(USE_MKL_DFTI)
using Impl = mkl::Impl<T>;
inline constexpr auto kBackend = FFTBackend::MKL;
#else
using Impl = pocket::Impl<T>;
inline constexpr auto kBackend = FFTBackend::PocketFFT;
#endif

}  // namespace fft2d

/// Supported floating-point types for FFT operations.
template <typename T>
concept FFTScalar = std::same_as<T, float> || std::same_as<T, double>;

/// 2D Fast Fourier Transform for real-valued grids.
///
/// Provides forward (R2C) and inverse (C2R) FFT operations on row-major
/// Eigen matrices. The implementation backend (MKL or PocketFFT) is selected
/// at compile time via the USE_MKL_DFTI preprocessor define.
///
/// Usage:
/// @code
///   FFT2D<double> fft(512, 512);
///   RowMajorMatrix<double> spatial(512, 512);
///   RowMajorComplexMatrix<double> frequency(512, 257);
///
///   fft.forward(spatial, frequency);
///   // ... modify frequency domain ...
///   fft.inverse(frequency, spatial);
/// @endcode
///
/// @tparam T Floating-point type (float or double)
template <FFTScalar T>
class FFT2D {
 public:
  using Scalar = T;
  using Complex = std::complex<T>;
  using RealMatrix = RowMajorMatrix<T>;
  using ComplexMatrix = RowMajorComplexMatrix<T>;
  using Implementation = fft2d::Impl<T>;

  /// The active FFT backend for this build.
  static constexpr auto kBackend = fft2d::kBackend;

  /// Creates a new 2D FFT plan.
  /// @param[in] rows Number of rows in the spatial domain grid
  /// @param[in] cols Number of columns in the spatial domain grid
  FFT2D(std::int64_t rows, std::int64_t cols) : impl_{rows, cols} {}

  /// Performs forward FFT (real to complex).
  ///
  /// Transforms a real-valued spatial domain grid to its complex frequency
  /// domain representation. The output grid has dimensions (rows, cols/2+1)
  /// due to Hermitian symmetry.
  ///
  /// @param[in] grid Input grid in spatial domain (rows × cols)
  /// @param[out] c_grid Output grid in frequency domain (rows × c_cols)
  /// @param[in] num_threads Number of threads for computation (default: 1)
  void forward(const Eigen::Ref<const RealMatrix>& grid,
               Eigen::Ref<ComplexMatrix> c_grid,
               std::size_t num_threads = 1) const {
    impl_.forward(grid, c_grid, num_threads);
  }

  /// Performs inverse FFT (complex to real).
  ///
  /// Transforms a complex frequency domain grid back to real spatial domain.
  /// The transform is normalized (divided by rows×cols).
  ///
  /// @param[in] c_grid Input grid in frequency domain (rows × c_cols)
  /// @param[out] grid Output grid in spatial domain (rows × cols)
  /// @param[in] num_threads Number of threads for computation (default: 1)
  void inverse(const Eigen::Ref<const ComplexMatrix>& c_grid,
               Eigen::Ref<RealMatrix> grid, std::size_t num_threads = 1) const {
    impl_.inverse(c_grid, grid, num_threads);
  }

  /// @return Number of rows in the spatial domain grid
  [[nodiscard]] constexpr auto rows() const noexcept -> std::int64_t {
    return impl_.rows();
  }

  /// @return Number of columns in the spatial domain grid
  [[nodiscard]] constexpr auto cols() const noexcept -> std::int64_t {
    return impl_.cols();
  }

  /// @return Number of columns in the frequency domain grid (cols/2 + 1)
  [[nodiscard]] constexpr auto c_cols() const noexcept -> std::int64_t {
    return impl_.c_cols();
  }

  /// @return Total number of elements in spatial domain (rows × cols)
  [[nodiscard]] constexpr auto size() const noexcept -> std::int64_t {
    return impl_.size();
  }

  /// @return The FFT backend being used
  [[nodiscard]] static constexpr auto backend() noexcept -> FFTBackend {
    return kBackend;
  }

  /// @return String name of the FFT backend
  [[nodiscard]] static constexpr auto backend_name() noexcept
      -> std::string_view {
    return to_string(kBackend);
  }

 private:
  Implementation impl_;
};

}  // namespace pyinterp::math
