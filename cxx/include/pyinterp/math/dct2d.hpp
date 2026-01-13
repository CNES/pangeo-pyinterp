// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <cstdint>
#include <string_view>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math/dct2d/pocketfft.hpp"

namespace pyinterp::math {

/// DCT backend identifier for runtime queries.
enum class DCTBackend : uint8_t {
  PocketFFT,
};

/// @return String representation of the DCT backend
[[nodiscard]] constexpr auto to_string(DCTBackend backend) noexcept
    -> std::string_view {
  switch (backend) {
    case DCTBackend::PocketFFT:
      return "PocketFFT";
  }
  return "Unknown";
}

namespace dct2d {

/// Backend-specific implementation type alias.
/// Currently only PocketFFT is supported; structure mirrors FFT2D
/// for future backend additions.
template <typename T>
using Impl = pocket::Impl<T>;

inline constexpr auto kBackend = DCTBackend::PocketFFT;

}  // namespace dct2d

/// Supported floating-point types for DCT operations.
template <typename T>
concept DCTScalar = std::same_as<T, float> || std::same_as<T, double>;

/// 2D Discrete Cosine Transform (DCT-II / DCT-III).
///
/// Provides in-place forward (DCT-II) and inverse (DCT-III) transforms
/// on row-major Eigen matrices. The inverse transform is normalized so
/// that forward→inverse returns the original data.
///
/// DCT is particularly useful for:
/// - Image compression (JPEG uses 8×8 DCT blocks)
/// - Spectral methods for PDEs with Neumann boundary conditions
/// - Signal processing where symmetry is exploited
///
/// Usage:
/// @code
///   DCT2D<double> dct(512, 512);
///   RowMajorMatrix<double> grid(512, 512);
///
///   dct.forward(grid);   // Transform to frequency domain (in-place)
///   // ... modify coefficients ...
///   dct.inverse(grid);   // Transform back to spatial domain (in-place)
/// @endcode
///
/// @tparam T Floating-point type (float or double)
template <DCTScalar T>
class DCT2D {
 public:
  using Scalar = T;
  using Matrix = RowMajorMatrix<T>;
  using Implementation = dct2d::Impl<T>;

  /// The active DCT backend for this build.
  static constexpr auto kBackend = dct2d::kBackend;

  /// Creates a new 2D DCT plan.
  /// @param[in] rows Number of rows in the grid
  /// @param[in] cols Number of columns in the grid
  DCT2D(std::int64_t rows, std::int64_t cols) : impl_{rows, cols} {}

  /// Performs forward DCT (DCT-II) in-place.
  ///
  /// Transforms spatial domain data to DCT coefficients. The (0,0)
  /// coefficient represents the DC component (mean value × size).
  ///
  /// @param[in,out] grid Grid to transform (modified in-place)
  /// @param[in] num_threads Number of threads for computation (default: 1)
  void forward(Eigen::Ref<Matrix> grid, std::size_t num_threads = 1) const {
    impl_.forward(grid, num_threads);
  }

  /// Performs inverse DCT (DCT-III) in-place.
  ///
  /// Transforms DCT coefficients back to spatial domain. Includes
  /// normalization factor of 1/(4·rows·cols) for round-trip consistency.
  ///
  /// @param[in,out] grid Grid to transform (modified in-place)
  /// @param[in] num_threads Number of threads for computation (default: 1)
  void inverse(Eigen::Ref<Matrix> grid, std::size_t num_threads = 1) const {
    impl_.inverse(grid, num_threads);
  }

  /// @return Number of rows in the grid
  [[nodiscard]] constexpr auto rows() const noexcept -> std::int64_t {
    return impl_.rows();
  }

  /// @return Number of columns in the grid
  [[nodiscard]] constexpr auto cols() const noexcept -> std::int64_t {
    return impl_.cols();
  }

  /// @return Total number of elements (rows × cols)
  [[nodiscard]] constexpr auto size() const noexcept -> std::int64_t {
    return impl_.size();
  }

  /// @return The DCT backend being used
  [[nodiscard]] static constexpr auto backend() noexcept -> DCTBackend {
    return kBackend;
  }

  /// @return String name of the DCT backend
  [[nodiscard]] static constexpr auto backend_name() noexcept
      -> std::string_view {
    return to_string(kBackend);
  }

 private:
  Implementation impl_;
};

}  // namespace pyinterp::math
