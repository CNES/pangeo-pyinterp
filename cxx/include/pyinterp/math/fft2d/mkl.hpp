// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <mkl_dfti.h>

#include <cassert>
#include <complex>
#include <concepts>
#include <cstdint>
#include <format>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

#include "pyinterp/eigen.hpp"

namespace pyinterp::math::fft2d::mkl {

/// Supported floating-point types for FFT operations.
template <typename T>
concept FFTScalar = std::same_as<T, float> || std::same_as<T, double>;

/// MKL precision selector - maps C++ types to MKL precision constants.
template <FFTScalar T>
struct MKLPrecision;

template <>
struct MKLPrecision<double> {
  static constexpr auto value = DFTI_DOUBLE;
};

template <>
struct MKLPrecision<float> {
  static constexpr auto value = DFTI_SINGLE;
};

template <FFTScalar T>
inline constexpr auto mkl_precision_v = MKLPrecision<T>::value;

/// RAII wrapper for MKL thread-local settings.
/// Restores default thread count on destruction.
class ScopedThreadCount {
 public:
  explicit ScopedThreadCount(std::size_t num_threads) noexcept {
    mkl_set_num_threads_local(static_cast<MKL_INT>(num_threads));
  }

  ~ScopedThreadCount() noexcept {
    mkl_set_num_threads_local(0);  // Reset to default
  }

  // Non-copyable, non-movable
  ScopedThreadCount(const ScopedThreadCount&) = delete;
  ScopedThreadCount& operator=(const ScopedThreadCount&) = delete;
  ScopedThreadCount(ScopedThreadCount&&) = delete;
  ScopedThreadCount& operator=(ScopedThreadCount&&) = delete;
};

/// Custom deleter for MKL DFTI descriptor handles.
struct DftiDescriptorDeleter {
  void operator()(DFTI_DESCRIPTOR_HANDLE handle) const noexcept {
    if (handle != nullptr) {
      DftiFreeDescriptor(&handle);
    }
  }
};

using DftiDescriptorPtr =
    std::unique_ptr<DFTI_DESCRIPTOR, DftiDescriptorDeleter>;

/// Throws std::runtime_error if MKL operation failed.
/// @param[in] status MKL return status
/// @param[in] operation Description of the operation for error context
inline void check_mkl_status(MKL_LONG status, std::string_view operation) {
  if (status != 0) {
    throw std::runtime_error(
        std::format("MKL FFT2D error (code {}): {}", status, operation));
  }
}

/// Configuration for DFTI descriptor stride setup.
struct StrideConfig {
  std::array<MKL_LONG, 3> input_strides;
  std::array<MKL_LONG, 3> output_strides;
};

/// 2D FFT (R2C/C2R) implementation using Intel MKL.
///
/// Performs out-of-place real-to-complex (forward) and complex-to-real
/// (inverse) transforms on row-major Eigen matrices.
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
  /// @throws std::runtime_error if MKL descriptor creation fails
  Impl(std::int64_t rows, std::int64_t cols)
      : rows_{rows},
        cols_{cols},
        c_cols_{cols / 2 + 1},
        forward_plan_{create_forward_plan()},
        inverse_plan_{create_inverse_plan()} {}

  /// Performs forward FFT (real to complex).
  /// @param[in] grid Input grid in spatial domain
  /// @param[out] c_grid Output grid in frequency domain
  /// @param[in] num_threads Number of threads for computation (default: 1)
  void forward(const Eigen::Ref<const RealMatrix>& grid,
               Eigen::Ref<ComplexMatrix> c_grid,
               std::size_t num_threads = 1) const {
    assert(grid.rows() == rows_ && grid.cols() == cols_);
    assert(c_grid.rows() == rows_ && c_grid.cols() == c_cols_);

    const ScopedThreadCount thread_guard{num_threads};

    // MKL API requires non-const void* even for input buffers.
    // This is safe as DftiComputeForward only reads from input.
    check_mkl_status(
        DftiComputeForward(
            forward_plan_.get(),
            const_cast<void*>(static_cast<const void*>(grid.data())),
            static_cast<void*>(c_grid.data())),
        "DftiComputeForward");
  }

  /// Performs inverse FFT (complex to real).
  /// @param[in] c_grid Input grid in frequency domain
  /// @param[out] grid Output grid in spatial domain
  /// @param[in] num_threads Number of threads for computation (default: 1)
  void inverse(const Eigen::Ref<const ComplexMatrix>& c_grid,
               Eigen::Ref<RealMatrix> grid, std::size_t num_threads = 1) const {
    assert(c_grid.rows() == rows_ && c_grid.cols() == c_cols_);
    assert(grid.rows() == rows_ && grid.cols() == cols_);

    const ScopedThreadCount thread_guard{num_threads};

    check_mkl_status(
        DftiComputeBackward(
            inverse_plan_.get(),
            const_cast<void*>(static_cast<const void*>(c_grid.data())),
            static_cast<void*>(grid.data())),
        "DftiComputeBackward");
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
  DftiDescriptorPtr forward_plan_;
  DftiDescriptorPtr inverse_plan_;

  /// Creates and configures a DFTI descriptor.
  [[nodiscard]] auto create_descriptor() const -> DftiDescriptorPtr {
    DFTI_DESCRIPTOR_HANDLE handle = nullptr;
    const std::array<MKL_LONG, 2> dims{static_cast<MKL_LONG>(rows_),
                                       static_cast<MKL_LONG>(cols_)};

    check_mkl_status(DftiCreateDescriptor(&handle, mkl_precision_v<T>,
                                          DFTI_REAL, 2, dims.data()),
                     "DftiCreateDescriptor");

    return DftiDescriptorPtr{handle};
  }

  /// Configures common descriptor settings.
  void configure_descriptor(DFTI_DESCRIPTOR_HANDLE handle,
                            const StrideConfig& strides) const {
    check_mkl_status(
        DftiSetValue(handle, DFTI_INPUT_STRIDES, strides.input_strides.data()),
        "DftiSetValue INPUT_STRIDES");

    check_mkl_status(DftiSetValue(handle, DFTI_OUTPUT_STRIDES,
                                  strides.output_strides.data()),
                     "DftiSetValue OUTPUT_STRIDES");

    check_mkl_status(DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE),
                     "DftiSetValue DFTI_PLACEMENT");
  }

  /// Creates the forward (R2C) FFT plan.
  [[nodiscard]] auto create_forward_plan() const -> DftiDescriptorPtr {
    auto plan = create_descriptor();

    const StrideConfig strides{
        .input_strides = {0, static_cast<MKL_LONG>(cols_), 1},
        .output_strides = {0, static_cast<MKL_LONG>(c_cols_), 1}};

    configure_descriptor(plan.get(), strides);

    check_mkl_status(DftiCommitDescriptor(plan.get()),
                     "DftiCommitDescriptor (forward)");

    return plan;
  }

  /// Creates the inverse (C2R) FFT plan with normalization.
  [[nodiscard]] auto create_inverse_plan() const -> DftiDescriptorPtr {
    auto plan = create_descriptor();

    const StrideConfig strides{
        .input_strides = {0, static_cast<MKL_LONG>(c_cols_), 1},
        .output_strides = {0, static_cast<MKL_LONG>(cols_), 1}};

    configure_descriptor(plan.get(), strides);

    // Apply normalization: 1/(rows*cols) scaling on inverse transform
    const auto scale = static_cast<T>(1) / static_cast<T>(rows_ * cols_);
    check_mkl_status(DftiSetValue(plan.get(), DFTI_BACKWARD_SCALE, scale),
                     "DftiSetValue DFTI_BACKWARD_SCALE");

    check_mkl_status(DftiCommitDescriptor(plan.get()),
                     "DftiCommitDescriptor (inverse)");

    return plan;
  }
};

}  // namespace pyinterp::math::fft2d::mkl
