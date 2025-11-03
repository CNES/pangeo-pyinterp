// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <mkl_dfti.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include "pyinterp/eigen.hpp"

namespace pyinterp::detail::math::fft2d::mkl {

// Helper to cast away const for MKL C API which expects non-const void*
template <typename T>
inline void* remove_const_ptr(const T* p) {
  return const_cast<void*>(static_cast<const void*>(p));
}

// Helpers to cast std::complex<T> pointers to MKL complex types while
// removing constness. These mirror the helpers used in Eigen's MKL backend.
inline MKL_Complex16* complex_cast(const std::complex<double>* p) {
  return const_cast<MKL_Complex16*>(reinterpret_cast<const MKL_Complex16*>(p));
}

inline MKL_Complex8* complex_cast(const std::complex<float>* p) {
  return const_cast<MKL_Complex8*>(reinterpret_cast<const MKL_Complex8*>(p));
}

/// @brief Runs a function and throws a `std::runtime_error` if the status is
/// not zero.
/// @param[in] status The status returned by a function.
/// @param[in] msg The error message to be thrown.
constexpr auto run_or_throw(MKL_LONG status, const char* const msg) -> void {
  if (status != 0) {
    throw std::runtime_error(msg);
  }
}

/// @brief Custom deleter for MKL descriptor handles
struct DftiDescriptorDeleter {
  void operator()(DFTI_DESCRIPTOR_HANDLE handle) const {
    if (handle) {
      DftiFreeDescriptor(&handle);
    }
  }
};
using DftiDescriptorPtr =
    std::unique_ptr<DFTI_DESCRIPTOR, DftiDescriptorDeleter>;

/// @brief 2D FFT (R2C/C2R) implementation using MKL.
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
      : rows_(rows), cols_(cols), c_cols_(cols / 2 + 1) {
    auto precision = std::is_same_v<T, double> ? DFTI_DOUBLE : DFTI_SINGLE;
    MKL_LONG dims[2] = {static_cast<MKL_LONG>(rows),
                        static_cast<MKL_LONG>(cols)};

    // Create Forward Plan (Real-to-Complex)
    DFTI_DESCRIPTOR_HANDLE fwd_handle = nullptr;
    run_or_throw(
        DftiCreateDescriptor(&fwd_handle, precision, DFTI_REAL, 2, dims),
        "MKL DftiCreateDescriptor failed for forward FFT2D");
    forward_plan_.reset(fwd_handle);

    // MKL expects row-major (C-style) strides
    MKL_LONG fwd_r_strides[3] = {0, static_cast<MKL_LONG>(cols), 1};
    MKL_LONG fwd_c_strides[3] = {0, static_cast<MKL_LONG>(c_cols_), 1};
    run_or_throw(
        DftiSetValue(forward_plan_.get(), DFTI_INPUT_STRIDES, fwd_r_strides),
        "MKL DftiSetValue failed for INPUT_STRIDES (forward)");
    run_or_throw(
        DftiSetValue(forward_plan_.get(), DFTI_OUTPUT_STRIDES, fwd_c_strides),
        "MKL DftiSetValue failed for OUTPUT_STRIDES (forward)");
    // We perform out-of-place real->complex transforms, ensure MKL knows
    // the transform will be not-inplace.
    run_or_throw(
        DftiSetValue(forward_plan_.get(), DFTI_PLACEMENT, DFTI_NOT_INPLACE),
        "MKL DftiSetValue failed for DFTI_PLACEMENT (forward)");
    run_or_throw(DftiCommitDescriptor(forward_plan_.get()),
                 "MKL DftiCommitDescriptor failed (forward)");

    // Create Inverse Plan (Complex-to-Real)
    DFTI_DESCRIPTOR_HANDLE inv_handle = nullptr;
    run_or_throw(
        DftiCreateDescriptor(&inv_handle, precision, DFTI_REAL, 2, dims),
        "MKL DftiCreateDescriptor failed for inverse FFT2D");
    inverse_plan_.reset(inv_handle);

    MKL_LONG inv_c_strides[3] = {0, static_cast<MKL_LONG>(c_cols_), 1};
    MKL_LONG inv_r_strides[3] = {0, static_cast<MKL_LONG>(cols_), 1};
    run_or_throw(
        DftiSetValue(inverse_plan_.get(), DFTI_INPUT_STRIDES, inv_c_strides),
        "MKL DftiSetValue failed for INPUT_STRIDES (inverse)");
    run_or_throw(
        DftiSetValue(inverse_plan_.get(), DFTI_OUTPUT_STRIDES, inv_r_strides),
        "MKL DftiSetValue failed for OUTPUT_STRIDES (inverse)");
    run_or_throw(
        DftiSetValue(inverse_plan_.get(), DFTI_PLACEMENT, DFTI_NOT_INPLACE),
        "MKL DftiSetValue failed for DFTI_PLACEMENT (inverse)");

    // Set normalization factor for inverse
    run_or_throw(DftiSetValue(inverse_plan_.get(), DFTI_BACKWARD_SCALE,
                              static_cast<T>(1.0) / (rows_ * cols_)),
                 "MKL DftiSetValue failed for DFTI_BACKWARD_SCALE");

    run_or_throw(DftiCommitDescriptor(inverse_plan_.get()),
                 "MKL DftiCommitDescriptor failed (inverse)");
  }

  /// @brief Performs the forward FFT.
  /// @param grid The input grid (spatial domain).
  /// @param c_grid The output grid (frequency domain).
  /// @param num_threads The number of threads to use.
  void forward(const Eigen::Ref<const RowMajorMatrix<T>>& grid,
               Eigen::Ref<RowMajorComplexMatrix<T>>& c_grid,
               const size_t num_threads = 1) {
    assert(grid.rows() == rows_ && grid.cols() == cols_);
    assert(c_grid.rows() == rows_ && c_grid.cols() == c_cols_);

    mkl_set_num_threads_local(static_cast<MKL_INT>(num_threads));

    // DftiComputeForward expects a non-const void* for the input/output
    // pointers. Eigen provides const pointers for const refs, so cast away
    // constness safely because MKL's API requires mutable pointers even
    // for read-only input in certain signatures.
    run_or_throw(
        DftiComputeForward(forward_plan_.get(), remove_const_ptr(grid.data()),
                           static_cast<void*>(c_grid.data())),
        "MKL DftiComputeForward failed for FFT2D");

    mkl_set_num_threads_local(0);  // reset to default
  }

  /// @brief Performs the inverse FFT.
  /// @param c_grid The input grid (frequency domain).
  /// @param grid The output grid (spatial domain).
  /// @param size_t num_threads The number of threads to use.
  void inverse(const Eigen::Ref<const RowMajorComplexMatrix<T>>& c_grid,
               Eigen::Ref<RowMajorMatrix<T>>& grid,
               const size_t num_threads = 1) {
    assert(c_grid.rows() == rows_ && c_grid.cols() == c_cols_);
    assert(grid.rows() == rows_ && grid.cols() == cols_);

    mkl_set_num_threads_local(static_cast<MKL_INT>(num_threads));

    run_or_throw(DftiComputeBackward(inverse_plan_.get(),
                                     remove_const_ptr(c_grid.data()),
                                     static_cast<void*>(grid.data())),
                 "MKL DftiComputeBackward failed for FFT2D");

    mkl_set_num_threads_local(0);  // reset to default
  }

  /// @brief Returns the number of rows in the grid.
  constexpr auto rows() const noexcept -> int64_t { return rows_; }
  /// @brief Returns the number of columns in the grid.
  constexpr auto cols() const noexcept -> int64_t { return cols_; }
  /// @brief Returns the number of columns in the complex grid.
  constexpr auto c_cols() const noexcept -> int64_t { return c_cols_; }

 private:
  /// @brief Number of rows of the grid
  int64_t rows_;
  /// @brief Number of columns of the grid
  int64_t cols_;
  /// @brief Number of columns of the complex grid
  int64_t c_cols_;
  /// @brief Unique pointer for forward FFTW plan
  DftiDescriptorPtr forward_plan_;
  /// @brief Unique pointer for inverse FFTW plan
  DftiDescriptorPtr inverse_plan_;
};

}  // namespace pyinterp::detail::math::fft2d::mkl
