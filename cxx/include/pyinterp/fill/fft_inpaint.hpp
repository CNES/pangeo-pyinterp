// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <tuple>

#include "pyinterp/config/fill.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/fill/helpers.hpp"
#include "pyinterp/math/dct2d.hpp"
#include "pyinterp/math/fft2d.hpp"

namespace pyinterp::fill {

/// Replaces all undefined values (NaN) in a grid using spectral in-painting.
///
/// Uses iterative Gaussian filtering in frequency space. For periodic grids,
/// uses FFT; for non-periodic grids, uses DCT (which implicitly handles
/// symmetric boundary conditions).
///
/// @tparam Type Data type (must be floating point).
/// @param[in,out] grid The grid to be processed.
/// @param[in] config Configuration parameters.
/// @return A tuple containing the number of iterations performed and the
/// maximum residual value.
template <std::floating_point Type>
[[nodiscard]] auto fft_inpaint(Eigen::Ref<RowMajorMatrix<Type>> grid,
                               const config::fill::FFTInpaint& config)
    -> std::tuple<size_t, Type> {
  // Early return if no undefined values
  if (!grid.hasNaN()) {
    return std::make_tuple(0, Type(0));
  }

  const auto x_size = grid.rows();
  const auto y_size = grid.cols();

  // Identify masked (NaN) locations: true = unknown, false = known
  Matrix<bool> mask = grid.array().isNaN();

  // Store original known values (unknown positions set to 0)
  RowMajorMatrix<Type> original_values = mask.array().select(Type(0), grid);

  // Set initial guess for unknown positions
  switch (config.first_guess()) {
    case config::fill::FirstGuess::kZero:
      grid = mask.array().select(Type(0), grid);
      break;
    case config::fill::FirstGuess::kZonalAverage:
      set_zonal_average<Type>(grid, mask, config.num_threads());
      break;
    default:
      throw std::invalid_argument(
          "Unsupported first guess method: " +
          std::to_string(static_cast<int>(config.first_guess())));
  }

  const auto sigma_sq =
      Type(2) * static_cast<Type>(config.sigma() * config.sigma());

  // Gaussian weight computation
  auto gaussian_weight = [sigma_sq](Type u, Type v) -> Type {
    return std::exp(-(u * u + v * v) / sigma_sq);
  };

  // Transform workspaces
  std::optional<math::FFT2D<Type>> fft;
  std::optional<math::DCT2D<Type>> dct;
  RowMajorComplexMatrix<Type> grid_fft;
  RowMajorComplexMatrix<Type> kernel_fft;
  RowMajorMatrix<Type> kernel_dct;

  if (config.is_periodic()) {
    // Periodic case: FFT with wrapped frequencies
    fft.emplace(x_size, y_size);
    grid_fft.resize(fft->rows(), fft->c_cols());
    kernel_fft.resize(fft->rows(), fft->c_cols());

    // Pre-compute FFT kernel (Gaussian in frequency space)
    for (int64_t ix = 0; ix < kernel_fft.rows(); ++ix) {
      // Frequency u with FFT shift: [0, N/2] positive, (N/2, N) negative
      const Type u = (ix <= x_size / 2) ? Type(ix) : Type(ix - x_size);
      for (int64_t iy = 0; iy < kernel_fft.cols(); ++iy) {
        // Frequency v (R2C layout: only non-negative frequencies stored)
        kernel_fft(ix, iy) = {gaussian_weight(u, Type(iy)), Type(0)};
      }
    }
  } else {
    // Non-periodic case: DCT (symmetric extension, no wrap-around)
    dct.emplace(x_size, y_size);
    kernel_dct.resize(x_size, y_size);

    // Pre-compute DCT kernel (frequencies are non-negative)
    for (int64_t ix = 0; ix < x_size; ++ix) {
      for (int64_t iy = 0; iy < y_size; ++iy) {
        kernel_dct(ix, iy) = gaussian_weight(Type(ix), Type(iy));
      }
    }
  }

  // Iterative in-painting
  RowMajorMatrix<Type> previous_grid = grid;
  Type max_residual = std::numeric_limits<Type>::max();

  for (size_t iteration = 1; iteration <= config.max_iterations();
       ++iteration) {
    // Apply Gaussian filter in frequency space
    if (config.is_periodic()) {
      fft->forward(grid, grid_fft, config.num_threads());
      grid_fft.array() *= kernel_fft.array();
      fft->inverse(grid_fft, grid, config.num_threads());
    } else {
      dct->forward(grid, config.num_threads());
      grid.array() *= kernel_dct.array();
      dct->inverse(grid, config.num_threads());
    }

    // Restore known values (keep smoothed values only at unknown positions)
    grid = mask.array().select(grid, original_values);

    // Check convergence (residual only at unknown positions)
    auto diff = (grid.array() - previous_grid.array()).abs();
    max_residual = mask.array().select(diff, Type(0)).maxCoeff();

    if (max_residual < config.epsilon()) {
      return std::make_tuple(iteration, max_residual);
    }

    previous_grid = grid;
  }

  return std::make_tuple(config.max_iterations(), max_residual);
}

}  // namespace pyinterp::fill
