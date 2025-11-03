#pragma once

#include <complex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <tuple>

#include "pyinterp/detail/math.hpp"
#include "pyinterp/detail/math/dct2d.hpp"
#include "pyinterp/detail/math/fft2d.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/fill/enum.hpp"
#include "pyinterp/fill/utils.hpp"

namespace pyinterp::fill {

/// Replaces all undefined values (NaN) in a grid using spectral in-painting.
///
/// @param grid The grid to be processed
/// @param first_guess Method to use for the first guess.
/// @param is_circle If true, uses a Fast Fourier Transform (FFT) assuming
/// periodic boundaries. If false, uses a Discrete Cosine Transform (DCT)
/// assuming reflective boundaries.
/// @param max_iterations Maximum number of iterations.
/// @param epsilon Tolerance for ending relaxation.
/// @param sigma Standard deviation of the Gaussian low-pass filter in
/// pixel units. Controls the smoothness of the fill.
/// @param num_threads The number of threads to use for the computation.
/// @return A tuple containing the number of iterations performed and the
/// maximum residual value.
template <typename Type>
auto fft_inpaint(Eigen::Ref<RowMajorMatrix<Type>> &grid,
                 const FirstGuess first_guess, const bool is_circle,
                 const size_t max_iterations, const Type epsilon,
                 const Type sigma, size_t num_threads)
    -> std::tuple<size_t, Type> {
  // If the grid doesn't have an undefined value, this routine has nothing more
  // to do.
  if (!grid.hasNaN()) {
    return std::make_tuple(0, Type(0));
  }

  // Calculation of the maximum number of threads if the user chooses.
  if (num_threads == 0) {
    num_threads = std::thread::hardware_concurrency();
  }

  // Calculation of the position of the undefined values on the grid.
  auto mask = Matrix<bool>(grid.array().isNaN());

  // Keep a copy of the original non-NaN values BEFORE modifying the grid
  auto original_values = RowMajorMatrix<Type>(grid);

  // Replace NaN in original_values with 0 to avoid issues in select operations
  // These values won't be used (we only select from non-NaN positions)
  original_values = (mask.array()).select(0, original_values);

  // Calculation of the first guess with the chosen method
  switch (first_guess) {
    case FirstGuess::kZero:
      grid = (mask.array()).select(0, grid);
      break;
    case FirstGuess::kZonalAverage:
      set_zonal_average<Type>(grid, mask, num_threads);
      break;
    default:
      throw std::invalid_argument("Invalid guess type: " +
                                  std::to_string(first_guess));
  }
  const auto x_size = grid.rows();
  const auto y_size = grid.cols();
  const Type sigma_sq = 2.0 * sigma * sigma;

  // Workspaces and transform objects
  std::optional<detail::math::FFT2D<Type>> fft;
  std::optional<detail::math::DCT2D<Type>> dct;

  RowMajorComplexMatrix<Type> grid_fft;
  RowMajorComplexMatrix<Type> kernel_fft;
  RowMajorMatrix<Type> kernel_dct;

  if (is_circle) {
    // Periodic Case: FFT
    fft.emplace(x_size, y_size);
    grid_fft.resize(fft->rows(), fft->c_cols());
    kernel_fft.resize(fft->rows(), fft->c_cols());

    // Pre-compute FFT Kernel (Gaussian in frequency space)
    for (int64_t ix = 0; ix < kernel_fft.rows(); ++ix) {
      // Frequency u (with shift for FFT)
      const Type u = (ix < x_size / 2) ? ix : (ix - x_size);
      for (int64_t iy = 0; iy < kernel_fft.cols(); ++iy) {
        // Frequency v (no shift for R2C complex layout)
        const Type v = iy;
        const Type dist_sq = (u * u) + (v * v);
        kernel_fft(ix, iy) =
            std::complex<Type>(std::exp(-dist_sq / sigma_sq), 0.0);
      }
    }
  } else {
    // Non-Periodic Case: DCT
    dct.emplace(x_size, y_size);
    kernel_dct.resize(x_size, y_size);

    // Pre-compute DCT Kernel (Gaussian in frequency space)
    for (int64_t ix = 0; ix < kernel_dct.rows(); ++ix) {
      // Frequency u (no shift for DCT)
      const Type u = ix;
      for (int64_t iy = 0; iy < kernel_dct.cols(); ++iy) {
        // Frequency v (no shift for DCT)
        const Type v = iy;
        const Type dist_sq = (u * u) + (v * v);
        kernel_dct(ix, iy) = std::exp(-dist_sq / sigma_sq);
      }
    }
  }

  // Initialization of the function results.
  size_t iteration = 0;
  Type max_residual = 0;
  auto previous_grid = RowMajorMatrix<Type>(grid);

  for (; iteration < max_iterations; ++iteration) {
    if (is_circle) {
      // FFT Path
      fft->forward(grid, grid_fft, num_threads);
      grid_fft.array() *= kernel_fft.array();
      fft->inverse(grid_fft, grid, num_threads);
    } else {
      // DCT Path
      dct->forward(grid, num_threads);  // In-place
      grid.array() *= kernel_dct.array();
      dct->inverse(grid, num_threads);  // In-place (and normalized)
    }

    // In-painting Step: Re-apply known values
    grid = (mask.array()).select(grid, original_values);

    // Check Convergence
    auto diff = (grid.array() - previous_grid.array()).abs();
    // Select residual only from undefined pixels
    auto masked_diff = (mask.array()).select(diff, 0.0);
    max_residual = masked_diff.maxCoeff();

    if (max_residual < epsilon) {
      ++iteration;  // Count this iteration before breaking
      break;
    }
    previous_grid = grid;
  }

  return std::make_tuple(iteration, max_residual);
}

}  // namespace pyinterp::fill
