// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <cmath>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "pyinterp/config/fill.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/fill/boundary.hpp"
#include "pyinterp/fill/helpers.hpp"

namespace pyinterp::fill {
namespace detail {

/// @brief Performs red-black Gauss-Seidel smoothing with optional RHS.
///
/// @tparam Type Data type
/// @tparam GridType Type of the grid to be updated
/// @tparam RhsType Type of the right-hand side
/// @tparam MaskType Type of the boolean mask
/// @param grid The grid to be updated
/// @param rhs The right-hand side (f in Au = f). Use zeros for homogeneous
/// case.
/// @param mask Boolean mask indicating which cells to update
/// @param is_periodic True if the X axis defines a periodic boundary condition
/// @param relaxation SOR relaxation parameter (1.0 = pure Gauss-Seidel)
/// @param num_threads Number of threads for parallel execution
/// @return Maximum residual value across all updated cells
template <typename Type, typename GridType, typename RhsType, typename MaskType>
auto gauss_seidel_core(GridType &grid, const RhsType &rhs, const MaskType &mask,
                       const bool is_periodic, const Type relaxation,
                       const size_t num_threads) -> Type {
  // Shape of the grid
  auto x_size = grid.rows();
  auto y_size = grid.cols();

  // Captures the detected exceptions in the calculation function
  // (only the last exception captured is kept)
  auto except = std::exception_ptr(nullptr);

  // Thread worker responsible for processing red or black cells.
  // @param y_start First index y of the band to be processed.
  // @param y_end Last index y, excluded, of the band to be processed.
  // @param red_black Whether the band is red or black.
  // @param max_residual Maximum residual of this strip.
  auto worker = [&](const int64_t y_start, const int64_t y_end,
                    const int red_black, Type *max_residual) -> void {
    // Update the cell grid[ix, iy] using the Gauss-Seidel method.
    //
    // @param nbr Neighbor indices structure
    // @param ix Index x of the pixel to be modified.
    // @param iy Index y of the pixel to be modified.
    auto cell_fill = [&grid, &rhs, &relaxation, &max_residual](
                         const fill::DynamicNeighbors &nbr, const int64_t ix,
                         const int64_t iy) {
      auto &cell = grid(ix, iy);
      // For Au = f: u_new = 0.25 * (neighbors + f)
      // residual = (u_new - u_old) is the correction applied
      auto residual =
          (Type(0.25) * (grid(nbr.ix0, iy) + grid(nbr.ix1, iy) +
                         grid(ix, nbr.iy0) + grid(ix, nbr.iy1) + rhs(ix, iy)) -
           cell) *
          relaxation;
      cell += residual;
      *max_residual = std::max(*max_residual, std::fabs(residual));
    };

    // Initialization of the maximum value of the residuals of the processed
    // thread.
    *max_residual = Type(0);

    try {
      // Create boundary helper once for this thread
      fill::DynamicNeighbors nbr(x_size, y_size, is_periodic, false);

      for (auto ix = 0; ix < x_size; ++ix) {
        // Update x-neighbors once per row
        nbr.update_x(ix);

        for (auto iy = y_start; iy < y_end; ++iy) {
          if (mask(ix, iy) && ((ix + iy) % 2) == red_black) {
            // Update y-neighbors for current cell
            nbr.update_y(iy);
            cell_fill(nbr, ix, iy);
          }
        }
      }
    } catch (...) {
      except = std::current_exception();
    }
  };

  if (num_threads == 1) {
    // Single thread processing.
    //
    // @param red_black Whether the band is red or black.
    // @return Maximum residual of this strip.
    auto calculate = [&](int red_black) -> Type {
      auto max_residual = Type(0);
      worker(0, y_size, red_black, &max_residual);
      if (except != nullptr) {
        std::rethrow_exception(except);
      }
      return max_residual;
    };
    return std::max(calculate(0), calculate(1));
  }

  // Launches the threads for a red or black band.
  //
  // @param red_black Whether the band is red or black.
  // @return The maximum residual of the processed band.
  auto calculate = [&](const int red_black) -> Type {
    int64_t start = 0;
    int64_t shift = y_size / num_threads;

    // Handled threads
    std::vector<std::thread> threads;

    // Maximum residual values for each thread.
    std::vector<Type> max_residuals(num_threads);

    for (size_t index = 0; index < num_threads - 1; ++index) {
      threads.emplace_back(std::thread(worker, start, start + shift, red_black,
                                       &max_residuals[index]));
      start += shift;
    }
    threads.emplace_back(std::thread(worker, start, y_size, red_black,
                                     &max_residuals[num_threads - 1]));
    for (auto &&item : threads) {
      item.join();
    }
    if (except != nullptr) {
      std::rethrow_exception(except);
    }
    return *std::max_element(max_residuals.begin(), max_residuals.end());
  };
  return std::max(calculate(0), calculate(1));
}

/// @brief Wrapper for backward compatibility: Gauss-Seidel for homogeneous case
/// (f=0).
template <typename Type>
auto gauss_seidel(EigenDRef<RowMajorMatrix<Type>> grid, Matrix<bool> &mask,
                  const bool is_periodic, const Type relaxation,
                  const size_t num_threads) -> Type {
  // Create zero RHS for homogeneous case
  Matrix<Type> rhs = Matrix<Type>::Zero(grid.rows(), grid.cols());
  return gauss_seidel_core<Type>(grid, rhs, mask, is_periodic, relaxation,
                                 num_threads);
}

}  // namespace detail

/// Replaces all undefined values (NaN) in a grid using the Gauss-Seidel
/// method by relaxation.
///
/// @param[in,out] grid The grid to be processed
/// @param[in] config Configuration parameters for the Gauss-Seidel method.
/// @return A tuple containing the number of iterations performed and the
/// maximum residual value.
template <typename Type>
[[nodiscard]] auto gauss_seidel(EigenDRef<RowMajorMatrix<Type>> grid,
                                const config::fill::GaussSeidel &config)
    -> std::tuple<size_t, Type> {
  /// If the grid doesn't have an undefined value, this routine has nothing more
  /// to do.
  if (!grid.hasNaN()) {
    return std::make_tuple(0, Type(0));
  }

  /// Calculation of the position of the undefined values on the grid.
  auto mask = Matrix<bool>(grid.array().isNaN());

  /// Calculation of the first guess with the chosen method
  switch (config.first_guess()) {
    case config::fill::FirstGuess::kZero:
      grid = (mask.array()).select(0, grid);
      break;
    case config::fill::FirstGuess::kZonalAverage:
      set_zonal_average<Type>(grid, mask, config.num_threads());
      break;
    default:
      throw std::invalid_argument(
          "Unsupported first guess method: " +
          std::to_string(static_cast<int>(config.first_guess())));
  }

  // Initialization of the function results.
  size_t iteration = 0;
  Type max_residual = 0;

  for (size_t it = 0; it < config.max_iterations(); ++it) {
    ++iteration;
    max_residual = detail::gauss_seidel<Type>(
        grid, mask, config.is_periodic(),
        static_cast<Type>(config.relaxation()), config.num_threads());
    if (max_residual < static_cast<Type>(config.epsilon())) {
      break;
    }
  }
  return std::make_tuple(iteration, max_residual);
}

}  // namespace pyinterp::fill
