// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <concepts>
#include <stdexcept>
#include <tuple>

#include "pyinterp/config/fill.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/fill/boundary.hpp"
#include "pyinterp/fill/gauss_seidel.hpp"
#include "pyinterp/fill/helpers.hpp"

namespace pyinterp::fill {
namespace detail {

/// @brief Number of iterations used to solve on the coarsest grid level.
constexpr size_t kCoarseSolveIterations = 50;

/// @brief Computes the residual r = f - Au for the Poisson equation.
///
/// For the discrete Laplacian with 5-point stencil:
///   Au = u(i,j) - 0.25 * [u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1)]
///
/// @tparam Type Data type (must be floating point).
/// @tparam GridType Type of the input grid.
/// @tparam RhsType Type of the right-hand side.
/// @param[out] residual Output residual grid.
/// @param[in] grid Input grid (current solution u).
/// @param[in] rhs Right-hand side (f in Au = f).
/// @param[in] mask Mask indicating known (false) and unknown (true) points.
/// @param[in] is_periodic Whether the grid is periodic in the x-direction.
template <std::floating_point Type, typename GridType, typename RhsType>
void compute_residual(Matrix<Type>& residual, const GridType& grid,
                      const RhsType& rhs, const Matrix<bool>& mask,
                      const bool is_periodic) {
  const auto x_size = grid.rows();
  const auto y_size = grid.cols();
  constexpr auto scale = Type(0.25);

  fill::DynamicNeighbors nbr(x_size, y_size, is_periodic, false);

  for (int64_t ix = 0; ix < x_size; ++ix) {
    nbr.update_x(ix);

    for (int64_t iy = 0; iy < y_size; ++iy) {
      if (mask(ix, iy)) {
        nbr.update_y(iy);

        // Au = u - 0.25 * (sum of neighbors)
        const Type Au =
            grid(ix, iy) - scale * (grid(nbr.ix0, iy) + grid(nbr.ix1, iy) +
                                    grid(ix, nbr.iy0) + grid(ix, nbr.iy1));
        // r = f - Au
        residual(ix, iy) = rhs(ix, iy) - Au;
      } else {
        residual(ix, iy) = Type(0);  // No residual at known points
      }
    }
  }
}

/// @brief Restricts a grid from fine (N) to coarse (N/2) using full weighting.
///
/// Uses 2x2 box averaging. For odd-sized grids, the boundary helper ensures
/// proper handling of edge cells via reflection.
///
/// @tparam Type Data type (must be floating point).
/// @tparam GridType Type of the input grid.
/// @param[out] coarse_grid Output coarse grid.
/// @param[in] fine_grid Input fine grid.
/// @param[in] is_periodic Whether the grid is circular in the x-direction.
template <std::floating_point Type, typename GridType>
void restrict_grid(Matrix<Type>& coarse_grid, const GridType& fine_grid,
                   const bool is_periodic) {
  const auto coarse_x = coarse_grid.rows();
  const auto coarse_y = coarse_grid.cols();
  const auto fine_x = fine_grid.rows();
  const auto fine_y = fine_grid.cols();

  constexpr Type scale = Type(0.25);

  // Create boundary helper for fine grid indices
  fill::DynamicNeighbors fine_nbr(fine_x, fine_y, is_periodic, false);

  for (int64_t ix = 0; ix < coarse_x; ++ix) {
    // Update x-neighbor once per row (for 2*ix index in fine grid)
    fine_nbr.update_x(2 * ix);
    const auto jx0 = 2 * ix;
    const auto jx1 = fine_nbr.ix1;

    for (int64_t iy = 0; iy < coarse_y; ++iy) {
      // Update y-neighbor for current cell (for 2*iy index in fine grid)
      fine_nbr.update_y(2 * iy);
      const auto jy0 = 2 * iy;
      const auto jy1 = fine_nbr.iy1;

      // 2x2 box average (full weighting)
      coarse_grid(ix, iy) = scale * (fine_grid(jx0, jy0) + fine_grid(jx1, jy0) +
                                     fine_grid(jx0, jy1) + fine_grid(jx1, jy1));
    }
  }
}

/// @brief Restricts the mask from fine to coarse.
///
/// A coarse pixel is masked if any of its 4 fine children are masked.
///
/// @param[out] coarse_mask Output coarse mask.
/// @param[in] fine_mask Input fine mask.
/// @param[in] is_periodic Whether the grid is periodic in the x-direction.
inline void restrict_mask(Matrix<bool>& coarse_mask,
                          const Matrix<bool>& fine_mask,
                          const bool is_periodic) {
  const auto coarse_x = coarse_mask.rows();
  const auto coarse_y = coarse_mask.cols();
  const auto fine_x = fine_mask.rows();
  const auto fine_y = fine_mask.cols();

  // Create boundary helper once (for fine grid indices)
  fill::DynamicNeighbors fine_nbr(fine_x, fine_y, is_periodic, false);

  for (int64_t ix = 0; ix < coarse_x; ++ix) {
    // Update x-neighbor once per row (for 2*ix index in fine grid)
    fine_nbr.update_x(2 * ix);
    const auto jx0 = 2 * ix;
    const auto jx1 = fine_nbr.ix1;

    for (int64_t iy = 0; iy < coarse_y; ++iy) {
      // Update y-neighbor for current cell (for 2*iy index in fine grid)
      fine_nbr.update_y(2 * iy);
      const auto jy0 = 2 * iy;
      const auto jy1 = fine_nbr.iy1;

      coarse_mask(ix, iy) = fine_mask(jx0, jy0) || fine_mask(jx1, jy0) ||
                            fine_mask(jx0, jy1) || fine_mask(jx1, jy1);
    }
  }
}

/// @brief Prolongates a grid from coarse (N/2) to fine (N) using bilinear
/// interpolation.
///
/// @tparam Type Data type (must be floating point).
/// @tparam GridType Type of the input grid.
/// @param[out] fine_grid Output fine grid (values are added, not replaced).
/// @param[in] coarse_grid Input coarse grid.
/// @param[in] is_periodic Whether the grid is periodic in the x-direction.
template <std::floating_point Type, typename GridType>
void prolong_grid(Matrix<Type>& fine_grid, const GridType& coarse_grid,
                  const bool is_periodic) {
  const auto fine_x = fine_grid.rows();
  const auto fine_y = fine_grid.cols();
  const auto coarse_x = coarse_grid.rows();
  const auto coarse_y = coarse_grid.cols();

  // Create boundary helpers once
  fill::DynamicNeighbors coarse_nbr(coarse_x, coarse_y, is_periodic, false);
  fill::DynamicNeighbors fine_nbr(fine_x, fine_y, is_periodic, false);

  for (int64_t ix = 0; ix < coarse_x; ++ix) {
    // Update x-neighbors once per row
    coarse_nbr.update_x(ix);
    fine_nbr.update_x(2 * ix);

    const auto jx0 = 2 * ix;
    const auto jx1 = fine_nbr.ix1;

    for (int64_t iy = 0; iy < coarse_y; ++iy) {
      // Update y-neighbors for current cell
      coarse_nbr.update_y(iy);
      fine_nbr.update_y(2 * iy);

      // Get coarse grid values at 2x2 stencil
      const auto c00 = coarse_grid(ix, iy);
      const auto c10 = coarse_grid(coarse_nbr.ix1, iy);
      const auto c01 = coarse_grid(ix, coarse_nbr.iy1);
      const auto c11 = coarse_grid(coarse_nbr.ix1, coarse_nbr.iy1);

      // Get fine grid indices
      const auto jy0 = 2 * iy;
      const auto jy1 = fine_nbr.iy1;

      // Bilinear interpolation weights
      fine_grid(jx0, jy0) += c00;
      fine_grid(jx1, jy0) += (c00 + c10) * Type(0.5);
      fine_grid(jx0, jy1) += (c00 + c01) * Type(0.5);
      fine_grid(jx1, jy1) += (c00 + c10 + c01 + c11) * Type(0.25);
    }
  }
}

/// @brief Performs smoothing passes using Gauss-Seidel with optional
/// relaxation for the equation Au = f.
///
/// @tparam Type Data type (must be floating point).
/// @tparam GridType Type of the grid to be updated.
/// @tparam RhsType Type of the right-hand side.
/// @param[out] grid The grid to be updated.
/// @param[in] rhs The right-hand side (f in Au = f).
/// @param[in] mask Boolean mask indicating known (false) and unknown (true)
/// points.
/// @param[in] is_periodic Whether the grid is periodic in the x-direction.
/// @param[in] relaxation Relaxation factor (1.0 = no relaxation).
/// @param[in] num_sweeps Number of Gauss-Seidel sweeps to perform.
/// @param[in] num_threads The number of threads to use.
template <std::floating_point Type, typename GridType, typename RhsType>
void smooth(GridType& grid, const RhsType& rhs, const Matrix<bool>& mask,
            const bool is_periodic, const Type relaxation,
            const size_t num_sweeps, const size_t num_threads) {
  // Use the shared red-black Gauss-Seidel implementation
  for (size_t sweep = 0; sweep < num_sweeps; ++sweep) {
    detail::gauss_seidel_core<Type>(grid, rhs, mask, is_periodic, relaxation,
                                    num_threads);
  }
}

/// @brief Recursive V-Cycle for Multigrid solving Au = f.
///
/// @tparam Type Data type (must be floating point).
/// @tparam GridType Type of the grid to be updated.
/// @tparam RhsType Type of the right-hand side.
/// @param[out] grid The grid to be updated.
/// @param[in] rhs The right-hand side (f in Au = f).
/// @param[in] mask Boolean mask indicating known (false) and unknown (true)
/// points.
/// @param[in] is_periodic Whether the grid is periodic in the x-direction.
/// @param[in] relaxation Relaxation factor (1.0 = no relaxation).
/// @param[in] pre_smooth Number of pre-smoothing iterations per level.
/// @param[in] post_smooth Number of post-smoothing iterations per level.
/// @param[in] min_size Minimum grid size to stop recursion.
/// @param[in] num_threads The number of threads to use.
template <std::floating_point Type, typename GridType, typename RhsType>
void v_cycle(GridType& grid, const RhsType& rhs, const Matrix<bool>& mask,
             const bool is_periodic, const Type relaxation,
             const size_t pre_smooth, const size_t post_smooth,
             const int64_t min_size, const size_t num_threads) {
  const auto x_size = grid.rows();
  const auto y_size = grid.cols();

  // Base case: solve directly on coarsest grid
  if (x_size <= min_size || y_size <= min_size) {
    smooth<Type>(grid, rhs, mask, is_periodic, Type(1.0),
                 kCoarseSolveIterations, num_threads);
    return;
  }

  // 1. Pre-smoothing
  smooth<Type>(grid, rhs, mask, is_periodic, relaxation, pre_smooth,
               num_threads);

  // 2. Compute residual: r = f - Au
  Matrix<Type> residual(x_size, y_size);
  compute_residual<Type>(residual, grid, rhs, mask, is_periodic);

  // 3. Restrict residual and mask to coarse grid
  const auto coarse_x = (x_size + 1) / 2;
  const auto coarse_y = (y_size + 1) / 2;
  Matrix<Type> coarse_rhs(coarse_x, coarse_y);
  Matrix<bool> coarse_mask(coarse_x, coarse_y);
  restrict_grid<Type>(coarse_rhs, residual, is_periodic);
  restrict_mask(coarse_mask, mask, is_periodic);

  // 4. Solve error equation on coarse grid: A*e = r
  Matrix<Type> coarse_error = Matrix<Type>::Zero(coarse_x, coarse_y);
  v_cycle<Type>(coarse_error, coarse_rhs, coarse_mask, is_periodic, relaxation,
                pre_smooth, post_smooth, min_size, num_threads);

  // 5. Prolongate and correct (only at masked points)
  Matrix<Type> fine_error = Matrix<Type>::Zero(x_size, y_size);
  prolong_grid<Type>(fine_error, coarse_error, is_periodic);
  grid = mask.array().select(grid.array() + fine_error.array(), grid);

  // 6. Post-smoothing
  smooth<Type>(grid, rhs, mask, is_periodic, relaxation, post_smooth,
               num_threads);
}

}  // namespace detail

/// Replaces all undefined values (NaN) in a grid using the Multigrid V-cycle
/// method.
///
/// Solves the homogeneous Laplace equation (f = 0) at masked points using
/// geometric multigrid with V-cycles. The solution smoothly interpolates
/// the known boundary values into the unknown regions.
///
/// @tparam Type Data type (must be floating point).
/// @param[in,out] grid The grid to be processed.
/// @param[in] config Multigrid configuration.
/// @return A tuple containing the number of iterations performed and the
/// maximum residual value.
template <std::floating_point Type>
[[nodiscard]] auto multigrid(EigenDRef<RowMajorMatrix<Type>> grid,
                             const config::fill::Multigrid& config)
    -> std::tuple<size_t, Type> {
  // Early return if no NaN values
  if (!grid.hasNaN()) {
    return std::make_tuple(0, Type(0));
  }

  // Identify masked (NaN) locations
  Matrix<bool> mask = grid.array().isNaN();

  // Set initial guess
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

  // RHS is zero for fill problems (homogeneous Laplace equation)
  Matrix<Type> rhs = Matrix<Type>::Zero(grid.rows(), grid.cols());

  // Multigrid parameters
  constexpr int64_t min_size = 4;
  const Type relaxation = Type(1.0);  // SOR parameter (1.0 = Gauss-Seidel)
  size_t iteration = 0;
  Type max_residual = std::numeric_limits<Type>::max();

  // V-cycle iterations
  for (size_t it = 0; it < config.max_iterations(); ++it) {
    ++iteration;

    detail::v_cycle<Type>(grid, rhs, mask, config.is_periodic(), relaxation,
                          config.pre_smooth(), config.post_smooth(), min_size,
                          config.num_threads());

    // Check convergence
    Matrix<Type> residual(grid.rows(), grid.cols());
    detail::compute_residual<Type>(residual, grid, rhs, mask,
                                   config.is_periodic());
    max_residual = residual.array().abs().maxCoeff();

    if (max_residual < config.epsilon()) {
      break;
    }
  }

  return std::make_tuple(iteration, max_residual);
}

}  // namespace pyinterp::fill
