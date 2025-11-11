#pragma once

#include "pyinterp/eigen.hpp"
#include "pyinterp/fill/boundary.hpp"
#include "pyinterp/fill/enum.hpp"
#include "pyinterp/fill/gauss_seidel.hpp"
#include "pyinterp/fill/utils.hpp"

namespace pyinterp {
namespace detail::multigrid {

/// @brief Computes the residual (r = f - Au) for the Poisson equation.
/// For fill problems, f = 0 at masked points, so r = -Au there.
/// @tparam Type Data type.
/// @tparam GridType Type of the input grid.
/// @param[out] residual Output residual grid.
/// @param[in] grid Input grid.
/// @param[in] mask Mask indicating known (false) and unknown (true) points.
/// @param[in] is_circle Whether the grid is circular in the x-direction.
template <typename Type, typename GridType>
void compute_residual(Matrix<Type>& residual, const GridType& grid,
                      const Matrix<bool>& mask, const bool is_circle) {
  const auto x_size = grid.rows();
  const auto y_size = grid.cols();
  constexpr auto scale = Type(0.25);

  // Create boundary helper once
  fill::DynamicNeighbors nbr(x_size, y_size, is_circle, false);

  for (int64_t ix = 0; ix < x_size; ++ix) {
    // Update x-neighbors once per row
    nbr.update_x(ix);

    for (int64_t iy = 0; iy < y_size; ++iy) {
      if (mask(ix, iy)) {
        // Update y-neighbors for current cell
        nbr.update_y(iy);

        // Residual: r = -Au = u - 0.25 * (sum of neighbors)
        residual(ix, iy) =
            grid(ix, iy) - scale * (grid(nbr.ix0, iy) + grid(nbr.ix1, iy) +
                                    grid(ix, nbr.iy0) + grid(ix, nbr.iy1));
      } else {
        residual(ix, iy) = Type(0);  // No residual at known points
      }
    }
  }
}

/// @brief Restricts a grid from fine (N) to coarse (N/2) using full weighting.
/// @tparam Type Data type.
/// @tparam GridType Type of the input grid.
/// @param[out] coarse_grid Output coarse grid.
/// @param[in] fine_grid Input fine grid.
/// @param[in] is_circle Whether the grid is circular in the x-direction.
template <typename Type, typename GridType>
void restrict_grid(Matrix<Type>& coarse_grid, const GridType& fine_grid,
                   const bool is_circle) {
  const auto coarse_x = coarse_grid.rows();
  const auto coarse_y = coarse_grid.cols();
  const auto fine_x = fine_grid.rows();
  const auto fine_y = fine_grid.cols();

  constexpr Type scale = Type(0.25);

  // Create boundary helper once (for fine grid indices)
  fill::DynamicNeighbors fine_nbr(fine_x, fine_y, is_circle, false);

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
/// A coarse pixel is masked if any of its 4 fine children are masked.
/// @param[out] coarse_mask Output coarse mask.
/// @param[in] fine_mask Input fine mask.
/// @param[in] is_circle Whether the grid is circular in the x-direction.
inline void restrict_mask(Matrix<bool>& coarse_mask,
                          const Matrix<bool>& fine_mask, const bool is_circle) {
  const auto coarse_x = coarse_mask.rows();
  const auto coarse_y = coarse_mask.cols();
  const auto fine_x = fine_mask.rows();
  const auto fine_y = fine_mask.cols();

  // Create boundary helper once (for fine grid indices)
  fill::DynamicNeighbors fine_nbr(fine_x, fine_y, is_circle, false);

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
/// @tparam Type Data type.
/// @tparam GridType Type of the input grid.
/// @param[out] fine_grid Output fine grid.
/// @param[in] coarse_grid Input coarse grid.
/// @param[in] is_circle Whether the grid is circular in the x-direction.
template <typename Type, typename GridType>
void prolong_grid(Matrix<Type>& fine_grid, const GridType& coarse_grid,
                  const bool is_circle) {
  const auto fine_x = fine_grid.rows();
  const auto fine_y = fine_grid.cols();
  const auto coarse_x = coarse_grid.rows();
  const auto coarse_y = coarse_grid.cols();

  // Create boundary helpers once
  fill::DynamicNeighbors coarse_nbr(coarse_x, coarse_y, is_circle, false);
  fill::DynamicNeighbors fine_nbr(fine_x, fine_y, is_circle, false);

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

/// @brief Performs one smoothing pass using Gauss-Seidel with optional
/// relaxation for the equation Au = f.
/// @tparam Type Data type
/// @tparam GridType Type of the grid to be updated
/// @tparam RhsType Type of the right-hand side
/// @tparam MaskType Type of the boolean mask
/// @param[out] grid The grid to be updated
/// @param[in] rhs The right-hand side (f in Au = f). Use zeros for homogeneous
/// case.
/// @param[in] mask Boolean mask indicating known (false) and unknown (true)
/// points.
/// @param[in] is_circle Whether the grid is circular in the x-direction.
/// @param[in] relaxation Relaxation factor (1.0 = no relaxation).
/// @param[in] num_sweeps Number of Gauss-Seidel sweeps to perform.
/// @param[in] num_threads The number of threads to use.
template <typename Type, typename GridType, typename RhsType, typename MaskType>
void smooth(GridType& grid, const RhsType& rhs, const MaskType& mask,
            const bool is_circle, const Type relaxation,
            const size_t num_sweeps, const size_t num_threads) {
  // Use the shared red-black Gauss-Seidel implementation
  for (size_t sweep = 0; sweep < num_sweeps; ++sweep) {
    detail::gauss_seidel_core<Type>(grid, rhs, mask, is_circle, relaxation,
                                    num_threads);
  }
}

/// @brief Recursive V-Cycle for Multigrid solving Au = f.
/// @tparam Type Data type
/// @tparam GridType Type of the grid to be updated
/// @tparam RhsType Type of the right-hand side
/// @tparam MaskType Type of the boolean mask
/// @param[out] grid The grid to be updated
/// @param[in] rhs The right-hand side (f in Au = f). Use zeros for homogeneous
/// case.
/// @param[in] mask Boolean mask indicating known (false) and unknown (true)
/// points.
/// @param[in] is_circle Whether the grid is circular in the x-direction.
/// @param[in] relaxation Relaxation factor (1.0 = no relaxation).
/// @param[in] pre_smooth Number of pre-smoothing iterations per level.
/// @param[in] post_smooth Number of post-smoothing iterations per level.
/// @param[in] min_size Minimum grid size to stop recursion.
/// @param[in] num_threads The number of threads to use.
template <typename Type, typename GridType, typename RhsType, typename MaskType>
void v_cycle(GridType& grid, const RhsType& rhs, const MaskType& mask,
             const bool is_circle, const Type relaxation,
             const size_t pre_smooth, const size_t post_smooth,
             const int min_size, const size_t num_threads) {
  const auto x_size = grid.rows();
  const auto y_size = grid.cols();

  // Base case: solve directly on coarsest grid
  if (x_size <= min_size || y_size <= min_size) {
    smooth<Type, GridType, RhsType, MaskType>(grid, rhs, mask, is_circle,
                                              Type(1.0), 50, num_threads);
    return;
  }

  // 1. Pre-smoothing
  smooth<Type, GridType, RhsType, MaskType>(
      grid, rhs, mask, is_circle, relaxation, pre_smooth, num_threads);

  // 2. Compute residual: r = f - Au
  Matrix<Type> residual(x_size, y_size);
  compute_residual(residual, grid, mask, is_circle);
  // Add back the RHS: r = f - Au, but compute_residual gives -Au
  for (int64_t ix = 0; ix < x_size; ++ix) {
    for (int64_t iy = 0; iy < y_size; ++iy) {
      if (mask(ix, iy)) {
        residual(ix, iy) = rhs(ix, iy) - residual(ix, iy);
      }
    }
  }

  // 3. Restrict residual and mask to coarse grid
  const auto coarse_x = (x_size + 1) / 2;
  const auto coarse_y = (y_size + 1) / 2;
  Matrix<Type> coarse_rhs(coarse_x, coarse_y);
  Matrix<bool> coarse_mask(coarse_x, coarse_y);
  restrict_grid(coarse_rhs, residual, is_circle);
  restrict_mask(coarse_mask, mask, is_circle);

  // 4. Solve error equation on coarse grid: A*e = r
  Matrix<Type> coarse_error = Matrix<Type>::Zero(coarse_x, coarse_y);
  v_cycle<Type>(coarse_error, coarse_rhs, coarse_mask, is_circle, relaxation,
                pre_smooth, post_smooth, min_size, num_threads);

  // 5. Prolongate and correct
  Matrix<Type> fine_error(x_size, y_size);
  fine_error.setZero();
  prolong_grid(fine_error, coarse_error, is_circle);
  // Only apply correction to masked (undefined) points
  grid = mask.array().select(grid.array() + fine_error.array(), grid);

  // 6. Post-smoothing
  smooth<Type, GridType, RhsType, MaskType>(
      grid, rhs, mask, is_circle, relaxation, post_smooth, num_threads);
}

}  // namespace detail::multigrid

namespace fill {

/// Replaces all undefined values (NaN) in a grid using the Multigrid V-cycle
/// method.
///
/// @tparam Type Data type.
/// @param grid The grid to be processed
/// @param first_guess Method to use for the first guess.
/// @param is_circle True if the X axis of the grid defines a circle.
/// @param max_iterations Maximum number of V-Cycles to perform.
/// @param epsilon Tolerance for convergence.
/// @param pre_smooth Number of pre-smoothing iterations per level.
/// @param post_smooth Number of post-smoothing iterations per level.
/// @param num_threads The number of threads to use (currently not parallelized
/// in multigrid).
/// @return A tuple containing the number of iterations performed and the
/// maximum residual value.
template <typename Type>
auto multigrid(EigenDRef<RowMajorMatrix<Type>>& grid,
               const FirstGuess first_guess, const bool is_circle,
               const size_t max_iterations, const Type epsilon,
               const size_t pre_smooth = 2, const size_t post_smooth = 2,
               size_t num_threads = 0) -> std::tuple<size_t, Type> {
  // Early return if no NaN values
  if (!grid.hasNaN()) {
    return std::make_tuple(0, Type(0));
  }

  // Thread count determination
  if (num_threads == 0) {
    num_threads = std::thread::hardware_concurrency();
  }

  // Identify masked (NaN) locations
  auto mask = Matrix<bool>(grid.array().isNaN());

  // Set initial guess
  switch (first_guess) {
    case FirstGuess::kZero:
      grid = (mask.array()).select(Type(0), grid);
      break;
    case FirstGuess::kZonalAverage:
      set_zonal_average<Type>(grid, mask, num_threads);
      break;
    default:
      throw std::invalid_argument("Invalid guess type: " +
                                  std::to_string(first_guess));
  }

  // RHS is zero for fill problems (homogeneous Laplace equation at masked
  // points)
  Matrix<Type> rhs = Matrix<Type>::Zero(grid.rows(), grid.cols());

  // Multigrid parameters
  const int min_size = 4;
  const Type relaxation = Type(1.0);  // SOR parameter (1.0 = Gauss-Seidel)
  size_t iteration = 0;
  Type max_residual = std::numeric_limits<Type>::max();

  // V-cycle iterations
  for (size_t it = 0; it < max_iterations; ++it) {
    ++iteration;

    detail::multigrid::v_cycle<Type, EigenDRef<RowMajorMatrix<Type>>,
                               Matrix<Type>, Matrix<bool>>(
        grid, rhs, mask, is_circle, relaxation, pre_smooth, post_smooth,
        min_size, num_threads);

    // Check convergence
    Matrix<Type> residual(grid.rows(), grid.cols());
    detail::multigrid::compute_residual<Type, EigenDRef<RowMajorMatrix<Type>>>(
        residual, grid, mask, is_circle);
    max_residual = residual.array().abs().maxCoeff();

    if (max_residual < epsilon) {
      break;
    }
  }

  return std::make_tuple(iteration, max_residual);
}

}  // namespace fill
}  // namespace pyinterp
