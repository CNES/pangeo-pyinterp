// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "pyinterp/config/fill.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/fill/boundary.hpp"
#include "pyinterp/fill/helpers.hpp"
#include "pyinterp/parallel_for.hpp"

namespace pyinterp::fill {
namespace detail {

/// @brief Unknown cells of a grid, split into the red and black colours of
/// the Gauss-Seidel ordering, with their neighbours resolved.
///
/// Building the list once replaces, on every sweep, a scan of the whole grid
/// and the evaluation of the boundary conditions for each unknown cell.
struct RedBlackCells {
  /// An unknown cell and the indexes of its four neighbours
  struct Cell {
    int64_t ix;
    int64_t iy;
    int64_t ix0;
    int64_t ix1;
    int64_t iy0;
    int64_t iy1;
  };

  /// Cells with (ix + iy) even, then odd
  std::array<std::vector<Cell>, 2> cells;

  /// @param[in] mask Boolean mask indicating which cells to update
  /// @param[in] is_periodic True if the X axis is periodic
  template <typename MaskType>
  RedBlackCells(const MaskType &mask, const bool is_periodic) {
    const auto x_size = mask.rows();
    const auto y_size = mask.cols();
    fill::DynamicNeighbors nbr(x_size, y_size, is_periodic, false);
    for (int64_t ix = 0; ix < x_size; ++ix) {
      nbr.update_x(ix);
      for (int64_t iy = 0; iy < y_size; ++iy) {
        if (mask(ix, iy)) {
          nbr.update_y(iy);
          cells[(ix + iy) % 2].push_back({.ix = ix,
                                          .iy = iy,
                                          .ix0 = nbr.ix0,
                                          .ix1 = nbr.ix1,
                                          .iy0 = nbr.iy0,
                                          .iy1 = nbr.iy1});
        }
      }
    }
  }
};

/// @brief Performs one red-black Gauss-Seidel sweep over the unknown cells.
///
/// The red cells are updated first, then the black ones. Cells of one colour
/// only read cells of the other, so each colour is split between the threads
/// without changing the result.
///
/// @tparam Type Data type
/// @tparam GridType Type of the grid to be updated
/// @tparam RhsType Type of the right-hand side, or std::nullptr_t for the
/// homogeneous case (f = 0)
/// @param grid The grid to be updated
/// @param rhs The right-hand side (f in Au = f), or nullptr
/// @param cells Unknown cells of the grid
/// @param relaxation SOR relaxation parameter (1.0 = pure Gauss-Seidel)
/// @param num_threads Number of threads for parallel execution
/// @return Maximum correction applied to a cell
template <typename Type, typename GridType, typename RhsType>
auto gauss_seidel_sweep(GridType &grid, const RhsType &rhs,
                        const RedBlackCells &cells, const Type relaxation,
                        const size_t num_threads) -> Type {
  auto max_residual = Type(0);
  auto mutex = std::mutex();

  for (const auto &colour : cells.cells) {
    parallel_for(
        static_cast<int64_t>(colour.size()),
        [&](const int64_t start, const int64_t end) {
          auto local_max = Type(0);
          for (auto ix = start; ix < end; ++ix) {
            const auto &item = colour[ix];
            // For Au = f: u_new = 0.25 * (neighbors + f)
            // residual = (u_new - u_old) is the correction applied
            auto sum = grid(item.ix0, item.iy) + grid(item.ix1, item.iy) +
                       grid(item.ix, item.iy0) + grid(item.ix, item.iy1);
            if constexpr (!std::is_same_v<RhsType, std::nullptr_t>) {
              sum += rhs(item.ix, item.iy);
            }
            auto &cell = grid(item.ix, item.iy);
            const auto residual = (Type(0.25) * sum - cell) * relaxation;
            cell += residual;
            local_max = std::max(local_max, std::fabs(residual));
          }
          auto lock = std::scoped_lock(mutex);
          max_residual = std::max(max_residual, local_max);
        },
        static_cast<int64_t>(num_threads));
  }
  return max_residual;
}

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
  return gauss_seidel_sweep<Type>(grid, rhs, RedBlackCells(mask, is_periodic),
                                  relaxation, num_threads);
}

/// @brief Wrapper for backward compatibility: Gauss-Seidel for homogeneous case
/// (f=0).
template <typename Type>
auto gauss_seidel(EigenDRef<RowMajorMatrix<Type>> grid, Matrix<bool> &mask,
                  const bool is_periodic, const Type relaxation,
                  const size_t num_threads) -> Type {
  return gauss_seidel_sweep<Type>(
      grid, nullptr, RedBlackCells(mask, is_periodic), relaxation, num_threads);
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

  const auto cells = detail::RedBlackCells(mask, config.is_periodic());
  for (size_t it = 0; it < config.max_iterations(); ++it) {
    ++iteration;
    max_residual = detail::gauss_seidel_sweep<Type>(
        grid, nullptr, cells, static_cast<Type>(config.relaxation()),
        config.num_threads());
    if (max_residual < static_cast<Type>(config.epsilon())) {
      break;
    }
  }
  return std::make_tuple(iteration, max_residual);
}

}  // namespace pyinterp::fill
