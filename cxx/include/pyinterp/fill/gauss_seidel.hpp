#pragma once

#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>

#include "pyinterp/eigen.hpp"
#include "pyinterp/fill/enum.hpp"
#include "pyinterp/fill/utils.hpp"

namespace pyinterp {
namespace detail {

///  Replaces all undefined values (NaN) in a grid using the Gauss-Seidel
///  method by relaxation.
///
/// @param grid The grid to be processed
/// @param is_circle True if the X axis of the grid defines a circle.
/// @param relaxation Relaxation constant
/// @return maximum residual value
template <typename Type>
auto gauss_seidel(EigenDRef<RowMajorMatrix<Type>> &grid, Matrix<bool> &mask,
                  const bool is_circle, const Type relaxation,
                  const size_t num_threads) -> Type {
  // Shape of the grid
  auto x_size = grid.rows();
  auto y_size = grid.cols();

  // Captures the detected exceptions in the calculation function
  // (only the last exception captured is kept)
  auto except = std::exception_ptr(nullptr);

  // Thread worker responsible for processing red or black cells.
  //
  // @param y_start First index y of the band to be processed.
  // @param y_end Last index y, excluded, of the band to be processed.
  // @param red_black Whether the band is red or black.
  // @param max_residual Maximum residual of this strip.
  auto worker = [&](const int64_t y_start, const int64_t y_end,
                    const int red_black, Type *max_residual) -> void {
    // Update the cell grid[ix, iy] using the Gauss-Seidel method.
    //
    // @param ix0 ix - 1
    // @param ix Index of the pixel to be modified.
    // @param ix1 ix + 1
    // @param iy0 iy - 1
    // @param iy Index of the pixel to be modified.
    // @param iy1 iy + 1
    auto cell_fill = [&grid, &relaxation, &max_residual](
                         const int64_t ix0, const int64_t ix, const int64_t ix1,
                         const int64_t iy0, const int64_t iy,
                         const int64_t iy1) {
      auto &cell = grid(ix, iy);
      auto residual = (Type(0.25) * (grid(ix0, iy) + grid(ix1, iy) +
                                     grid(ix, iy0) + grid(ix, iy1)) -
                       cell) *
                      relaxation;
      cell += residual;
      *max_residual = std::max(*max_residual, std::fabs(residual));
    };

    // Initialization of the maximum value of the residuals of the processed
    // thread.
    *max_residual = Type(0);

    try {
      for (auto ix = 0; ix < x_size; ++ix) {
        auto ix0 = ix == 0 ? (is_circle ? x_size - 1 : 1) : ix - 1;
        auto ix1 = ix == x_size - 1 ? (is_circle ? 0 : x_size - 2) : ix + 1;

        for (auto iy = y_start; iy < y_end; ++iy) {
          if (mask(ix, iy) && ((ix + iy) % 2) == red_black) {
            auto iy0 = iy == 0 ? 1 : iy - 1;
            auto iy1 = iy == y_size - 1 ? y_size - 2 : iy + 1;

            cell_fill(ix0, ix, ix1, iy0, iy, iy1);
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

}  // namespace detail

namespace fill {

/// Replaces all undefined values (NaN) in a grid using the Gauss-Seidel
/// method by relaxation.
///
/// @param grid The grid to be processed
/// @param is_circle True if the X axis of the grid defines a circle.
/// @param max_iterations Maximum number of iterations to be used by relaxation.
/// @param epsilon Tolerance for ending relaxation before the maximum number of
/// iterations limit.
/// @param relaxation Relaxation constant
/// @param num_threads The number of threads to use for the computation. If 0
/// all CPUs are used. If 1 is given, no parallel computing code is used at all,
/// which is useful for debugging.
/// @return A tuple containing the number of iterations performed and the
/// maximum residual value.
template <typename Type>
auto gauss_seidel(EigenDRef<RowMajorMatrix<Type>> &grid,
                  const FirstGuess first_guess, const bool is_circle,
                  const size_t max_iterations, const Type epsilon,
                  const Type relaxation, size_t num_threads)
    -> std::tuple<size_t, Type> {
  /// If the grid doesn't have an undefined value, this routine has nothing more
  /// to do.
  if (!grid.hasNaN()) {
    return std::make_tuple(0, Type(0));
  }

  /// Calculation of the maximum number of threads if the user chooses.
  if (num_threads == 0) {
    num_threads = std::thread::hardware_concurrency();
  }

  /// Calculation of the position of the undefined values on the grid.
  auto mask = Matrix<bool>(grid.array().isNaN());

  /// Calculation of the first guess with the chosen method
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

  // Initialization of the function results.
  size_t iteration = 0;
  Type max_residual = 0;

  for (size_t it = 0; it < max_iterations; ++it) {
    ++iteration;
    max_residual = detail::gauss_seidel<Type>(grid, mask, is_circle, relaxation,
                                              num_threads);
    if (max_residual < epsilon) {
      break;
    }
  }
  return std::make_tuple(iteration, max_residual);
}

}  // namespace fill
}  // namespace pyinterp
