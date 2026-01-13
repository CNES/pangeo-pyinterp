// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math.hpp"
#include "pyinterp/parallel_for.hpp"

namespace pyinterp::fill {

/// @brief Calculate the zonal average in x direction.
///
/// Replaces all missing (_FillValue) values in a grid with the mean value
/// computed from valid data in the same latitude band.
///
/// @param[in] grid The grid to be processed.
/// @param[in,out] mask Matrix describing the undefined pixels of the grid.
/// @param[in] num_threads Number of threads used for the calculation.
template <typename Type, typename Derived>
void set_zonal_average(Eigen::RefBase<Derived> &grid_ref, Matrix<bool> &mask,
                       const size_t num_threads) {
  auto &grid = grid_ref.derived();
  static_assert(std::is_same_v<typename Derived::Scalar, Type>,
                "Type mismatch");

  // Captures the detected exceptions in the calculation function
  // (only the last exception captured is kept)
  auto except = std::exception_ptr(nullptr);

  parallel_for(
      grid.cols(),
      [&](size_t y_start, size_t y_end) {
        try {
          // Calculation of longitude band means.
          for (auto iy = static_cast<int64_t>(y_start);
               iy < static_cast<int64_t>(y_end); ++iy) {
            auto acc = boost::accumulators::accumulator_set<
                Type,
                boost::accumulators::stats<boost::accumulators::tag::count,
                                           boost::accumulators::tag::mean>>();
            for (int64_t ix = 0; ix < grid.rows(); ++ix) {
              if (!mask(ix, iy)) {
                acc(grid(ix, iy));
              }
            }

            // The masked value is replaced by the average of the longitude band
            // if it is defined; otherwise it is replaced by zero.
            auto first_guess = boost::accumulators::count(acc)
                                   ? boost::accumulators::mean(acc)
                                   : Type(0);
            for (int64_t ix = 0; ix < grid.rows(); ++ix) {
              if (mask(ix, iy)) {
                grid(ix, iy) = first_guess;
              }
            }
          }
        } catch (...) {
          except = std::current_exception();
        }
      },
      num_threads);

  if (except != nullptr) {
    std::rethrow_exception(except);
  }
}

// Get the indexes that frame a given index.
inline auto frame_index(const int64_t index, const int64_t size,
                        const bool is_angle, std::vector<int64_t> &frame)
    -> void {
  // Index in the center of the window
  auto center = static_cast<int64_t>(frame.size() / 2);

  for (int64_t ix = 0; ix < static_cast<int64_t>(frame.size()); ++ix) {
    auto idx = index - center + ix;

    // Normalizing longitude?
    if (is_angle) {
      idx = math::remainder(idx, size);
    } else {
      // Otherwise, the symmetrical indexes are used if the indexes are outside
      // the domain definition.
      if (idx < 0 || idx >= size) {
        // Special case: if size == 1, all indices map to 0
        if (size == 1) {
          idx = 0;
        } else {
          auto where = math::remainder(idx, (size - 1) * 2);
          if (where >= size) {
            idx = size - 2 - math::remainder(where, size);
          } else {
            idx = math::remainder(where, size);
          }
        }
      }
    }
    frame[ix] = idx;
  }
}

/// @brief Checking the size of the filter window.
///
/// @param[in] name1 Name of the first window parameter.
/// @param[in] size1 Size of the first window parameter.
/// @throw std::invalid_argument if size1 < 1.
constexpr auto check_windows_size(const std::string &name1, const uint32_t size)
    -> void {
  if (size < 1) {
    throw std::invalid_argument(name1 + " must be >= 1");
  }
}

/// @brief Checking the size of the filter window.
///
/// @tparam Args Variadic arguments.
/// @param[in] name1 Name of the first window parameter.
/// @param[in] size Size of the first window parameter.
/// @param[in] args Remaining arguments (name, size) pairs.
/// @throw std::invalid_argument if any size < 1.
template <typename... Args>
constexpr auto check_windows_size(const std::string &name1, uint32_t size,
                                  Args... args) -> void {
  check_windows_size(name1, size);
  check_windows_size(args...);
}

}  // namespace pyinterp::fill
