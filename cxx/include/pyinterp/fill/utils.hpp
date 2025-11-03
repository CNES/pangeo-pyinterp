#pragma once

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "pyinterp/detail/math.hpp"
#include "pyinterp/detail/thread.hpp"
#include "pyinterp/eigen.hpp"

namespace pyinterp::fill {
/// Calculate the zonal average in x direction
///
/// @param grid The grid to be processed.
/// @param mask Matrix describing the undefined pixels of the grid providedNaN
/// NumberReplaces all missing (_FillValue) values in a grid with values derived
/// from solving Poisson's equation via relaxation. of threads used for the
/// calculation
template <typename Type, typename Derived>
void set_zonal_average(Eigen::RefBase<Derived> &grid_ref, Matrix<bool> &mask,
                       const size_t num_threads) {
  auto &grid = grid_ref.derived();
  static_assert(std::is_same_v<typename Derived::Scalar, Type>,
                "Type mismatch");

  // Captures the detected exceptions in the calculation function
  // (only the last exception captured is kept)
  auto except = std::exception_ptr(nullptr);

  detail::dispatch(
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
      grid.cols(), num_threads);

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
      idx = detail::math::remainder(idx, size);
    } else {
      // Otherwise, the symmetrical indexes are used if the indexes are outside
      // the domain definition.
      if (idx < 0 || idx >= size) {
        auto where = detail::math::remainder(idx, (size - 1) * 2);
        if (where >= size) {
          idx = size - 2 - detail::math::remainder(where, size);
        } else {
          idx = detail::math::remainder(where, size);
        }
      }
    }
    frame[ix] = idx;
  }
}

/// Checking the size of the filter window.
constexpr auto check_windows_size(const std::string &name1, const uint32_t size)
    -> void {
  if (size < 1) {
    throw std::invalid_argument(name1 + " must be >= 1");
  }
}

/// Checking the size of the filter window.
template <typename... Args>
constexpr auto check_windows_size(const std::string &name1, uint32_t size,
                                  Args... args) -> void {
  check_windows_size(name1, size);
  check_windows_size(args...);
}

}  // namespace pyinterp::fill
