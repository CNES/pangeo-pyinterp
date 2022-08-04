// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <algorithm>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "pyinterp/detail/math.hpp"
#include "pyinterp/detail/thread.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/grid.hpp"

namespace pyinterp {
namespace detail {

/// Calculate the zonal average in x direction
///
/// @param grid The grid to be processed.
/// @param mask Matrix describing the undefined pixels of the grid providedNaN
/// NumberReplaces all missing (_FillValue) values in a grid with values derived
/// from solving Poisson's equation via relaxation. of threads used for the
/// calculation
///
/// @param grid
template <typename Type>
void set_zonal_average(pybind11::EigenDRef<Matrix<Type>> &grid,
                       Matrix<bool> &mask, const size_t num_threads) {
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

///  Replaces all undefined values (NaN) in a grid using the Gauss-Seidel
///  method by relaxation.
///
/// @param grid The grid to be processed
/// @param is_circle True if the X axis of the grid defines a circle.
/// @param relaxation Relaxation constant
/// @return maximum residual value
template <typename Type>
auto gauss_seidel(pybind11::EigenDRef<pyinterp::Matrix<Type>> &grid,
                  Matrix<bool> &mask, const bool is_circle,
                  const Type relaxation, const size_t num_threads) -> Type {
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

/// Type of first guess grid.
enum FirstGuess {
  kZero,          //!< Use 0.0 as an initial guess
  kZonalAverage,  //!< Use zonal average in x direction
};

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
auto gauss_seidel(pybind11::EigenDRef<Matrix<Type>> &grid,
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
      detail::set_zonal_average(grid, mask, num_threads);
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

/// Type of values processed by the Loess filter.
enum ValueType {
  kUndefined,  //!< Undefined values (fill undefined values)
  kDefined,    //!< Defined values (smooth values)
  kAll         //!< Smooth and fill values
};

/// Fills undefined values using a locally weighted regression function or
/// LOESS. The weight function used for LOESS is the tri-cube weight
/// function, w(x)=(1-|d|^{3})^{3}
///
/// @param grid Grid Function on a uniform 2-dimensional grid to be filled.
/// @param nx Number of points of the half-window to be taken into account
/// along the longitude axis.
/// @param nx Number of points of the half-window to be taken into account
/// along the latitude axis.
/// @param value_type Type of values processed by the filter
/// @param num_threads The number of threads to use for the computation. If
/// 0 all CPUs are used. If 1 is given, no parallel computing code is used
/// at all, which is useful for debugging.
/// @return The grid will have all the NaN filled with extrapolated values.
template <typename Type>
auto loess(const Grid2D<Type> &grid, const uint32_t nx, const uint32_t ny,
           const ValueType value_type, const size_t num_threads)
    -> pybind11::array_t<Type> {
  check_windows_size("nx", nx, "ny", ny);
  auto result = pybind11::array_t<Type>(
      pybind11::array::ShapeContainer{grid.x()->size(), grid.y()->size()});
  auto _result = result.template mutable_unchecked<2>();

  // Captures the detected exceptions in the calculation function
  // (only the last exception captured is kept)
  auto except = std::exception_ptr(nullptr);

  auto worker = [&](const size_t start, const size_t end) {
    try {
      // Access to the shared pointer outside the loop to avoid data races
      const auto &x_axis = *grid.x();
      const auto &y_axis = *grid.y();
      auto x_frame = std::vector<int64_t>(nx * 2 + 1);
      auto y_frame = std::vector<int64_t>(ny * 2 + 1);

      for (size_t ix = start; ix < end; ++ix) {
        auto x = x_axis(ix);

        // We retrieve the indexes framing the current value.
        frame_index(ix, x_axis.size(), x_axis.is_angle(), x_frame);

        // Read the first value of the calculated window.
        const auto x0 = x_axis(x_frame[0]);

        // The current value is normalized to the first value in the
        // window.
        if (x_axis.is_angle()) {
          x = detail::math::normalize_angle(x, x0, 360.0);
        }

        for (int64_t iy = 0; iy < y_axis.size(); ++iy) {
          auto z = grid.value(ix, iy);

          // If the current value is masked.
          const auto undefined = std::isnan(z);
          if (value_type == kAll || (value_type == kDefined && !undefined) ||
              (value_type == kUndefined && undefined)) {
            auto y = y_axis(iy);

            // We retrieve the indexes framing the current value.
            frame_index(iy, y_axis.size(), false, y_frame);

            // Initialization of values to calculate the extrapolated
            // value.
            auto value = Type(0);
            auto weight = Type(0);

            // For all the coordinates of the frame.
            for (auto wx : x_frame) {
              auto xi = x_axis(wx);

              // We normalize the window's coordinates to its first value.
              if (x_axis.is_angle()) {
                xi = detail::math::normalize_angle(xi, x0, 360.0);
              }

              for (auto wy : y_frame) {
                auto zi = grid.value(wx, wy);

                // If the value is not masked, its weight is calculated from
                // the tri-cube weight function
                if (!std::isnan(zi)) {
                  const auto power = 3.0;
                  auto d =
                      std::sqrt(detail::math::sqr(((xi - x)) / nx) +
                                detail::math::sqr(((y_axis(wy) - y)) / ny));
                  auto wi = d <= 1 ? std::pow((1.0 - std::pow(d, power)), power)
                                   : 0.0;
                  value += static_cast<Type>(wi * zi);
                  weight += static_cast<Type>(wi);
                }
              }
            }

            // Finally, we calculate the extrapolated value if possible,
            // otherwise we will recopy the masked original value.
            if (weight != 0) {
              z = value / weight;
            }
          }
          _result(ix, iy) = z;
        }
      }
    } catch (...) {
      except = std::current_exception();
    }
  };

  {
    pybind11::gil_scoped_release release;
    detail::dispatch(worker, grid.x()->size(), num_threads);
  }
  return result;
}

template <typename Type, typename GridType, typename... Index>
auto loess_(const GridType &grid, const uint32_t nx, const uint32_t ny,
            const ValueType value_type, const Axis<double> &x_axis,
            const Axis<double> &y_axis, const std::vector<int64_t> &x_frame,
            std::vector<int64_t> &y_frame, const double x0, const double x,
            const int64_t ix, const int64_t iy, Index &&...index) -> Type {
  auto z = grid.value(ix, iy, index...);

  // If the current value is masked.
  const auto undefined = std::isnan(z);
  if (value_type == kAll || (value_type == kDefined && !undefined) ||
      (value_type == kUndefined && undefined)) {
    auto y = y_axis(iy);

    // We retrieve the indexes framing the current value.
    frame_index(iy, y_axis.size(), false, y_frame);

    // Initialization of values to calculate the extrapolated
    // value.
    auto value = Type(0);
    auto weight = Type(0);

    // For all the coordinates of the frame.
    for (auto wx : x_frame) {
      auto xi = x_axis(wx);

      // We normalize the window's coordinates to its first value.
      if (x_axis.is_angle()) {
        xi = detail::math::normalize_angle(xi, x0, 360.0);
      }

      for (auto wy : y_frame) {
        auto zi = grid.value(wx, wy, index...);

        // If the value is not masked, its weight is calculated
        // from the tri-cube weight function
        if (!std::isnan(zi)) {
          const auto power = 3.0;
          auto d = std::sqrt(detail::math::sqr(((xi - x)) / nx) +
                             detail::math::sqr(((y_axis(wy) - y)) / ny));
          auto wi = d <= 1 ? std::pow((1.0 - std::pow(d, power)), power) : 0.0;
          value += static_cast<Type>(wi * zi);
          weight += static_cast<Type>(wi);
        }
      }
    }
    // Finally, we calculate the extrapolated value if possible,
    // otherwise we will recopy the masked original value.
    if (weight != 0) {
      z = value / weight;
    }
  }
  return z;
}

template <typename Type, typename AxisType>
auto loess(const Grid3D<Type, AxisType> &grid, const uint32_t nx,
           const uint32_t ny, const ValueType value_type,
           const size_t num_threads) -> pybind11::array_t<Type> {
  check_windows_size("nx", nx, "ny", ny);
  auto result = pybind11::array_t<Type>(pybind11::array::ShapeContainer{
      grid.x()->size(), grid.y()->size(), grid.z()->size()});
  auto _result = result.template mutable_unchecked<3>();

  // Captures the detected exceptions in the calculation function
  // (only the last exception captured is kept)
  auto except = std::exception_ptr(nullptr);

  auto worker = [&](const size_t start, const size_t end) {
    try {
      // Access to the shared pointer outside the loop to avoid data races
      const auto &x_axis = *grid.x();
      const auto &y_axis = *grid.y();
      auto x_frame = std::vector<int64_t>(nx * 2 + 1);
      auto y_frame = std::vector<int64_t>(ny * 2 + 1);

      for (size_t iz = start; iz < end; ++iz) {
        for (int64_t ix = 0; ix < x_axis.size(); ++ix) {
          auto x = x_axis(ix);

          // We retrieve the indexes framing the current value.
          frame_index(ix, x_axis.size(), x_axis.is_angle(), x_frame);

          // Read the first value of the calculated window.
          const auto x0 = x_axis(x_frame[0]);

          // The current value is normalized to the first value in the
          // window.
          if (x_axis.is_angle()) {
            x = detail::math::normalize_angle(x, x0, 360.0);
          }

          for (int64_t iy = 0; iy < y_axis.size(); ++iy) {
            _result(ix, iy, iz) = loess_<Type, Grid3D<Type, AxisType>>(
                grid, nx, ny, value_type, x_axis, y_axis, x_frame, y_frame, x0,
                x, ix, iy, iz);
          }
        }
      }
    } catch (...) {
      except = std::current_exception();
    }
  };

  {
    pybind11::gil_scoped_release release;
    detail::dispatch(worker, grid.z()->size(), num_threads);
  }
  return result;
}

template <typename Type, typename AxisType>
auto loess(const Grid4D<Type, AxisType> &grid, const uint32_t nx,
           const uint32_t ny, const ValueType value_type,
           const size_t num_threads) -> pybind11::array_t<Type> {
  check_windows_size("nx", nx, "ny", ny);
  auto result = pybind11::array_t<Type>(pybind11::array::ShapeContainer{
      grid.x()->size(), grid.y()->size(), grid.z()->size(), grid.u()->size()});
  auto _result = result.template mutable_unchecked<4>();

  // Captures the detected exceptions in the calculation function
  // (only the last exception captured is kept)
  auto except = std::exception_ptr(nullptr);

  auto worker = [&](const size_t start, const size_t end) {
    try {
      // Access to the shared pointer outside the loop to avoid data races
      const auto &x_axis = *grid.x();
      const auto &y_axis = *grid.y();
      const auto &z_axis = *grid.z();
      auto x_frame = std::vector<int64_t>(nx * 2 + 1);
      auto y_frame = std::vector<int64_t>(ny * 2 + 1);

      for (size_t iu = start; iu < end; ++iu) {
        for (int64_t ix = 0; ix < x_axis.size(); ++ix) {
          auto x = x_axis(ix);

          // We retrieve the indexes framing the current value.
          frame_index(ix, x_axis.size(), x_axis.is_angle(), x_frame);

          // Read the first value of the calculated window.
          const auto x0 = x_axis(x_frame[0]);

          // The current value is normalized to the first value in the
          // window.
          if (x_axis.is_angle()) {
            x = detail::math::normalize_angle(x, x0, 360.0);
          }

          for (int64_t iy = 0; iy < y_axis.size(); ++iy) {
            for (int64_t iz = 0; iz < z_axis.size(); ++iz) {
              _result(ix, iy, iz, iu) = loess_<Type, Grid4D<Type, AxisType>>(
                  grid, nx, ny, value_type, x_axis, y_axis, x_frame, y_frame,
                  x0, x, ix, iy, iz, iu);
            }
          }
        }
      }
    } catch (...) {
      except = std::current_exception();
    }
  };

  {
    pybind11::gil_scoped_release release;
    detail::dispatch(worker, grid.u()->size(), num_threads);
  }
  return result;
}

}  // namespace fill
}  // namespace pyinterp
