// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <vector>

#include "pyinterp/config/fill.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/fill/helpers.hpp"
#include "pyinterp/parallel_for.hpp"

namespace pyinterp::fill {

/// Supported floating-point types for LOESS operations.
template <typename T>
concept LoessScalar = std::floating_point<T>;

namespace detail {

/// Tri-cube weight function: w(d) = (1 - |d|³)³ for |d| ≤ 1, else 0.
template <LoessScalar T>
[[nodiscard]] constexpr auto tricube_weight(T distance) noexcept -> T {
  if (distance > T{1}) {
    return T{0};
  }
  // (1 - d^3)^3
  const auto d3 = distance * distance * distance;
  const auto tmp = T{1} - d3;
  return tmp * tmp * tmp;
}

/// Determines if a value should be processed based on its defined status.
template <LoessScalar T>
[[nodiscard]] constexpr auto should_process(
    T value, config::fill::LoessValueType value_type) noexcept -> bool {
  const bool is_undefined = std::isnan(value);
  switch (value_type) {
    case config::fill::LoessValueType::kAll:
      return true;
    case config::fill::LoessValueType::kDefined:
      return !is_undefined;
    case config::fill::LoessValueType::kUndefined:
      return is_undefined;
  }
  std::unreachable();
}

/// Thread-local workspace for LOESS computation.
struct LoessWorkspace {
  std::vector<std::int64_t> x_frame;
  std::vector<std::int64_t> y_frame;

  LoessWorkspace(std::uint32_t nx, std::uint32_t ny)
      : x_frame(2 * static_cast<std::size_t>(nx) + 1),
        y_frame(2 * static_cast<std::size_t>(ny) + 1) {}
};

/// Compute LOESS value for a single point.
/// @param[in] data_values Matrix containing the values for neighbor
/// calculation.
/// @param[in] config LOESS configuration parameters.
/// @param[in] workspace Thread-local workspace for computations.
/// @param[in] ix Row index of the target point.
/// @param[in] iy Column index of the target point.
/// @param[in] current_value Current value at (ix, iy) to return if no neighbors
/// found.
/// @return Computed LOESS value for the point at (ix, iy).
template <LoessScalar T, typename Derived>
[[nodiscard]] auto loess_point(const Eigen::MatrixBase<Derived>& data_values,
                               const config::fill::Loess& config,
                               const LoessWorkspace& workspace,
                               const int64_t ix, const int64_t iy,
                               const T current_value) -> T {
  T weighted_sum{0};
  T weight_sum{0};

  const auto nx_inv = T{1} / static_cast<T>(config.nx());
  const auto ny_inv = T{1} / static_cast<T>(config.ny());

  for (const auto wx : workspace.x_frame) {
    for (const auto wy : workspace.y_frame) {
      const auto zi = data_values(wx, wy);

      if (!std::isnan(zi)) {
        // Normalize distances
        const auto dx = static_cast<T>(wx - ix) * nx_inv;
        const auto dy = static_cast<T>(wy - iy) * ny_inv;
        const auto distance = std::sqrt(dx * dx + dy * dy);

        // Apply tri-cube weight
        const auto weight = tricube_weight(distance);

        weighted_sum += weight * zi;
        weight_sum += weight;
      }
    }
  }

  return weight_sum != T{0} ? weighted_sum / weight_sum : current_value;
}

/// Process a single row.
/// @param[in] data_values Matrix providing neighbor values.
/// @param[in] data_validity Matrix providing validity checks (should_process).
/// @param[out] result Matrix to store computed LOESS values.
/// @param[in] config LOESS configuration parameters.
/// @param[in,out] workspace Thread-local workspace for computations.
/// @param[in] ix Row index to process.
template <LoessScalar T, typename Derived1, typename Derived2>
void process_row(const Eigen::MatrixBase<Derived1>& data_values,
                 const Eigen::MatrixBase<Derived2>& data_validity,
                 RowMajorMatrix<T>& result, const config::fill::Loess& config,
                 LoessWorkspace& workspace, const int64_t ix) {
  const auto num_rows = data_values.rows();
  const auto num_cols = data_values.cols();

  // Build frame indices for x-axis
  frame_index(ix, num_rows, config.is_periodic(), workspace.x_frame);

  // Process all columns for this row
  for (int64_t iy = 0; iy < num_cols; ++iy) {
    // Check validity against the specific validity matrix
    if (!should_process(data_validity(ix, iy), config.value_type())) {
      result(ix, iy) = data_values(ix, iy);
      continue;
    }

    frame_index(iy, num_cols, /*is_angle=*/false, workspace.y_frame);
    result(ix, iy) = loess_point<T>(data_values, config, workspace, ix, iy,
                                    data_values(ix, iy));
  }
}

/// Compute zonal average (mean of all defined values).
template <LoessScalar T>
[[nodiscard]] auto compute_zonal_average(const RowMajorMatrix<T>& data) -> T {
  const auto valid_mask = data.array().isFinite();
  const auto count = valid_mask.count();

  if (count == 0) {
    return T{0};
  }

  const auto sum = valid_mask.select(data.array(), T{0}).sum();
  return sum / static_cast<T>(count);
}

/// Apply initial guess to undefined values.
template <LoessScalar T>
void apply_first_guess(RowMajorMatrix<T>& data,
                       config::fill::FirstGuess first_guess) {
  const T fill_value = (first_guess == config::fill::FirstGuess::kZonalAverage)
                           ? compute_zonal_average(data)
                           : T{0};

  data.array() = data.array().isNaN().select(fill_value, data.array());
}

/// Compute maximum absolute difference between two matrices.
template <LoessScalar T>
[[nodiscard]] auto compute_max_difference(const RowMajorMatrix<T>& current,
                                          const RowMajorMatrix<T>& previous)
    -> T {
  const auto diff = (current.array() - previous.array()).abs();
  const auto valid_mask = !diff.isNaN();

  if (valid_mask.count() == 0) {
    return T{0};
  }

  return valid_mask.select(diff, T{0}).maxCoeff();
}

/// Single-pass LOESS processing.
/// @param[in] data_values Matrix containing the values for neighbor
/// calculation.
/// @param[in] data_validity Matrix providing validity checks (should_process).
/// @param[out] result Matrix to store computed LOESS values.
/// @param[in] config LOESS configuration parameters.
template <LoessScalar T, typename Derived1, typename Derived2>
void loess_single_pass(const Eigen::MatrixBase<Derived1>& data_values,
                       const Eigen::MatrixBase<Derived2>& data_validity,
                       RowMajorMatrix<T>& result,
                       const config::fill::Loess& config) {
  parallel_for(
      data_values.rows(),
      [&](std::int64_t start, std::int64_t end) {
        LoessWorkspace workspace(config.nx(), config.ny());
        for (int64_t ix = start; ix < end; ++ix) {
          process_row(data_values, data_validity, result, config, workspace,
                      ix);
        }
      },
      config.num_threads());
}

}  // namespace detail

/// Fills undefined values using locally weighted regression (LOESS).
///
/// The weight function is the tri-cube: w(d) = (1 - |d|³)³
///
/// @tparam T Floating-point scalar type
/// @param[in] data Input matrix to process
/// @param[in] nx Half-window size along x-axis (rows)
/// @param[in] ny Half-window size along y-axis (columns)
/// @param[in] value_type Which values to process
/// @param[in] config LOESS configuration
/// @return New matrix with processed values
template <LoessScalar T>
[[nodiscard]] auto loess(const EigenDRef<const RowMajorMatrix<T>>& data,
                         const config::fill::Loess& config)
    -> RowMajorMatrix<T> {
  RowMajorMatrix<T> result(data);

  // Single Pass Filling
  if (config.max_iterations() == 1) {
    detail::loess_single_pass(data, data, result, config);
    return result;
  }

  // Iterative Filling

  // Apply first guess (replaces NaNs with 0 or Zonal Average)
  // 'result' now contains NO NaNs.
  detail::apply_first_guess(result, config.first_guess());

  // If max_iterations is 0, we just return the guess.
  if (config.max_iterations() == 0) {
    return result;
  }

  RowMajorMatrix<T> previous;

  for (uint32_t iter = 0; iter < config.max_iterations(); ++iter) {
    previous = result;

    detail::loess_single_pass(previous, data, result, config);

    // Check convergence
    if (detail::compute_max_difference(result, previous) <
        static_cast<T>(config.epsilon())) {
      break;
    }
  }

  return result;
}

}  // namespace pyinterp::fill
